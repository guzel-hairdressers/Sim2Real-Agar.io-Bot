"""Reproducible imitation + PPO and outcome-supervised diffusion training.

Run as a module: python -m src.train_arena_v2 --output outputs/arena_v2
No champion files are overwritten. Evaluation/promotion is a separate command.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import multiprocessing as mp
import time
import copy
import numpy as np
import torch
from torch import nn

from envs.arena_config import ArenaConfig, OBSERVATION_DIM, event_reward
from src.arena_tactics import teacher_action, safety_filter
from src.train_champion_ppo import HybridActorCritic
from src.agar_diffusion_policy import TorchAgarDiffusionNet, TorchAgarCritic, AgarDiffusionSchedule


def training_tactics(version):
    if version == 'initial_v2':
        from benchmarks.training_tactics_v2_initial import teacher_action, safety_filter
    elif version == 'current':
        from src.arena_tactics import teacher_action, safety_filter
    else:
        raise ValueError(version)
    return teacher_action, safety_filter


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def curriculum(env, rng):
    """Half of training resets include larger cells; keep the mass budget."""
    if rng.random() < .5:
        masses = rng.uniform(15, 70, env.num_players)
        masses *= 300 / masses.sum()
        for player, mass in zip(env.players.values(), masses):
            player.mass = float(mass)
            player.peak_mass = float(mass)
        env.reserve_mass = env.total_world_mass_cap - sum(p.mass for p in env.players.values()) - len(env.food_positions) * env.pellet_mass
    return {p: env._get_obs(p) for p in env.player_ids}


def harvest_episode(args):
    seed, steps, version = args
    teacher_action, _ = training_tactics(version)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    env = ArenaConfig(max_steps=steps).make()
    env.reset(seed=seed)
    obs = curriculum(env, rng)
    records = []
    previous = {p: {'kills': 0, 'deaths': 0} for p in env.player_ids}
    for _ in range(steps):
        actions = {p: teacher_action(o) for p, o in obs.items()}
        next_obs, rewards, _, _, infos = env.step(actions)
        values = [event_reward(previous[p], infos[p], rewards[p]) for p in env.player_ids]
        deaths = [infos[p]['deaths'] > previous[p]['deaths'] for p in env.player_ids]
        records.append((np.array(list(obs.values())), np.array(list(actions.values())), values, deaths))
        previous, obs = infos, next_obs
    observations = np.asarray([r[0] for r in records], np.float32)
    actions = np.asarray([r[1] for r in records], np.float32)
    rewards = np.asarray([r[2] for r in records], np.float32)
    deaths = np.asarray([r[3] for r in records], bool)
    chunks, conditions, returns = [], [], []
    for t in range(0, steps - 16 + 1, 3):
        for p in range(env.num_players):
            # Never teach a plan that crosses a player respawn.
            if not deaths[t:t+16, p].any():
                chunks.append(actions[t:t+16, p])
                conditions.append(observations[t, p])
                returns.append(float(np.dot(rewards[t:t+16, p], .99 ** np.arange(16))))
    return observations.reshape(-1, OBSERVATION_DIM), actions.reshape(-1, 3), np.array(conditions), np.array(chunks), np.array(returns)


def save_model(path, net, algorithm, metadata):
    package = dict(schema_version=2, observation_version=2, obs_dim=OBSERVATION_DIM,
        algorithm=algorithm, arena_config=asdict(ArenaConfig()), training=metadata,
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()})
    torch.save(package, path)


def gae(rewards, values, next_values, deaths, truncations, gamma=.99, lam=.95):
    advantage = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1:], np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1-deaths[t]) - values[t]
        carry = delta + gamma * lam * (1-deaths[t]) * (1-truncations[t]) * carry
        advantage[t] = carry
    return advantage, advantage + values


def worker(pipe, seed, version):
    from src.arena_policies import LegacyPolicy
    from src.heuristic_agent import MasterHeuristicAgarBot
    teacher_action, _ = training_tactics(version)
    seed_all(seed)
    rng = np.random.default_rng(seed)
    env = ArenaConfig().make()
    obs, _ = env.reset(seed=seed)
    obs = curriculum(env, rng)
    previous = {p: {'kills': 0, 'deaths': 0} for p in env.player_ids}
    legacy = LegacyPolicy('ppo')
    profiles = [MasterHeuristicAgarBot(profile='apex'), MasterHeuristicAgarBot(profile='hunter')]
    pipe.send(np.array(list(obs.values())))
    while True:
        message = pipe.recv()
        if message is None:
            break
        actions, learners = message
        opponents = [i for i in range(7) if i not in learners]
        for j, i in enumerate(opponents):
            p = env.player_ids[i]
            actions[p] = teacher_action(obs[p]) if j == 0 else (legacy.predict(obs[p]) if j == 3 else profiles[j-1].predict(obs[p]))
        next_obs, reward, _, trunc, infos = env.step(actions)
        rews = [event_reward(previous[p], infos[p], reward[p]) for p in env.player_ids]
        deaths = [infos[p]['deaths'] > previous[p]['deaths'] for p in env.player_ids]
        final_obs = np.array(list(next_obs.values()))
        done = any(trunc.values())
        if done:
            next_obs, _ = env.reset()
            next_obs = curriculum(env, rng)
            previous = {p: {'kills': 0, 'deaths': 0} for p in env.player_ids}
        else:
            previous = infos
        obs = next_obs
        pipe.send((np.array(list(obs.values())), final_obs, np.array(rews), np.array(deaths), done))
    pipe.close()


def train_ppo(net, device, output, args, metadata):
    teacher_action, safety_filter = training_tactics(args.teacher_version)
    ctx = mp.get_context('spawn')
    parents, processes = [], []
    for w in range(args.workers):
        parent, child = ctx.Pipe()
        process = ctx.Process(target=worker, args=(child, args.seed + 5000 + w, args.teacher_version))
        process.start()
        child.close()
        parents.append(parent); processes.append(process)
    obs = np.array([p.recv() for p in parents])
    learners = np.array([[(w + k*2) % 7 for k in range(3)] for w in range(args.workers)])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, eps=1e-5)
    total, updates = 0, 0
    started = time.monotonic()
    try:
        while total < args.steps:
            records = []
            for _ in range(128):
                batch = obs[np.arange(args.workers)[:, None], learners].reshape(-1, OBSERVATION_DIM)
                with torch.no_grad():
                    action, logp, _, value = net.get_action_and_value(torch.as_tensor(batch, device=device))
                raw = action.cpu().numpy()
                safe = [safety_filter(a, o) for a, o in zip(raw, batch)]
                for w, pipe in enumerate(parents):
                    pipe.send(({f'player_{i}': safe[w*3+k] for k, i in enumerate(learners[w])}, learners[w].tolist()))
                result = [p.recv() for p in parents]
                final = np.array([r[1] for r in result])[np.arange(args.workers)[:, None], learners].reshape(-1, OBSERVATION_DIM)
                reward = np.array([r[2] for r in result])[np.arange(args.workers)[:, None], learners].reshape(-1)
                dead = np.array([r[3] for r in result])[np.arange(args.workers)[:, None], learners].reshape(-1)
                trunc = np.repeat([r[4] for r in result], 3)
                records.append((batch, raw, logp.cpu().numpy(), value.cpu().numpy(), reward, dead, trunc, final))
                obs = np.array([r[0] for r in result])
                total += len(batch)
            arrays = [np.asarray([r[k] for r in records]) for k in range(8)]
            shape = arrays[4].shape
            with torch.no_grad():
                next_values = net(torch.as_tensor(arrays[7].reshape(-1, OBSERVATION_DIM), device=device))[-1].cpu().numpy().reshape(shape)
            advantages, returns = gae(arrays[4], arrays[3], next_values, arrays[5], arrays[6])
            conditions = arrays[0].reshape(-1, OBSERVATION_DIM)
            inputs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in
                      (conditions, arrays[1].reshape(-1, 3), arrays[2].reshape(-1), advantages.reshape(-1), returns.reshape(-1))]
            b_o, b_a, b_lp, adv, ret = inputs
            adv = (adv-adv.mean()) / (adv.std()+1e-8)
            demonstrations = torch.as_tensor(np.array([teacher_action(o) for o in conditions]), device=device)
            for _ in range(4):
                for indices in torch.randperm(len(b_o), device=device).split(256):
                    lp, entropy, value = net.evaluate_actions(b_o[indices], b_a[indices])
                    ratio = (lp-b_lp[indices]).exp()
                    objective = torch.minimum(ratio*adv[indices], ratio.clamp(.8, 1.2)*adv[indices])
                    mu, _, split, _ = net(b_o[indices])
                    imitation = nn.functional.mse_loss(mu, demonstrations[indices, :2])
                    loss = -objective.mean() + .25*nn.functional.mse_loss(value, ret[indices]) - .005*entropy.mean() + .05*imitation
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), .5); optimizer.step()
            updates += 1
            if updates % 10 == 0 or total >= args.steps:
                print(json.dumps({'stage': 'ppo', 'steps': total, 'reward': float(arrays[4].mean()),
                    'deaths': int(arrays[5].sum()), 'steps_per_second': round(total/(time.monotonic()-started), 1)}), flush=True)
                save_model(output/'ppo.pt', net, 'ppo', {**metadata, 'learner_steps': total})
    finally:
        for p in parents:
            try:
                p.send(None)
            except (BrokenPipeError, EOFError):
                pass
        for p in processes:
            p.join(timeout=10)
            if p.is_alive(): p.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/arena_v2'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=24)
    parser.add_argument('--episode-steps', type=int, default=600)
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--seed', type=int, default=260271)
    parser.add_argument('--bc-epochs', type=int, default=15)
    parser.add_argument('--diffusion-epochs', type=int, default=25)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--diffusion-only', action='store_true')
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='cosine')
    parser.add_argument('--symmetric-actions', action='store_true')
    parser.add_argument('--teacher-version', choices=['initial_v2', 'current'], default='initial_v2')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(args.device)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output/'training_config.json').write_text(json.dumps(metadata, indent=2))
    data_path = args.data or args.output/'demonstrations.npz'
    if not args.reuse_data:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
            results = list(pool.map(harvest_episode, [(args.seed+i, args.episode_steps, args.teacher_version) for i in range(args.episodes)]))
        # Split whole episodes before building overlapping trajectory chunks.
        split = max(1, int(len(results)*.8))
        keys = ['obs', 'acts', 'conditions', 'chunks', 'returns']
        dataset = {prefix+k: np.concatenate([r[j] for r in group]).astype(np.float32)
                   for prefix, group in [('', results[:split]), ('val_', results[split:])]
                   for j, k in enumerate(keys)}
        np.savez_compressed(data_path, **dataset)
    data = np.load(data_path)
    net = HybridActorCritic(OBSERVATION_DIM).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=6e-4)
    obs = torch.as_tensor(data['obs'], device=device)
    acts = torch.as_tensor(data['acts'], device=device)
    for epoch in range(0 if args.diffusion_only else args.bc_epochs):
        losses = []
        for idx in torch.randperm(len(obs), device=device).split(512):
            mu, _, logits, _ = net(obs[idx])
            loss = nn.functional.mse_loss(mu, acts[idx, :2]) + .2*nn.functional.binary_cross_entropy_with_logits(logits, acts[idx, 2], pos_weight=torch.tensor(5., device=device))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        print(json.dumps({'stage':'imitation','epoch':epoch+1,'loss':float(np.mean(losses))}), flush=True)
    if not args.diffusion_only:
        save_model(args.output/'ppo_bc.pt', net, 'ppo', metadata)
    if args.steps and not args.diffusion_only:
        train_ppo(net, device, args.output, args, metadata)
    elif not args.diffusion_only:
        save_model(args.output/'ppo.pt', net, 'ppo', metadata)
    # Denoiser and critic learn the same physically timed demonstration chunks.
    # Critic labels are actual discounted 16-step environment rewards, not a
    # hand-written function of the first observation/action.
    diff = TorchAgarDiffusionNet(obs_dim=OBSERVATION_DIM).to(device)
    ema = copy.deepcopy(diff)
    critic = TorchAgarCritic(obs_dim=OBSERVATION_DIM).to(device)
    optimizer = torch.optim.AdamW(diff.parameters(), lr=5e-4)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=5e-4)
    cond = torch.as_tensor(data['conditions'], device=device)
    chunks = torch.as_tensor(data['chunks'], device=device)
    if args.symmetric_actions:
        chunks = chunks.clone()
        chunks[..., 0] = chunks[..., 0]*2-1
        chunks[..., 2] = chunks[..., 2]*2-1
    targets = torch.as_tensor(data['returns'], device=device)
    alpha = torch.as_tensor(AgarDiffusionSchedule(schedule=args.schedule).alphas_cumprod, dtype=torch.float32, device=device)
    for epoch in range(args.diffusion_epochs):
        losses, critic_losses = [], []
        for idx in torch.randperm(len(cond), device=device).split(256):
            t = torch.randint(100, (len(idx),), device=device)
            noise = torch.randn_like(chunks[idx]); a = alpha[t, None, None]
            noisy = a.sqrt()*chunks[idx] + (1-a).sqrt()*noise
            predicted = diff(noisy, t[:, None]/100, cond[idx])
            loss = nn.functional.mse_loss(predicted, noise)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(diff.parameters(), 1); optimizer.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), diff.parameters()):
                    averaged.lerp_(current, .05)
            physical_chunks = torch.as_tensor(data['chunks'][idx.cpu().numpy()], device=device)
            qloss = nn.functional.smooth_l1_loss(critic(cond[idx], physical_chunks), targets[idx])
            critic_optimizer.zero_grad(); qloss.backward(); critic_optimizer.step()
            losses.append(loss.item()); critic_losses.append(qloss.item())
        with torch.no_grad():
            vcond = torch.as_tensor(data['val_conditions'], device=device)
            vchunk = torch.as_tensor(data['val_chunks'], device=device)
            vtarget = torch.as_tensor(data['val_returns'], device=device)
            prediction = critic(vcond, vchunk)
            validation = float(((prediction-vtarget)**2).mean())
        print(json.dumps({'stage':'diffusion','epoch':epoch+1,'noise_loss':float(np.mean(losses)),
            'critic_loss':float(np.mean(critic_losses)), 'critic_validation_mse': validation}), flush=True)
    save_model(args.output/'diffusion.pt', ema, 'diffusion', metadata)
    save_model(args.output/'critic.pt', critic, 'return_critic', {**metadata, 'target':'observed_discounted_16_step_return'})


if __name__ == '__main__':
    main()
