"""Long-horizon PPO split learning with frozen historical movement.

Only the discrete split head and value function learn. Movement assistance
is explicit and unchanged; it cannot veto or invent a feasible split.
"""
import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Bernoulli, kl_divergence

from src.arena_policies import load_package
from src.benchmark_showcase import ROOT
from src.learned_risk import physical_split_filter
from src.train_arena_v2 import gae
from src.train_champion_ppo import HybridActorCritic


def worker(pipe, index, seed):
    # This dependency substitution is confined to a spawned child. Reuse the
    # exact arena/curriculum/reward collector, changing only action execution.
    from src import train_balanced_ppo as rollout
    original = rollout.physical_action
    try:
        rollout.physical_action = physical_split_filter
        rollout.worker(pipe, index, seed)
    finally:
        rollout.physical_action = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT / 'outputs/learned_risk_v8/ppo.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--seed', type=int, default=6300001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    package = load_package(args.base, 'ppo')
    net = HybridActorCritic(70, split_architecture='obs_mlp')
    net.load_state_dict(package['state_dict'])
    reference = copy.deepcopy(net).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    head = list(net.split_head.parameters())
    critic = list(net.critic_trunk.parameters()) + list(net.critic_head.parameters())
    for parameter in head + critic:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': head, 'lr': 1e-4},
                                  {'params': critic, 'lr': 1e-4}], eps=1e-5)
    gamma = math.exp(-(1 / 30) / 20)
    # Align the stochastic policy's mode with the existing .65 deployment
    # threshold, rather than using different decision boundaries in fitting.
    threshold_logit = math.log(.65 / .35)
    sources = ['src/train_balanced_splits.py', 'src/train_balanced_ppo.py',
               'src/learned_risk.py', 'src/arena_policies.py', 'src/arena_tactics.py',
               'src/balanced_policies.py', 'src/agar_diffusion_policy.py',
               'src/train_arena_v2.py', 'src/train_champion_ppo.py', 'src/split_head.py',
               'src/heuristic_agent.py', 'envs/arena_config.py', 'envs/agar_env.py']
    config = dict(seed=args.seed, steps_requested=args.steps, workers=args.workers,
                  rollout=args.rollout, reward_discount_seconds=20, gamma=gamma,
                  gae_lambda=.98, actor_continuous_frozen=True, split_head_frozen=False,
                  movement_control='historical assistance; frozen continuous neural weights',
                  split_control='learned_risk_physical_only',
                  reward='log mass gain + .10 piece captures - 3 full deaths; no respawn growth credit',
                  head_learning_rate=1e-4, critic_learning_rate=1e-4,
                  head_reference_kl_coefficient=.01, entropy_coefficient=.001,
                  threshold_logit=threshold_logit,
                  base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
                  source_hashes={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources})
    (args.output / 'training_config.json').write_text(json.dumps(config, indent=2))
    ctx = mp.get_context('spawn')
    parents, processes, seeds = [], [], set()
    for index in range(args.workers):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker, args=(child, index, args.seed + index * 10000))
        proc.start()
        child.close()
        parents.append(parent)
        processes.append(proc)
    def begin(pipe):
        row = pipe.recv()
        seeds.add(row['episode_seed'])
        return row['obs']
    obs = np.array([begin(pipe) for pipe in parents])
    steps, updates, history = 0, 0, []
    started = time.monotonic()
    try:
        while steps < args.steps:
            records = []
            for _ in range(args.rollout):
                batch = obs.reshape(-1, 70)
                tensor = torch.tensor(batch, dtype=torch.float32)
                with torch.no_grad():
                    mu, _, logits, values = net(tensor)
                    eligible = (tensor[:, 4] > .5).float()
                    distribution = Bernoulli(logits=logits - threshold_logit)
                    splits = distribution.sample() * eligible
                    logp = distribution.log_prob(splits) * eligible
                    actions = torch.cat([mu, splits[:, None]], 1).numpy()
                for index, pipe in enumerate(parents):
                    pipe.send(actions[index * 2:index * 2 + 2])
                results = [pipe.recv() for pipe in parents]
                final = np.array([r['final_obs'] for r in results]).reshape(-1, 70)
                rewards = np.array([r['reward'] for r in results]).reshape(-1)
                deaths = np.array([r['deaths'] for r in results]).reshape(-1)
                truncated = np.repeat([r['truncated'] for r in results], 2)
                records.append((batch, splits.numpy(), logp.numpy(), values.numpy(),
                                rewards, deaths, truncated, final))
                obs = np.array([begin(pipe) if r['truncated'] else r['final_obs']
                                for pipe, r in zip(parents, results)])
                steps += len(batch)
            arrays = [np.asarray([r[k] for r in records]) for k in range(8)]
            with torch.no_grad():
                nv = net(torch.tensor(arrays[7].reshape(-1, 70), dtype=torch.float32))[-1]
            advantages, returns = gae(arrays[4], arrays[3], nv.numpy().reshape(arrays[4].shape),
                                      arrays[5], arrays[6], gamma=gamma, lam=.98)
            observations = torch.tensor(arrays[0].reshape(-1, 70), dtype=torch.float32)
            actions = torch.tensor(arrays[1].reshape(-1), dtype=torch.float32)
            old_logp = torch.tensor(arrays[2].reshape(-1), dtype=torch.float32)
            advantage = torch.tensor(advantages.reshape(-1), dtype=torch.float32)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            target = torch.tensor(returns.reshape(-1), dtype=torch.float32)
            losses = []
            for _ in range(8):
                for ids in torch.randperm(len(observations)).split(256):
                    batch = observations[ids]
                    eligible = (batch[:, 4] > .5).float()
                    denominator = eligible.sum().clamp(min=1)
                    _, _, logits, value = net(batch)
                    distribution = Bernoulli(logits=logits - threshold_logit)
                    lp = distribution.log_prob(actions[ids]) * eligible
                    ratio = (lp - old_logp[ids]).exp()
                    surrogate = torch.minimum(ratio * advantage[ids],
                                              ratio.clamp(.9, 1.1) * advantage[ids])
                    with torch.no_grad():
                        prior = Bernoulli(logits=reference(batch)[2] - threshold_logit)
                    anchor = (kl_divergence(prior, distribution) * eligible).sum() / denominator
                    entropy = (distribution.entropy() * eligible).sum() / denominator
                    loss = -(surrogate * eligible).sum() / denominator
                    loss = loss + .25 * (value - target[ids]).square().mean() + .01 * anchor - .001 * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head + critic, .5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            updates += 1
            row = dict(steps=steps, update=updates, reward=float(arrays[4].mean()),
                       deaths=int(arrays[5].sum()), eligible_frames=int((arrays[0][:, :, 4] > .5).sum()),
                       split_requests=int(arrays[1].sum()), loss=float(np.mean(losses)),
                       steps_per_second=steps / (time.monotonic() - started))
            history.append(row)
            print(json.dumps(row), flush=True)
            if updates % 15 == 0 or steps >= args.steps:
                checkpoint = copy.deepcopy(package)
                checkpoint['state_dict'] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                checkpoint['movement_control'] = 'assisted'
                checkpoint['split_control'] = 'learned_risk_physical_only'
                checkpoint['balanced_split_training'] = {**config, 'steps_completed': steps,
                                                        'episode_seeds': sorted(seeds)}
                torch.save(checkpoint, args.output / f'ppo_{steps}.pt')
                torch.save(checkpoint, args.output / 'ppo.pt')
                (args.output / 'training_history.json').write_text(json.dumps(history, indent=2))
    finally:
        for pipe in parents:
            try:
                pipe.send(None)
            except (BrokenPipeError, EOFError):
                pass
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


if __name__ == '__main__':
    main()
