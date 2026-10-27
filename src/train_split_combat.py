"""Train binary split decisions from paired, physical split/hold trials.

Continuous PPO/diffusion movement weights stay frozen. Positive labels require
an actual projectile consumption, no actor damage, and a benefit over holding.
The existing safety filter is retained without relaxing its thresholds.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from torch import nn

from envs.agar_env import PartiallyObservableAgarEnv
from src.arena_policies import ArenaPPO, ArenaDiffusion
from src.arena_tactics import split_is_safe, safety_filter
from src.split_head import BinarySplitHead
from src.train_champion_ppo import HybridActorCritic

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {}


def initialize_workers():
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    POLICIES['ppo'] = ArenaPPO(ROOT/'outputs/arena_v2/ppo.pt')
    POLICIES['diffusion'] = ArenaDiffusion(ROOT/'outputs/arena_v2_cosine/diffusion.pt', ROOT/'outputs/arena_v2_cosine/critic.pt')


def make_encounter(seed):
    """Randomized mass, motion, alignment, food, walls, viruses and rivals."""
    rng = np.random.default_rng(seed)
    env = PartiallyObservableAgarEnv(arena_size=18, num_players=3, num_food=0,
        num_viruses=0, observation_version=2, dt=1/30, max_steps=90)
    env.reset(seed=seed)
    a, b, rival = env.players.values()
    a.mass = float(rng.uniform(60, 148))
    b.mass = float(rng.uniform(12, min(55, a.mass*.32)))
    angle = float(rng.uniform(-.12, .12))
    distance = float(rng.uniform(1.15, 2.3))
    kind = int(rng.integers(0, 8))
    if kind == 1:  # tempting prey too large for the launched half
        b.mass = float(a.mass*rng.uniform(.48, .78))
    elif kind == 2:
        angle = float(rng.uniform(.22, 1.4)*rng.choice([-1, 1]))
    elif kind == 3:  # split is wasteful when a normal bite already works
        distance = float(rng.uniform(.5, 1.0))
    elif kind == 4:
        a.split_cooldown = float(rng.uniform(.1, .8))
    yaw = float(rng.uniform(-np.pi, np.pi))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    origin = rng.uniform(-4.5, 4.5, 2)
    if kind == 5:
        origin = np.array([7.8, rng.uniform(-6, 6)])
        yaw = 0.
        rotation = np.eye(2)
    a.pos = origin.astype(np.float32); a.yaw = yaw
    target = origin + rotation@np.array([distance*np.cos(angle), distance*np.sin(angle)])
    b.pos = np.clip(target, -8.3, 8.3).astype(np.float32)
    motion = int(rng.integers(0, 4))
    b.yaw = yaw + [0, np.pi, np.pi/2, -np.pi/2][motion]
    target_thrust = float(rng.choice([0., .35, .65, 1.]))
    b.linear_vel = target_thrust*b.max_speed*(1.35 if target_thrust>.88 and b.mass>18 else 1)
    a.linear_vel = float(rng.uniform(0, min(1.3, a.max_speed)))
    rival.mass = float(rng.uniform(15, 110))
    rival.pos = np.array([-8., -8.], np.float32)
    if kind == 6:
        rival.pos = np.clip(origin + rotation@np.array([rng.uniform(1.4, 2.8), rng.uniform(.8, 1.6)]), -8.2, 8.2).astype(np.float32)
        rival.mass = float(max(a.mass*.7, b.mass+10))
    if kind == 7:
        env.viruses = np.array([[*(origin+rotation@np.array([distance*.7, 0])), .65]], np.float32)
    count = int(rng.choice([0, 30, 100, 180]))
    env.food_positions = rng.uniform(-8.5, 8.5, (count, 2)).astype(np.float32)
    env.reserve_mass = max(0, env.total_world_mass_cap-sum(p.mass for p in env.players.values())-count*.5)
    return env, np.array([target_thrust, 0, 0], np.float32), kind


def branch(env, policy, target_action, split, seed, horizon=18):
    env = copy.deepcopy(env)
    policy.reset(seed=seed)
    first_capture = horizon+1
    projectile_capture = False
    damage = False
    consumed = False
    for t in range(horizon):
        obs = env._get_obs('player_0')
        action = policy.predict(obs)
        action[2] = float(split and t == 0)
        # Both branches use identical observation-only movement and guards.
        action = safety_filter(action, obs)
        if t > 0:
            action[2] = 0
        env.step({'player_0':action, 'player_1':target_action,
                  'player_2':np.zeros(3, np.float32)})
        for event in env.predation_events:
            if event['prey'] == 'player_0':
                damage = True
            if event['predator'] == 'player_0' and event['prey'] == 'player_1' and not consumed:
                first_capture = t+1; consumed = True
                projectile_capture = event['projectile_age_steps'] >= 0
    a = env.players['player_0']
    return dict(capture_step=first_capture, projectile_capture=projectile_capture,
                damage=damage, deaths=a.deaths, mass=a.mass)


def collect_batch(job):
    seeds, = job
    results = []
    for seed in seeds:
        env, target, kind = make_encounter(seed)
        if env.total_world_mass > env.total_world_mass_cap + 1e-4:
            continue
        obs = env._get_obs('player_0')
        safe = split_is_safe(obs)
        for algorithm, policy in POLICIES.items():
            # Rejection examples include the full safety contract. All positive
            # examples, and a sample of negative examples, get actual rollouts.
            split = hold = None
            if safe or seed % 10 == 0:
                split = branch(env, policy, target, True, seed)
                hold = branch(env, policy, target, False, seed)
            label = bool(safe and split and split['projectile_capture'] and
                not split['damage'] and not split['deaths'] and
                split['capture_step']+3 <= hold['capture_step'])
            results.append(dict(seed=seed, kind=kind, algorithm=algorithm,
                obs=obs.tolist(), safe=safe, label=label, split=split, hold=hold))
    return results


def train_head(rows, replay, seed, epochs, validation):
    torch.manual_seed(seed); torch.set_num_threads(1)
    observations = np.array([r['obs'] for r in rows], np.float32)
    labels = np.array([r['label'] for r in rows], np.float32)
    # Replay keeps unopposed foraging and missing-prey states negative. Do not
    # relabel any of the original safe opportunities as negative examples.
    replay = replay[np.array([not split_is_safe(o) for o in replay])]
    observations = np.concatenate([observations, replay])
    labels = np.concatenate([labels, np.zeros(len(replay), np.float32)])
    positive = np.flatnonzero(labels > .5)
    negative = np.flatnonzero(labels < .5)
    if len(positive) < 100:
        raise ValueError(f'Insufficient successful split examples: {len(positive)}')
    rng = np.random.default_rng(seed)
    model = BinarySplitHead()
    model.mean.copy_(torch.from_numpy(observations.mean(axis=0)))
    model.scale.copy_(torch.from_numpy(observations.std(axis=0).clip(.04)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    o = torch.from_numpy(observations); y = torch.from_numpy(labels)
    vo = torch.tensor([r['obs'] for r in validation],dtype=torch.float32)
    vy = torch.tensor([r['label'] for r in validation],dtype=torch.float32)
    best_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        losses = []
        for _ in range(40):
            indices = np.concatenate([rng.choice(positive, 128), rng.choice(negative, 128)])
            logits = model(o[indices]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        if epoch%10 == 0 or epoch == epochs-1:
            with torch.inference_mode():
                vloss = nn.functional.binary_cross_entropy_with_logits(model(vo).squeeze(-1),vy,reduction='none')
                balanced = float(.5*vloss[vy>.5].mean()+.5*vloss[vy<.5].mean())
            if balanced < best_loss:
                best_loss = balanced
                best_state = copy.deepcopy(model.state_dict())
            print(json.dumps(dict(stage='split_head', epoch=epoch+1, loss=float(np.mean(losses)),
                validation_balanced_loss=balanced, positive_examples=len(positive), negative_examples=len(negative))), flush=True)
    model.load_state_dict(best_state)
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('outputs/split_combat_v3'))
    p.add_argument('--cases', type=int, default=6000)
    p.add_argument('--seed', type=int, default=2800001)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--reuse-data', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'counterfactuals.jsonl'
    if not args.reuse_data:
        batches = [(list(range(start, min(start+32, args.seed+args.cases))),)
                   for start in range(args.seed, args.seed+args.cases, 32)]
        with path.open('w') as output:
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize_workers) as pool:
                for i, batch in enumerate(pool.map(collect_batch, batches)):
                    for row in batch: output.write(json.dumps(row)+'\n')
                    if i%20 == 0: print(json.dumps({'stage':'collect','batches':i+1,'total':len(batches)}),flush=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Also validate files collected by an earlier worker version. Reject a
    # randomized negative fixture if its initial allocation exceeded the cap.
    valid_seeds = set()
    for seed in set(r['seed'] for r in rows):
        env, _, _ = make_encounter(seed)
        if env.total_world_mass <= env.total_world_mass_cap+1e-4:
            valid_seeds.add(seed)
    rows = [r for r in rows if r['seed'] in valid_seeds]
    base = np.load(ROOT/'outputs/arena_v2/demonstrations.npz')['obs']
    replay = base[np.random.default_rng(args.seed).choice(len(base), 10000, replace=False)]
    metadata = dict(seed=args.seed, cases=args.cases, epochs=args.epochs,
        target='paired projectile capture at least 3 steps earlier, without actor damage',
        movement_frozen=True, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for algorithm, original in [('ppo', ROOT/'outputs/arena_v2/ppo.pt'),
                                ('diffusion', ROOT/'outputs/arena_v2_cosine/diffusion.pt')]:
        subset = [r for r in rows if r['algorithm']==algorithm]
        # Entire cases are separated, not duplicated minibatch rows.
        split = int(args.cases*.8)
        train = [r for r in subset if r['seed'] < args.seed+split]
        validation = [r for r in subset if r['seed'] >= args.seed+split]
        head = train_head(train, replay, args.seed+(algorithm=='diffusion'), args.epochs, validation)
        with torch.inference_mode():
            probabilities = head(torch.tensor([r['obs'] for r in validation])).sigmoid().squeeze(-1).numpy()
        truth = np.array([r['label'] for r in validation])
        proposal = probabilities > .65
        metrics = dict(algorithm=algorithm, train_positive=sum(r['label'] for r in train),
            validation_positive=int(truth.sum()), validation_proposals=int(proposal.sum()),
            recall=float((proposal&truth).sum()/max(1,truth.sum())),
            precision=float((proposal&truth).sum()/max(1,proposal.sum())))
        package = torch.load(original, map_location='cpu', weights_only=True)
        package['split_architecture'] = 'obs_mlp'
        package['combat_training'] = metadata
        if algorithm == 'ppo':
            old = package['state_dict']
            net = HybridActorCritic(70, split_architecture='obs_mlp')
            net.load_state_dict({k:v for k,v in old.items() if not k.startswith('split_head.')},strict=False)
            net.split_head.load_state_dict(head.state_dict())
            for key,value in old.items():
                if not key.startswith('split_head.'):
                    assert torch.equal(value,net.state_dict()[key]), key
            package['state_dict'] = net.state_dict()
        else:
            package['split_head_state_dict'] = head.state_dict()
        torch.save(package,args.output/f'{algorithm}.pt')
        (args.output/f'{algorithm}_validation.json').write_text(json.dumps(metrics,indent=2))
        print(json.dumps(metrics),flush=True)
    (args.output/'training_config.json').write_text(json.dumps({**metadata,'workers':args.workers},indent=2))


if __name__ == '__main__':
    main()
