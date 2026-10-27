"""Matched-seat tournaments against frozen original or arena-v2 policies.

Development seeds: 810001 + 37*i and the initial 910001 + 37*i run.
Fresh final test seeds after the clearance revision: 1010001 + 37*i.
Independent matches, not individual players, are the statistical unit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch

from envs.arena_config import ArenaConfig
from src.arena_policies import ArenaPPO, ArenaDiffusion, LegacyPolicy
from src.heuristic_agent import MasterHeuristicAgarBot


def run_match(job):
    algorithm, checkpoint, critic, seed, steps, assisted = job[:6]
    baseline_version = job[6] if len(job) > 6 else 'original'
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    env = ArenaConfig(max_steps=steps).make()
    obs, _ = env.reset(seed=seed)
    slots = np.random.default_rng(seed).permutation(7)
    policies, groups = {}, {}
    for j, seat in enumerate(slots):
        pid = f'player_{seat}'
        group = 'new' if j < 2 else ('legacy' if j < 4 else 'heuristic')
        groups[pid] = group
        if group == 'new':
            policies[pid] = ArenaPPO(checkpoint, assisted) if algorithm == 'ppo' else ArenaDiffusion(checkpoint, critic, seed+j, assisted)
        elif group == 'legacy':
            if baseline_version == 'arena_v2':
                policies[pid] = ArenaPPO('outputs/arena_v2/ppo.pt') if algorithm == 'ppo' else ArenaDiffusion(
                    'outputs/arena_v2_cosine/diffusion.pt', 'outputs/arena_v2_cosine/critic.pt', seed=seed+j)
            else:
                policies[pid] = LegacyPolicy(algorithm, seed=seed+j)
        else:
            policies[pid] = MasterHeuristicAgarBot(profile=['apex', 'hunter', 'survivor'][j-4], seed=seed+j)
    start_mass = {p: x.mass for p, x in env.players.items()}
    times = {'new': [], 'legacy': []}
    previous_deaths = {p: 0 for p in env.player_ids}
    splits = {g: 0 for g in ('new', 'legacy', 'heuristic')}
    projectile_kills = dict(splits)
    for _ in range(steps):
        actions = {}
        for pid, policy in policies.items():
            t0 = time.perf_counter()
            actions[pid] = policy.predict(obs[pid])
            if groups[pid] in times:
                times[groups[pid]].append((time.perf_counter()-t0)*1000)
        for pid, action in actions.items():
            splits[groups[pid]] += int(action[2] > .5 and env.players[pid].can_split)
        obs, _, _, _, infos = env.step(actions)
        for event in env.predation_events:
            if 0 <= event['projectile_age_steps'] <= 18:
                projectile_kills[groups[event['predator']]] += 1
        for pid, policy in policies.items():
            if infos[pid]['deaths'] > previous_deaths[pid] and hasattr(policy, 'reset'):
                policy.reset(seed + env.step_count + int(pid[-1]))
            previous_deaths[pid] = infos[pid]['deaths']
    result = {'seed': seed, 'groups': {}, 'algorithm': algorithm}
    for group in ('new', 'legacy', 'heuristic'):
        pids = [p for p in env.player_ids if groups[p] == group]
        players = [env.players[p] for p in pids]
        result['groups'][group] = dict(
            final_mass=float(np.mean([p.mass for p in players])),
            mass_gain=float(np.mean([env.players[p].mass-start_mass[p] for p in pids])),
            piece_kills=sum(p.kills for p in players), deaths=sum(p.deaths for p in players),
            wall_contacts=sum(p.wall_contacts for p in players),
            executed_splits=splits[group], projectile_kills=projectile_kills[group],
            latency_ms=float(np.mean(times[group])) if group in times else None)
    result['mass_difference'] = result['groups']['new']['mass_gain'] - result['groups']['legacy']['mass_gain']
    result['new_wins_mass'] = result['mass_difference'] > 0
    return result


def summarize(rows):
    differences = np.array([r['mass_difference'] for r in rows])
    rng = np.random.default_rng(7741)
    bootstrap = rng.choice(differences, (10000, len(rows)), replace=True).mean(axis=1)
    result = dict(matches=len(rows), mass_gain_difference=float(differences.mean()),
                  bootstrap_95_ci=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                  mass_wins=sum(r['new_wins_mass'] for r in rows))
    for group in ('new', 'legacy'):
        result[group] = {key: float(np.mean([r['groups'][group][key] for r in rows]))
                         for key in rows[0]['groups'][group] if r_value(rows[0]['groups'][group][key])}
    result['improved'] = bool(result['bootstrap_95_ci'][0] > 0 and result['new']['deaths'] <= result['legacy']['deaths'])
    return result


def r_value(value):
    return value is not None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', choices=['ppo', 'diffusion'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--critic')
    p.add_argument('--steps', type=int, default=1800)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seeds', nargs='+', type=int, default=[1010001+37*i for i in range(12)])
    p.add_argument('--raw', action='store_true')
    p.add_argument('--baseline-version', choices=['original', 'arena_v2'], default='original')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    jobs = [(args.algorithm, args.checkpoint, args.critic, seed, args.steps, not args.raw, args.baseline_version) for seed in args.seeds]
    results = []
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_match, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result(); results.append(row)
            print(json.dumps(row), flush=True)
    summary = summarize(results)
    if args.baseline_version == 'arena_v2':
        # Predeclared split-combat retention gate: this is not a claim of
        # statistically significant growth superiority over the v2 policy.
        summary['retention_passed'] = bool(len(results) >= 12 and args.steps >= 1800 and
            summary['new']['final_mass'] >= .9*summary['legacy']['final_mass'] and
            sum(r['groups']['new']['deaths'] for r in results) <=
            sum(r['groups']['legacy']['deaths'] for r in results)+2)
    baseline_paths = [str(Path('outputs')/name) for name in ('agar_ppo_champion.pt', 'agar_diffusion_champion.pt', 'agar_diffusion_critic_champion.pt')]
    if args.baseline_version == 'arena_v2':
        baseline_paths = ['outputs/arena_v2/ppo.pt', 'outputs/arena_v2_cosine/diffusion.pt', 'outputs/arena_v2_cosine/critic.pt']
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [args.checkpoint, args.critic, *baseline_paths] if path}
    sources = ['envs/agar_env.py', 'envs/arena_config.py', 'src/arena_tactics.py',
               'src/arena_policies.py', 'src/agar_diffusion_policy.py', 'src/train_champion_ppo.py',
               'src/evaluate_arena_v2.py', 'benchmarks/legacy_reflex_ca3f24d.py']
    sources.append('src/split_head.py')
    sources.append('src/heuristic_agent.py')
    report = dict(config=asdict(ArenaConfig(max_steps=args.steps)), assisted=not args.raw,
                  algorithm=args.algorithm, checkpoint_hashes=hashes,
                  baseline_version=args.baseline_version,
                  source_hashes={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
                  rows=sorted(results, key=lambda r: r['seed']), summary=summary,
                  wall_seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
