"""Compare a PPO replacement against the exact same initial control seats.

Each seed has one control world and two worlds with a single replacement.
The two focus cases share a seed and are bootstrapped as one cluster.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
from src.arena_policies import ArenaPPO
from src.balanced_policies import BalancedPPO, load_policy
from src.benchmark_showcase import ROOT
from src.heuristic_agent import MasterHeuristicAgarBot
from src.learned_risk import LearnedRiskPPO

SOURCES = ['src/evaluate_balanced_ppo.py', 'src/balanced_policies.py',
           'src/learned_risk.py', 'src/arena_policies.py', 'src/arena_tactics.py',
           'src/train_champion_ppo.py', 'src/split_head.py',
           'src/heuristic_agent.py', 'envs/arena_config.py', 'envs/agar_env.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world(seed, steps, candidate=None, focus=None):
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ArenaConfig(max_steps=steps).make()
    obs, _ = env.reset(seed=seed)
    initial = hashlib.sha256(np.array([obs[p] for p in env.player_ids]).tobytes()).hexdigest()
    order = np.random.default_rng(seed).permutation(7)
    policies = {}
    for j, seat in enumerate(order):
        pid = f'player_{seat}'
        if j < 4:
            path = candidate if pid == focus else ROOT / 'outputs/split_combat_v3/ppo.pt'
            policies[pid] = load_policy('ppo', path)
        else:
            policies[pid] = MasterHeuristicAgarBot(
                profile=['apex', 'hunter', 'survivor'][j - 4], seed=seed + j)
    splits = dict.fromkeys(env.player_ids, 0)
    violations = dict(splits)
    projectiles = dict(splits)
    # Only the fully learned adapter is independently audited here. Historical
    # controllers retain steering assistance: unavailable must never mean zero.
    overrides = {pid: 0 if isinstance(policy, BalancedPPO) else None
                 for pid, policy in policies.items()}
    split_changes = {pid: 0 if isinstance(policy, ArenaPPO) else None
                     for pid, policy in policies.items()}
    timing = []
    for _ in range(steps):
        before = {pid: p.deaths for pid, p in env.players.items()}
        actions = {}
        for pid, policy in policies.items():
            start = time.perf_counter()
            action = policy.predict(obs[pid])
            if pid == focus:
                timing.append((time.perf_counter() - start) * 1000)
            if isinstance(policy, ArenaPPO):
                # Independently check the emitted command against the raw model.
                with torch.inference_mode():
                    raw = policy.net.get_action_and_value(
                        torch.as_tensor(obs[pid]).unsqueeze(0), True)[0][0].numpy()
                feasible = np.clip(raw, [0, -1, 0], [1, 1, 1])
                if not env.players[pid].can_split:
                    feasible[2] = 0
                split_changes[pid] += int((action[2] > .5) != (feasible[2] > .5))
                if isinstance(policy, BalancedPPO):
                    overrides[pid] += int(not np.allclose(action, feasible, atol=1e-6, rtol=0))
            actions[pid] = action
            splits[pid] += int(action[2] > .5 and env.players[pid].can_split)
            violations[pid] += int(action[2] > .5 and not env.players[pid].can_split)
        obs, _, _, _, _ = env.step(actions)
        for event in env.predation_events:
            if 0 <= event['projectile_age_steps'] <= 18:
                projectiles[event['predator']] += 1
        for pid, policy in policies.items():
            if env.players[pid].deaths > before[pid] and hasattr(policy, 'reset'):
                policy.reset(seed + env.step_count + int(pid[-1]))
    players = {pid: dict(final_mass=p.mass, full_deaths=p.deaths,
                         captures=p.kills, splits=splits[pid],
                         projectile_captures=projectiles[pid],
                         wall_contact_frames=p.wall_contacts,
                         physical_violations=violations[pid],
                         nonphysical_split_changes=split_changes[pid],
                         control_mode=('fully_learned' if isinstance(policies[pid], BalancedPPO)
                                       else 'assisted_movement_learned_split' if isinstance(policies[pid], LearnedRiskPPO)
                                       else 'historical_assisted' if isinstance(policies[pid], ArenaPPO)
                                       else 'heuristic'),
                         nonphysical_overrides=overrides[pid])
               for pid, p in env.players.items()}
    return dict(initial_observations_sha256=initial, players=players,
                focus_latency_ms=None if not timing else dict(
                    mean=float(np.mean(timing)), p95=float(np.percentile(timing, 95))))


def paired_seed(job):
    seed, steps, candidate = job
    control = world(seed, steps)
    order = np.random.default_rng(seed).permutation(7)
    cases = []
    for seat in order[:2]:
        focus = f'player_{seat}'
        replacement = world(seed, steps, candidate, focus)
        if replacement['initial_observations_sha256'] != control['initial_observations_sha256']:
            raise ValueError('Paired worlds have different initial observations')
        cases.append(dict(focus=focus, baseline=control['players'][focus],
                          candidate=replacement['players'][focus],
                          latency_ms=replacement['focus_latency_ms'],
                          replacement_world=replacement['players']))
    return dict(seed=seed, initial_observations_sha256=control['initial_observations_sha256'],
                control_world=control['players'], cases=cases)


def summarize(rows):
    if len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Duplicate seed clusters')
    result = dict(seed_clusters=len(rows), paired_focus_cases=sum(len(r['cases']) for r in rows))
    for side in ('candidate', 'baseline'):
        players = [c[side] for r in rows for c in r['cases']]
        result[side] = dict(mean_final_mass=float(np.mean([p['final_mass'] for p in players])),
                            death_free_runs=sum(p['full_deaths'] == 0 for p in players))
        for key in ('full_deaths', 'captures', 'splits', 'projectile_captures',
                    'wall_contact_frames', 'physical_violations', 'nonphysical_overrides',
                    'nonphysical_split_changes'):
            values = [p[key] for p in players]
            result[side][key] = None if any(v is None for v in values) else sum(values)
    differences = np.array([np.mean([c['candidate']['final_mass'] - c['baseline']['final_mass']
                                     for c in r['cases']]) for r in rows])
    boot = np.random.default_rng(9417).choice(differences, (10000, len(rows)), replace=True).mean(1)
    result['mean_final_mass_difference'] = float(differences.mean())
    result['mass_difference_seed_bootstrap_95_ci'] = np.percentile(boot, [2.5, 97.5]).tolist()
    c, b = result['candidate'], result['baseline']
    result['balance_thresholds_met'] = bool(c['mean_final_mass'] >= .9 * b['mean_final_mass']
                                          and c['full_deaths'] <= b['full_deaths'] + 2)
    players = [case['candidate'] for row in rows for case in row['cases']]
    result['control_contract_verified'] = all(
        p['control_mode'] in ('fully_learned', 'assisted_movement_learned_split')
        and p['physical_violations'] == 0 and p['nonphysical_split_changes'] == 0
        and (p['control_mode'] != 'fully_learned' or p['nonphysical_overrides'] == 0)
        for p in players)
    result['retention_thresholds_met'] = bool(result['balance_thresholds_met']
                                             and result['control_contract_verified'])
    result['passed'] = bool(len(rows) >= 12 and result['paired_focus_cases'] == 24
                             and result['retention_thresholds_met'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--purpose', choices=['development', 'final'], default='development')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seed clusters must be unique')
    if args.purpose == 'final' and (len(args.seeds) != 12 or args.steps != 1800):
        parser.error('Final evaluation requires 12 seeds and 1800 steps')
    args.candidate = args.candidate.resolve()
    hashes = {s: sha(ROOT / s) for s in SOURCES}
    model_hash = sha(args.candidate)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = [(s, args.steps, str(args.candidate)) for s in args.seeds]
        for row in pool.map(paired_seed, jobs):
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), total=len(jobs), seed=row['seed'])), flush=True)
    if hashes != {s: sha(ROOT / s) for s in SOURCES} or model_hash != sha(args.candidate):
        raise ValueError('Sources or candidate changed during evaluation')
    report = dict(purpose=args.purpose, candidate_path=str(args.candidate.relative_to(ROOT)),
                  candidate_sha256=model_hash,
                  baseline_sha256=sha(ROOT / 'outputs/split_combat_v3/ppo.pt'),
                  source_hashes=hashes, config=asdict(ArenaConfig(max_steps=args.steps)),
                  control_audit_scope='Split-bit equality is audited for all PPO controllers. Full command equality is audited only for BalancedPPO. Assisted movement full-command counter: null (not measured).',
                  protocol=dict(seeds=args.seeds, steps=args.steps, workers=args.workers,
                                observation_hz=30, all_observations_processed=True,
                                roster='Control: four v3 PPO + three heuristics; replace only one focus seat per world',
                                focus_seats_per_seed=2, worlds_per_seed=3,
                                seed_cluster_bootstrap=True),
                  wall_seconds=time.perf_counter() - started, rows=rows, summary=summarize(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
