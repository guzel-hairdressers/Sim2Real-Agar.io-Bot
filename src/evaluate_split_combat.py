"""Held-out physical attack opportunities and hard rejection cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

from src.train_split_combat import initialize_workers, collect_batch, make_encounter, POLICIES
from src.arena_policies import ArenaPPO, ArenaDiffusion
from src.arena_tactics import split_is_safe

CANDIDATES = {}


def initialize_candidates(directory):
    initialize_workers()
    CANDIDATES['ppo'] = ArenaPPO(Path(directory)/'ppo.pt')
    CANDIDATES['diffusion'] = ArenaDiffusion(Path(directory)/'diffusion.pt', 'outputs/arena_v2_cosine/critic.pt')


def run_policy(env, target, policy, seed):
    policy.reset(seed=seed)
    split_count = 0
    unsafe_splits = 0
    hit = damage = False
    for _ in range(18):
        obs = env._get_obs('player_0')
        action = policy.predict(obs)
        executed = int(action[2] > .5 and env.players['player_0'].can_split)
        split_count += executed
        unsafe_splits += int(executed and not split_is_safe(obs))
        env.step({'player_0':action, 'player_1':target, 'player_2':np.zeros(3, np.float32)})
        for event in env.predation_events:
            damage |= event['prey'] == 'player_0'
            hit |= event['predator'] == 'player_0' and event['prey'] == 'player_1' and 0 <= event['projectile_age_steps'] <= 18
    return dict(splits=split_count, unsafe_splits=unsafe_splits, projectile_capture=hit, damage=damage,
                deaths=env.players['player_0'].deaths)


def evaluate_batch(job):
    trials = collect_batch(job)
    rows = []
    for trial in trials:
        algorithm, seed = trial['algorithm'], trial['seed']
        env, target, _ = make_encounter(seed)
        obs = torch.tensor(trial['obs'], dtype=torch.float32).unsqueeze(0)
        candidate = CANDIDATES[algorithm]
        with torch.inference_mode():
            if algorithm == 'ppo':
                probability = float(candidate.net(obs)[2].sigmoid().item())
            else:
                probability = float(candidate.split_head(obs).sigmoid().item())
        result = run_policy(env, target, candidate, seed)
        env, target, _ = make_encounter(seed)
        baseline = run_policy(env, target, POLICIES[algorithm], seed)
        rows.append({k:v for k,v in trial.items() if k != 'obs'} | {
            'proposal_probability':probability, 'candidate':result, 'baseline':baseline})
    return rows


def summarize(rows):
    result = {}
    for algorithm in ('ppo', 'diffusion'):
        subset = [r for r in rows if r['algorithm']==algorithm]
        positive = [r for r in subset if r['label']]
        unsafe = [r for r in subset if not r['safe']]
        executed = [r for r in subset if r['candidate']['splits']]
        hits = sum(r['candidate']['projectile_capture'] for r in positive)
        proposals = [r for r in subset if r['proposal_probability']>.65]
        result[algorithm] = dict(cases=len(subset), positive_opportunities=len(positive),
            safe_opportunity_fire_rate=sum(bool(r['candidate']['splits']) for r in positive)/max(1,len(positive)),
            positive_projectile_capture_rate=hits/max(1,len(positive)),
            unsafe_cases=len(unsafe), unsafe_executed_splits=sum(r['candidate']['unsafe_splits'] for r in subset),
            executed_attacks=len(executed), projectile_captures=sum(r['candidate']['projectile_capture'] for r in subset),
            baseline_attacks=sum(r['baseline']['splits'] for r in subset),
            baseline_projectile_captures=sum(r['baseline']['projectile_capture'] for r in subset),
            raw_proposal_precision=sum(r['label'] for r in proposals)/max(1,len(proposals)),
            actor_damage=sum(r['candidate']['damage'] for r in subset),
            baseline_damage=sum(r['baseline']['damage'] for r in subset))
        r = result[algorithm]
        r['passed'] = bool(len(positive)>=50 and r['safe_opportunity_fire_rate']>=.85 and
            r['positive_projectile_capture_rate']>=.8 and r['unsafe_executed_splits']==0 and
            r['projectile_captures']>=r['baseline_projectile_captures']+30)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=Path('outputs/split_combat_v3'))
    p.add_argument('--cases', type=int, default=1000)
    p.add_argument('--seed', type=int, default=3100001)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    jobs = [(list(range(start,min(start+16,args.seed+args.cases))),)
            for start in range(args.seed,args.seed+args.cases,16)]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
            initializer=initialize_candidates, initargs=(str(args.directory),)) as pool:
        for i, batch in enumerate(pool.map(evaluate_batch,jobs)):
            rows.extend(batch)
            if i%10==0: print(json.dumps({'batches':i+1,'total':len(jobs)}),flush=True)
    sources = ['src/train_split_combat.py', 'src/evaluate_split_combat.py', 'src/split_head.py',
               'src/arena_policies.py', 'src/arena_tactics.py', 'envs/agar_env.py', 'src/train_champion_ppo.py']
    report = dict(seed=args.seed, summary=summarize(rows), rows=rows,
        source_hashes={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        checkpoint_hashes={name:hashlib.sha256((args.directory/name).read_bytes()).hexdigest() for name in ('ppo.pt','diffusion.pt')})
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)


if __name__ == '__main__':
    main()
