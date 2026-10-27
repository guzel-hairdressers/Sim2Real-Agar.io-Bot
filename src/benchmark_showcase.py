"""Absolute current-policy benchmark and reproducible spectator recording.

Two PPO, two diffusion and three heuristic agents; no historical policies.
Every policy processes every 30 Hz observation. Trace rendering is separate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

from envs.arena_config import ArenaConfig
from src.arena_policies import ArenaPPO, ArenaDiffusion
from src.heuristic_agent import MasterHeuristicAgarBot

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT/'outputs/split_combat_v3'


def model_paths():
    manifest = json.loads((MODEL_DIR/'deployment.json').read_text())
    paths = {}
    for name, entry in manifest['models'].items():
        path = MODEL_DIR/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Accepted model hash mismatch: {name}')
        paths[name] = path
    return paths, manifest


def snapshot(env, seats, splits, actions):
    return dict(step=env.step_count, seconds=env.step_count*env.dt,
        food=env.food_positions.tolist(), viruses=env.viruses.tolist(),
        players=[dict(id=pid, group=seats[pid]['group'], name=seats[pid]['name'],
            pos=p.pos.tolist(), yaw=p.yaw, mass=p.mass, vision_radius=p.vision_radius,
            kills=p.kills, deaths=p.deaths, splits=splits[pid],
            action=np.asarray(actions.get(pid,np.zeros(3))).tolist(),
            pieces=[dict(pos=pc.pos.tolist(),radius=pc.radius,mass=pc.mass) for pc in p.pieces])
            for pid,p in env.players.items()])


def run_match(job):
    seed, steps, trace_path = job
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    paths, _ = model_paths()
    env = ArenaConfig(max_steps=steps).make()
    obs, _ = env.reset(seed=seed)
    order = np.random.default_rng(seed).permutation(7)
    policies, seats = {}, {}
    for j, seat in enumerate(order):
        pid = f'player_{seat}'
        if j < 2:
            group, name = 'ppo', f'PPO-{j+1}'
            policies[pid] = ArenaPPO(paths['ppo'])
        elif j < 4:
            group, name = 'diffusion', f'DIFF-{j-1}'
            policies[pid] = ArenaDiffusion(paths['diffusion'],paths['critic'],seed=seed+j)
        else:
            profile = ['apex','hunter','survivor'][j-4]
            group, name = 'heuristic', f'RULE-{profile.upper()}'
            policies[pid] = MasterHeuristicAgarBot(profile=profile,seed=seed+j)
        seats[pid] = dict(group=group,name=name)
    initial = {pid:p.mass for pid,p in env.players.items()}
    splits = dict.fromkeys(env.player_ids,0)
    projectile = dict(splits)
    latency = {g:[] for g in ('ppo','diffusion','heuristic')}
    frame_times, timeline, trace = [], [], []
    if trace_path:
        trace.append(snapshot(env,seats,splits,{}))
    for step in range(steps):
        start = time.perf_counter()
        actions = {}
        deaths = {pid:p.deaths for pid,p in env.players.items()}
        for pid, policy in policies.items():
            before = time.perf_counter()
            actions[pid] = policy.predict(obs[pid])
            latency[seats[pid]['group']].append((time.perf_counter()-before)*1000)
            splits[pid] += int(actions[pid][2]>.5 and env.players[pid].can_split)
        obs, _, _, _, _ = env.step(actions)
        for event in env.predation_events:
            if 0 <= event['projectile_age_steps'] <= 18:
                projectile[event['predator']] += 1
        for pid, policy in policies.items():
            if env.players[pid].deaths > deaths[pid] and hasattr(policy,'reset'):
                policy.reset(seed+env.step_count+int(pid[-1]))
        frame_times.append((time.perf_counter()-start)*1000)
        if (step+1)%30 == 0:
            timeline.append(dict(seconds=(step+1)/30,
                **{g:float(np.mean([p.mass for pid,p in env.players.items() if seats[pid]['group']==g]))
                   for g in latency}))
        if trace_path and (step+1)%2 == 0:
            trace.append(snapshot(env,seats,splits,actions))
    if trace_path:
        with gzip.open(trace_path,'wt') as f:
            json.dump(dict(seed=seed,config=asdict(ArenaConfig(max_steps=steps)),
                           capture_hz=15,frames=trace),f,separators=(',',':'))
    winner = max(env.players,key=lambda pid:env.players[pid].mass)
    players = [dict(id=pid,**seats[pid],initial_mass=initial[pid],final_mass=p.mass,
        piece_captures=p.kills,full_deaths=p.deaths,wall_contact_frames=p.wall_contacts,
        executed_splits=splits[pid],projectile_piece_captures=projectile[pid]) for pid,p in env.players.items()]
    return dict(seed=seed,players=players,winner_group=seats[winner]['group'],timeline=timeline,
        latency_ms={g:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)))
                    for g,v in latency.items()},
        processing_step_ms=dict(mean=float(np.mean(frame_times)),p95=float(np.percentile(frame_times,95))))


def summarize(rows, steps):
    summary = {}
    for group in ('ppo','diffusion','heuristic'):
        players = [p for r in rows for p in r['players'] if p['group']==group]
        means = np.array([np.mean([p['final_mass'] for p in r['players'] if p['group']==group]) for r in rows])
        boot = np.random.default_rng(417).choice(means,(10000,len(means)),replace=True).mean(axis=1)
        summary[group] = dict(agent_runs=len(players),agent_minutes=len(players)*steps/1800,
            mean_initial_mass=float(np.mean([p['initial_mass'] for p in players])),
            mean_final_mass=float(means.mean()),final_mass_match_bootstrap_95_ci=np.percentile(boot,[2.5,97.5]).tolist(),
            zero_death_runs=sum(p['full_deaths']==0 for p in players),
            full_deaths=sum(p['full_deaths'] for p in players),piece_captures=sum(p['piece_captures'] for p in players),
            executed_splits=sum(p['executed_splits'] for p in players),
            projectile_piece_captures=sum(p['projectile_piece_captures'] for p in players),
            final_mass_winners=sum(r['winner_group']==group for r in rows),
            inference_mean_ms=float(np.mean([r['latency_ms'][group]['mean'] for r in rows])),
            mean_per_match_inference_p95_ms=float(np.mean([r['latency_ms'][group]['p95'] for r in rows])))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/showcase_v1/benchmark.json')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--seeds',type=int,nargs='+',default=[4100001+37*i for i in range(12)])
    args = p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trace = args.output.parent/'demo_trace.json.gz'
    started = time.perf_counter(); rows=[]
    jobs = [(seed,args.steps,str(trace) if i==0 else None) for i,seed in enumerate(args.seeds)]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(run_match,jobs):
            rows.append(row)
            print(json.dumps(dict(seed=row['seed'],completed=len(rows),total=len(jobs))),flush=True)
    _, manifest = model_paths()
    sources = ['src/benchmark_showcase.py','envs/agar_env.py','envs/arena_config.py',
               'src/arena_policies.py','src/arena_tactics.py','src/train_champion_ppo.py',
               'src/split_head.py','src/agar_diffusion_policy.py','src/heuristic_agent.py']
    report = dict(protocol=dict(matches=len(rows),steps_per_match=args.steps,seconds_per_match=args.steps/30,
        roster='2 current PPO + 2 current diffusion + 3 heuristics; shuffled seats; no cooperation',
        observation_hz=30,recording_hz=15,all_observations_processed=True,workers=args.workers,
        confidence_unit='independent matches; bootstrap 10,000 resamples',
        win_definition='one individual with largest final mass; slots per group are 2/2/3',
        latency_definition='policy.predict including safety checks; CPU under concurrent evaluation load; excludes rendering',
        demo_selection='first requested seed, fixed before outcomes; no selection by score'),
        config=asdict(ArenaConfig(max_steps=args.steps)),
        machine=dict(platform=platform.platform(),machine=platform.machine(),python=sys.version.split()[0],
                     torch=torch.__version__,numpy=np.__version__),
        models=manifest['models'],source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),wall_seconds=time.perf_counter()-started,
        summary=summarize(rows,args.steps),rows=rows)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)


if __name__ == '__main__':
    main()
