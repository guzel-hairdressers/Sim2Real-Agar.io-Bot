"""Record active matches and a paired PPO/diffusion demonstration.

Uses ordinary ArenaConfig resets, reactive opponents and deployed policies.
No scripted split commands, stationary targets or edited combat outcomes.
This is demonstration footage, separate from the twelve-match benchmark.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

from envs.arena_config import ArenaConfig
from src.arena_policies import ArenaPPO, ArenaDiffusion
from src.benchmark_showcase import ROOT, model_paths, snapshot
from src.heuristic_agent import MasterHeuristicAgarBot

OUT=ROOT/'outputs/showcase_clashes'


def record(job):
    kind,seed=job
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    paths,_=model_paths();env=ArenaConfig(max_steps=3600).make()
    obs,_=env.reset(seed=seed);policies={};seats={}
    order=np.random.default_rng(seed).permutation(7) if kind=='mixed' else np.arange(7)
    for j,seat in enumerate(order):
        pid=f'player_{seat}'
        if kind=='mixed':
            group='ppo' if j<2 else 'diffusion' if j<4 else 'heuristic'
            number=j+1 if j<2 else j-1 if j<4 else j-3
        else:
            group=kind if j==0 else 'heuristic';number=1 if j==0 else j
        if group=='ppo':
            policy=ArenaPPO(paths['ppo']);name=f'PPO-{number}'
        elif group=='diffusion':
            policy=ArenaDiffusion(paths['diffusion'],paths['critic'],seed=seed+j);name=f'DIFF-{number}'
        else:
            profile=['apex','hunter','survivor'][(j-4 if kind=='mixed' else j-1)%3]
            policy=MasterHeuristicAgarBot(profile=profile,seed=seed+j)
            name=f'RULE-{profile.upper()}' if kind=='mixed' else f'RULE-{profile.upper()}-{number}'
        policies[pid]=policy;seats[pid]=dict(group=group,name=name)
    splits=dict.fromkeys(env.player_ids,0);frames=[snapshot(env,seats,splits,{})];events=[]
    for _ in range(3600):
        before={pid:dict(deaths=p.deaths,pos=p.pos.tolist()) for pid,p in env.players.items()}
        actions={pid:policy.predict(obs[pid]) for pid,policy in policies.items()}
        launched=[]
        for pid,action in actions.items():
            if action[2]>.5 and env.players[pid].can_split:
                splits[pid]+=1;launched.append(pid)
        obs,_,_,_,_=env.step(actions)
        for pid in launched:
            events.append(dict(kind='split',step=env.step_count,seconds=env.step_count/30,
                               player=pid,pos=before[pid]['pos']))
        for event in env.predation_events:
            events.append(dict(kind='capture',step=env.step_count,seconds=env.step_count/30,
                predator=event['predator'],prey=event['prey'],pos=event['pos'].tolist(),
                projectile_age_steps=int(event['projectile_age_steps']),mass_gained=float(event['mass_gained'])))
        for pid,policy in policies.items():
            if env.players[pid].deaths>before[pid]['deaths']:
                events.append(dict(kind='death',step=env.step_count,seconds=env.step_count/30,
                                   player=pid,pos=before[pid]['pos']))
                if hasattr(policy,'reset'):policy.reset(seed+env.step_count+int(pid[-1]))
        if env.step_count%2==0:frames.append(snapshot(env,seats,splits,actions))
    trace=dict(kind=kind,seed=seed,config=asdict(ArenaConfig(max_steps=3600)),capture_hz=15,frames=frames,events=events)
    path=OUT/f'{kind}_trace.json.gz'
    with gzip.open(path,'wt') as f:json.dump(trace,f,separators=(',',':'))
    return dict(kind=kind,seed=seed,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                captures=sum(e['kind']=='capture' for e in events),
                splits=sum(e['kind']=='split' for e in events),
                full_deaths=sum(e['kind']=='death' for e in events),
                focus_agent=next(p for p in frames[-1]['players'] if p['id']=='player_0') if kind!='mixed' else None)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[('mixed',4100001),('ppo',4200001),('diffusion',4200001)]
    with ProcessPoolExecutor(3,mp_context=mp.get_context('spawn')) as pool:
        recordings=list(pool.map(record,jobs))
    _,models=model_paths()
    sources=['src/record_clashes.py','src/benchmark_showcase.py','envs/agar_env.py','envs/arena_config.py',
             'src/arena_policies.py','src/arena_tactics.py','src/train_champion_ppo.py',
             'src/split_head.py','src/agar_diffusion_policy.py','src/heuristic_agent.py']
    report=dict(recordings=recordings,models=models['models'],simulation_hz=30,recording_hz=15,
        simulation_seconds=120,playback_speed=2,video_seconds=60,
        comparison='Same standard initial state, one learned focus agent plus six reactive heuristics; separate rollouts, not a direct duel',
        footage_is_benchmark=False,source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (OUT/'recordings.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
