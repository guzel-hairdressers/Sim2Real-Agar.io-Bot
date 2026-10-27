"""Full-match retention and absolute mixed-league evaluation for learned risk."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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
from src.benchmark_showcase import ROOT,summarize as league_summary
from src.learned_risk import load_policy
from src.heuristic_agent import MasterHeuristicAgarBot


def match(job):
    mode,algorithm,directory,seed,steps=job
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    env=ArenaConfig(max_steps=steps).make();obs,_=env.reset(seed=seed)
    order=np.random.default_rng(seed).permutation(7);policies={};groups={}
    for j,seat in enumerate(order):
        pid=f'player_{seat}'
        if j<4:
            group=('ppo' if j<2 else 'diffusion') if mode=='league' else ('candidate' if j<2 else 'baseline')
            alg=group if mode=='league' else algorithm
            folder=Path(directory) if group!='baseline' else ROOT/'outputs/split_combat_v3'
            policies[pid]=load_policy(alg,folder/f'{alg}.pt',ROOT/'outputs/arena_v2_cosine/critic.pt',seed+j)
        else:
            group='heuristic';policies[pid]=MasterHeuristicAgarBot(profile=['apex','hunter','survivor'][j-4],seed=seed+j)
        groups[pid]=group
    initial={pid:p.mass for pid,p in env.players.items()};splits=dict.fromkeys(env.player_ids,0)
    projectile=dict(splits);violations=dict(splits);latency={g:[] for g in set(groups.values())}
    timeline=[];frame_times=[]
    for step in range(steps):
        before=time.perf_counter();actions={};deaths={pid:p.deaths for pid,p in env.players.items()}
        for pid,policy in policies.items():
            start=time.perf_counter();action=policy.predict(obs[pid]);latency[groups[pid]].append((time.perf_counter()-start)*1000)
            actions[pid]=action;splits[pid]+=int(action[2]>.5 and env.players[pid].can_split)
            if groups[pid]!='heuristic':violations[pid]+=int(action[2]>.5 and not env.players[pid].can_split)
        obs,_,_,_,_=env.step(actions)
        for e in env.predation_events:
            if 0<=e['projectile_age_steps']<=18:projectile[e['predator']]+=1
        for pid,policy in policies.items():
            if env.players[pid].deaths>deaths[pid] and hasattr(policy,'reset'):policy.reset(seed+env.step_count+int(pid[-1]))
        frame_times.append((time.perf_counter()-before)*1000)
        if (step+1)%30==0:
            timeline.append(dict(seconds=(step+1)/30,**{g:float(np.mean([p.mass for pid,p in env.players.items() if groups[pid]==g])) for g in latency}))
    players=[dict(id=pid,group=groups[pid],initial_mass=initial[pid],final_mass=p.mass,
        piece_captures=p.kills,full_deaths=p.deaths,wall_contact_frames=p.wall_contacts,
        executed_splits=splits[pid],projectile_piece_captures=projectile[pid],physical_violations=violations[pid])
        for pid,p in env.players.items()]
    winner=max(env.players,key=lambda pid:env.players[pid].mass)
    return dict(seed=seed,players=players,winner_group=groups[winner],timeline=timeline,
        latency_ms={g:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for g,v in latency.items()},
        processing_step_ms=dict(mean=float(np.mean(frame_times)),p95=float(np.percentile(frame_times,95))))


def retention_summary(rows):
    result=dict(matches=len(rows))
    for group in ('candidate','baseline'):
        players=[p for r in rows for p in r['players'] if p['group']==group]
        result[group]=dict(mean_final_mass=float(np.mean([p['final_mass'] for p in players])),
            full_deaths=sum(p['full_deaths'] for p in players),captures=sum(p['piece_captures'] for p in players),
            splits=sum(p['executed_splits'] for p in players),physical_violations=sum(p['physical_violations'] for p in players))
    differences=np.array([np.mean([p['final_mass'] for p in r['players'] if p['group']=='candidate'])-
                          np.mean([p['final_mass'] for p in r['players'] if p['group']=='baseline']) for r in rows])
    boot=np.random.default_rng(417).choice(differences,(10000,len(rows)),replace=True).mean(1)
    result['mean_final_mass_difference']=float(differences.mean());result['mass_difference_95_ci']=np.percentile(boot,[2.5,97.5]).tolist()
    result['passed']=bool(len(rows)>=12 and result['candidate']['mean_final_mass']>=.9*result['baseline']['mean_final_mass'] and
        result['candidate']['full_deaths']<=result['baseline']['full_deaths']+2 and result['candidate']['physical_violations']==0)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/learned_risk_v8')
    p.add_argument('--mode',choices=['retention','league'],required=True);p.add_argument('--algorithm',choices=['ppo','diffusion'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--steps',type=int,default=1800);p.add_argument('--seeds',type=int,nargs='+')
    args=p.parse_args()
    if args.mode=='retention' and not args.algorithm:p.error('Retention needs --algorithm')
    seeds=args.seeds or [(5300001 if args.mode=='retention' else 5400001)+37*i for i in range(12)]
    started=time.perf_counter();jobs=[(args.mode,args.algorithm,str(args.directory),s,args.steps) for s in seeds];rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(match,jobs):
            rows.append(row);print(json.dumps(dict(seed=row['seed'],completed=len(rows),total=len(jobs))),flush=True)
    models={name:dict(file=f'{name}.pt',sha256=hashlib.sha256((args.directory/f'{name}.pt').read_bytes()).hexdigest()) for name in ('ppo','diffusion')}
    critic=ROOT/'outputs/arena_v2_cosine/critic.pt';models['critic']=dict(file='../arena_v2_cosine/critic.pt',sha256=hashlib.sha256(critic.read_bytes()).hexdigest())
    sources=['src/evaluate_risk_league.py','src/benchmark_showcase.py','src/learned_risk.py','envs/agar_env.py',
             'envs/arena_config.py','src/arena_policies.py','src/arena_tactics.py','src/split_head.py',
             'src/train_champion_ppo.py','src/agar_diffusion_policy.py','src/heuristic_agent.py']
    report=dict(mode=args.mode,algorithm=args.algorithm,protocol=dict(matches=len(rows),steps_per_match=args.steps,
        seconds_per_match=args.steps/30,observation_hz=30,workers=args.workers,all_observations_processed=True,
        roster='2 current PPO + 2 current diffusion + 3 heuristics' if args.mode=='league' else '2 candidates + 2 v3 baseline + 3 heuristics; shuffled seats'),
        config=asdict(ArenaConfig(max_steps=args.steps)),models=models,rows=rows,
        source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        baseline_hashes={n:hashlib.sha256((ROOT/f'outputs/split_combat_v3/{n}.pt').read_bytes()).hexdigest() for n in ('ppo','diffusion')},
        machine=dict(platform=platform.platform(),machine=platform.machine(),python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__),
        wall_seconds=time.perf_counter()-started,
        summary=league_summary(rows,args.steps) if args.mode=='league' else retention_summary(rows))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
