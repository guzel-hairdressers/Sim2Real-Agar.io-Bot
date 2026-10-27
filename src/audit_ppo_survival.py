"""Replay failed PPO cases with identical seats and controlled split choices.

These are consumed v8 test seeds, now explicitly development diagnostics.
The hold intervention is confined to this experiment, never runtime control.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from envs.arena_config import ArenaConfig
from src.benchmark_showcase import ROOT
from src.learned_risk import load_policy
from src.heuristic_agent import MasterHeuristicAgarBot


def replay(job):
    seed,focus,variant,directory=job
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    env=ArenaConfig().make();obs,_=env.reset(seed=seed);policies={};groups={}
    for j,seat in enumerate(np.random.default_rng(seed).permutation(7)):
        pid=f'player_{seat}';group='candidate' if j<2 else 'baseline' if j<4 else 'heuristic'
        if j<4:
            folder=Path(directory) if group=='candidate' else ROOT/'outputs/split_combat_v3'
            if pid==focus and variant=='baseline':folder=ROOT/'outputs/split_combat_v3'
            policies[pid]=load_policy('ppo',folder/'ppo.pt')
        else:policies[pid]=MasterHeuristicAgarBot(profile=['apex','hunter','survivor'][j-4],seed=seed+j)
        groups[pid]=group
    observations=[];commands=[];events=[];splits=dict.fromkeys(env.player_ids,0);last_split=None
    for _ in range(1800):
        observations.append(obs[focus].copy())
        actions={pid:policy.predict(obs[pid]) for pid,policy in policies.items()}
        if variant=='hold':actions[focus][2]=0.  # Diagnostic intervention only.
        commands.append(actions[focus].copy())
        before={pid:p.deaths for pid,p in env.players.items()}
        pose=env.players[focus].pos.copy();masses=[p.mass for p in env.players[focus].pieces]
        for pid,a in actions.items():
            if a[2]>.5 and env.players[pid].can_split:
                splits[pid]+=1
                if pid==focus:
                    last_split=env.step_count
                    events.append(dict(kind='split',step=env.step_count,position=pose.tolist(),piece_masses=masses,
                                       obs=obs[focus].tolist(),action=a.tolist()))
        obs,_,_,_,_=env.step(actions)
        if env.players[focus].deaths>before[focus]:
            events.append(dict(kind='death',step=env.step_count,position=pose.tolist(),piece_masses=masses,
                steps_since_last_split=None if last_split is None else env.step_count-last_split,
                last_observation=observations[-1].tolist(),action=commands[-1].tolist(),
                captures=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in e.items()}
                          for e in env.predation_events if e['prey']==focus]))
        for pid,policy in policies.items():
            if env.players[pid].deaths>before[pid] and hasattr(policy,'reset'):
                policy.reset(seed+env.step_count+int(pid[-1]))
    players=[dict(id=pid,group=groups[pid],final_mass=p.mass,full_deaths=p.deaths,
                  piece_captures=p.kills,executed_splits=splits[pid]) for pid,p in env.players.items()]
    return dict(seed=seed,focus=focus,variant=variant,players=players,events=events,
                observations=np.array(observations),commands=np.array(commands))


def compare(original,alternative):
    changed=np.flatnonzero(np.any(original['commands']!=alternative['commands'],axis=1))
    first=None
    if len(changed):
        step=int(changed[0])
        first=dict(step=step,original_obs=original['observations'][step].tolist(),
            alternative_obs=alternative['observations'][step].tolist(),
            original_action=original['commands'][step].tolist(),alternative_action=alternative['commands'][step].tolist())
    return dict(first_action_difference=first,identical_focus_actions=not len(changed))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/learned_risk_v8')
    p.add_argument('--report',type=Path,default=ROOT/'outputs/learned_risk_v8/ppo_retention.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/balanced_v9/ppo_death_audit.json')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    reference=json.loads(args.report.read_text())
    cases=[(row['seed'],player['id']) for row in reference['rows'] for player in row['players']
           if player['group']=='candidate' and player['full_deaths']]
    jobs=[(s,pid,v,str(args.directory)) for s,pid in cases for v in ('candidate','baseline','hold')]
    results={}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for i,row in enumerate(pool.map(replay,jobs)):
            results[row['seed'],row['focus'],row['variant']]=row
            print(json.dumps(dict(completed=i+1,total=len(jobs),seed=row['seed'],focus=row['focus'],variant=row['variant'],
                                 result=next(p for p in row['players'] if p['id']==row['focus']))),flush=True)
    rows=[]
    for seed,focus in cases:
        original=results[seed,focus,'candidate'];old=next(r for r in reference['rows'] if r['seed']==seed)
        for got in original['players']:
            expected=next(p for p in old['players'] if p['id']==got['id'])
            for key in ('final_mass','full_deaths','piece_captures','executed_splits'):
                if not np.isclose(got[key],expected[key],atol=1e-8,rtol=0):
                    raise ValueError(f'Replay differs from measured v8 row: {seed}/{focus}/{got["id"]}/{key}')
        row=dict(seed=seed,focus=focus,original_replay_matches_all_players=True,variants={})
        for variant in ('candidate','baseline','hold'):
            result=results[seed,focus,variant]
            row['variants'][variant]=dict(focus_result=next(p for p in result['players'] if p['id']==focus),
                                          events=result['events'],comparison_to_original=compare(original,result))
        rows.append(row)
    sources=['src/audit_ppo_survival.py',*reference['source_hashes']]
    output=dict(purpose='Causal development audit; not fresh acceptance evidence',
        consumed_prior_test_seeds=sorted({s for s,_ in cases}),
        interventions='Only focus controller changes; same initial seats, other controllers and seeds. Hold disables only the focus split bit for diagnosis.',
        reference_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),models=reference['models'],rows=rows,
        source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2))


if __name__=='__main__':main()
