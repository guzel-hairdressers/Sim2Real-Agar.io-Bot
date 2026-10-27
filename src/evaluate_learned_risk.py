"""Outcome evaluation for physics-only learned split decisions.

Old risk rules are logged for comparison, never used to veto candidate actions
or label training examples. Actual utility, damage and deaths decide quality.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from src.train_learned_risk import initialize,collect,risk_scene,ROOT
from src.learned_risk import load_policy
from src.arena_policies import ArenaPPO,ArenaDiffusion
from src.arena_tactics import split_is_safe
from src.heuristic_agent import MasterHeuristicAgarBot

CANDIDATES={};BASELINES={}


def initialize_eval(directory):
    initialize()
    critic=ROOT/'outputs/arena_v2_cosine/critic.pt'
    for algorithm in ('ppo','diffusion'):
        CANDIDATES[algorithm]=load_policy(algorithm,Path(directory)/f'{algorithm}.pt',critic)
        BASELINES[algorithm]=load_policy(algorithm,ROOT/f'outputs/split_combat_v3/{algorithm}.pt',critic)


def probability(policy,algorithm,obs):
    with torch.inference_mode():
        value=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
        return float((policy.net(value)[2] if algorithm=='ppo' else policy.split_head(value)).sigmoid().item())


def simulate(env,policy,algorithm,seed,horizon):
    np.random.seed(seed)
    env=copy.deepcopy(env);policy.reset(seed=seed);initial=env.players['player_0'].mass
    opponents={pid:MasterHeuristicAgarBot(profile='hunter' if pid=='player_2' else 'survivor',seed=seed+i)
               for i,pid in enumerate(['player_1','player_2'])}
    obs={pid:env._get_obs(pid) for pid in env.player_ids}
    samples=[];splits=0;physical_violations=0;hard_vetoes=0;deaths=0;projectile=0;lost=0.
    for _ in range(horizon):
        o=obs['player_0'];request=probability(policy,algorithm,o)>.65
        actions={'player_0':policy.predict(o)}
        physical_violations+=int(actions['player_0'][2]>.5 and o[4]<.5)
        hard_vetoes+=int(request and o[4]>.5 and actions['player_0'][2]<=.5)
        splits+=int(actions['player_0'][2]>.5 and o[4]>.5)
        actions.update({pid:p.predict(obs[pid]) for pid,p in opponents.items()})
        obs,_,_,_,_=env.step(actions)
        for e in env.predation_events:
            if e['prey']=='player_0':lost+=e['mass_gained']/.7
            if e['predator']=='player_0' and 0<=e['projectile_age_steps']<=18:projectile+=1
        a=env.players['player_0'];samples.append(a.mass)
        if a.deaths>deaths:policy.reset(seed+env.step_count);deaths=a.deaths
    a=env.players['player_0'];utility=.6*(a.mass-initial)/initial+.4*(np.mean(samples)-initial)/initial+.06*a.kills-2*a.deaths
    return dict(utility=float(utility),mass=a.mass,deaths=a.deaths,captures=a.kills,splits=splits,
                physical_violations=physical_violations,hard_vetoes=hard_vetoes,projectile_captures=projectile,lost_mass=lost)


def evaluate(job):
    seeds,horizon=job;truth=collect(job);rows=[]
    for row in truth:
        env,_=risk_scene(row['seed']);algorithm=row['algorithm']
        o=env._get_obs('player_0');candidate=CANDIDATES[algorithm]
        row['initial_probability']=probability(candidate,algorithm,o)
        row['old_guard_allows']=bool(split_is_safe(o))
        row['candidate']=simulate(env,candidate,algorithm,row['seed'],horizon)
        row['baseline']=simulate(env,BASELINES[algorithm],algorithm,row['seed'],horizon)
        rows.append(row)
    return rows


def summarize(rows):
    result={}
    for algorithm in ('ppo','diffusion'):
        subset=[r for r in rows if r['algorithm']==algorithm]
        truth=np.array([r['label'] for r in subset]);proposals=np.array([r['initial_probability']>.65 and r['physically_possible'] for r in subset])
        difference=np.array([r['candidate']['utility']-r['baseline']['utility'] for r in subset])
        boot=np.random.default_rng(417).choice(difference,(10000,len(difference)),replace=True).mean(1)
        s=dict(cases=len(subset),positive_opportunities=int(truth.sum()),initial_proposals=int(proposals.sum()),
            initial_precision=float((truth&proposals).sum()/max(1,proposals.sum())),initial_recall=float((truth&proposals).sum()/max(1,truth.sum())),
            mean_utility_difference=float(difference.mean()),utility_difference_95_ci=np.percentile(boot,[2.5,97.5]).tolist(),
            useful_initial_splits_outside_old_guard=sum(bool(r['label'] and not r['old_guard_allows'] and r['initial_probability']>.65 and r['physically_possible']) for r in subset))
        for group in ('candidate','baseline'):
            s[group]={k:sum(r[group][k] for r in subset) for k in ('deaths','captures','splits','physical_violations','hard_vetoes','projectile_captures')}
            s[group]['mean_utility']=float(np.mean([r[group]['utility'] for r in subset]))
        s['passed']=bool(s['cases']>=550 and s['positive_opportunities']>=50 and s['mean_utility_difference']>=-.03 and
            s['candidate']['deaths']<=s['baseline']['deaths']+max(2,int(.01*s['cases'])) and
            s['candidate']['physical_violations']==0 and s['candidate']['hard_vetoes']==0 and
            s['useful_initial_splits_outside_old_guard']>=10)
        result[algorithm]=s
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/learned_risk_v5')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=600)
    p.add_argument('--seed',type=int,default=4900001);p.add_argument('--horizon',type=int,default=60);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();jobs=[(list(range(s,min(s+8,args.seed+args.cases))),args.horizon) for s in range(args.seed,args.seed+args.cases,8)];rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_eval,initargs=(str(args.directory),)) as pool:
        for i,batch in enumerate(pool.map(evaluate,jobs)):
            rows.extend(batch)
            if i%10==0:print(json.dumps(dict(batches=i+1,total=len(jobs))),flush=True)
    sources=['src/evaluate_learned_risk.py','src/train_learned_risk.py','src/learned_risk.py','src/arena_policies.py','src/arena_tactics.py','envs/agar_env.py','src/heuristic_agent.py']
    report=dict(seed=args.seed,cases_requested=args.cases,horizon=args.horizon,rows=rows,summary=summarize(rows),
        source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        checkpoint_hashes={n:hashlib.sha256((args.directory/f'{n}.pt').read_bytes()).hexdigest() for n in ('ppo','diffusion')})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
