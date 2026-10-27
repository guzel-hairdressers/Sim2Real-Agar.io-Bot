"""Correct learned split risk using actual policy states and longer outcomes.

Queries the first/second requested split during a reactive rollout. Labels are
four-second split/hold returns, not manually forbidden states or risk rules.
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
from src.benchmark_showcase import ROOT
from src.learned_risk import load_policy
from src.train_learned_risk import risk_scene,fit
from src.heuristic_agent import MasterHeuristicAgarBot

POLICIES={}


def initialize(directory):
    logging.getLogger().setLevel(logging.ERROR);torch.set_num_threads(1)
    for a in ('ppo','diffusion'):
        POLICIES[a]=load_policy(a,Path(directory)/f'{a}.pt',ROOT/'outputs/arena_v2_cosine/critic.pt')


def branch(context,split,horizon=120):
    env,policy,opponents,first,random_state=context
    env=copy.deepcopy(env);policy=copy.deepcopy(policy);opponents=copy.deepcopy(opponents)
    np.random.set_state(random_state)
    actor=env.players['player_0'];initial=actor.mass;kills=actor.kills;deaths=actor.deaths;last_deaths=deaths
    samples=[];obs={pid:env._get_obs(pid) for pid in env.player_ids}
    for step in range(horizon):
        action=first.copy() if step==0 else policy.predict(obs['player_0'])
        action[2]=float(split and step==0 and obs['player_0'][4]>.5)
        actions={'player_0':action,**{pid:p.predict(obs[pid]) for pid,p in opponents.items()}}
        obs,_,_,_,_=env.step(actions);samples.append(actor.mass)
        if actor.deaths>last_deaths:policy.reset(env.step_count+1);last_deaths=actor.deaths
    utility=.6*(actor.mass-initial)/initial+.4*(np.mean(samples)-initial)/initial+.06*(actor.kills-kills)-2*(actor.deaths-deaths)
    return dict(utility=float(utility),mass=actor.mass,captures=actor.kills-kills,deaths=actor.deaths-deaths)


def context_for(seed,algorithm):
    np.random.seed(seed);env,_=risk_scene(seed)
    if env.total_world_mass>env.total_world_mass_cap+1e-4:return None
    policy=POLICIES[algorithm];policy.reset(seed=seed)
    opponents={pid:MasterHeuristicAgarBot(profile='hunter' if pid=='player_2' else 'survivor',seed=seed+i)
               for i,pid in enumerate(['player_1','player_2'])}
    obs={pid:env._get_obs(pid) for pid in env.player_ids};requests=0;fallback=None;deaths=0
    desired=1 if seed%2 else 2
    for step in range(90):
        action=policy.predict(obs['player_0'])
        possible=obs['player_0'][4]>.5
        if possible and action[2]>.5:requests+=1
        if possible and ((action[2]>.5 and requests>=desired) or (step>=30 and fallback is None)):
            context=(copy.deepcopy(env),copy.deepcopy(policy),copy.deepcopy(opponents),action.copy(),np.random.get_state())
            if action[2]>.5 and requests>=desired:return context,step,requests
            fallback=(context,step,requests)
        obs,_,_,_,_=env.step({'player_0':action,**{pid:p.predict(obs[pid]) for pid,p in opponents.items()}})
        if env.players['player_0'].deaths>deaths:policy.reset(seed+env.step_count);deaths=env.players['player_0'].deaths
    return fallback


def collect(job):
    seeds,=job;rows=[]
    for seed in seeds:
        for algorithm in ('ppo','diffusion'):
            selected=context_for(seed,algorithm)
            if selected is None:continue
            context,step,request=selected;obs=context[0]._get_obs('player_0')
            split=branch(context,True);hold=branch(context,False);advantage=split['utility']-hold['utility']
            rows.append(dict(seed=seed,algorithm=algorithm,obs=obs.tolist(),label=bool(advantage>.03),
                advantage=advantage,split=split,hold=hold,query_step=step,query_request=request,
                physically_possible=True,source='on-policy four-second corrective trial'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'outputs/learned_risk_v4')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/learned_risk_v4_refined')
    p.add_argument('--cases',type=int,default=600);p.add_argument('--seed',type=int,default=4820001)
    p.add_argument('--epochs',type=int,default=240);p.add_argument('--workers',type=int,default=4);p.add_argument('--reuse-data',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);path=args.output/'counterfactuals.jsonl'
    if not args.reuse_data:
        jobs=[(list(range(s,min(s+8,args.seed+args.cases))),) for s in range(args.seed,args.seed+args.cases,8)]
        with path.open('w') as f,ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(args.base),)) as pool:
            for i,batch in enumerate(pool.map(collect,jobs)):
                for row in batch:f.write(json.dumps(row)+'\n')
                f.flush()
                if i%10==0:print(json.dumps(dict(stage='corrective_collect',batches=i+1,total=len(jobs))),flush=True)
    corrections=[json.loads(s) for s in path.read_text().splitlines()]
    original=[json.loads(s) for s in (args.base/'counterfactuals.jsonl').read_text().splitlines()]
    protocol=json.loads((args.base/'acceptance_protocol.json').read_text())
    (args.output/'acceptance_protocol.json').write_text(json.dumps(protocol,indent=2))
    for algorithm in ('ppo','diffusion'):
        package=torch.load(args.base/f'{algorithm}.pt',map_location='cpu',weights_only=True)
        old=package['risk_training'];cut=old['seed']+int(old['cases']*.8);newcut=args.seed+int(args.cases*.8)
        train=[r for r in original if r['algorithm']==algorithm and r['seed']<cut]
        validation=[r for r in original if r['algorithm']==algorithm and r['seed']>=cut]
        newtrain=[r for r in corrections if r['algorithm']==algorithm and r['seed']<newcut]
        newval=[r for r in corrections if r['algorithm']==algorithm and r['seed']>=newcut]
        # Emphasize encountered policy states; these are replay weights, not
        # additional independent cases. Unique counts are reported separately.
        head,history=fit(train+newtrain*4,validation+newval*4,args.seed+(algorithm=='diffusion'),args.epochs)
        if algorithm=='ppo':
            package['state_dict']={k:v for k,v in package['state_dict'].items() if not k.startswith('split_head.')}
            package['state_dict'].update({'split_head.'+k:v for k,v in head.state_dict().items()})
        else:package['split_head_state_dict']=head.state_dict()
        config=copy.deepcopy(old);config['refinement']=dict(seed=args.seed,cases=args.cases,horizon=120,epochs=args.epochs,
            unique_training_states=len(newtrain),unique_validation_states=len(newval),replay_weight=4,
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_model_sha256=hashlib.sha256((args.base/f'{algorithm}.pt').read_bytes()).hexdigest())
        config['source_hashes']['src/refine_learned_risk.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        package['risk_training']=config;torch.save(package,args.output/f'{algorithm}.pt')
        report=dict(unique_original_training_states=len(train),unique_corrective_training_states=len(newtrain),
                    unique_original_validation_states=len(validation),unique_corrective_validation_states=len(newval),history=history)
        (args.output/f'{algorithm}_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(algorithm=algorithm,corrective_train=len(newtrain),corrective_validation=len(newval))),flush=True)
    (args.output/'training_config.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
