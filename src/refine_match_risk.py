"""On-policy correction at actual split decisions in full seven-agent matches."""
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
from envs.arena_config import ArenaConfig
from src.benchmark_showcase import ROOT
from src.learned_risk import load_policy
from src.train_learned_risk import fit
from src.heuristic_agent import MasterHeuristicAgarBot


def branch(env,policies,first,pid,random_state,split,horizon=120):
    env=copy.deepcopy(env);policies=copy.deepcopy(policies);np.random.set_state(random_state)
    actor=env.players[pid];mass=actor.mass;kills=actor.kills;deaths=actor.deaths;samples=[]
    obs={p:env._get_obs(p) for p in env.player_ids}
    for step in range(horizon):
        old_deaths={p:a.deaths for p,a in env.players.items()}
        actions={p:a.copy() for p,a in first.items()} if step==0 else {p:a.predict(obs[p]) for p,a in policies.items()}
        actions[pid][2]=float(split and step==0 and obs[pid][4]>.5)
        obs,_,_,_,_=env.step(actions);samples.append(actor.mass)
        for p,policy in policies.items():
            if env.players[p].deaths>old_deaths[p] and hasattr(policy,'reset'):policy.reset(1+env.step_count+int(p[-1]))
    utility=.6*(actor.mass-mass)/mass+.4*(np.mean(samples)-mass)/mass+.06*(actor.kills-kills)-2*(actor.deaths-deaths)
    return dict(utility=float(utility),mass=actor.mass,captures=actor.kills-kills,deaths=actor.deaths-deaths)


def collect(job):
    seed,directory,horizon,sample_holds=job;logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    env=ArenaConfig().make();obs,_=env.reset(seed=seed);policies={};groups={};rows=[]
    for j,seat in enumerate(np.random.default_rng(seed).permutation(7)):
        pid=f'player_{seat}'
        if j<4:
            algorithm='ppo' if j<2 else 'diffusion';groups[pid]=algorithm
            policies[pid]=load_policy(algorithm,Path(directory)/f'{algorithm}.pt',ROOT/'outputs/arena_v2_cosine/critic.pt',seed+j)
        else:policies[pid]=MasterHeuristicAgarBot(profile=['apex','hunter','survivor'][j-4],seed=seed+j)
    for _ in range(1800):
        actions={p:a.predict(obs[p]) for p,a in policies.items()};old_deaths={p:a.deaths for p,a in env.players.items()}
        for pid,algorithm in groups.items():
            requested=bool(actions[pid][2]>.5)
            if len(rows)>=24 or obs[pid][4]<=.5:continue
            if not requested and not (sample_holds and env.step_count%300==0):continue
            random_state=np.random.get_state()
            split=branch(env,policies,actions,pid,random_state,True,horizon)
            hold=branch(env,policies,actions,pid,random_state,False,horizon)
            np.random.set_state(random_state)  # Queries cannot alter the real rollout.
            advantage=split['utility']-hold['utility']
            rows.append(dict(seed=seed,query_step=env.step_count,actor_id=pid,algorithm=algorithm,
                obs=obs[pid].tolist(),split=split,hold=hold,advantage=advantage,label=bool(advantage>.03),
                physically_possible=True,requested=requested,horizon=horizon,
                source=f'actual full-match state; {horizon/30:g}-second paired outcomes'))
        obs,_,_,_,_=env.step(actions)
        for pid,policy in policies.items():
            if env.players[pid].deaths>old_deaths[pid] and hasattr(policy,'reset'):policy.reset(seed+env.step_count+int(pid[-1]))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'outputs/learned_risk_v4')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/learned_risk_v5');p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--seed',type=int,default=4830001);p.add_argument('--epochs',type=int,default=240)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--reuse-data',action='store_true')
    p.add_argument('--selection',choices=['utility','loss','calibrated_utility'],default='utility')
    p.add_argument('--policy-directory',type=Path);p.add_argument('--horizon',type=int,default=120)
    p.add_argument('--sample-holds',action='store_true');p.add_argument('--prior-corrections',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);path=args.output/'counterfactuals.jsonl'
    if not args.reuse_data:
        jobs=[(args.seed+i,str(args.policy_directory or args.base),args.horizon,args.sample_holds) for i in range(args.episodes)]
        with path.open('w') as f,ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
            for i,batch in enumerate(pool.map(collect,jobs)):
                for row in batch:f.write(json.dumps(row)+'\n')
                f.flush();print(json.dumps(dict(stage='full_match_corrections',episodes=i+1,total=args.episodes,queries=len(batch))),flush=True)
    corrections=[json.loads(s) for s in path.read_text().splitlines()]
    prior=[json.loads(s) for s in args.prior_corrections.read_text().splitlines()] if args.prior_corrections else []
    prior_seeds=sorted({r['seed'] for r in prior});prior_train=set(prior_seeds[:int(len(prior_seeds)*.8)])
    original=[json.loads(s) for s in (args.base/'counterfactuals.jsonl').read_text().splitlines()]
    generic_path=ROOT/'outputs/learned_risk_v4_refined/counterfactuals.jsonl';generic=[];generic_source=generic_path
    if generic_path.exists():
        for line in generic_path.read_text().splitlines():
            try:generic.append(json.loads(line))
            except json.JSONDecodeError:pass
        # Retain only complete, flushed batches from the exploratory sampler.
        cutoff=4820001+((max(r['seed'] for r in generic)-4820001)//8)*8
        generic=[r for r in generic if r['seed']<cutoff]
    elif (args.output/'generic_corrections_used.json').exists():
        generic_source=args.output/'generic_corrections_used.json'
        generic=json.loads(generic_source.read_text())['records']
    generic_seeds=sorted({r['seed'] for r in generic});generic_train=set(generic_seeds[:int(len(generic_seeds)*.8)])
    (args.output/'acceptance_protocol.json').write_bytes((args.base/'acceptance_protocol.json').read_bytes())
    metadata={}
    for algorithm in ('ppo','diffusion'):
        package=torch.load(args.base/f'{algorithm}.pt',map_location='cpu',weights_only=True);old=package['risk_training']
        cut=old['seed']+int(old['cases']*.8);newcut=args.seed+int(args.episodes*.8)
        train=[r for r in original if r['algorithm']==algorithm and r['seed']<cut]
        validation=[r for r in original if r['algorithm']==algorithm and r['seed']>=cut]
        newtrain=[r for r in corrections if r['algorithm']==algorithm and r['seed']<newcut]+[r for r in prior if r['algorithm']==algorithm and r['seed'] in prior_train]
        newval=[r for r in corrections if r['algorithm']==algorithm and r['seed']>=newcut]+[r for r in prior if r['algorithm']==algorithm and r['seed'] not in prior_train]
        gtrain=[r for r in generic if r['algorithm']==algorithm and r['seed'] in generic_train]
        gval=[r for r in generic if r['algorithm']==algorithm and r['seed'] not in generic_train]
        head,history=fit(train+gtrain*2+newtrain*6,validation+gval*2+newval*6,args.seed+(algorithm=='diffusion'),args.epochs,args.selection)
        if algorithm=='ppo':
            package['state_dict']={k:v for k,v in package['state_dict'].items() if not k.startswith('split_head.')}
            package['state_dict'].update({'split_head.'+k:v for k,v in head.state_dict().items()})
        else:package['split_head_state_dict']=head.state_dict()
        config=copy.deepcopy(old);config['base_source_hashes']=copy.deepcopy(old['source_hashes'])
        config['match_refinement']=dict(seed=args.seed,episodes=args.episodes,horizon=args.horizon,selection=args.selection,
            sample_holds=args.sample_holds,policy_directory=str(args.policy_directory or args.base),
            policy_model_sha256={a:hashlib.sha256(((args.policy_directory or args.base)/f'{a}.pt').read_bytes()).hexdigest() for a in ('ppo','diffusion')},
            prior_seeds=prior_seeds,prior_data_sha256=hashlib.sha256(args.prior_corrections.read_bytes()).hexdigest() if prior else None,
            epochs=args.epochs,unique_training_states=len(newtrain),unique_validation_states=len(newval),
            replay_weight=6,generic_training_states=len(gtrain),generic_validation_states=len(gval),generic_replay_weight=2,
            generic_data_sha256=hashlib.sha256(generic_source.read_bytes()).hexdigest() if generic else None,
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_model_sha256=hashlib.sha256((args.base/f'{algorithm}.pt').read_bytes()).hexdigest())
        for file in ('src/refine_match_risk.py','src/refine_learned_risk.py','src/train_learned_risk.py'):
            config['source_hashes'][file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        package['risk_training']=config;torch.save(package,args.output/f'{algorithm}.pt')
        report=dict(unique_original_training_states=len(train),unique_full_match_training_states=len(newtrain),
                    unique_full_match_validation_states=len(newval),generic_training_states=len(gtrain),generic_validation_states=len(gval),history=history)
        (args.output/f'{algorithm}_validation.json').write_text(json.dumps(report,indent=2));metadata[algorithm]=config
        print(json.dumps(dict(algorithm=algorithm,match_train=len(newtrain),match_validation=len(newval))),flush=True)
    (args.output/'training_config.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
