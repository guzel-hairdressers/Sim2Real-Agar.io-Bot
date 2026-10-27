"""Learn split risk from reactive split/hold outcome rollouts, not risk rules."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn

from src.arena_policies import ArenaPPO, ArenaDiffusion
from src.heuristic_agent import MasterHeuristicAgarBot
from src.learned_risk import physical_split_filter
from src.split_head import BinarySplitHead
from src.train_split_combat import make_encounter, ROOT

POLICIES={}


def initialize():
    logging.getLogger().setLevel(logging.ERROR);torch.set_num_threads(1)
    POLICIES['ppo']=ArenaPPO(ROOT/'outputs/split_combat_v3/ppo.pt',assisted=False)
    POLICIES['diffusion']=ArenaDiffusion(ROOT/'outputs/split_combat_v3/diffusion.pt',ROOT/'outputs/arena_v2_cosine/critic.pt',assisted=False)


def risk_scene(seed):
    env,_,kind=make_encounter(seed)
    rng=np.random.default_rng(seed+91000000);a,b,rival=env.players.values()
    if rng.random()<.3:
        a.mass=float(rng.uniform(36,175));b.mass=float(rng.uniform(8,min(75,a.mass*.85)))
    if rng.random()<.25:
        angle=float(rng.uniform(-np.pi,np.pi));distance=float(rng.uniform(1.3,4.))
        rival.pos=np.clip(a.pos+distance*np.array([np.cos(angle),np.sin(angle)]),-8.2,8.2).astype(np.float32)
        rival.mass=float(rng.uniform(.6,1.5)*a.mass)
        rival.yaw=angle+np.pi;rival.linear_vel=float(rng.uniform(0,rival.max_speed))
    if rng.random()<.2:
        b.pos=np.clip(a.pos+rng.uniform(4,7)*np.array([np.cos(a.yaw),np.sin(a.yaw)]),-8.2,8.2).astype(np.float32)
    if rng.random()<.25:
        a.split_cooldown=0.;a.split(step_index=0)
        a.split_cooldown=float(rng.choice([0.,0.,.2,.6]))
        for piece in a.pieces:
            piece.split_birth_step=-1;piece.vel*=.25
            piece.merge_cooldown=min(piece.merge_cooldown,7.2)
    env.reserve_mass=max(0,env.total_world_mass_cap-sum(p.mass for p in env.players.values())-len(env.food_positions)*.5)
    return env,kind


def outcome(env,policy,seed,split,horizon):
    np.random.seed(seed)
    env=copy.deepcopy(env);policy.reset(seed=seed)
    initial=env.players['player_0'].mass
    opponents={pid:MasterHeuristicAgarBot(profile='hunter' if pid=='player_2' else 'survivor',seed=seed+i)
               for i,pid in enumerate(['player_1','player_2'])}
    mass_samples=[];lost=0.;projectile=0;deaths=0
    obs={pid:env._get_obs(pid) for pid in env.player_ids}
    for step in range(horizon):
        raw=policy.predict(obs['player_0']);raw[2]=float(split and step==0)
        actions={'player_0':physical_split_filter(raw,obs['player_0'])}
        actions.update({pid:p.predict(obs[pid]) for pid,p in opponents.items()})
        obs,_,_,_,_=env.step(actions)
        for e in env.predation_events:
            if e['prey']=='player_0':lost+=e['mass_gained']/.7
            if e['predator']=='player_0' and 0<=e['projectile_age_steps']<=18:projectile+=1
        mass_samples.append(env.players['player_0'].mass)
        if env.players['player_0'].deaths>deaths:
            policy.reset(seed+env.step_count);deaths=env.players['player_0'].deaths
    actor=env.players['player_0']
    # Reward trajectory mass as well as the end state, so an early capture
    # has value without labeling nearby threats as forbidden by definition.
    utility=.6*(actor.mass-initial)/initial+.4*(np.mean(mass_samples)-initial)/initial
    utility+=.06*actor.kills-2.*actor.deaths
    return dict(utility=float(utility),mass=actor.mass,deaths=actor.deaths,
                captures=actor.kills,lost_mass=lost,projectile_captures=projectile)


def collect(job):
    seeds,horizon=job;rows=[]
    for seed in seeds:
        env,kind=risk_scene(seed)
        if env.total_world_mass>env.total_world_mass_cap+1e-4:continue
        obs=env._get_obs('player_0');possible=bool(obs[4]>.5)
        for algorithm,policy in POLICIES.items():
            split=hold=None;advantage=0.
            if possible:
                split=outcome(env,policy,seed,True,horizon)
                hold=outcome(env,policy,seed,False,horizon)
                advantage=split['utility']-hold['utility']
            rows.append(dict(seed=seed,kind=kind,algorithm=algorithm,obs=obs.tolist(),
                physically_possible=possible,split=split,hold=hold,advantage=advantage,
                label=bool(possible and advantage>.03)))
    return rows


def fit(train,validation,seed,epochs,selection_metric='utility'):
    if selection_metric not in ('utility','loss','calibrated_utility'):
        raise ValueError('Unknown development selection metric')
    torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(1)
    model=BinarySplitHead();o=torch.tensor([r['obs'] for r in train],dtype=torch.float32)
    y=torch.tensor([r['label'] for r in train],dtype=torch.float32)
    vo=torch.tensor([r['obs'] for r in validation],dtype=torch.float32)
    vy=torch.tensor([r['label'] for r in validation],dtype=torch.float32)
    model.mean.copy_(o.mean(0));model.scale.copy_(o.std(0).clamp_min(.04))
    pos=np.flatnonzero(y.numpy());neg=np.flatnonzero(1-y.numpy())
    if len(pos)<40:raise ValueError(f'Not enough measured positive outcomes: {len(pos)}')
    weights=torch.tensor([1+min(5,abs(r['advantage'])*2) for r in train],dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=5e-4)
    with torch.no_grad():model.net[-1].bias.fill_(float(np.log(len(pos)/len(neg))))
    vadv=torch.tensor([r['advantage'] for r in validation],dtype=torch.float32)
    vweights=1+torch.clamp(vadv.abs()*2,max=5)
    best=None;state=None;history=[];candidates=[]
    for epoch in range(epochs):
        losses=[]
        for _ in range(40):
            # Preserve the observed class prior. Artificial 50/50 balancing
            # made rare beneficial actions look far too common at inference.
            ids=rng.integers(0,len(train),256)
            loss=(nn.functional.binary_cross_entropy_with_logits(model(o[ids]).squeeze(-1),y[ids],reduction='none')*weights[ids]).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
        if epoch%10==0 or epoch==epochs-1:
            with torch.inference_mode():
                logits=model(vo).squeeze(-1)
                vl=nn.functional.binary_cross_entropy_with_logits(logits,vy,reduction='none')
                score=float((vl*vweights).mean())
                proposal=logits.sigmoid()>.65
                gain=float((proposal*vadv).mean())
            selection=(gain,-score) if selection_metric=='utility' else (-score,gain)
            if best is None or selection>best:best=selection;state=copy.deepcopy(model.state_dict())
            if selection_metric=='calibrated_utility':candidates.append((score,gain,copy.deepcopy(model.state_dict())))
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_weighted_bce=score,
                     validation_policy_advantage=gain,validation_proposals=int(proposal.sum()))
            history.append(row);print(json.dumps(row),flush=True)
    if selection_metric=='calibrated_utility':
        # Limit probability overfitting without selecting the trivial no-split
        # head solely for its loss on a predominantly negative dataset.
        ceiling=min(c[0] for c in candidates)*1.25
        state=max((c for c in candidates if c[0]<=ceiling),key=lambda c:(c[1],-c[0]))[2]
    model.load_state_dict(state);return model.eval(),history


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/learned_risk_v4')
    p.add_argument('--cases',type=int,default=4000);p.add_argument('--seed',type=int,default=4800001)
    p.add_argument('--horizon',type=int,default=60);p.add_argument('--epochs',type=int,default=240)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--reuse-data',action='store_true')
    p.add_argument('--resume-data',action='store_true',help='Keep complete matching non-virus cases; regenerate virus and unfinished cases')
    args=p.parse_args()
    if args.reuse_data and args.resume_data:p.error('Choose either --reuse-data or --resume-data')
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'counterfactuals.jsonl'
    preserved=[];kept=set()
    if args.resume_data and path.exists():
        cached=[]
        for line in path.read_text().splitlines():
            try:cached.append(json.loads(line))
            except json.JSONDecodeError:pass  # A terminated writer may leave one partial last line.
        for seed in sorted({r['seed'] for r in cached}):
            pair=[r for r in cached if r['seed']==seed]
            env,_=risk_scene(seed)
            if len(pair)==2 and len(env.viruses)==0 and all(np.array_equal(np.array(r['obs'],np.float32),env._get_obs('player_0')) for r in pair):
                preserved.extend(pair);kept.add(seed)
        print(json.dumps(dict(stage='resume',preserved_nonvirus_cases=len(kept))),flush=True)
    if not args.reuse_data:
        pending=[s for s in range(args.seed,args.seed+args.cases) if s not in kept]
        jobs=[(pending[i:i+16],args.horizon) for i in range(0,len(pending),16)]
        with path.open('w') as f,ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            for row in preserved:f.write(json.dumps(row)+'\n')
            f.flush()
            for i,batch in enumerate(pool.map(collect,jobs)):
                for row in batch:f.write(json.dumps(row)+'\n')
                f.flush()
                if i%10==0:print(json.dumps(dict(stage='collect',batches=i+1,total=len(jobs))),flush=True)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows.sort(key=lambda r:(r['seed'],r['algorithm']))
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    config=dict(vars(args));config['output']=os.path.relpath(args.output.resolve(),ROOT);config['data_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    config['labels']='Measured split-minus-hold utility > 0.03; no risk-heuristic labels'
    config['movement_weights_frozen']=True
    config['sampling']='Uniform observed case prior; weighted BCE; development selection by measured action advantage'
    config['randomness']='Global NumPy reset per branch; cloned environment RNG; seeded policy and opponent RNGs'
    config['preserved_nonvirus_cases']=len(kept)
    config['curriculum']='v4 reactive rivals, varied masses, distant targets and already split states'
    config['source_hashes']={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in (
        'src/train_learned_risk.py','src/learned_risk.py','src/train_split_combat.py','envs/agar_env.py',
        'src/arena_policies.py','src/arena_tactics.py','src/heuristic_agent.py')}
    (args.output/'training_config.json').write_text(json.dumps(config,indent=2))
    for algorithm in ('ppo','diffusion'):
        subset=[r for r in rows if r['algorithm']==algorithm];cut=args.seed+int(args.cases*.8)
        train=[r for r in subset if r['seed']<cut];validation=[r for r in subset if r['seed']>=cut]
        head,history=fit(train,validation,args.seed+(algorithm=='diffusion'),args.epochs)
        with torch.inference_mode():proposals=head(torch.tensor([r['obs'] for r in validation],dtype=torch.float32)).sigmoid().squeeze(-1).numpy()>.65
        truth=np.array([r['label'] for r in validation]);metrics=dict(train_cases=len(train),train_positive=sum(r['label'] for r in train),validation_cases=len(validation),validation_positive=int(truth.sum()),recall=float((proposals&truth).sum()/max(1,truth.sum())),precision=float((proposals&truth).sum()/max(1,proposals.sum())))
        package=torch.load(ROOT/f'outputs/split_combat_v3/{algorithm}.pt',map_location='cpu',weights_only=True)
        package['split_control']='learned_risk_physical_only';package['risk_training']=config
        if algorithm=='ppo':
            package['state_dict']={k:v for k,v in package['state_dict'].items() if not k.startswith('split_head.')}
            package['state_dict'].update({'split_head.'+k:v for k,v in head.state_dict().items()})
        else:package['split_head_state_dict']=head.state_dict()
        torch.save(package,args.output/f'{algorithm}.pt')
        (args.output/f'{algorithm}_validation.json').write_text(json.dumps(dict(metrics=metrics,history=history),indent=2))
        print(json.dumps(dict(algorithm=algorithm,**metrics)),flush=True)


if __name__=='__main__':main()
