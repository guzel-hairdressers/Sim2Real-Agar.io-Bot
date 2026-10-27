"""Continue PPO movement with long-horizon growth and survival rewards.

The v8 split head is frozen. Only actuator bounds and can_split constrain
executed actions. Ordinary matches mix with escape curricula during fitting;
curriculum geometry and reward shaping are never inference-time rules.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from envs.arena_config import ArenaConfig
from src.arena_policies import load_package
from src.balanced_policies import physical_action
from src.benchmark_showcase import ROOT
from src.heuristic_agent import MasterHeuristicAgarBot
from src.learned_risk import load_policy
from src.train_arena_v2 import gae
from src.train_champion_ppo import HybridActorCritic


def growth_survival_reward(before_mass,after_mass,kills,deaths):
    growth=math.log(max(after_mass,1.)/max(before_mass,1.))
    if deaths:growth=min(growth,0.)  # Respawn mass is never growth credit.
    return float(growth+.10*kills-3.*deaths)


def worker(pipe,index,seed):
    logging.getLogger().setLevel(logging.ERROR);torch.set_num_threads(1)
    rng=np.random.default_rng(seed);episode=-1
    baseline=ROOT/'outputs/split_combat_v3';v8=ROOT/'outputs/learned_risk_v8'
    while True:
        episode+=1;episode_seed=seed+episode
        np.random.seed(episode_seed)
        env=ArenaConfig().make();obs,_=env.reset(seed=episode_seed)
        order=rng.permutation(7);learners=[f'player_{s}' for s in order[:2]];opponents={}
        for j,seat in enumerate(order[2:]):
            pid=f'player_{seat}'
            if j<2:
                alg='ppo' if episode%2==0 else 'diffusion'
                folder=baseline if alg=='ppo' else v8
                opponents[pid]=load_policy(alg,folder/f'{alg}.pt',ROOT/'outputs/arena_v2_cosine/critic.pt',episode_seed+j)
            else:opponents[pid]=MasterHeuristicAgarBot(profile=['apex','hunter','survivor'][j-2],seed=episode_seed+j)
        escape=bool(rng.random()<.35)
        if escape:
            actor=env.players[learners[int(rng.integers(2))]];predator=env.players[f'player_{order[4]}']
            signs=rng.choice([-1,1],2);actor.pos=(signs*rng.uniform(6.0,7.6,2)).astype(np.float32)
            actor.mass=float(rng.uniform(14,34));actor.yaw=float(rng.uniform(-np.pi,np.pi));actor.linear_vel=float(rng.uniform(0,1))
            displacement=-signs*rng.uniform(2.2,3.5,2)
            predator.pos=(actor.pos+displacement).astype(np.float32);predator.mass=float(rng.uniform(48,95))
            predator.yaw=float(math.atan2(-displacement[1],-displacement[0]));predator.linear_vel=.8
            env.reserve_mass=max(0,env.total_world_mass_cap-sum(p.mass for p in env.players.values())-len(env.food_positions)*.5)
            obs={p:env._get_obs(p) for p in env.player_ids}
        limit=600 if escape else 1800
        pipe.send(dict(obs=np.array([obs[p] for p in learners]),episode_seed=episode_seed,escape=escape))
        for step in range(limit):
            message=pipe.recv()
            if message is None:pipe.close();return
            before={p:(a.mass,a.kills,a.deaths) for p,a in env.players.items()}
            actions={p:physical_action(a,obs[p]) for p,a in zip(learners,message)}
            actions.update({p:a.predict(obs[p]) for p,a in opponents.items()})
            next_obs,_,_,_,_=env.step(actions)
            rewards=[];dead=[]
            for pid in learners:
                mass,kills,deaths=before[pid];actor=env.players[pid]
                rewards.append(growth_survival_reward(mass,actor.mass,actor.kills-kills,actor.deaths-deaths))
                dead.append(actor.deaths>deaths)
            for pid,policy in opponents.items():
                if env.players[pid].deaths>before[pid][2] and hasattr(policy,'reset'):policy.reset(episode_seed+env.step_count+int(pid[-1]))
            final=np.array([next_obs[p] for p in learners]);done=step+1==limit
            pipe.send(dict(final_obs=final,reward=np.array(rewards),deaths=np.array(dead),truncated=done))
            obs=next_obs
            if done:break


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/learned_risk_v8/ppo.pt')
    p.add_argument('--steps',type=int,default=120000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=6000001);p.add_argument('--rollout',type=int,default=256)
    p.add_argument('--time-horizon',type=float,default=20.);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(args.seed)
    package=load_package(args.base,'ppo');net=HybridActorCritic(70,split_architecture='obs_mlp')
    net.load_state_dict(package['state_dict']);net.train()
    for parameter in net.split_head.parameters():parameter.requires_grad_(False)
    reference=copy.deepcopy(net).eval()
    actor=list(net.actor_trunk.parameters())+list(net.cont_head.parameters())+[net.log_std]
    critic=list(net.critic_trunk.parameters())+list(net.critic_head.parameters())
    optimizer=torch.optim.Adam([{'params':actor,'lr':3e-5},{'params':critic,'lr':1e-4}],eps=1e-5)
    gamma=math.exp(-(1/30)/args.time_horizon)
    sources=['src/train_balanced_ppo.py','src/balanced_policies.py','src/train_champion_ppo.py',
             'src/train_arena_v2.py','envs/agar_env.py','envs/arena_config.py','src/heuristic_agent.py','src/learned_risk.py']
    config=dict(seed=args.seed,steps_requested=args.steps,workers=args.workers,rollout=args.rollout,
        discount_horizon_seconds=args.time_horizon,gamma=gamma,gae_lambda=.98,
        split_head_frozen=True,continuous_safety_override=False,escape_curriculum_fraction=.35,
        reward='log mass gain + 0.10 piece captures - 3.0 full deaths; no respawn growth credit',
        actor_learning_rate=3e-5,critic_learning_rate=1e-4,anchor_coefficient=.05,
        source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),arena_config=asdict(ArenaConfig()))
    (args.output/'training_config.json').write_text(json.dumps(config,indent=2))
    ctx=mp.get_context('spawn');parents=[];processes=[];episode_seeds=set()
    for i in range(args.workers):
        a,b=ctx.Pipe();proc=ctx.Process(target=worker,args=(b,i,args.seed+i*10000));proc.start();b.close()
        parents.append(a);processes.append(proc)
    def begin(pipe):
        result=pipe.recv();episode_seeds.add(result['episode_seed']);return result['obs']
    obs=np.array([begin(pipe) for pipe in parents]);total=0;updates=0;started=time.monotonic();history=[]
    try:
        while total<args.steps:
            records=[]
            for _ in range(args.rollout):
                batch=obs.reshape(-1,70);tensor=torch.tensor(batch,dtype=torch.float32)
                with torch.no_grad():
                    mu,std,logits,value=net(tensor);distribution=Normal(mu,std)
                    raw=distribution.sample();logp=distribution.log_prob(raw).sum(-1)
                    split=((logits.sigmoid()>.65)&(tensor[:,4]>.5)).float()
                    actions=torch.cat([raw,split[:,None]],1).numpy()
                for i,pipe in enumerate(parents):pipe.send(actions[i*2:i*2+2])
                result=[pipe.recv() for pipe in parents]
                final=np.array([r['final_obs'] for r in result]).reshape(-1,70)
                rewards=np.array([r['reward'] for r in result]).reshape(-1)
                deaths=np.array([r['deaths'] for r in result]).reshape(-1)
                trunc=np.repeat([r['truncated'] for r in result],2)
                records.append((batch,raw.numpy(),logp.numpy(),value.numpy(),rewards,deaths,trunc,final))
                obs=np.array([begin(pipe) if r['truncated'] else r['final_obs'] for pipe,r in zip(parents,result)])
                total+=len(batch)
            arrays=[np.asarray([r[k] for r in records]) for k in range(8)];shape=arrays[4].shape
            with torch.no_grad():next_values=net(torch.tensor(arrays[7].reshape(-1,70),dtype=torch.float32))[-1].numpy().reshape(shape)
            advantages,returns=gae(arrays[4],arrays[3],next_values,arrays[5],arrays[6],gamma=gamma,lam=.98)
            bo=torch.tensor(arrays[0].reshape(-1,70),dtype=torch.float32);ba=torch.tensor(arrays[1].reshape(-1,2),dtype=torch.float32)
            oldlp=torch.tensor(arrays[2].reshape(-1),dtype=torch.float32);adv=torch.tensor(advantages.reshape(-1),dtype=torch.float32)
            ret=torch.tensor(returns.reshape(-1),dtype=torch.float32);adv=(adv-adv.mean())/(adv.std()+1e-8)
            losses=[];kls=[]
            for _ in range(4):
                for ids in torch.randperm(len(bo)).split(256):
                    mu,std,_,value=net(bo[ids]);distribution=Normal(mu,std)
                    lp=distribution.log_prob(ba[ids]).sum(-1);ratio=(lp-oldlp[ids]).exp()
                    objective=torch.minimum(ratio*adv[ids],ratio.clamp(.9,1.1)*adv[ids]).mean()
                    with torch.no_grad():anchor=reference(bo[ids])[0]
                    imitation=nn.functional.mse_loss(mu,anchor)
                    loss=-objective+.25*nn.functional.mse_loss(value,ret[ids])+.05*imitation-.003*distribution.entropy().sum(-1).mean()
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],.5);optimizer.step()
                    losses.append(float(loss.detach()));kls.append(float((oldlp[ids]-lp).mean().detach()))
            updates+=1
            row=dict(steps=total,update=updates,reward=float(arrays[4].mean()),deaths=int(arrays[5].sum()),
                     loss=float(np.mean(losses)),approximate_kl=float(np.mean(kls)),steps_per_second=total/(time.monotonic()-started))
            history.append(row);print(json.dumps(row),flush=True)
            if updates%15==0 or total>=args.steps:
                checkpoint=copy.deepcopy(package);checkpoint['state_dict']={k:v.detach().cpu() for k,v in net.state_dict().items()}
                checkpoint['movement_control']='learned';checkpoint['split_control']='learned_full_control_physical_only'
                checkpoint['balanced_training']={**config,'steps_completed':total,'episode_seeds':sorted(episode_seeds)}
                torch.save(checkpoint,args.output/f'ppo_{total}.pt');torch.save(checkpoint,args.output/'ppo.pt')
                (args.output/'training_history.json').write_text(json.dumps(history,indent=2))
    finally:
        for pipe in parents:
            try:pipe.send(None)
            except (BrokenPipeError,EOFError):pass
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():proc.terminate()


if __name__=='__main__':main()
