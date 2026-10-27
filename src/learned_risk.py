"""Learned split risk with only simulator feasibility enforced at execution.

The existing movement interventions are retained, but cannot overwrite the
model's split choice. No threat distance, wall, virus, prey mass, attack angle
or interception test is allowed to veto a physically possible split here.
"""
import numpy as np
from src.arena_policies import ArenaPPO, ArenaDiffusion, load_package
from src.arena_tactics import safety_filter


def physical_split_filter(raw,obs):
    raw=np.asarray(raw,dtype=np.float32)
    action=safety_filter(raw,obs)
    # obs[4] is the simulator's can_split flag: eligible mass, piece capacity
    # and cooldown. Restore the learned command after movement intervention.
    action[2]=np.clip(raw[2],0,1) if obs[4]>.5 else 0.
    return action


class LearnedRiskPPO(ArenaPPO):
    def __init__(self,path):
        super().__init__(path,assisted=False)

    def predict(self,obs,deterministic=True):
        self.assisted=False
        raw=super().predict(obs,deterministic)
        action=physical_split_filter(raw,obs)
        self.interventions+=int(np.max(np.abs(action-raw))>1e-5)
        return action


class LearnedRiskDiffusion(ArenaDiffusion):
    def __init__(self,path,critic_path=None,seed=42):
        super().__init__(path,critic_path,seed=seed,assisted=False)

    def predict(self,obs,deterministic=True):
        self.assisted=False
        raw=super().predict(obs,deterministic)
        action=physical_split_filter(raw,obs)
        self.interventions+=int(np.max(np.abs(action-raw))>1e-5)
        return action


def load_policy(algorithm,path,critic_path=None,seed=42):
    package=load_package(path,algorithm)
    mode=package.get('split_control','legacy_risk_filter')
    if mode not in ('legacy_risk_filter','learned_risk_physical_only'):
        raise ValueError('Unsupported split control mode')
    learned=mode=='learned_risk_physical_only'
    if algorithm=='ppo':return LearnedRiskPPO(path) if learned else ArenaPPO(path)
    if algorithm=='diffusion':
        cls=LearnedRiskDiffusion if learned else ArenaDiffusion
        return cls(path,critic_path,seed=seed)
    raise ValueError('Unsupported algorithm')
