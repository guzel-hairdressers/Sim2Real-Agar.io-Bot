"""PPO continuation with learned movement and physical action limits only.

Historical packages retain their original inference adapters. New PPO
packages explicitly declare learned movement; no threat/wall steering rule
may overwrite their action. Diffusion is unchanged in this continuation.
"""
import numpy as np
from src.arena_policies import ArenaPPO,load_package
from src.learned_risk import load_policy as historical_load


def physical_action(raw,obs):
    action=np.clip(np.asarray(raw,dtype=np.float32),[0,-1,0],[1,1,1]).astype(np.float32)
    if obs[4]<=.5:action[2]=0.
    return action


class BalancedPPO(ArenaPPO):
    def __init__(self,path):
        package=load_package(path,'ppo')
        if package.get('split_control')!='learned_full_control_physical_only':
            raise ValueError('Learned movement requires physical-only split control')
        super().__init__(path,assisted=False)

    def predict(self,obs,deterministic=True):
        self.assisted=False
        raw=super().predict(obs,deterministic)
        action=physical_action(raw,obs)
        self.interventions+=int(np.max(np.abs(raw-action))>1e-5)
        return action


def load_policy(algorithm,path,critic_path=None,seed=42):
    package=load_package(path,algorithm)
    mode=package.get('movement_control','assisted')
    if mode=='learned':
        if algorithm!='ppo':raise ValueError('Learned movement continuation supports PPO only')
        return BalancedPPO(path)
    if mode!='assisted':raise ValueError('Unknown movement control mode')
    return historical_load(algorithm,path,critic_path,seed)
