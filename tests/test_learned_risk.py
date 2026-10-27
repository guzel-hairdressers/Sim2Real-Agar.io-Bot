import unittest
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from envs.agar_env import PartiallyObservableAgarEnv,SubPiece
from src.learned_risk import physical_split_filter,load_policy,LearnedRiskPPO,LearnedRiskDiffusion
from src.train_learned_risk import outcome
from src.arena_policies import TacticalTeacher
from src.refine_match_risk import branch as match_branch
from src.heuristic_agent import MasterHeuristicAgarBot
from envs.arena_config import ArenaConfig

ROOT=Path(__file__).resolve().parents[1]


def scene(mass=90,other=20,pos=(0,0),target=(1.5,0)):
    env=PartiallyObservableAgarEnv(arena_size=18,num_players=2,num_food=0,num_viruses=0,
                                 observation_version=2,dt=1/30)
    env.reset(seed=79);a,b=env.players.values();a.mass=mass;b.mass=other
    a.pos=np.array(pos,np.float32);b.pos=np.array(target,np.float32);a.yaw=0
    return env


class PhysicalSplitTests(unittest.TestCase):
    def test_counterfactual_query_cannot_mutate_live_world_or_policy(self):
        env=ArenaConfig().make();obs,_=env.reset(seed=79)
        policies={'player_0':LearnedRiskDiffusion(ROOT/'outputs/split_combat_v3/diffusion.pt',ROOT/'outputs/arena_v2_cosine/critic.pt')}
        policies.update({pid:MasterHeuristicAgarBot(seed=79+i) for i,pid in enumerate(env.player_ids[1:])})
        first={pid:p.predict(obs[pid]) for pid,p in policies.items()}
        poses={pid:p.pos.copy() for pid,p in env.players.items()};mass={pid:p.mass for pid,p in env.players.items()}
        generator=policies['player_0'].policy.generator.get_state().clone()
        observations=policies['player_0'].observations
        buffers=[a.copy() for a in policies['player_0'].policy.action_buffer]
        env_rng=copy.deepcopy(env.np_random.bit_generator.state)
        random_state=np.random.get_state()
        match_branch(env,policies,first,'player_0',random_state,True)
        np.random.set_state(random_state)
        self.assertEqual(env.step_count,0)
        for pid,p in env.players.items():
            np.testing.assert_array_equal(p.pos,poses[pid]);self.assertEqual(p.mass,mass[pid])
        self.assertEqual(env.np_random.bit_generator.state,env_rng)
        self.assertEqual(policies['player_0'].observations,observations)
        torch.testing.assert_close(policies['player_0'].policy.generator.get_state(),generator,rtol=0,atol=0)
        self.assertEqual(len(policies['player_0'].policy.action_buffer),len(buffers))
        for a,b in zip(policies['player_0'].policy.action_buffer,buffers):np.testing.assert_array_equal(a,b)

    def test_virus_fragmentation_outcome_uses_controlled_randomness(self):
        env=PartiallyObservableAgarEnv(arena_size=18,num_players=3,num_food=0,num_viruses=0,
                                     observation_version=2,dt=1/30,max_steps=90)
        env.reset(seed=79);a,b,c=env.players.values();a.mass,b.mass,c.mass=90,20,20
        a.pos=np.array([0.,0.],np.float32);a.yaw=0.
        b.pos=np.array([1.5,0.],np.float32);c.pos=np.array([-8.,-8.],np.float32)
        env.viruses=np.array([[1.,0.,.65]],np.float32);env.reserve_mass=270.
        np.random.seed(123)
        with patch.object(np.random,'uniform',wraps=np.random.uniform) as draws:
            first=outcome(env,TacticalTeacher(),79,True,60)
            self.assertGreater(draws.call_count,0)
        np.random.seed(987)
        second=outcome(env,TacticalTeacher(),79,True,60)
        self.assertEqual(first,second)

    def test_both_loaded_models_cannot_apply_legacy_risk_veto(self):
        for algorithm,expected in [('ppo',LearnedRiskPPO),('diffusion',LearnedRiskDiffusion)]:
            package=torch.load(ROOT/f'outputs/split_combat_v3/{algorithm}.pt',map_location='cpu',weights_only=True)
            package['split_control']='learned_risk_physical_only'
            head=package['state_dict'] if algorithm=='ppo' else package['split_head_state_dict']
            prefix='split_head.' if algorithm=='ppo' else ''
            for key,tensor in head.items():
                if key.startswith(prefix+'net.'):tensor.zero_()
            head[prefix+'net.4.bias'].fill_(10.)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.pt';torch.save(package,path)
                model=load_policy(algorithm,path,ROOT/'outputs/arena_v2_cosine/critic.pt')
                self.assertIsInstance(model,expected)
                model.assisted=True  # Generic callers cannot re-enable the old veto.
                self.assertEqual(model.predict(scene(other=180)._get_obs('player_0'))[2],1)

    def test_unknown_control_mode_is_rejected(self):
        package=torch.load(ROOT/'outputs/split_combat_v3/ppo.pt',map_location='cpu',weights_only=True)
        package['split_control']='unknown'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save(package,path)
            with self.assertRaisesRegex(ValueError,'Unsupported split control'):
                load_policy('ppo',path)

    def test_nearby_predator_cannot_veto_requested_split(self):
        env=scene(other=180)
        obs=env._get_obs('player_0');self.assertEqual(obs[4],1)
        self.assertEqual(physical_split_filter([.8,0,1],obs)[2],1)

    def test_wall_and_virus_cannot_veto_requested_split(self):
        env=scene(pos=(7.8,0),target=(6,0))
        env.viruses=np.array([[7.7,.5,.65]],np.float32)
        obs=env._get_obs('player_0');self.assertEqual(obs[4],1)
        self.assertEqual(physical_split_filter([.8,0,1],obs)[2],1)

    def test_target_mass_and_angle_are_not_execution_rules(self):
        env=scene(mass=45,other=30,target=(-1.5,0))
        obs=env._get_obs('player_0');self.assertEqual(obs[4],1)
        self.assertEqual(physical_split_filter([.8,0,1],obs)[2],1)

    def test_no_rule_can_invent_a_split(self):
        for env in (scene(),scene(other=180),scene(pos=(7.8,0),target=(6,0))):
            self.assertEqual(physical_split_filter([.8,0,0],env._get_obs('player_0'))[2],0)

    def test_physical_mass_and_cooldown_limits_are_enforced(self):
        env=scene(mass=25,other=10)
        self.assertEqual(physical_split_filter([.8,0,1],env._get_obs('player_0'))[2],0)
        env.players['player_0'].mass=90;env.players['player_0'].split_cooldown=.2
        self.assertEqual(physical_split_filter([.8,0,1],env._get_obs('player_0'))[2],0)

    def test_piece_capacity_limit_is_enforced(self):
        env=scene();a=env.players['player_0']
        a.pieces=[SubPiece(pos=np.array([i*.2,0],np.float32),mass=40,vel=np.zeros(2,np.float32)) for i in range(4)]
        self.assertEqual(physical_split_filter([.8,0,1],env._get_obs('player_0'))[2],0)


if __name__=='__main__':unittest.main()
