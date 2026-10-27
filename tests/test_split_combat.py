import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from envs.agar_env import PartiallyObservableAgarEnv
from src.arena_policies import ArenaPPO, ArenaDiffusion
from src.split_head import BinarySplitHead
from src.train_champion_ppo import HybridActorCritic


ROOT = Path(__file__).resolve().parents[1]


def scene(mass=90, prey=20, distance=1.5):
    env = PartiallyObservableAgarEnv(arena_size=18, num_players=2, num_food=0,
        num_viruses=0, observation_version=2, dt=1/30)
    env.reset(seed=79)
    a, b = env.players.values()
    a.mass, b.mass = mass, prey
    a.pos = np.array([0., 0.], np.float32); a.yaw = 0
    b.pos = np.array([distance, 0.], np.float32)
    env.reserve_mass = 0
    return env


class SplitCombatTests(unittest.TestCase):
    def test_trained_agents_attack_safe_prey_and_reject_oversized_prey(self):
        policies = [ArenaPPO(ROOT/'outputs/split_combat_v3/ppo.pt'),
                    ArenaDiffusion(ROOT/'outputs/split_combat_v3/diffusion.pt',
                                   ROOT/'outputs/arena_v2_cosine/critic.pt')]
        for policy in policies:
            with self.subTest(policy=type(policy).__name__):
                env = scene()
                self.assertEqual(policy.predict(env._get_obs('player_0'))[2],1)
                self.assertEqual(policy.predict(scene(45,30)._get_obs('player_0'))[2],0)

    def test_saved_split_models_preserve_previous_movement_weights(self):
        for name, original in [('ppo','outputs/arena_v2/ppo.pt'),
                               ('diffusion','outputs/arena_v2_cosine/diffusion.pt')]:
            old = torch.load(ROOT/original,map_location='cpu',weights_only=True)['state_dict']
            new = torch.load(ROOT/f'outputs/split_combat_v3/{name}.pt',map_location='cpu',weights_only=True)['state_dict']
            for key,value in old.items():
                if name=='ppo' and key.startswith('split_head.'):
                    continue
                with self.subTest(algorithm=name,weight=key):
                    self.assertTrue(torch.equal(value,new[key]))

    def test_actual_projectile_consumption_is_identified(self):
        env = scene()
        events = []
        for step in range(18):
            env.step({'player_0':np.array([1,0,float(step==0)]), 'player_1':np.zeros(3)})
            events.extend(env.predation_events)
        self.assertTrue(any(e['predator']=='player_0' and 0<=e['projectile_age_steps']<=18 for e in events))

    def test_body_consumption_is_not_counted_as_projectile(self):
        env = scene(distance=.8)
        env.step({'player_0':np.array([.5,0,0]), 'player_1':np.zeros(3)})
        self.assertTrue(env.predation_events)
        self.assertTrue(all(e['projectile_age_steps']==-1 for e in env.predation_events))

    def test_new_ppo_head_does_not_change_continuous_policy_or_value(self):
        original = ArenaPPO(ROOT/'outputs/arena_v2/ppo.pt')
        new = HybridActorCritic(70, split_architecture='obs_mlp')
        new.load_state_dict({k:v for k,v in original.net.state_dict().items() if not k.startswith('split_head.')},strict=False)
        observations = torch.randn(64,70)
        with torch.no_grad():
            old_mu, old_std, _, old_value = original.net(observations)
            mu, std, _, value = new(observations)
        for first, second in [(mu,old_mu),(std,old_std),(value,old_value)]:
            torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_diffusion_binary_head_leaves_navigation_unchanged_and_guard_blocks_bait(self):
        original_path = ROOT/'outputs/arena_v2_cosine/diffusion.pt'
        critic = ROOT/'outputs/arena_v2_cosine/critic.pt'
        package = torch.load(original_path,map_location='cpu',weights_only=True)
        head = BinarySplitHead()
        with torch.no_grad():
            for parameter in head.parameters(): parameter.zero_()
            head.net[-1].bias.fill_(10)
        package['split_architecture']='obs_mlp'
        package['split_head_state_dict']=head.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'; torch.save(package,path)
            new=ArenaDiffusion(path,critic,assisted=False)
            old=ArenaDiffusion(original_path,critic,assisted=False)
            obs=scene()._get_obs('player_0')
            new.reset(seed=73); old.reset(seed=73)
            new_action, old_action = new.predict(obs), old.predict(obs)
            np.testing.assert_array_equal(new_action[:2],old_action[:2])
            self.assertEqual(new_action[2],1)
            new.assisted=True
            self.assertEqual(new.predict(scene(45,30)._get_obs('player_0'))[2],0)


if __name__ == '__main__':
    unittest.main()
