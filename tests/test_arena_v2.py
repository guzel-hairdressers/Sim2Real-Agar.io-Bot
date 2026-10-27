import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from envs.agar_env import PartiallyObservableAgarEnv, SubPiece
from envs.arena_config import ArenaConfig, event_reward
from src.arena_tactics import teacher_action, safety_filter, split_is_safe
from src.train_champion_ppo import HybridActorCritic, ChampionPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy, TorchAgarDiffusionNet, AgarDiffusionSchedule
from src.train_arena_v2 import gae


class ArenaV2Tests(unittest.TestCase):
    def setUp(self):
        self.env = PartiallyObservableAgarEnv(num_players=2, num_food=0, num_viruses=0,
                                             observation_version=2, dt=1/30)
        self.env.reset(seed=73)
        self.a, self.b = self.env.players.values()
        self.a.pos = np.array([0., 0.], np.float32)
        self.b.pos = np.array([1.5, 0.], np.float32)
        self.a.yaw = 0

    def test_seed_reproduces_full_trajectory(self):
        def rollout():
            obs, _ = self.env.reset(seed=79)
            result = [obs]
            for _ in range(25):
                obs, rewards, _, _, _ = self.env.step({p: teacher_action(o) for p, o in obs.items()})
                result.append((obs, rewards))
            return repr(result)
        self.assertEqual(rollout(), rollout())

    def test_motion_and_fragmentation_are_observable(self):
        self.a.mass, self.b.mass = 35, 50
        self.b.linear_vel = 2
        first = self.env._get_obs('player_0')
        self.b.linear_vel = -2
        second = self.env._get_obs('player_0')
        self.assertFalse(np.array_equal(first[42:48], second[42:48]))
        self.b.pieces = [SubPiece(np.array([1.4, 0.], np.float32), 25., np.zeros(2)),
                         SubPiece(np.array([1.6, 0.], np.float32), 25., np.zeros(2))]
        self.assertEqual(self.env._get_obs('player_0')[47], 0)

    def test_unseen_pieces_do_not_leak(self):
        self.b.pos = np.array([10., 0.], np.float32)
        first = self.env._get_obs('player_0')
        self.b.mass = 140
        self.b.linear_vel = 4
        np.testing.assert_array_equal(first, self.env._get_obs('player_0'))

    def test_predator_ahead_causes_braking_and_turn(self):
        self.a.mass, self.b.mass = 20, 40
        obs = self.env._get_obs('player_0')
        action = safety_filter([1, 0, 1], obs)
        self.assertLess(action[0], .4)
        self.assertGreater(abs(action[1]), .8)
        self.assertEqual(action[2], 0)

    def test_unsafe_split_bait_rejected(self):
        self.a.mass, self.b.mass = 45, 30
        obs = self.env._get_obs('player_0')
        self.assertFalse(split_is_safe(obs))
        self.assertEqual(safety_filter([1, 0, 1], obs)[2], 0)

    def test_safe_interception_allowed_but_retreat_out_of_range_rejected(self):
        self.a.mass, self.b.mass = 90, 20
        obs = self.env._get_obs('player_0')
        self.assertTrue(split_is_safe(obs))
        self.b.pos = np.array([4., 0.], np.float32)
        self.b.yaw = 0
        self.b.linear_vel = 2.2
        self.assertFalse(split_is_safe(self.env._get_obs('player_0')))

    def test_safety_preserves_uncontested_neural_action(self):
        self.a.mass = self.b.mass = 20
        raw = np.array([.6, .2, 0], np.float32)
        np.testing.assert_array_equal(raw, safety_filter(raw, self.env._get_obs('player_0')))

    def test_probability_ratio_is_one_without_update_even_when_clipped(self):
        torch.manual_seed(73)
        net = HybridActorCritic(70)
        obs = torch.zeros(4096, 70)
        with torch.no_grad():
            net.cont_head.bias[0] = 8
            action, old, _, _ = net.get_action_and_value(obs)
            new, _, _ = net.evaluate_actions(obs, action)
        self.assertTrue(torch.any(action[:, 0] > 1))
        torch.testing.assert_close(torch.exp(new-old), torch.ones_like(old), rtol=0, atol=1e-6)

    def test_rewards_pay_only_new_events(self):
        previous = {'kills': 1, 'deaths': 1}
        self.assertEqual(event_reward(previous, previous, 0), 0)
        self.assertEqual(event_reward(previous, {'kills': 2, 'deaths': 1}, 0), 4.5)
        self.assertEqual(event_reward(previous, {'kills': 1, 'deaths': 2}, 0), -3)

    def test_explicit_missing_checkpoints_are_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'missing.pt')
            with self.assertRaises(FileNotFoundError):
                ChampionPPOAgent(path)
            with self.assertRaises(FileNotFoundError):
                AgarDiffusionPolicy(model_path=path)

    def test_shared_configuration(self):
        env = ArenaConfig().make()
        obs, _ = env.reset(seed=7)
        self.assertEqual(env.dt, 1/30)
        self.assertEqual(len(obs), 7)
        self.assertTrue(all(o.shape == (70,) and env.observation_space.contains(o) for o in obs.values()))

    def test_wall_braking_accounts_for_cell_radius(self):
        self.a.mass = 90
        self.b.pos = np.array([-8., 0.], np.float32)
        self.a.pos = np.array([self.env.half_arena-1.5, 0.], np.float32)
        action = safety_filter([1, 0, 0], self.env._get_obs('player_0'))
        self.assertLess(action[0], .4)
        self.assertGreater(abs(action[1]), .8)

    def test_time_limit_bootstraps_but_does_not_cross_reset(self):
        advantage, _ = gae(np.array([[1.], [2.]]), np.array([[10.], [20.]]),
            np.array([[20.], [30.]]), np.zeros((2, 1)), np.array([[1.], [0.]]))
        self.assertAlmostEqual(advantage[0, 0], 10.8, places=5)
        self.assertAlmostEqual(advantage[1, 0], 11.7, places=5)

    def test_death_does_not_bootstrap_the_respawn(self):
        advantage, _ = gae(np.array([[1.], [2.]]), np.array([[10.], [20.]]),
            np.array([[20.], [30.]]), np.array([[1.], [0.]]), np.zeros((2, 1)))
        self.assertAlmostEqual(advantage[0, 0], -9, places=5)

    def test_diffusion_private_rng_repeats_without_global_rng_dependency(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'diffusion.pt')
            torch.save(TorchAgarDiffusionNet(obs_dim=70).state_dict(), path)
            policy = AgarDiffusionPolicy(obs_dim=70, model_path=path, critic_path='')
            obs = self.env._get_obs('player_0')
            policy.reset(seed=79)
            a = policy.sample_trajectory_torch(obs)
            torch.randn(300)
            policy.reset(seed=79)
            np.testing.assert_array_equal(a, policy.sample_trajectory_torch(obs))

    def test_cosine_schedule_reaches_approximately_pure_noise(self):
        schedule = AgarDiffusionSchedule(schedule='cosine')
        self.assertLess(schedule.alphas_cumprod[-1], 1e-5)
        self.assertTrue(np.all(np.diff(schedule.alphas_cumprod) < 0))


if __name__ == '__main__':
    unittest.main()
