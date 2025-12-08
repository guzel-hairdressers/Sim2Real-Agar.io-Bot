"""
Unit tests for Diffusion Policy (DDIM Trajectory Denoising & Receding Horizon Control).
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.diffusion_policy import (
    DiffusionSchedule,
    TemporalDenoisingNet,
    DiffusionPolicy,
    MSEBehavioralCloningPolicy,
    MultimodalDemonstrationGenerator
)


class TestDiffusionPolicy(unittest.TestCase):

    def setUp(self):
        self.schedule = DiffusionSchedule(num_timesteps=100)
        self.policy = DiffusionPolicy(
            action_horizon=16,
            exec_horizon=8,
            action_dim=2,
            obs_dim=12,
            num_ddim_steps=10,
            seed=42
        )

    def test_schedule_properties(self):
        self.assertEqual(len(self.schedule.betas), 100)
        self.assertEqual(len(self.schedule.alphas), 100)
        self.assertEqual(len(self.schedule.alphas_cumprod), 100)
        # Cumulative alphas must be monotonically decreasing
        self.assertTrue(np.all(np.diff(self.schedule.alphas_cumprod) <= 0.0))
        # Bounds must be within [0, 1]
        self.assertTrue(np.all(self.schedule.alphas_cumprod >= 0.0))
        self.assertTrue(np.all(self.schedule.alphas_cumprod <= 1.0))

    def test_forward_q_sampling(self):
        clean_traj = np.ones((16, 2), dtype=np.float32)
        corrupted_t0, noise = self.schedule.q_sample(clean_traj, t=0)
        self.assertEqual(corrupted_t0.shape, (16, 2))
        self.assertEqual(noise.shape, (16, 2))

        # At t=99 (near pure noise), variance should be dominated by noise
        corrupted_t99, _ = self.schedule.q_sample(clean_traj, t=99)
        self.assertEqual(corrupted_t99.shape, (16, 2))
        self.assertFalse(np.allclose(corrupted_t99, clean_traj))

    def test_denoising_network_forward(self):
        net = TemporalDenoisingNet(action_horizon=16, action_dim=2, obs_dim=12, hidden_dim=64, seed=42)
        a_k = np.zeros((16, 2), dtype=np.float32)
        obs = np.ones(12, dtype=np.float32)

        eps = net.forward(a_k, t=50, obs=obs)
        self.assertEqual(eps.shape, (16, 2))
        self.assertTrue(np.all(np.isfinite(eps)))

        # Batched forward pass
        batch_a_k = np.zeros((4, 16, 2), dtype=np.float32)
        batch_obs = np.ones((4, 12), dtype=np.float32)
        batch_eps = net.forward(batch_a_k, t=50, obs=batch_obs)
        self.assertEqual(batch_eps.shape, (4, 16, 2))

    def test_ddim_sampling_and_trajectory_shape(self):
        obs = np.random.randn(12).astype(np.float32)
        traj = self.policy.sample_trajectory_ddim(obs, steps=10)
        self.assertEqual(traj.shape, (16, 2))
        # All actions must be bounded in [-1, 1]
        self.assertTrue(np.all(traj >= -1.0))
        self.assertTrue(np.all(traj <= 1.0))

    def test_receding_horizon_control(self):
        self.policy.reset()
        self.assertEqual(len(self.policy.action_buffer), 0)

        obs = np.zeros(12, dtype=np.float32)
        # Step 0: Should trigger trajectory generation and buffer pop
        act0 = self.policy.predict(obs)
        self.assertEqual(act0.shape, (2,))
        # Buffer should now hold 16 - 1 = 15 actions
        self.assertEqual(len(self.policy.action_buffer), 15)

        # Step through execution horizon (T_e = 8)
        for _ in range(7):
            _ = self.policy.predict(obs)
        self.assertEqual(self.policy.steps_since_plan, 8)

        # 9th step should trigger replan because steps_since_plan >= exec_horizon
        _ = self.policy.predict(obs)
        # Buffer replenished to 15
        self.assertEqual(len(self.policy.action_buffer), 15)
        self.assertEqual(self.policy.steps_since_plan, 1)

    def test_multimodal_demonstration_generator(self):
        generator = MultimodalDemonstrationGenerator(action_horizon=16)
        demos = generator.generate_demonstrations(num_episodes=10, seed=42)
        self.assertEqual(len(demos), 10)

        modes = [d["mode"] for d in demos]
        self.assertIn("left", modes)
        self.assertIn("right", modes)

        for d in demos:
            self.assertEqual(d["observations"].shape[1], 12)
            self.assertEqual(d["actions"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
