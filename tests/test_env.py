"""
Unit tests for Continuous Mobile Robot Navigation Environment & Sim2Real Domain Randomization.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.navigation_env import ContinuousNavigationEnv
from envs.domain_randomization import DomainRandomizationWrapper


class TestNavigationEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = ContinuousNavigationEnv(arena_size=10.0, num_obstacles=4, num_lidar_rays=8)

    def test_environment_specs(self):
        self.assertEqual(self.env.action_space.shape, (2,))
        self.assertEqual(self.env.observation_space.shape, (12,))
        self.assertTrue(np.all(self.env.action_space.low == -1.0))
        self.assertTrue(np.all(self.env.action_space.high == 1.0))

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (12,))
        self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 1.0))
        # Initial robot position should be at origin
        self.assertAlmostEqual(self.env.robot_pos[0], 0.0, places=3)
        self.assertAlmostEqual(self.env.robot_pos[1], 0.0, places=3)

    def test_step_execution(self):
        self.env.reset(seed=42)
        action = np.array([0.5, 0.2], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs.shape, (12,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("is_success", info)
        self.assertIn("collision", info)

    def test_lidar_raycasting(self):
        self.env.reset(seed=42)
        lidar = self.env._compute_lidar_readings()
        self.assertEqual(len(lidar), 8)
        # All LiDAR beams must be positive and bounded by 1.0 (normalized)
        self.assertTrue(np.all(lidar >= 0.0))
        self.assertTrue(np.all(lidar <= 1.0))

    def test_domain_randomization_wrapper(self):
        wrapped_env = DomainRandomizationWrapper(
            self.env,
            friction_range=(0.3, 1.2),
            mass_range=(1.5, 4.0),
            sensor_noise_range=(0.01, 0.05),
            max_action_delay=2
        )

        frictions = []
        masses = []
        delays = []

        for seed in range(5):
            _, info = wrapped_env.reset(seed=seed)
            params = info["domain_params"]
            frictions.append(params["friction_coeff"])
            masses.append(params["robot_mass"])
            delays.append(params["action_delay_steps"])

        # Physical parameters must vary across randomized episodes
        self.assertGreater(max(frictions) - min(frictions), 0.05)
        self.assertGreater(max(masses) - min(masses), 0.1)

        # Test action stepping with latency queue
        action = np.array([0.8, -0.4], dtype=np.float32)
        obs, reward, term, trunc, _ = wrapped_env.step(action)
        self.assertEqual(obs.shape, (12,))


if __name__ == "__main__":
    unittest.main()
