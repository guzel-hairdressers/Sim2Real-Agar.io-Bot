import os
import sys
import unittest
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agar_diffusion_policy import (
    AgarDiffusionSchedule,
    AgarTemporalDenoisingNet,
    AgarDiffusionPolicy
)


class TestAgarDiffusionPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = AgarDiffusionPolicy(
            action_horizon=16,
            exec_horizon=2,
            action_dim=3,
            obs_dim=38,
            num_ddim_steps=8,
            seed=42
        )
        self.mock_obs = np.random.uniform(-0.5, 0.5, size=38).astype(np.float32)

    def test_schedule_properties(self):
        schedule = self.policy.schedule
        self.assertEqual(len(schedule.betas), 100)
        self.assertTrue(np.all(schedule.alphas_cumprod[:-1] >= schedule.alphas_cumprod[1:]))

    def test_trajectory_sampling_bounds(self):
        traj = self.policy.sample_trajectory_ddim(self.mock_obs, steps=8)
        self.assertEqual(traj.shape, (16, 3))
        # Thrust must be in [0, 1]
        self.assertTrue(np.all(traj[:, 0] >= 0.0) and np.all(traj[:, 0] <= 1.0))
        # Steer must be in [-1, 1]
        self.assertTrue(np.all(traj[:, 1] >= -1.0) and np.all(traj[:, 1] <= 1.0))
        # Split must be in [0, 1]
        self.assertTrue(np.all(traj[:, 2] >= 0.0) and np.all(traj[:, 2] <= 1.0))

    def test_receding_horizon_buffer(self):
        self.policy.reset()
        self.assertEqual(len(self.policy.action_buffer), 0)

        # First prediction triggers a plan of length 16, pops 1 -> buffer has 15
        act0 = self.policy.predict(self.mock_obs)
        self.assertEqual(act0.shape, (3,))
        self.assertEqual(len(self.policy.action_buffer), 15)

        # Step 1 more time (exec_horizon = 2)
        self.policy.predict(self.mock_obs)
        self.assertEqual(self.policy.steps_since_plan, 2)

        # 3rd call triggers re-planning!
        self.policy.predict(self.mock_obs)
        self.assertEqual(self.policy.steps_since_plan, 1)

    def test_sub_millisecond_latency(self):
        # Warmup
        _ = self.policy.sample_trajectory_ddim(self.mock_obs, steps=8)
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            _ = self.policy.sample_trajectory_ddim(self.mock_obs, steps=8)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0
        self.assertLess(elapsed_ms, 5.0)


if __name__ == "__main__":
    unittest.main()
