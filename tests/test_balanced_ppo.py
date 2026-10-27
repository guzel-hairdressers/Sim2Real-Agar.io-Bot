import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.arena_policies import ArenaPPO
from src.balanced_policies import BalancedPPO, load_policy, physical_action
from src.learned_risk import load_policy as historical_load
from src.train_balanced_ppo import growth_survival_reward

ROOT = Path(__file__).resolve().parents[1]


class BalancedPPOTests(unittest.TestCase):
    def test_risky_observation_cannot_change_neural_movement_or_request(self):
        # A wall/threat observation may inform the model, never overwrite it.
        obs = np.ones(70, np.float32)
        obs[47] = 0.01
        np.testing.assert_array_equal(physical_action([0.9, -0.7, 1], obs),
                                      np.array([0.9, -0.7, 1], np.float32))
        obs[4] = 0
        np.testing.assert_array_equal(physical_action([0.9, -0.7, 1], obs),
                                      np.array([0.9, -0.7, 0], np.float32))

    def test_full_control_checkpoint_cannot_silently_use_old_adapter(self):
        package = torch.load(ROOT / 'outputs/learned_risk_v8/ppo.pt',
                             map_location='cpu', weights_only=True)
        package['movement_control'] = 'learned'
        package['split_control'] = 'learned_full_control_physical_only'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ppo.pt'
            torch.save(package, path)
            model = load_policy('ppo', path)
            self.assertIsInstance(model, BalancedPPO)
            with self.assertRaisesRegex(ValueError, 'Unsupported split control'):
                historical_load('ppo', path)
            raw = ArenaPPO(path, assisted=False)
            obs = np.zeros(70, np.float32)
            obs[4] = 1
            expected = raw.predict(obs)
            model.assisted = True
            np.testing.assert_array_equal(model.predict(obs), physical_action(expected, obs))

    def test_old_packages_keep_their_original_control(self):
        path = ROOT / 'outputs/learned_risk_v8/ppo.pt'
        self.assertIs(type(load_policy('ppo', path)), type(historical_load('ppo', path)))

    def test_respawning_cannot_earn_growth_reward(self):
        self.assertEqual(growth_survival_reward(2, 16, 0, 1), -3)
        self.assertLess(growth_survival_reward(30, 16, 0, 1), -3)
        self.assertGreater(growth_survival_reward(16, 20, 0, 0), 0)
        self.assertEqual(growth_survival_reward(16, 16, 0, 0), 0)


if __name__ == '__main__':
    unittest.main()
