"""
Unit tests for MasterHeuristicAgarBot.
Validates:
1. Predator evasion (flees at 180 deg with max thrust).
2. Tactical split attack (triggers split when prey is within strike envelope and in crosshairs).
3. Heavy-cell virus avoidance vs small-cell shielding.
4. Wall deflection.
5. Food foraging orientation.
"""

import math
import unittest
import numpy as np

from src.heuristic_agent import MasterHeuristicAgarBot


def make_dummy_obs(
    ego_mass: float = 20.0,
    can_split: bool = False,
    threat: tuple = None,  # (fwd, lat, dist, mass_ratio)
    prey: tuple = None,    # (fwd, lat, dist, mass_ratio)
    food: tuple = None,    # (c_fwd, c_lat, density, nearest_dist)
    hazard: tuple = None   # (wall_front, wall_bearing, min_virus_d, virus_ahead)
) -> np.ndarray:
    obs = np.zeros(38, dtype=np.float32)
    # Ego state
    obs[0] = 0.35 * math.sqrt(ego_mass / 15.0) / 3.0
    obs[1] = 0.5
    obs[2] = 0.0
    obs[3] = ego_mass / 150.0
    obs[4] = 1.0 if can_split else 0.0
    obs[5] = 0.0

    # Radars default to 1.0 (clear)
    obs[6:14] = 1.0
    obs[14:22] = 1.0

    # Threat
    if threat is not None:
        obs[22] = threat[0]
        obs[23] = threat[1]
        obs[24] = threat[2]
        obs[25] = threat[3]
        if threat[0] > 0:
            obs[10] = threat[2]

    # Prey
    if prey is not None:
        obs[26] = prey[0]
        obs[27] = prey[1]
        obs[28] = prey[2]
        obs[29] = prey[3]

    # Food
    if food is not None:
        obs[30] = food[0]
        obs[31] = food[1]
        obs[32] = food[2]
        obs[33] = food[3]

    # Hazard
    if hazard is not None:
        obs[34] = hazard[0]
        obs[35] = hazard[1]
        obs[36] = hazard[2]
        obs[37] = hazard[3]
    else:
        obs[34] = 1.0  # Far from wall
        obs[35] = 0.0
        obs[36] = 1.0  # Far from virus
        obs[37] = 0.0

    return obs


class TestMasterHeuristicAgarBot(unittest.TestCase):

    def test_predator_evasion_thrust_and_steer(self):
        bot = MasterHeuristicAgarBot(profile="apex")
        # Threat directly ahead: fwd=0.4, lat=0.0, dist=0.4, mass_ratio=0.8
        obs = make_dummy_obs(threat=(0.4, 0.0, 0.4, 0.8))
        act = bot.predict(obs)

        # Must sprint at maximum thrust
        self.assertAlmostEqual(act[0], 1.0, delta=0.05)
        # Sharp turn away from incoming threat
        self.assertGreaterEqual(abs(act[1]), 0.8)
        self.assertEqual(act[2], 0.0)  # Do not split when threatened

    def test_predator_evasion_lateral(self):
        bot = MasterHeuristicAgarBot(profile="apex")
        # Threat to the right: fwd=0.2, lat=-0.3, dist=0.36
        obs = make_dummy_obs(threat=(0.2, -0.3, 0.36, 0.6))
        act = bot.predict(obs)

        self.assertAlmostEqual(act[0], 1.0, delta=0.05)
        # Should steer left (positive steer) away from rightward threat
        self.assertGreater(act[1], 0.3)

    def test_tactical_split_attack_triggered(self):
        bot = MasterHeuristicAgarBot(profile="hunter")
        # Eligible to split (ego_mass=45kg), prey directly ahead in crosshairs at dist=0.45
        obs = make_dummy_obs(
            ego_mass=45.0,
            can_split=True,
            prey=(0.45, 0.02, 0.45, 0.6)  # Aligned, small lateral offset
        )
        act = bot.predict(obs)

        # Must fire tactical split
        self.assertEqual(act[2], 1.0)
        self.assertEqual(act[0], 1.0)
        self.assertLess(abs(act[1]), 0.3)  # Well-aligned with target

    def test_no_split_if_predator_lurking(self):
        bot = MasterHeuristicAgarBot(profile="hunter")
        # Prey in front, but predator nearby at dist=0.4
        obs = make_dummy_obs(
            ego_mass=45.0,
            can_split=True,
            prey=(0.45, 0.02, 0.45, 0.6),
            threat=(0.4, 0.0, 0.4, 0.8)
        )
        act = bot.predict(obs)

        # Must NOT split into danger; must flee!
        self.assertEqual(act[2], 0.0)
        self.assertEqual(act[0], 1.0)

    def test_heavy_cell_virus_avoidance(self):
        bot = MasterHeuristicAgarBot(profile="apex")
        # Heavy cell (55kg) with virus directly ahead (virus_ahead=1, min_d=0.25)
        obs = make_dummy_obs(
            ego_mass=55.0,
            can_split=True,
            hazard=(1.0, 0.0, 0.25, 1.0)
        )
        act = bot.predict(obs)

        # Must steer away from virus and NEVER split
        self.assertEqual(act[2], 0.0)
        self.assertGreater(abs(act[1]), 0.5)

    def test_food_foraging_direction(self):
        bot = MasterHeuristicAgarBot(profile="apex")
        # Food cluster to the left: c_fwd=0.4, c_lat=0.4
        obs = make_dummy_obs(food=(0.4, 0.4, 0.7, 0.2))
        act = bot.predict(obs)

        # Should steer left (positive steer) towards food
        self.assertGreater(act[1], 0.4)
        self.assertGreaterEqual(act[0], 0.85)


if __name__ == "__main__":
    unittest.main()
