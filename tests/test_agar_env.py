import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv, PlayerState


class TestPartiallyObservableAgarEnv(unittest.TestCase):

    def setUp(self):
        self.env = PartiallyObservableAgarEnv(arena_size=20.0, num_players=6, num_food=80, num_viruses=4, max_steps=50)

    def test_environment_spaces(self):
        self.assertEqual(self.env.action_space.shape, (3,))
        self.assertEqual(self.env.observation_space.shape, (38,))
        self.assertEqual(len(self.env.player_ids), 6)

    def test_reset_and_obs_dimensions(self):
        obs_dict, info = self.env.reset(seed=42)
        self.assertEqual(len(obs_dict), 6)
        for pid, obs in obs_dict.items():
            self.assertEqual(obs.shape, (38,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(obs)))
            # Ego state in [0, 1] range for radius and velocity
            self.assertGreaterEqual(obs[0], 0.0)

    def test_fog_of_war_visibility(self):
        self.env.reset(seed=42)
        p0 = self.env.players["player_0"]
        p1 = self.env.players["player_1"]

        # Place p0 at center and p1 far away (15 meters, well outside 5.5m vision)
        p0.pos = np.array([0.0, 0.0], dtype=np.float32)
        p0.mass = 20.0
        p1.pos = np.array([15.0, 0.0], dtype=np.float32)
        p1.mass = 50.0 # Huge predator, but outside FoV

        obs_p0 = self.env._get_obs("player_0")
        # Threat vector (features 22..25) should be zero because p1 is outside vision radius!
        np.testing.assert_array_almost_equal(obs_p0[22:26], np.zeros(4, dtype=np.float32))

        # Now bring p1 inside vision radius (3.0 meters ahead)
        p1.pos = np.array([3.0, 0.0], dtype=np.float32)
        obs_p0_near = self.env._get_obs("player_0")
        # Threat vector should now detect the predator!
        self.assertGreater(obs_p0_near[24], 0.0) # non-zero distance feature
        self.assertGreater(obs_p0_near[25], 0.0) # mass ratio feature

    def test_mass_dependent_kinematics(self):
        small_player = PlayerState("small", np.array([0.0, 0.0]), (255, 0, 0), initial_mass=15.0)
        large_player = PlayerState("large", np.array([0.0, 0.0]), (0, 255, 0), initial_mass=150.0)

        # Larger player must have larger radius
        self.assertGreater(large_player.radius, small_player.radius)
        # Larger player must be slower
        self.assertLess(large_player.max_speed, small_player.max_speed)
        # Larger player must have slightly expanded vision
        self.assertGreater(large_player.vision_radius, small_player.vision_radius)

    def test_food_foraging(self):
        self.env.reset(seed=42)
        p0 = self.env.players["player_0"]
        initial_mass = p0.mass

        # Place a food pellet directly on top of p0
        self.env.food_positions[0] = p0.pos.copy()

        actions = {pid: np.array([0.0, 0.0, 0.0], dtype=np.float32) for pid in self.env.player_ids}
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        # p0 must have gained mass from the food
        self.assertGreater(p0.mass, initial_mass)
        self.assertGreater(rewards["player_0"], 0.02)

    def test_predation_consumption(self):
        self.env.reset(seed=42)
        p0 = self.env.players["player_0"]
        p1 = self.env.players["player_1"]

        # Sub-test 1: Insufficient mass difference (< 8.0 kg) must NOT trigger a kill
        p0.mass = 23.0
        p1.mass = 20.0 # Only 3kg difference
        p0.pos = np.array([0.0, 0.0], dtype=np.float32)
        p1.pos = np.array([0.05, 0.0], dtype=np.float32)
        actions = {pid: np.array([0.0, 0.0, 0.0], dtype=np.float32) for pid in self.env.player_ids}
        self.env.step(actions)
        self.assertEqual(p0.kills, 0, "Kill should NOT trigger if mass gap is only 3kg!")

        # Sub-test 2: Substantial mass difference (>= 8.0 kg and >= 1.25x) triggers a genuine kill
        p0.mass = 35.0 # 15kg difference
        p1.mass = 20.0
        p0.pos = np.array([0.0, 0.0], dtype=np.float32)
        p1.pos = np.array([0.05, 0.0], dtype=np.float32) # Overlapping
        self.env.step(actions)
        self.assertEqual(p0.kills, 1, "Substantial predator must score kill!")
        self.assertEqual(p1.deaths, 1)

    def test_splitting_mechanic(self):
        self.env.reset(seed=42)
        p0 = self.env.players["player_0"]
        p0.mass = 44.0 # Eligible to split (>= 36kg)
        self.assertTrue(p0.can_split)

        # Execute split action (thrust=0.8, steer=0.0, split=1.0)
        actions = {pid: np.array([0.5, 0.0, 0.0], dtype=np.float32) for pid in self.env.player_ids}
        actions["player_0"] = np.array([0.8, 0.0, 1.0], dtype=np.float32)

        self.env.step(actions)

        # Player 0 must now have 2 pieces
        self.assertEqual(len(p0.pieces), 2)
        piece_a, piece_b = p0.pieces
        # Each piece should have roughly half mass
        self.assertAlmostEqual(piece_a.mass, 22.0, delta=1.5)
        self.assertAlmostEqual(piece_b.mass, 22.0, delta=1.5)
        # Front piece should have moved forward
        self.assertGreater(np.linalg.norm(piece_b.pos - piece_a.pos), 0.3)

    def test_mass_conservation_invariant(self):
        """Verify that total world mass is strictly conserved over 200 simulation steps."""
        self.env.reset(seed=42)
        initial_world_mass = self.env.total_world_mass
        self.assertAlmostEqual(initial_world_mass, self.env.total_world_mass_cap, delta=1e-3)

        for step in range(200):
            actions = {pid: self.env.action_space.sample() for pid in self.env.player_ids}
            if step % 20 == 0:
                for pid in self.env.player_ids:
                    actions[pid][2] = 1.0  # Trigger splits
            self.env.step(actions)
            curr_mass = self.env.total_world_mass
            self.assertAlmostEqual(curr_mass, self.env.total_world_mass_cap, delta=1e-3,
                                   msg=f"Mass drift detected at step {step}: {curr_mass} vs {self.env.total_world_mass_cap}")

    def test_density_field_grid_resolution(self):
        """Verify that the potential field grid has resolution >= 50x50 and normalizes to 1."""
        self.env.reset(seed=42)
        prob, grid_pts = self.env._compute_spawn_density_grid()
        self.assertGreaterEqual(prob.shape[0], 50)
        self.assertGreaterEqual(prob.shape[1], 50)
        self.assertAlmostEqual(float(np.sum(prob)), 1.0, delta=1e-4)
        # Verify clearance shadow: probability directly under players should be suppressed
        for player in self.env.players.values():
            for pc in player.pieces:
                diff = grid_pts - pc.pos
                dists = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                # Cell closest to player center should have very low spawn density
                self.assertLess(prob[min_idx], 0.001)


if __name__ == "__main__":
    unittest.main()

