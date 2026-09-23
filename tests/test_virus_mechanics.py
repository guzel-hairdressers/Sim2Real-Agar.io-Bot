"""
Automated Physics & Balance Tests for Agar.io Environment:
1. Virus shredding for max-piece (4-clone) cells (no ghosting through viruses).
2. Small cell viral sanctuary & line-of-sight wall blocking during predation.
3. Superlinear metabolic decay for colossal cells (M >= 400kg).
4. Virus spawn & player respawn guaranteed clearance.
"""

import math
import numpy as np
from envs.agar_env import PartiallyObservableAgarEnv, SubPiece, PlayerState


def test_max_pieces_virus_shredding():
    """Verify that a 4-clone cell colliding with a virus loses 25% mass and recoils, rather than ghosting."""
    env = PartiallyObservableAgarEnv(arena_size=18.0, num_players=2, num_viruses=1, dt=0.1)
    env.reset(seed=42)

    # Set virus at center
    env.viruses = np.array([[0.0, 0.0, 0.65]], dtype=np.float32)

    # Create a 4-piece cell for player_0 with mass 50kg each
    p0 = env.players["player_0"]
    p0.pieces = [
        SubPiece(pos=np.array([0.7, 0.0], dtype=np.float32), mass=50.0, vel=np.array([-1.0, 0.0], dtype=np.float32)),
        SubPiece(pos=np.array([4.0, 4.0], dtype=np.float32), mass=50.0, vel=np.zeros(2, dtype=np.float32)),
        SubPiece(pos=np.array([-4.0, 4.0], dtype=np.float32), mass=50.0, vel=np.zeros(2, dtype=np.float32)),
        SubPiece(pos=np.array([4.0, -4.0], dtype=np.float32), mass=50.0, vel=np.zeros(2, dtype=np.float32)),
    ]
    assert len(p0.pieces) == 4, "Player must have 4 pieces (max_pieces)"

    # Clear food pellets to isolate virus shredding penalty
    env.food_positions = np.empty((0, 2), dtype=np.float32)

    init_mass = p0.pieces[0].mass

    # Step environment: piece 0 collides with virus at (0, 0)
    actions = {pid: np.array([0.0, 0.0, 0.0], dtype=np.float32) for pid in env.player_ids}
    obs, rewards, _, _, _ = env.step(actions)

    new_mass = p0.pieces[0].mass
    assert new_mass < init_mass * 0.80, f"Mass should have shredded by 25%, was {init_mass} -> {new_mass}"
    assert p0.pieces[0].pos[0] > 0.65, f"Piece should have been recoiled away from virus, pos={p0.pieces[0].pos}"
    assert rewards["player_0"] <= -7.5, f"Shredding penalty should be applied, got reward {rewards['player_0']}"
    print(f"Test max_pieces_virus_shredding PASSED: mass {init_mass:.1f}kg -> {new_mass:.1f}kg (-25%), recoiled to {p0.pieces[0].pos[0]:.2f}m")


def test_small_cell_virus_sanctuary():
    """Verify that a small cell (<36kg) sheltering inside or behind a virus cannot be eaten by a 100kg predator."""
    env = PartiallyObservableAgarEnv(arena_size=18.0, num_players=2, num_viruses=1, dt=0.1)
    env.reset(seed=42)

    # Virus at center
    env.viruses = np.array([[0.0, 0.0, 0.65]], dtype=np.float32)

    # Predator at x = 0.8, mass = 100kg (radius ~0.90m)
    p_pred = env.players["player_0"]
    p_pred.pieces = [
        SubPiece(pos=np.array([0.8, 0.0], dtype=np.float32), mass=100.0, vel=np.zeros(2, dtype=np.float32))
    ]

    # Prey at center (0, 0) sheltering inside the virus! Mass = 15kg
    p_prey = env.players["player_1"]
    p_prey.pieces = [
        SubPiece(pos=np.array([0.0, 0.0], dtype=np.float32), mass=15.0, vel=np.zeros(2, dtype=np.float32))
    ]

    dist = np.linalg.norm(p_pred.pieces[0].pos - p_prey.pieces[0].pos)
    assert dist < p_pred.pieces[0].radius, "Predator overlaps prey location"

    # Step environment:
    actions = {pid: np.array([0.0, 0.0, 0.0], dtype=np.float32) for pid in env.player_ids}
    env.step(actions)

    assert len(p_prey.pieces) > 0, "Prey sheltered inside virus MUST NOT be eaten!"
    assert p_prey.pieces[0].mass >= 14.0, "Prey should not lose mass"
    print("Test small_cell_virus_sanctuary PASSED: prey inside virus was protected from 100kg predator!")


def test_virus_line_of_sight_wall():
    """Verify that a virus between predator and prey blocks predation like a wall."""
    env = PartiallyObservableAgarEnv(arena_size=18.0, num_players=2, num_viruses=1, dt=0.1)
    env.reset(seed=42)

    # Virus at (0.0, 0.0), radius = 0.65
    env.viruses = np.array([[0.0, 0.0, 0.65]], dtype=np.float32)

    # Predator at (-0.6, 0.0), mass = 120kg (radius = 0.99m)
    p_pred = env.players["player_0"]
    p_pred.pieces = [
        SubPiece(pos=np.array([-0.6, 0.0], dtype=np.float32), mass=120.0, vel=np.zeros(2, dtype=np.float32))
    ]

    # Prey at (0.6, 0.0), mass = 15kg (radius = 0.35m)
    p_prey = env.players["player_1"]
    p_prey.pieces = [
        SubPiece(pos=np.array([0.6, 0.0], dtype=np.float32), mass=15.0, vel=np.zeros(2, dtype=np.float32))
    ]

    actions = {pid: np.array([0.0, 0.0, 0.0], dtype=np.float32) for pid in env.player_ids}
    env.step(actions)

    assert len(p_prey.pieces) > 0, "Prey behind virus wall MUST NOT be eaten!"
    print("Test virus_line_of_sight_wall PASSED: virus wall obstructed predation!")


def test_superlinear_mass_decay_scaling():
    """Verify that a 400kg colossal cell decays > 15 kg/s while a 20kg cell has negligible decay."""
    env = PartiallyObservableAgarEnv(arena_size=18.0, num_players=2, dt=0.1)
    env.reset(seed=42)

    # Player 0: 400kg giant
    p0 = env.players["player_0"]
    p0.pieces = [SubPiece(pos=np.array([5.0, 5.0], dtype=np.float32), mass=400.0, vel=np.zeros(2, dtype=np.float32))]

    # Player 1: 20kg small cell
    p1 = env.players["player_1"]
    p1.pieces = [SubPiece(pos=np.array([-5.0, -5.0], dtype=np.float32), mass=20.0, vel=np.zeros(2, dtype=np.float32))]

    # Step for 1.0 second (10 ticks)
    actions = {pid: np.array([0.0, 0.0, 0.0], dtype=np.float32) for pid in env.player_ids}
    for _ in range(10):
        env.step(actions)

    decay_400 = 400.0 - p0.pieces[0].mass
    decay_20 = 20.0 - p1.pieces[0].mass

    print(f"Decay in 1.0s: 400kg cell lost {decay_400:.2f}kg ({decay_400:.2f} kg/s), 20kg cell lost {decay_20:.2f}kg ({decay_20:.2f} kg/s)")

    assert decay_400 > 15.0, f"400kg cell must decay rapidly (> 15 kg/s), lost only {decay_400:.2f}kg"
    assert decay_20 < 0.20, f"20kg cell should have negligible decay, lost {decay_20:.2f}kg"
    print("Test superlinear_mass_decay_scaling PASSED!")


def test_player_respawn_virus_clearance():
    """Verify that player respawns never place a player within 2.8m of any virus."""
    env = PartiallyObservableAgarEnv(arena_size=18.0, num_players=4, num_viruses=6, dt=0.1)
    env.reset(seed=42)

    for i in range(20):
        env._respawn_player("player_0")
        p0_pos = env.players["player_0"].pieces[0].pos
        for v in env.viruses:
            dist = np.linalg.norm(p0_pos - v[:2])
            assert dist >= (v[2] + 2.5), f"Respawned too close to virus: dist={dist:.2f}m < {v[2] + 2.5:.2f}m"

    print("Test player_respawn_virus_clearance PASSED: all 20 respawns kept safe virus clearance!")
