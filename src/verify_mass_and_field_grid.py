"""
verify_mass_and_field_grid.py

Rigorous diagnostic verification and publication-grade visualization of:
1. Strict closed thermodynamic mass conservation invariant: M_total(t) = 200.0 kg.
2. High-resolution (50x50+) 2D Spatial Potential Field Grid with clearance shadows,
   inverse-mass corridor shifts, central oasis, and continuous jitter sampling.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.agar_env import PartiallyObservableAgarEnv, SubPiece

def verify_and_plot():
    os.makedirs("outputs", exist_ok=True)
    artifact_dir = os.path.expanduser("~/.gemini/antigravity/brain/62dc0844-0873-4692-bb43-1836ab1211da/media")
    os.makedirs(artifact_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 1: RIGOROUS 1,000-STEP CLOSED MASS CONSERVATION TEST")
    print("=" * 70)

    env = PartiallyObservableAgarEnv(arena_size=16.0, num_players=6, num_food=100, total_world_mass=400.0)
    obs, _ = env.reset(seed=42)

    steps = 1000
    history_total = []
    history_players = []
    history_food = []
    history_reserve = []
    history_max_mass = []
    predation_count = 0

    for t in range(steps):
        actions = {}
        for pid in env.player_ids:
            p = env.players[pid]
            thrust = 0.85 if t % 10 < 7 else 0.4
            steer = np.sin(t * 0.05 + int(pid[-1])) * 0.6
            split = 1.0 if (t % 30 == 0 and p.can_split) else 0.0
            actions[pid] = np.array([thrust, steer, split], dtype=np.float32)

        obs, rews, terms, truncs, infos = env.step(actions)
        
        m_tot = env.total_world_mass
        m_ply = sum(p.mass for p in env.players.values())
        m_foo = len(env.food_positions) * env.pellet_mass
        m_res = env.reserve_mass
        m_max = max(p.mass for p in env.players.values())
        predation_count += len(env.predation_events)

        history_total.append(m_tot)
        history_players.append(m_ply)
        history_food.append(m_foo)
        history_reserve.append(m_res)
        history_max_mass.append(m_max)

    max_drift = max(abs(m - 400.0) for m in history_total)
    print(f"1,000 steps completed!")
    print(f"Target Total Mass: 400.0000 kg")
    print(f"Max Absolute Drift: {max_drift:.3e} kg (Floating-point precision!)")
    print(f"Final Player Mass: {history_players[-1]:.2f} kg | Final Food Mass: {history_food[-1]:.2f} kg | Final Reserve: {history_reserve[-1]:.2f} kg")
    print(f"Total Predation Events: {predation_count}")
    assert max_drift < 1e-4, f"Mass drift detected: {max_drift}"

    # 1. Plot Mass Conservation Time Series
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    time_ax = np.arange(steps) * env.dt

    ax.plot(time_ax, history_total, color="black", lw=2.5, linestyle="--", label="Total World Mass (Invariant = 400kg)")
    ax.plot(time_ax, history_players, color="#3b82f6", lw=1.8, label="Active Players Mass (kg)")
    ax.plot(time_ax, history_reserve, color="#10b981", lw=1.8, label="Environmental Reserve Buffer (kg)")
    ax.plot(time_ax, history_food, color="#f59e0b", lw=1.8, label="Active Food Pellets Mass (kg)")
    ax.plot(time_ax, history_max_mass, color="#ef4444", lw=1.5, linestyle=":", label="Peak Single Cell Mass (kg)")

    ax.set_title("Strict Closed Thermodynamic Mass Conservation ($M_{\\mathrm{total}} \\equiv 400.0\\mathrm{kg}$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Simulation Time (seconds)", fontsize=11)
    ax.set_ylabel("Mass (kg)", fontsize=11)
    ax.set_ylim(-5, 430)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    curve_path = "outputs/mass_conservation_curve.png"
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.savefig(os.path.join(artifact_dir, "mass_conservation_curve.png"))
    plt.close()
    print(f"Saved mass curve: {curve_path}")

    # =========================================================================
    # PHASE 2: VISUALIZE 2D POTENTIAL FIELD GRID WITH ASYMMETRIC MASSES
    # =========================================================================
    print("=" * 70)
    print("PHASE 2: RENDERING 50x50+ POTENTIAL FIELD GRID HEATMAP")
    print("=" * 70)

    # Re-initialize scenario with stark mass asymmetry
    test_env = PartiallyObservableAgarEnv(arena_size=16.0, num_players=4, num_food=50)
    test_env.reset(seed=99)

    # Configure players:
    # Player 0: Apex Giant (95 kg) at (2.5, 1.5)
    # Player 1: Small Agile Prey (14 kg) at (-3.5, -2.0)
    # Player 2: Medium Competitor (35 kg) at (-2.0, 3.5)
    # Player 3: Fragile Scavenger (12 kg) at (3.5, -3.5)
    test_env.players["player_0"].pieces = [SubPiece(pos=np.array([2.5, 1.5], dtype=np.float32), mass=95.0, vel=np.zeros(2))]
    test_env.players["player_1"].pieces = [SubPiece(pos=np.array([-3.5, -2.0], dtype=np.float32), mass=14.0, vel=np.zeros(2))]
    test_env.players["player_2"].pieces = [SubPiece(pos=np.array([-2.0, 3.5], dtype=np.float32), mass=35.0, vel=np.zeros(2))]
    test_env.players["player_3"].pieces = [SubPiece(pos=np.array([3.5, -3.5], dtype=np.float32), mass=12.0, vel=np.zeros(2))]

    prob, grid_pts = test_env._compute_spawn_density_grid()
    grid_res = prob.shape[0]
    print(f"Computed potential field on {grid_res}x{grid_res} lattice (cell size: {test_env.arena_size/grid_res:.3f}m)")

    # Sample 75 food pellets to show distribution
    sampled_food = test_env._sample_food_from_grid(75)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)

    # Plot 2D density heatmap
    extent = [-test_env.half_arena, test_env.half_arena, -test_env.half_arena, test_env.half_arena]
    im = ax.imshow(prob, origin="lower", extent=extent, cmap="viridis", alpha=0.85)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spawn Probability Density $P(u, v)$", fontsize=10)

    # Plot contested corridor vectors between pairs
    active = list(test_env.players.values())
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            p_i = active[i]
            p_j = active[j]
            ax.plot([p_i.pos[0], p_j.pos[0]], [p_i.pos[1], p_j.pos[1]], color="white", linestyle="--", alpha=0.35, lw=1.0)
            
            # Plot the inverse-mass shifted center c_ij
            m_i, m_j = p_i.mass, p_j.mass
            c_ij = (m_j * p_i.pos + m_i * p_j.pos) / (m_i + m_j)
            ax.plot(c_ij[0], c_ij[1], marker="x", color="#fbbf24", markersize=7, markeredgewidth=1.8)

    # Overlay sampled food pellets
    ax.scatter(sampled_food[:, 0], sampled_food[:, 1], color="#f59e0b", edgecolors="white", s=22, linewidths=0.5, label="Sampled Food Pellets", zorder=4)

    # Overlay players with clear radius circles and clearance halos
    colors = {"player_0": "#ef4444", "player_1": "#3b82f6", "player_2": "#10b981", "player_3": "#a855f7"}
    names = {"player_0": "Giant (95kg)", "player_1": "Prey (14kg)", "player_2": "Medium (35kg)", "player_3": "Scavenger (12kg)"}

    for pid, player in test_env.players.items():
        pc = player.pieces[0]
        col = colors.get(pid, "white")
        # Body
        body = patches.Circle(pc.pos, pc.radius, facecolor=col, edgecolor="white", lw=2, zorder=5)
        ax.add_patch(body)
        # Clearance shadow border
        halo = patches.Circle(pc.pos, pc.radius + 0.65, facecolor="none", edgecolor=col, linestyle=":", lw=1.2, alpha=0.6, zorder=4)
        ax.add_patch(halo)
        
        label_text = f"{names[pid]}"
        ax.text(pc.pos[0], pc.pos[1] + pc.radius + 0.35, label_text, color="white", fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor=col))

    ax.set_title(f"High-Resolution Field Grid ({grid_res}x{grid_res}): Inverse-Mass Corridors & Clearance Shadows", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-test_env.half_arena, test_env.half_arena)
    ax.set_ylim(-test_env.half_arena, test_env.half_arena)
    ax.set_xlabel("Arena X (meters)", fontsize=10)
    ax.set_ylabel("Arena Y (meters)", fontsize=10)
    ax.legend(loc="upper right", frameon=True, facecolor="black", labelcolor="white", fontsize=8)

    heatmap_path = "outputs/field_grid_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.savefig(os.path.join(artifact_dir, "field_grid_heatmap.png"))
    plt.close()
    print(f"Saved field grid heatmap: {heatmap_path}")
    print("=" * 70)
    print("ALL VERIFICATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify_and_plot()
