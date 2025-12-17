"""
Scientific Head-to-Head Benchmark: Pure Neural PPO vs. Pure Neural Guided Diffusion.
Runs a fair, rigorous 30-round multi-agent tournament (N=180 player-episodes).
Evaluates:
1. Predatory Kills & K/D Ratio
2. Peak Mass & Nutrient Foraging
3. Kinetic Action Jerk (Smoothness)
4. Circling / Winding Index
5. Wall Contacts & Boundary Survival
6. Statistical Significance (Wilcoxon, Bootstrap 95% CI, Cohen's d)
Generates:
- outputs/pure_tournament_summary.csv
- outputs/pure_tournament_comparison.png
"""

import os
import sys
import time
import math
import csv
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PureTournament")


def bootstrap_ci(data: List[float], n_resamples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    if len(data) == 0:
        return 0.0, 0.0
    arr = np.array(data, dtype=np.float64)
    resamples = np.random.choice(arr, size=(n_resamples, len(arr)), replace=True)
    means = np.mean(resamples, axis=1)
    lower = float(np.percentile(means, (1.0 - ci) / 2.0 * 100.0))
    upper = float(np.percentile(means, (1.0 + ci) / 2.0 * 100.0))
    return lower, upper


def cohen_d(x: List[float], y: List[float]) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / max(1e-6, pooled_sd))


def compute_action_jerk(actions: List[np.ndarray], dt: float = 0.1) -> float:
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) / dt for i in range(1, len(actions))]
    return float(np.mean(diffs))


def main():
    logger.info("Initializing Pure Neural Tournament: PPO vs. Value-Guided Diffusion...")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    ppo_weights = "outputs/agar_ppo_champion.pt"
    diff_weights = "outputs/agar_diffusion_champion.pt"
    critic_weights = "outputs/agar_diffusion_critic_champion.pt"

    # Instantiate 3 PPO Agents and 3 Diffusion Agents
    ppo_agents = [
        SuperiorPPOAgent(weights_path=ppo_weights),
        SuperiorPPOAgent(weights_path=ppo_weights),
        SuperiorPPOAgent(weights_path=ppo_weights),
    ]

    diff_agents = [
        AgarDiffusionPolicy(action_horizon=16, exec_horizon=4, action_dim=3, obs_dim=38,
                            num_ddim_steps=10, seed=42, model_path=diff_weights, critic_path=critic_weights, num_candidates=8),
        AgarDiffusionPolicy(action_horizon=16, exec_horizon=4, action_dim=3, obs_dim=38,
                            num_ddim_steps=10, seed=105, model_path=diff_weights, critic_path=critic_weights, num_candidates=8),
        AgarDiffusionPolicy(action_horizon=16, exec_horizon=4, action_dim=3, obs_dim=38,
                            num_ddim_steps=10, seed=202, model_path=diff_weights, critic_path=critic_weights, num_candidates=8),
    ]

    competitors = {
        "player_0": ("PPO", ppo_agents[0]),
        "player_1": ("Guided Diffusion", diff_agents[0]),
        "player_2": ("PPO", ppo_agents[1]),
        "player_3": ("Guided Diffusion", diff_agents[1]),
        "player_4": ("PPO", ppo_agents[2]),
        "player_5": ("Guided Diffusion", diff_agents[2]),
    }

    num_rounds = 30
    episode_steps = 350
    arena_size = 14.0

    env = PartiallyObservableAgarEnv(
        arena_size=arena_size,
        num_players=6,
        num_food=100,
        max_steps=episode_steps,
        mass_decay_multiplier=1.35,
        max_pieces=4
    )

    round_records = []
    player_records = []

    # Aggregators by paradigm
    metrics = {
        "PPO": {"kills": [], "deaths": [], "peak_mass": [], "food": [], "jerk": [], "circling": [], "wall_contacts": [], "latency": []},
        "Guided Diffusion": {"kills": [], "deaths": [], "peak_mass": [], "food": [], "jerk": [], "circling": [], "wall_contacts": [], "latency": []},
    }

    print("\n" + "=" * 105)
    print(f"{'Round':<6} | {'PPO Kills':<10} | {'Diff Kills':<11} | {'PPO Mass':<10} | {'Diff Mass':<10} | {'PPO Jerk':<10} | {'Diff Jerk':<10}")
    print("=" * 105)

    for r in range(1, num_rounds + 1):
        obs_dict, _ = env.reset(seed=500 + r * 13)
        for _, pol in competitors.values():
            if hasattr(pol, "reset"):
                pol.reset()

        act_history = {pid: [] for pid in competitors}
        latencies = {pid: [] for pid in competitors}

        for step in range(episode_steps):
            actions = {}
            for pid, (_, pol) in competitors.items():
                t0 = time.perf_counter()
                act = pol.predict(obs_dict[pid], deterministic=True)
                lat = (time.perf_counter() - t0) * 1000.0
                actions[pid] = act
                act_history[pid].append(act.copy())
                latencies[pid].append(lat)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # Collect round metrics
        ppo_r_kills = sum(env.players[pid].kills for pid, (paradigm, _) in competitors.items() if paradigm == "PPO")
        diff_r_kills = sum(env.players[pid].kills for pid, (paradigm, _) in competitors.items() if paradigm == "Guided Diffusion")

        ppo_r_mass = [env.players[pid].peak_mass for pid, (paradigm, _) in competitors.items() if paradigm == "PPO"]
        diff_r_mass = [env.players[pid].peak_mass for pid, (paradigm, _) in competitors.items() if paradigm == "Guided Diffusion"]

        ppo_r_jerk = [compute_action_jerk(act_history[pid]) for pid, (paradigm, _) in competitors.items() if paradigm == "PPO"]
        diff_r_jerk = [compute_action_jerk(act_history[pid]) for pid, (paradigm, _) in competitors.items() if paradigm == "Guided Diffusion"]

        print(
            f"{r:<6d} | {ppo_r_kills:<10d} | {diff_r_kills:<11d} | "
            f"{np.mean(ppo_r_mass):<10.1f} | {np.mean(diff_r_mass):<10.1f} | "
            f"{np.mean(ppo_r_jerk):<10.3f} | {np.mean(diff_r_jerk):<10.3f}"
        )

        for pid, (paradigm, _) in competitors.items():
            p_state = env.players[pid]
            jerk = compute_action_jerk(act_history[pid])
            circling = float(np.mean([abs(a[1]) for a in act_history[pid]]))
            mean_lat = float(np.mean(latencies[pid]))

            metrics[paradigm]["kills"].append(p_state.kills)
            metrics[paradigm]["deaths"].append(p_state.deaths)
            metrics[paradigm]["peak_mass"].append(p_state.peak_mass)
            metrics[paradigm]["food"].append(p_state.food_eaten)
            metrics[paradigm]["jerk"].append(jerk)
            metrics[paradigm]["circling"].append(circling)
            metrics[paradigm]["wall_contacts"].append(p_state.wall_contacts)
            metrics[paradigm]["latency"].append(mean_lat)

            player_records.append({
                "round": r,
                "player_id": pid,
                "paradigm": paradigm,
                "kills": p_state.kills,
                "deaths": p_state.deaths,
                "peak_mass": round(p_state.peak_mass, 2),
                "food_eaten": p_state.food_eaten,
                "action_jerk": round(jerk, 4),
                "circling_index": round(circling, 4),
                "wall_contacts": p_state.wall_contacts,
                "latency_ms": round(mean_lat, 3)
            })

    print("=" * 105 + "\n")

    # Save round-by-round CSV
    with open("outputs/pure_tournament_round_by_round.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(player_records[0].keys()))
        writer.writeheader()
        writer.writerows(player_records)

    # Statistical Analysis
    summary_rows = []
    print("=" * 115)
    print("FORMAL STATISTICAL HYPOTHESIS TESTING (PURE NEURAL PPO VS. PURE GUIDED DIFFUSION)")
    print("=" * 115)

    headers = [
        "Metric", "PPO Mean", "PPO 95% CI", "Diffusion Mean", "Diffusion 95% CI",
        "Delta (%)", "Cohen's d", "p-value (Wilcoxon)", "Significance"
    ]
    summary_table_data = []

    metric_keys = [
        ("kills", "Predatory Kills / Ep"),
        ("deaths", "Deaths / Ep"),
        ("peak_mass", "Peak Cell Mass (kg)"),
        ("food", "Food Pellets Eaten"),
        ("jerk", "Kinematic Jerk (rad/s^2)"),
        ("circling", "Circling / Winding Index"),
        ("wall_contacts", "Wall Boundary Contacts"),
        ("latency", "Inference Latency (ms)"),
    ]

    for key, label in metric_keys:
        v_ppo = metrics["PPO"][key]
        v_diff = metrics["Guided Diffusion"][key]

        m_ppo = float(np.mean(v_ppo))
        m_diff = float(np.mean(v_diff))

        ci_ppo = bootstrap_ci(v_ppo)
        ci_diff = bootstrap_ci(v_diff)

        d = cohen_d(v_diff, v_ppo)
        try:
            _, p_val = stats.wilcoxon(v_diff, v_ppo)
        except Exception:
            _, p_val = stats.mannwhitneyu(v_diff, v_ppo)

        delta_pct = ((m_diff - m_ppo) / max(1e-6, abs(m_ppo))) * 100.0
        sig = "*** (p<0.001)" if p_val < 0.001 else ("** (p<0.01)" if p_val < 0.01 else ("* (p<0.05)" if p_val < 0.05 else "n.s."))

        summary_table_data.append({
            "metric": label,
            "ppo_mean": round(m_ppo, 3),
            "ppo_ci": f"[{ci_ppo[0]:.2f}, {ci_ppo[1]:.2f}]",
            "diff_mean": round(m_diff, 3),
            "diff_ci": f"[{ci_diff[0]:.2f}, {ci_diff[1]:.2f}]",
            "delta_pct": f"{delta_pct:+.1f}%",
            "cohens_d": round(d, 3),
            "p_value": f"{p_val:.4e}" if p_val < 1e-4 else f"{p_val:.4f}",
            "significance": sig
        })

        print(
            f"{label:<28} | PPO: {m_ppo:<6.2f} {str(summary_table_data[-1]['ppo_ci']):<14} | "
            f"Diff: {m_diff:<6.2f} {str(summary_table_data[-1]['diff_ci']):<14} | "
            f"{delta_pct:>+7.1f}% | d={d:<5.2f} | p={summary_table_data[-1]['p_value']:<8} | {sig}"
        )

    print("=" * 115 + "\n")

    with open("outputs/pure_tournament_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_table_data[0].keys()))
        writer.writeheader()
        writer.writerows(summary_table_data)

    # 4-Panel Publication Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0b0f19")
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    colors = {"PPO": "#ef4444", "Guided Diffusion": "#06b6d4"}

    # Panel 1: Predatory Kills & Deaths
    ax = axes[0, 0]
    ax.set_facecolor("#111827")
    labels = ["Kills / Ep", "Deaths / Ep"]
    x = np.arange(len(labels))
    w = 0.35
    ppo_vals = [np.mean(metrics["PPO"]["kills"]), np.mean(metrics["PPO"]["deaths"])]
    diff_vals = [np.mean(metrics["Guided Diffusion"]["kills"]), np.mean(metrics["Guided Diffusion"]["deaths"])]
    ax.bar(x - w / 2, ppo_vals, width=w, label="Pure PPO", color=colors["PPO"], edgecolor="white", alpha=0.85)
    ax.bar(x + w / 2, diff_vals, width=w, label="Guided Diffusion", color=colors["Guided Diffusion"], edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count per Episode", color="white", fontsize=11)
    ax.set_title("Combat Predation & Survival (Pure Neural)", color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#374151")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    # Panel 2: Peak Mass & Foraging
    ax = axes[0, 1]
    ax.set_facecolor("#111827")
    labels = ["Peak Mass (kg)", "Food Eaten / 2"]
    x = np.arange(len(labels))
    ppo_vals = [np.mean(metrics["PPO"]["peak_mass"]), np.mean(metrics["PPO"]["food"]) / 2.0]
    diff_vals = [np.mean(metrics["Guided Diffusion"]["peak_mass"]), np.mean(metrics["Guided Diffusion"]["food"]) / 2.0]
    ax.bar(x - w / 2, ppo_vals, width=w, label="Pure PPO", color=colors["PPO"], edgecolor="white", alpha=0.85)
    ax.bar(x + w / 2, diff_vals, width=w, label="Guided Diffusion", color=colors["Guided Diffusion"], edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", color="white", fontsize=11)
    ax.set_title("Ecosystem Dominance & Nutrient Foraging", color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#374151")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    # Panel 3: Kinematic Jerk & Circling Index
    ax = axes[1, 0]
    ax.set_facecolor("#111827")
    labels = ["Kinematic Jerk (rad/s²)", "Circling Index (|ω|)"]
    x = np.arange(len(labels))
    ppo_vals = [np.mean(metrics["PPO"]["jerk"]), np.mean(metrics["PPO"]["circling"])]
    diff_vals = [np.mean(metrics["Guided Diffusion"]["jerk"]), np.mean(metrics["Guided Diffusion"]["circling"])]
    ax.bar(x - w / 2, ppo_vals, width=w, label="Pure PPO", color=colors["PPO"], edgecolor="white", alpha=0.85)
    ax.bar(x + w / 2, diff_vals, width=w, label="Guided Diffusion", color=colors["Guided Diffusion"], edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Smoothness & Trajectory Metric", color="white", fontsize=11)
    ax.set_title("Kinematic Smoothness & Anti-Circling", color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#374151")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    # Panel 4: Wall Contacts & Boundary Safety
    ax = axes[1, 1]
    ax.set_facecolor("#111827")
    labels = ["Wall Contacts / Ep"]
    x = np.arange(len(labels))
    ppo_vals = [np.mean(metrics["PPO"]["wall_contacts"])]
    diff_vals = [np.mean(metrics["Guided Diffusion"]["wall_contacts"])]
    ax.bar(x - w / 2, ppo_vals, width=w, label="Pure PPO", color=colors["PPO"], edgecolor="white", alpha=0.85)
    ax.bar(x + w / 2, diff_vals, width=w, label="Guided Diffusion", color=colors["Guided Diffusion"], edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Wall Collisions", color="white", fontsize=11)
    ax.set_title("Boundary Obstacle Avoidance", color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#374151")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    plt.savefig("outputs/pure_tournament_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved outputs/pure_tournament_comparison.png")


if __name__ == "__main__":
    main()
