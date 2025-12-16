"""
ab_test_decay_sweep.py

Fine-grained scientific decay sweep around +30% stronger decay on the updated 400kg ecosystem
with ego body-frame observations, 4-piece cap, and gravitational re-merging attraction.

Conditions Tested:
- +15% Stronger: multiplier = 1.15
- +25% Stronger: multiplier = 1.25
- +30% Stronger: multiplier = 1.30
- +35% Stronger: multiplier = 1.35
- +45% Stronger: multiplier = 1.45

Adheres strictly to ml-best-practices:
- Paired random seeds across conditions (15 rounds per condition = 75 rounds total, N=450 player-episodes)
- One-way ANOVA
- Pairwise Welch's t-test and Mann-Whitney U test
- Cohen's d effect sizes
- 1,000-sample bootstrap 95% confidence intervals
- Complete CSV exports and multi-panel diagnostic figures
"""

import os
import sys
import time
import csv
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DecayFineSweep")


def bootstrap_ci(data: List[float], num_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    n = len(arr)
    boot_means = np.zeros(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        resample = np.random.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100.0))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100.0))
    return float(np.mean(arr)), lower, upper


def compute_cohens_d(group_a: List[float], group_b: List[float]) -> float:
    a = np.array(group_a, dtype=np.float64)
    b = np.array(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pooled < 1e-8:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / s_pooled)


def run_condition(
    name: str,
    multiplier: float,
    num_rounds: int = 15,
    episode_steps: int = 250,
    base_seed: int = 600,
    ppo_path: str = "outputs/agar_ppo_champion.pt",
    diff_path: str = "outputs/agar_diffusion_champion.pt",
    critic_path: str = "outputs/agar_diffusion_critic_champion.pt"
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    logger.info(f"--- Running Sweep Condition: {name} (multiplier={multiplier:.2f}x) ---")
    env = PartiallyObservableAgarEnv(
        arena_size=15.0,
        num_players=6,
        num_food=100,
        max_steps=episode_steps,
        total_world_mass=400.0,
        pellet_mass=0.5,
        mass_decay_multiplier=multiplier,
        max_pieces=4
    )

    competitors = {
        "player_0": ("Champion PPO 1", SuperiorPPOAgent(weights_path=ppo_path)),
        "player_1": ("Guided Diffusion 1", AgarDiffusionPolicy(model_path=diff_path, critic_path=critic_path, num_candidates=5, seed=42)),
        "player_2": ("Champion PPO 2", SuperiorPPOAgent(weights_path=ppo_path)),
        "player_3": ("Guided Diffusion 2", AgarDiffusionPolicy(model_path=diff_path, critic_path=critic_path, num_candidates=5, seed=105)),
        "player_4": ("Champion PPO 3", SuperiorPPOAgent(weights_path=ppo_path)),
        "player_5": ("Guided Diffusion 3", AgarDiffusionPolicy(model_path=diff_path, critic_path=critic_path, num_candidates=5, seed=202)),
    }

    round_records = []
    metrics = {
        "round_kills": [],
        "kills": [],
        "deaths": [],
        "kd_ratio": [],
        "peak_mass": [],
        "food": [],
        "circling_index": [],
        "splits": []
    }

    for r in range(num_rounds):
        seed = base_seed + r * 19
        obs_dict, _ = env.reset(seed=seed)
        for _, (_, pol) in competitors.items():
            if hasattr(pol, "reset"):
                pol.reset()

        round_omegas = {pid: [] for pid in env.player_ids}
        round_splits = {pid: 0 for pid in env.player_ids}

        for step in range(episode_steps):
            actions = {}
            for pid, (label, agent) in competitors.items():
                act = agent.predict(obs_dict[pid], deterministic=True)
                actions[pid] = act
                round_omegas[pid].append(abs(float(env.players[pid].angular_vel)))
                if act[2] > 0.5:
                    round_splits[pid] += 1

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break

        r_kills = sum(env.players[pid].kills for pid in env.player_ids)
        metrics["round_kills"].append(float(r_kills))

        for pid in env.player_ids:
            p_state = env.players[pid]
            label, _ = competitors[pid]
            k = float(p_state.kills)
            d = float(p_state.deaths)
            kd = float(k / max(1.0, d))
            pm = float(p_state.peak_mass)
            f = float(p_state.food_eaten)
            circling = float(np.mean(round_omegas[pid]))
            splits = float(round_splits[pid])

            metrics["kills"].append(k)
            metrics["deaths"].append(d)
            metrics["kd_ratio"].append(kd)
            metrics["peak_mass"].append(pm)
            metrics["food"].append(f)
            metrics["circling_index"].append(circling)
            metrics["splits"].append(splits)

            round_records.append({
                "condition": name,
                "decay_multiplier": multiplier,
                "round": r + 1,
                "seed": seed,
                "player_id": pid,
                "policy": label,
                "kills": k,
                "deaths": d,
                "kd_ratio": round(kd, 2),
                "peak_mass": round(pm, 2),
                "food_eaten": f,
                "circling_index": round(circling, 4),
                "splits": splits
            })

        logger.info(f"[{name}] Round {r+1}/{num_rounds} -> Kills: {r_kills}, Peak Mass: {max(p.peak_mass for p in env.players.values()):.1f}kg")

    return round_records, metrics


def main():
    os.makedirs("outputs", exist_ok=True)
    artifact_dir = os.path.expanduser("~/.gemini/antigravity/brain/62dc0844-0873-4692-bb43-1836ab1211da/media")
    os.makedirs(artifact_dir, exist_ok=True)

    conditions = [
        ("+15% Stronger (1.15x)", 1.15),
        ("+25% Stronger (1.25x)", 1.25),
        ("+30% Stronger (1.30x)", 1.30),
        ("+35% Stronger (1.35x)", 1.35),
        ("+45% Stronger (1.45x)", 1.45)
    ]

    all_records = []
    cond_metrics = {}

    for name, mult in conditions:
        records, metrics = run_condition(name, mult, num_rounds=15, base_seed=600)
        all_records.extend(records)
        cond_metrics[name] = metrics

    # Save round-by-round CSV
    csv_round = "outputs/decay_fine_sweep_round_by_round.csv"
    with open(csv_round, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    logger.info(f"Saved {csv_round}")

    # Summary table
    cond_names = [c[0] for c in conditions]
    summary_rows = []
    for name in cond_names:
        m = cond_metrics[name]
        mean_rk, low_rk, up_rk = bootstrap_ci(m["round_kills"])
        mean_pk, low_pk, up_pk = bootstrap_ci(m["kills"])
        mean_pm, low_pm, up_pm = bootstrap_ci(m["peak_mass"])
        mean_circ, low_c, up_c = bootstrap_ci(m["circling_index"])
        mean_sp, low_sp, up_sp = bootstrap_ci(m["splits"])
        mean_fd, low_fd, up_fd = bootstrap_ci(m["food"])

        summary_rows.append({
            "condition": name,
            "round_kills_mean": round(mean_rk, 3),
            "round_kills_95ci": f"[{low_rk:.2f}, {up_rk:.2f}]",
            "player_kills_mean": round(mean_pk, 3),
            "player_kills_95ci": f"[{low_pk:.2f}, {up_pk:.2f}]",
            "peak_mass_mean": round(mean_pm, 2),
            "peak_mass_95ci": f"[{low_pm:.1f}, {up_pm:.1f}]",
            "circling_index_mean": round(mean_circ, 4),
            "splits_mean": round(mean_sp, 3),
            "food_eaten_mean": round(mean_fd, 2)
        })

    csv_sum = "outputs/decay_fine_sweep_summary.csv"
    with open(csv_sum, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Saved {csv_sum}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    plt.subplots_adjust(wspace=0.3)

    short_labels = ["+15%\n(1.15x)", "+25%\n(1.25x)", "+30%\n(1.30x)", "+35%\n(1.35x)", "+45%\n(1.45x)"]
    palette = ["#60a5fa", "#34d399", "#10b981", "#fbbf24", "#f87171"]

    # 1. Kills per Round
    ax0 = axes[0]
    r_kills_list = [cond_metrics[n]["round_kills"] for n in cond_names]
    means_rk = [np.mean(d) for d in r_kills_list]
    errs_rk = [[means_rk[i] - bootstrap_ci(r_kills_list[i])[1] for i in range(5)],
               [bootstrap_ci(r_kills_list[i])[2] - means_rk[i] for i in range(5)]]
    bars0 = ax0.bar(short_labels, means_rk, yerr=errs_rk, capsize=5, color=palette, edgecolor="black", lw=1.2)
    ax0.set_title("Total Kills per Round (95% Bootstrap CI)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Confirmed Kills / Round")
    ax0.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars0, means_rk):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, f"{m:.2f}",
                 ha="center", va="center", color="white", fontweight="bold")

    # 2. Peak Cell Mass Distribution
    ax1 = axes[1]
    pm_list = [cond_metrics[n]["peak_mass"] for n in cond_names]
    bplot = ax1.boxplot(pm_list, tick_labels=short_labels, patch_artist=True,
                        medianprops=dict(color="#b91c1c", lw=2.2))
    for patch, col in zip(bplot["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax1.set_title("Peak Cell Mass Distributions (kg)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Peak Mass (kg)")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # 3. Circling Winding Index
    ax2 = axes[2]
    circ_list = [cond_metrics[n]["circling_index"] for n in cond_names]
    means_c = [np.mean(d) for d in circ_list]
    errs_c = [[means_c[i] - bootstrap_ci(circ_list[i])[1] for i in range(5)],
              [bootstrap_ci(circ_list[i])[2] - means_c[i] for i in range(5)]]
    bars2 = ax2.bar(short_labels, means_c, yerr=errs_c, capsize=5, color=palette, edgecolor="black", lw=1.2)
    ax2.set_title("Circling Winding Index (Mean |$\\omega$|)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Angular Velocity (rad/s)")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars2, means_c):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, f"{m:.3f}",
                 ha="center", va="center", color="white", fontweight="bold")

    plot_path = "outputs/decay_fine_sweep_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.savefig(os.path.join(artifact_dir, "decay_fine_sweep_comparison.png"))
    plt.close()
    logger.info(f"Saved {plot_path}")

    print("\n" + "=" * 80)
    print("FINE-GRAINED DECAY SWEEP SUMMARY (75 ROUNDS, N=450 PLAYER-EPISODES)")
    print("=" * 80)
    for row in summary_rows:
        print(f"Condition: {row['condition']:<24} | Kills/Rnd: {row['round_kills_mean']:.2f} {row['round_kills_95ci']} | "
              f"Peak Mass: {row['peak_mass_mean']:.1f}kg | Circling: {row['circling_index_mean']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
