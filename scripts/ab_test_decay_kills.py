"""
ab_test_decay_kills.py

Rigorous Scientific A/B/C Tournament testing the causal effect of metabolic mass decay rates
on predatory kills, circling behavior, and cell growth under a 400kg mass cap ecosystem.

Conditions:
- Treatment A (Baseline / 0% stronger): decay_multiplier = 1.0
- Treatment B (+30% stronger decay):   decay_multiplier = 1.3
- Treatment C (+50% stronger decay):   decay_multiplier = 1.5

Adheres to ml-best-practices:
- Paired identical random seeds across conditions
- One-way ANOVA across regimes
- Welch's t-test and Mann-Whitney U non-parametric test
- Cohen's d effect sizes
- 1,000-sample bootstrap 95% confidence intervals
- Complete CSV exports and multi-panel diagnostic visualizations
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
logger = logging.getLogger("DecayABTournament")


def bootstrap_ci(data: List[float], num_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence interval with replacement."""
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


def run_condition_tournament(
    condition_name: str,
    decay_multiplier: float,
    num_rounds: int = 20,
    episode_steps: int = 250,
    base_seed: int = 500,
    champ_ppo_path: str = "outputs/agar_ppo_champion.pt",
    champ_diff_path: str = "outputs/agar_diffusion_champion.pt",
    champ_critic_path: str = "outputs/agar_diffusion_critic_champion.pt"
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    logger.info(f"--- Running Tournament Condition: {condition_name} (Decay Multiplier: {decay_multiplier:.1f}x) ---")
    
    env = PartiallyObservableAgarEnv(
        arena_size=15.0,
        num_players=6,
        num_food=100,
        max_steps=episode_steps,
        total_world_mass=400.0,
        pellet_mass=0.5,
        mass_decay_multiplier=decay_multiplier
    )

    round_records = []
    condition_metrics = {
        "round_kills": [],
        "kills": [],
        "deaths": [],
        "kd_ratio": [],
        "peak_mass": [],
        "food": [],
        "circling_index": [],
        "splits": [],
        "wall_contacts": []
    }

    # Setup competitors: 3 PPO + 3 Diffusion
    competitors = {
        "player_0": ("Champion PPO 1", SuperiorPPOAgent(weights_path=champ_ppo_path)),
        "player_1": ("Guided Diffusion 1", AgarDiffusionPolicy(model_path=champ_diff_path, critic_path=champ_critic_path, num_candidates=5, seed=42)),
        "player_2": ("Champion PPO 2", SuperiorPPOAgent(weights_path=champ_ppo_path)),
        "player_3": ("Guided Diffusion 2", AgarDiffusionPolicy(model_path=champ_diff_path, critic_path=champ_critic_path, num_candidates=5, seed=105)),
        "player_4": ("Champion PPO 3", SuperiorPPOAgent(weights_path=champ_ppo_path)),
        "player_5": ("Guided Diffusion 3", AgarDiffusionPolicy(model_path=champ_diff_path, critic_path=champ_critic_path, num_candidates=5, seed=202)),
    }

    for r in range(num_rounds):
        seed = base_seed + r * 17
        obs_dict, _ = env.reset(seed=seed)
        for _, (_, pol) in competitors.items():
            if hasattr(pol, "reset"):
                pol.reset()

        round_actions = {pid: [] for pid in env.player_ids}
        round_omegas = {pid: [] for pid in env.player_ids}

        for step in range(episode_steps):
            actions = {}
            for pid, (label, agent) in competitors.items():
                act = agent.predict(obs_dict[pid], deterministic=True)
                actions[pid] = act
                round_actions[pid].append(act)
                round_omegas[pid].append(abs(float(env.players[pid].angular_vel)))

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break

        # Tally round totals
        total_round_kills = sum(env.players[pid].kills for pid in env.player_ids)
        condition_metrics["round_kills"].append(float(total_round_kills))

        for pid in env.player_ids:
            p_state = env.players[pid]
            label, _ = competitors[pid]
            k = float(p_state.kills)
            d = float(p_state.deaths)
            kd = float(k / max(1.0, d))
            pm = float(p_state.peak_mass)
            f = float(p_state.food_eaten)
            wc = float(p_state.wall_contacts)
            splits = float(sum(1 for a in round_actions[pid] if a[2] > 0.5))
            circling = float(np.mean(round_omegas[pid]))

            condition_metrics["kills"].append(k)
            condition_metrics["deaths"].append(d)
            condition_metrics["kd_ratio"].append(kd)
            condition_metrics["peak_mass"].append(pm)
            condition_metrics["food"].append(f)
            condition_metrics["circling_index"].append(circling)
            condition_metrics["splits"].append(splits)
            condition_metrics["wall_contacts"].append(wc)

            round_records.append({
                "condition": condition_name,
                "decay_multiplier": decay_multiplier,
                "round": r + 1,
                "seed": seed,
                "player_id": pid,
                "policy_label": label,
                "kills": k,
                "deaths": d,
                "kd_ratio": round(kd, 2),
                "peak_mass": round(pm, 2),
                "food_eaten": f,
                "circling_index": round(circling, 4),
                "splits_triggered": splits,
                "wall_contacts": wc
            })

        logger.info(f"[{condition_name}] Round {r+1}/{num_rounds} (Seed {seed}) -> Kills: {total_round_kills}, "
                    f"Max Mass: {max(p.peak_mass for p in env.players.values()):.1f}kg")

    return round_records, condition_metrics


def run_full_experiment(num_rounds: int = 20):
    os.makedirs("outputs", exist_ok=True)
    artifact_dir = os.path.expanduser("~/.gemini/antigravity/brain/62dc0844-0873-4692-bb43-1836ab1211da/media")
    os.makedirs(artifact_dir, exist_ok=True)

    champ_ppo_path = "outputs/agar_ppo_champion.pt"
    champ_diff_path = "outputs/agar_diffusion_champion.pt"
    champ_critic_path = "outputs/agar_diffusion_critic_champion.pt"

    # Execute conditions with identical paired seeds
    conditions = [
        ("0% Stronger (Baseline 1.0x)", 1.0),
        ("+30% Stronger (1.3x)", 1.3),
        ("+50% Stronger (1.5x)", 1.5)
    ]

    all_records = []
    condition_results = {}

    for name, mult in conditions:
        records, metrics = run_condition_tournament(
            condition_name=name,
            decay_multiplier=mult,
            num_rounds=num_rounds,
            episode_steps=250,
            base_seed=500,
            champ_ppo_path=champ_ppo_path,
            champ_diff_path=champ_diff_path,
            champ_critic_path=champ_critic_path
        )
        all_records.extend(records)
        condition_results[name] = metrics

    # 1. Export Round-by-Round Data
    csv_round_path = "outputs/decay_ab_round_by_round.csv"
    with open(csv_round_path, "w", newline="") as f:
        fieldnames = list(all_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    logger.info(f"Saved round-by-round CSV: {csv_round_path}")

    # 2. Compute Summary Statistics & Hypothesis Tests
    keys = ["0% Stronger (Baseline 1.0x)", "+30% Stronger (1.3x)", "+50% Stronger (1.5x)"]
    metrics_to_compare = [
        ("round_kills", "Total Kills per Round"),
        ("kills", "Player Kills per Episode"),
        ("circling_index", "Circling Index (Mean |omega|)"),
        ("peak_mass", "Peak Cell Mass (kg)"),
        ("food", "Food Pellets Foraged"),
        ("splits", "Split Actions Triggered"),
        ("kd_ratio", "Kill / Death Ratio")
    ]

    summary_rows = []
    print("\n" + "=" * 80)
    print("SCIENTIFIC STATISTICAL COMPARISON ACROSS DECAY REGIMES (N = 60 Rounds, 360 Player-Episodes)")
    print("=" * 80)

    for metric_key, metric_title in metrics_to_compare:
        a_data = condition_results[keys[0]][metric_key]
        b_data = condition_results[keys[1]][metric_key]
        c_data = condition_results[keys[2]][metric_key]

        mean_a, low_a, up_a = bootstrap_ci(a_data)
        mean_b, low_b, up_b = bootstrap_ci(b_data)
        mean_c, low_c, up_c = bootstrap_ci(c_data)

        # One-way ANOVA
        f_stat, p_anova = stats.f_oneway(a_data, b_data, c_data)

        # Pairwise A vs B
        t_ab, p_ttest_ab = stats.ttest_ind(a_data, b_data, equal_var=False)
        u_ab, p_mw_ab = stats.mannwhitneyu(a_data, b_data, alternative="two-sided")
        d_ab = compute_cohens_d(a_data, b_data)

        # Pairwise A vs C
        t_ac, p_ttest_ac = stats.ttest_ind(a_data, c_data, equal_var=False)
        u_ac, p_mw_ac = stats.mannwhitneyu(a_data, c_data, alternative="two-sided")
        d_ac = compute_cohens_d(a_data, c_data)

        # Pairwise B vs C
        t_bc, p_ttest_bc = stats.ttest_ind(b_data, c_data, equal_var=False)
        u_bc, p_mw_bc = stats.mannwhitneyu(b_data, c_data, alternative="two-sided")
        d_bc = compute_cohens_d(b_data, c_data)

        summary_rows.append({
            "metric": metric_title,
            "baseline_mean": round(mean_a, 3),
            "baseline_95ci": f"[{low_a:.2f}, {up_a:.2f}]",
            "decay_30_mean": round(mean_b, 3),
            "decay_30_95ci": f"[{low_b:.2f}, {up_b:.2f}]",
            "decay_50_mean": round(mean_c, 3),
            "decay_50_95ci": f"[{low_c:.2f}, {up_c:.2f}]",
            "anova_F": round(float(f_stat), 3),
            "anova_p": f"{p_anova:.4e}",
            "d_A_vs_B": round(d_ab, 3),
            "p_ttest_A_vs_B": f"{p_ttest_ab:.4e}",
            "d_A_vs_C": round(d_ac, 3),
            "p_ttest_A_vs_C": f"{p_ttest_ac:.4e}",
            "d_B_vs_C": round(d_bc, 3),
            "p_ttest_B_vs_C": f"{p_ttest_bc:.4e}"
        })

        print(f"\nMetric: {metric_title}")
        print(f"  Baseline (0%):   {mean_a:.3f} [95% CI: {low_a:.3f}, {up_a:.3f}]")
        print(f"  +30% Stronger:   {mean_b:.3f} [95% CI: {low_b:.3f}, {up_b:.3f}] (Cohen's d: {d_ab:+.2f}, p={p_ttest_ab:.4e})")
        print(f"  +50% Stronger:   {mean_c:.3f} [95% CI: {low_c:.3f}, {up_c:.3f}] (Cohen's d: {d_ac:+.2f}, p={p_ttest_ac:.4e})")
        print(f"  One-Way ANOVA:   F={f_stat:.3f}, p={p_anova:.4e}")

    # Export Summary CSV
    csv_summary_path = "outputs/decay_ab_statistical_summary.csv"
    with open(csv_summary_path, "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Saved statistical summary CSV: {csv_summary_path}")

    # 3. Publication-Grade Multi-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    labels = ["0% Stronger\n(Baseline 1.0x)", "+30% Stronger\n(1.3x)", "+50% Stronger\n(1.5x)"]
    palette = ["#3b82f6", "#10b981", "#ef4444"]

    # Panel A: Predatory Kills per Round
    ax_a = axes[0, 0]
    kills_data = [condition_results[k]["round_kills"] for k in keys]
    means_k = [np.mean(d) for d in kills_data]
    err_low_k = [means_k[i] - bootstrap_ci(kills_data[i])[1] for i in range(3)]
    err_up_k = [bootstrap_ci(kills_data[i])[2] - means_k[i] for i in range(3)]
    bars_a = ax_a.bar(labels, means_k, yerr=[err_low_k, err_up_k], capsize=6, color=palette, alpha=0.85, edgecolor="black", lw=1.2)
    ax_a.set_title("Predatory Kills per Round (95% Bootstrap CI)", fontsize=12, fontweight="bold", pad=10)
    ax_a.set_ylabel("Confirmed Kills / Round", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars_a, means_k):
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, f"{m:.2f}",
                  ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    # Panel B: Circling Index (Mean |omega|)
    ax_b = axes[0, 1]
    circ_data = [condition_results[k]["circling_index"] for k in keys]
    means_c = [np.mean(d) for d in circ_data]
    err_low_c = [means_c[i] - bootstrap_ci(circ_data[i])[1] for i in range(3)]
    err_up_c = [bootstrap_ci(circ_data[i])[2] - means_c[i] for i in range(3)]
    bars_b = ax_b.bar(labels, means_c, yerr=[err_low_c, err_up_c], capsize=6, color=palette, alpha=0.85, edgecolor="black", lw=1.2)
    ax_b.set_title("Circling Winding Index (Mean |$\\omega$|)", fontsize=12, fontweight="bold", pad=10)
    ax_b.set_ylabel("Mean Angular Velocity (rad/s)", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars_b, means_c):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, f"{m:.3f}",
                  ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    # Panel C: Peak Cell Mass Boxplot
    ax_c = axes[1, 0]
    mass_data = [condition_results[k]["peak_mass"] for k in keys]
    box = ax_c.boxplot(mass_data, tick_labels=labels, patch_artist=True,
                       boxprops=dict(facecolor="#e0e7ff", color="#3730a3", lw=1.5),
                       medianprops=dict(color="#b91c1c", lw=2.2))
    for patch, col in zip(box["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax_c.set_title("Peak Cell Mass Distribution (kg)", fontsize=12, fontweight="bold", pad=10)
    ax_c.set_ylabel("Peak Mass (kg)", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel D: Foraged Food Pellets
    ax_d = axes[1, 1]
    food_data = [condition_results[k]["food"] for k in keys]
    means_f = [np.mean(d) for d in food_data]
    err_low_f = [means_f[i] - bootstrap_ci(food_data[i])[1] for i in range(3)]
    err_up_f = [bootstrap_ci(food_data[i])[2] - means_f[i] for i in range(3)]
    bars_d = ax_d.bar(labels, means_f, yerr=[err_low_f, err_up_f], capsize=6, color=palette, alpha=0.85, edgecolor="black", lw=1.2)
    ax_d.set_title("Food Pellets Foraged per Episode", fontsize=12, fontweight="bold", pad=10)
    ax_d.set_ylabel("Pellets Eaten", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars_d, means_f):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, f"{m:.1f}",
                  ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    plot_path = "outputs/decay_ab_kills_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.savefig(os.path.join(artifact_dir, "decay_ab_kills_comparison.png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved multi-panel plot to: {plot_path}")

    print("\n" + "=" * 80)
    print("A/B/C MASS DECAY EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_experiment(num_rounds=20)
