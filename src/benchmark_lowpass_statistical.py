"""
Rigorous Multi-Iteration Statistical Benchmark for Low-Pass Action Filtering
Evaluates Raw Pre-Lowpass vs Light EMA (alpha=0.3) vs Medium EMA (alpha=0.5)
across 60 tournament rounds (20 rounds per condition with identical paired seeds).
Adheres strictly to ml-best-practices ("Comparing ML Models").
Uses standard Python csv module (zero pandas dependency).
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
import math
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from scipy import stats

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LowpassBenchmark")


def compute_action_jerk(actions: List[np.ndarray]) -> float:
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


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
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    s_pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if s_pooled < 1e-8:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / s_pooled)


def run_condition(
    condition_name: str,
    alpha_ppo: float,
    alpha_diff: float,
    num_rounds: int = 20,
    episode_steps: int = 250,
    base_seed: int = 500
) -> List[Dict[str, Any]]:
    logger.info(f"Executing Condition: {condition_name} ({num_rounds} rounds, {episode_steps} steps)...")
    env = PartiallyObservableAgarEnv(arena_size=14.0, num_players=6, num_food=120, max_steps=episode_steps)

    ppo_weights = "outputs/agar_ppo_champion.pt"
    if not os.path.exists(ppo_weights):
        ppo_weights = "projects/sim2real-ppo-navigation/outputs/agar_ppo_champion.pt"
    diff_weights = "outputs/agar_diffusion_champion.pt"
    if not os.path.exists(diff_weights):
        diff_weights = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_champion.pt"
    critic_weights = "outputs/agar_diffusion_critic_champion.pt"
    if not os.path.exists(critic_weights):
        critic_weights = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_critic_champion.pt"

    champ_ppo = SuperiorPPOAgent(weights_path=ppo_weights)
    champ_diff = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights)

    competitors = {
        "player_0": ("PPO-1", champ_ppo, "PPO"),
        "player_1": ("DIFF-1", champ_diff, "DIFF"),
        "player_2": ("PPO-2", champ_ppo, "PPO"),
        "player_3": ("DIFF-2", champ_diff, "DIFF"),
        "player_4": ("PPO-3", champ_ppo, "PPO"),
        "player_5": ("DIFF-3", champ_diff, "DIFF"),
    }

    records = []

    for r in range(num_rounds):
        round_seed = base_seed + r * 31
        obs_dict, _ = env.reset(seed=round_seed)
        prev_actions = {pid: np.array([0.92, 0.0, 0.0], dtype=np.float32) for pid in competitors}
        round_actions = {pid: [] for pid in competitors}

        for step in range(episode_steps):
            actions = {}
            for pid, (_, pol, arch) in competitors.items():
                raw_act = pol.predict(obs_dict[pid], deterministic=True)
                alpha = alpha_ppo if arch == "PPO" else alpha_diff
                if alpha > 0.0:
                    act = raw_act.copy()
                    act[0] = alpha * prev_actions[pid][0] + (1.0 - alpha) * raw_act[0]
                    act[1] = alpha * prev_actions[pid][1] + (1.0 - alpha) * raw_act[1]
                else:
                    act = raw_act
                prev_actions[pid] = act.copy()
                actions[pid] = act
                round_actions[pid].append(act.copy())

            obs_dict, _, _, _, _ = env.step(actions)

        for pid, (label, _, arch) in competitors.items():
            p_state = env.players[pid]
            jerk = compute_action_jerk(round_actions[pid])
            records.append({
                "condition": condition_name,
                "alpha_ppo": alpha_ppo,
                "alpha_diff": alpha_diff,
                "round": r + 1,
                "player_id": pid,
                "label": label,
                "arch": arch,
                "kills": p_state.kills,
                "deaths": p_state.deaths,
                "kd_ratio": p_state.kills / max(1, p_state.deaths),
                "peak_mass": p_state.peak_mass,
                "food_eaten": p_state.food_eaten,
                "actuator_jerk": jerk
            })

    return records


def run_full_statistical_study(num_rounds_per_condition: int = 20):
    conditions = [
        ("Raw_Pre_Lowpass", 0.0, 0.0),
        ("Light_EMA_PPO_0.3", 0.3, 0.0),
        ("Medium_EMA_PPO_0.5", 0.5, 0.0),
    ]

    all_records = []
    for cond_name, a_p, a_d in conditions:
        recs = run_condition(cond_name, a_p, a_d, num_rounds=num_rounds_per_condition)
        all_records.extend(recs)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dirs = [
        os.path.join(root_dir, "outputs"),
        os.path.join(root_dir, "projects", "sim2real-ppo-navigation", "outputs")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        csv_path = os.path.join(d, "lowpass_round_by_round.csv")
        if all_records:
            fieldnames = list(all_records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)

    logger.info(f"Collected {len(all_records)} total player-episode observations across 3 conditions.")

    # Partition PPO records
    raw_ppo = [r for r in all_records if r["arch"] == "PPO" and r["condition"] == "Raw_Pre_Lowpass"]
    light_ppo = [r for r in all_records if r["arch"] == "PPO" and r["condition"] == "Light_EMA_PPO_0.3"]
    med_ppo = [r for r in all_records if r["arch"] == "PPO" and r["condition"] == "Medium_EMA_PPO_0.5"]

    summary_rows = []
    metrics = ["actuator_jerk", "peak_mass", "food_eaten", "kills", "kd_ratio"]

    for m in metrics:
        raw_vals = [r[m] for r in raw_ppo]
        light_vals = [r[m] for r in light_ppo]
        med_vals = [r[m] for r in med_ppo]

        raw_m, raw_l, raw_u = bootstrap_ci(raw_vals)
        light_m, light_l, light_u = bootstrap_ci(light_vals)
        med_m, med_l, med_u = bootstrap_ci(med_vals)

        t_stat, p_val_t = stats.ttest_ind(light_vals, raw_vals, equal_var=False)
        u_stat, p_val_u = stats.mannwhitneyu(light_vals, raw_vals, alternative="two-sided")
        d_light = compute_cohens_d(raw_vals, light_vals)
        rel_delta_light = (light_m - raw_m) / max(1e-6, abs(raw_m)) * 100.0

        d_med = compute_cohens_d(raw_vals, med_vals)
        rel_delta_med = (med_m - raw_m) / max(1e-6, abs(raw_m)) * 100.0

        summary_rows.append({
            "metric": m,
            "raw_mean": round(raw_m, 4),
            "raw_ci95": f"[{raw_l:.3f}, {raw_u:.3f}]",
            "light_mean": round(light_m, 4),
            "light_ci95": f"[{light_l:.3f}, {light_u:.3f}]",
            "light_rel_delta_pct": round(rel_delta_light, 2),
            "light_p_val_welch": f"{p_val_t:.3e}",
            "light_p_val_mw": f"{p_val_u:.3e}",
            "light_cohens_d": round(d_light, 3),
            "med_mean": round(med_m, 4),
            "med_ci95": f"[{med_l:.3f}, {med_u:.3f}]",
            "med_rel_delta_pct": round(rel_delta_med, 2),
            "med_cohens_d": round(d_med, 3)
        })

    for d in out_dirs:
        summary_csv = os.path.join(d, "lowpass_statistical_summary.csv")
        if summary_rows:
            with open(summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

    print("\n" + "=" * 125)
    print("STATISTICAL SUMMARY: PRE-LOWPASS VS LIGHT EMA (alpha=0.3) VS MEDIUM EMA (alpha=0.5)")
    print("=" * 125)
    for row in summary_rows:
        print(f"{row['metric']:<16} | Raw: {row['raw_mean']:>8.4f} | Light: {row['light_mean']:>8.4f} ({row['light_rel_delta_pct']:>+6.1f}%, d={row['light_cohens_d']:>+5.2f}, p={row['light_p_val_welch']}) | Med: {row['med_mean']:>8.4f} ({row['med_rel_delta_pct']:>+6.1f}%, d={row['med_cohens_d']:>+5.2f})")
    print("=" * 125)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Impact of EMA Low-Pass Action Filtering on PPO Dynamics (N=60 rounds / 360 episodes)", fontsize=15, fontweight="bold", y=0.98)

    cond_labels = ["Raw (alpha=0.0)", "Light (alpha=0.3)", "Medium (alpha=0.5)"]
    palette = ["#f87171", "#34d399", "#60a5fa"]

    # 1. Jerk
    data_jerk = [[r["actuator_jerk"] for r in raw_ppo], [r["actuator_jerk"] for r in light_ppo], [r["actuator_jerk"] for r in med_ppo]]
    axes[0, 0].boxplot(data_jerk, labels=cond_labels, patch_artist=True, boxprops=dict(facecolor="#34d399", alpha=0.4))
    axes[0, 0].set_title("Actuator Jerk (Motor Chatter)", fontweight="bold")
    axes[0, 0].set_ylabel("L2 Delta Norm (lower is smoother)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Peak Mass
    data_mass = [[r["peak_mass"] for r in raw_ppo], [r["peak_mass"] for r in light_ppo], [r["peak_mass"] for r in med_ppo]]
    axes[0, 1].boxplot(data_mass, labels=cond_labels, patch_artist=True, boxprops=dict(facecolor="#60a5fa", alpha=0.4))
    axes[0, 1].set_title("Peak Cell Mass", fontweight="bold")
    axes[0, 1].set_ylabel("Mass (kg)")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Kills
    total_kills = [sum(r["kills"] for r in raw_ppo), sum(r["kills"] for r in light_ppo), sum(r["kills"] for r in med_ppo)]
    bars = axes[1, 0].bar(cond_labels, total_kills, color=palette, edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Cumulative Confirmed Predatory Kills", fontweight="bold")
    axes[1, 0].set_ylabel("Total Kills across Cohort")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar in bars:
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{int(bar.get_height())}", ha="center", fontweight="bold")

    # 4. Food Pellets Eaten
    data_food = [[r["food_eaten"] for r in raw_ppo], [r["food_eaten"] for r in light_ppo], [r["food_eaten"] for r in med_ppo]]
    axes[1, 1].boxplot(data_food, labels=cond_labels, patch_artist=True, boxprops=dict(facecolor="#fbbf24", alpha=0.4))
    axes[1, 1].set_title("Food Pellets Consumed", fontweight="bold")
    axes[1, 1].set_ylabel("Pellets Eaten")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(out_dirs[0], "lowpass_statistical_comparison.png")
    plt.savefig(chart_path, dpi=200)
    for d in out_dirs:
        plt.savefig(os.path.join(d, "lowpass_statistical_comparison.png"), dpi=200)
    plt.close()
    logger.info(f"Statistical visualization saved to: {chart_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds per condition (default 20)")
    args = parser.parse_args()
    run_full_statistical_study(num_rounds_per_condition=args.rounds)
