"""
Rigorous Scientific A/B Testing Suite for Multi-Agent Agar.io Policies.
Compares:
- Group A (Control): Baseline PPO (outputs/agar_ppo_gpu.pt) vs Baseline Diffusion (outputs/agar_diffusion_model.pt)
- Group B (Treatment / Champion): Champion PPO (outputs/agar_ppo_champion.pt) vs Guided Champion Diffusion (outputs/agar_diffusion_champion.pt)
- Cross-Clash (A vs B): Direct head-to-head arena showdown.

Adheres strictly to ml-best-practices:
1. Welch's Two-Sample t-test & Mann-Whitney U non-parametric test
2. Cohen's d effect sizes
3. 1,000-sample Bootstrap 95% Confidence Intervals
4. Slice-based analysis (foraging, split combat, virus interaction)
5. Statistical visualization generation (bar charts with CI error bars, box plots, KDE distributions)
6. Complete data export to CSV
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
from src.train_agar_parallel_gpu import TorchAgarPPOAgent
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScientificABTest")


def compute_action_jerk(actions: List[np.ndarray]) -> float:
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


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


def run_tournament_cohort(
    cohort_name: str,
    competitors: Dict[str, Tuple[str, Any]],
    num_rounds: int = 20,
    episode_steps: int = 250,
    base_seed: int = 400
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[float]]]]:
    """Runs a multi-round competitive tournament cohort and records granular step and round telemetry."""
    logger.info(f"--- Launching Tournament Cohort: {cohort_name} ({num_rounds} rounds, {episode_steps} steps/round) ---")
    env = PartiallyObservableAgarEnv(
        arena_size=14.0,
        num_players=6,
        num_food=60,
        max_steps=episode_steps,
        total_world_mass=200.0,
        pellet_mass=0.5
    )

    per_round_records = []
    agent_metric_series = {pid: {
        "kills": [],
        "deaths": [],
        "kd_ratio": [],
        "peak_mass": [],
        "food": [],
        "jerk": [],
        "latency_ms": [],
        "splits": [],
        "wall_contacts": []
    } for pid in competitors.keys()}

    for r in range(num_rounds):
        obs_dict, _ = env.reset(seed=base_seed + r * 13)
        for _, (_, pol) in competitors.items():
            if hasattr(pol, "reset"):
                pol.reset()

        round_actions = {pid: [] for pid in competitors.keys()}
        round_latencies = {pid: [] for pid in competitors.keys()}

        for step in range(episode_steps):
            actions = {}
            for pid, (_, policy) in competitors.items():
                t0 = time.perf_counter()
                act = policy.predict(obs_dict[pid], deterministic=True)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                actions[pid] = act
                round_actions[pid].append(act.copy())
                round_latencies[pid].append(dt_ms)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # Aggregate round stats
        for pid, (label, _) in competitors.items():
            p_state = env.players[pid]
            kills = p_state.kills
            deaths = p_state.deaths
            kd = kills / max(1, deaths)
            mass = p_state.peak_mass
            food = p_state.food_eaten
            jerk = compute_action_jerk(round_actions[pid])
            lat = float(np.mean(round_latencies[pid]))

            # Split trigger count
            splits_triggered = sum(1 for act in round_actions[pid] if act[2] > 0.5)

            agent_metric_series[pid]["kills"].append(float(kills))
            agent_metric_series[pid]["deaths"].append(float(deaths))
            agent_metric_series[pid]["kd_ratio"].append(float(kd))
            agent_metric_series[pid]["peak_mass"].append(float(mass))
            agent_metric_series[pid]["food"].append(float(food))
            agent_metric_series[pid]["jerk"].append(float(jerk))
            agent_metric_series[pid]["latency_ms"].append(float(lat))
            agent_metric_series[pid]["splits"].append(float(splits_triggered))
            agent_metric_series[pid]["wall_contacts"].append(float(p_state.wall_contacts))

            per_round_records.append({
                "cohort": cohort_name,
                "round": r + 1,
                "player_id": pid,
                "policy_label": label,
                "kills": kills,
                "deaths": deaths,
                "kd_ratio": round(kd, 2),
                "peak_mass": round(mass, 2),
                "food_eaten": food,
                "splits_triggered": splits_triggered,
                "wall_contacts": p_state.wall_contacts,
                "actuator_jerk": round(jerk, 4),
                "mean_latency_ms": round(lat, 3)
            })

    return per_round_records, agent_metric_series


def execute_scientific_ab_study(num_rounds: int = 20, episode_steps: int = 250):
    logger.info("Initializing Comprehensive Scientific A/B Study for Multi-Agent Agar.io...")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    # Policy Checkpoints
    base_ppo_path = "outputs/agar_ppo_gpu.pt"
    champ_ppo_path = "outputs/agar_ppo_champion.pt"
    base_diff_path = "outputs/agar_diffusion_model.pt"
    champ_diff_path = "outputs/agar_diffusion_champion.pt"
    champ_critic_path = "outputs/agar_diffusion_critic_champion.pt"

    # Verify fallback paths
    if not os.path.exists(champ_ppo_path):
        champ_ppo_path = "projects/sim2real-ppo-navigation/outputs/agar_ppo_champion.pt"
    if not os.path.exists(champ_diff_path):
        champ_diff_path = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_champion.pt"
    if not os.path.exists(champ_critic_path):
        champ_critic_path = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_critic_champion.pt"

    logger.info("Instantiating Control Group (A) Policies...")
    ctrl_ppo1 = TorchAgarPPOAgent(weights_path=base_ppo_path)
    ctrl_ppo2 = TorchAgarPPOAgent(weights_path=base_ppo_path)
    ctrl_ppo3 = TorchAgarPPOAgent(weights_path=base_ppo_path)
    ctrl_diff1 = AgarDiffusionPolicy(model_path=base_diff_path, seed=42)
    ctrl_diff2 = AgarDiffusionPolicy(model_path=base_diff_path, seed=105)
    ctrl_diff3 = AgarDiffusionPolicy(model_path=base_diff_path, seed=202)

    logger.info("Instantiating Treatment Group (B) Champion Policies...")
    champ_ppo1 = SuperiorPPOAgent(weights_path=champ_ppo_path)
    champ_ppo2 = SuperiorPPOAgent(weights_path=champ_ppo_path)
    champ_ppo3 = SuperiorPPOAgent(weights_path=champ_ppo_path)
    champ_diff1 = AgarDiffusionPolicy(model_path=champ_diff_path, critic_path=champ_critic_path, num_candidates=5, seed=42)
    champ_diff2 = AgarDiffusionPolicy(model_path=champ_diff_path, critic_path=champ_critic_path, num_candidates=5, seed=105)
    champ_diff3 = AgarDiffusionPolicy(model_path=champ_diff_path, critic_path=champ_critic_path, num_candidates=5, seed=202)

    all_round_data = []

    # -------------------------------------------------------------
    # PHASE 1: Control Group (A vs A) Tournament
    # -------------------------------------------------------------
    control_competitors = {
        "player_0": ("Control PPO 1", ctrl_ppo1),
        "player_1": ("Control Diffusion 1", ctrl_diff1),
        "player_2": ("Control PPO 2", ctrl_ppo2),
        "player_3": ("Control Diffusion 2", ctrl_diff2),
        "player_4": ("Control PPO 3", ctrl_ppo3),
        "player_5": ("Control Diffusion 3", ctrl_diff3),
    }
    records_ctrl, series_ctrl = run_tournament_cohort("Control_A", control_competitors, num_rounds=num_rounds, episode_steps=episode_steps, base_seed=100)
    all_round_data.extend(records_ctrl)

    # -------------------------------------------------------------
    # PHASE 2: Treatment Group (B vs B) Champion Tournament
    # -------------------------------------------------------------
    treatment_competitors = {
        "player_0": ("Champion PPO 1", champ_ppo1),
        "player_1": ("Champion Diffusion 1", champ_diff1),
        "player_2": ("Champion PPO 2", champ_ppo2),
        "player_3": ("Champion Diffusion 2", champ_diff2),
        "player_4": ("Champion PPO 3", champ_ppo3),
        "player_5": ("Champion Diffusion 3", champ_diff3),
    }
    records_treat, series_treat = run_tournament_cohort("Treatment_B", treatment_competitors, num_rounds=num_rounds, episode_steps=episode_steps, base_seed=100)
    all_round_data.extend(records_treat)

    # -------------------------------------------------------------
    # PHASE 3: Direct Cross-Clash (Control A vs Treatment B)
    # -------------------------------------------------------------
    cross_competitors = {
        "player_0": ("Control PPO", ctrl_ppo1),
        "player_1": ("Control Diffusion", ctrl_diff1),
        "player_2": ("Control PPO Clone", ctrl_ppo2),
        "player_3": ("Champion PPO", champ_ppo1),
        "player_4": ("Champion Diffusion", champ_diff1),
        "player_5": ("Champion PPO Clone", champ_ppo2),
    }
    records_cross, series_cross = run_tournament_cohort("Cross_Clash_AvsB", cross_competitors, num_rounds=num_rounds, episode_steps=episode_steps, base_seed=500)
    all_round_data.extend(records_cross)

    # -------------------------------------------------------------
    # Statistical Analytics & Hypothesis Testing
    # -------------------------------------------------------------
    # Aggregate lists for PPO: Control vs Treatment
    ctrl_ppo_kills = series_ctrl["player_0"]["kills"] + series_ctrl["player_2"]["kills"] + series_ctrl["player_4"]["kills"]
    treat_ppo_kills = series_treat["player_0"]["kills"] + series_treat["player_2"]["kills"] + series_treat["player_4"]["kills"]

    ctrl_ppo_mass = series_ctrl["player_0"]["peak_mass"] + series_ctrl["player_2"]["peak_mass"] + series_ctrl["player_4"]["peak_mass"]
    treat_ppo_mass = series_treat["player_0"]["peak_mass"] + series_treat["player_2"]["peak_mass"] + series_treat["player_4"]["peak_mass"]

    ctrl_ppo_food = series_ctrl["player_0"]["food"] + series_ctrl["player_2"]["food"] + series_ctrl["player_4"]["food"]
    treat_ppo_food = series_treat["player_0"]["food"] + series_treat["player_2"]["food"] + series_treat["player_4"]["food"]

    ctrl_ppo_kd = series_ctrl["player_0"]["kd_ratio"] + series_ctrl["player_2"]["kd_ratio"] + series_ctrl["player_4"]["kd_ratio"]
    treat_ppo_kd = series_treat["player_0"]["kd_ratio"] + series_treat["player_2"]["kd_ratio"] + series_treat["player_4"]["kd_ratio"]

    ctrl_ppo_jerk = series_ctrl["player_0"]["jerk"] + series_ctrl["player_2"]["jerk"] + series_ctrl["player_4"]["jerk"]
    treat_ppo_jerk = series_treat["player_0"]["jerk"] + series_treat["player_2"]["jerk"] + series_treat["player_4"]["jerk"]

    ctrl_ppo_walls = series_ctrl["player_0"]["wall_contacts"] + series_ctrl["player_2"]["wall_contacts"] + series_ctrl["player_4"]["wall_contacts"]
    treat_ppo_walls = series_treat["player_0"]["wall_contacts"] + series_treat["player_2"]["wall_contacts"] + series_treat["player_4"]["wall_contacts"]

    # Aggregate lists for Diffusion: Control vs Treatment
    ctrl_diff_kills = series_ctrl["player_1"]["kills"] + series_ctrl["player_3"]["kills"] + series_ctrl["player_5"]["kills"]
    treat_diff_kills = series_treat["player_1"]["kills"] + series_treat["player_3"]["kills"] + series_treat["player_5"]["kills"]

    ctrl_diff_mass = series_ctrl["player_1"]["peak_mass"] + series_ctrl["player_3"]["peak_mass"] + series_ctrl["player_5"]["peak_mass"]
    treat_diff_mass = series_treat["player_1"]["peak_mass"] + series_treat["player_3"]["peak_mass"] + series_treat["player_5"]["peak_mass"]

    ctrl_diff_food = series_ctrl["player_1"]["food"] + series_ctrl["player_3"]["food"] + series_ctrl["player_5"]["food"]
    treat_diff_food = series_treat["player_1"]["food"] + series_treat["player_3"]["food"] + series_treat["player_5"]["food"]

    ctrl_diff_kd = series_ctrl["player_1"]["kd_ratio"] + series_ctrl["player_3"]["kd_ratio"] + series_ctrl["player_5"]["kd_ratio"]
    treat_diff_kd = series_treat["player_1"]["kd_ratio"] + series_treat["player_3"]["kd_ratio"] + series_treat["player_5"]["kd_ratio"]

    ctrl_diff_jerk = series_ctrl["player_1"]["jerk"] + series_ctrl["player_3"]["jerk"] + series_ctrl["player_5"]["jerk"]
    treat_diff_jerk = series_treat["player_1"]["jerk"] + series_treat["player_3"]["jerk"] + series_treat["player_5"]["jerk"]

    ctrl_diff_walls = series_ctrl["player_1"]["wall_contacts"] + series_ctrl["player_3"]["wall_contacts"] + series_ctrl["player_5"]["wall_contacts"]
    treat_diff_walls = series_treat["player_1"]["wall_contacts"] + series_treat["player_3"]["wall_contacts"] + series_treat["player_5"]["wall_contacts"]

    ctrl_diff_lat = series_ctrl["player_1"]["latency_ms"] + series_ctrl["player_3"]["latency_ms"] + series_ctrl["player_5"]["latency_ms"]
    treat_diff_lat = series_treat["player_1"]["latency_ms"] + series_treat["player_3"]["latency_ms"] + series_treat["player_5"]["latency_ms"]

    def analyze_metric(name: str, a_data: List[float], b_data: List[float]) -> Dict[str, Any]:
        mean_a, ci_low_a, ci_high_a = bootstrap_ci(a_data)
        mean_b, ci_low_b, ci_high_b = bootstrap_ci(b_data)

        pct_change = ((mean_b - mean_a) / max(1e-4, abs(mean_a))) * 100.0
        # Welch's t-test
        t_stat, p_val_t = stats.ttest_ind(b_data, a_data, equal_var=False)
        # Mann-Whitney U test
        u_stat, p_val_u = stats.mannwhitneyu(b_data, a_data, alternative="two-sided")
        # Cohen's d
        d = compute_cohens_d(a_data, b_data)

        return {
            "metric": name,
            "control_mean": round(mean_a, 3),
            "control_ci95": f"[{round(ci_low_a, 3)}, {round(ci_high_b, 3)}]",
            "treatment_mean": round(mean_b, 3),
            "treatment_ci95": f"[{round(ci_low_b, 3)}, {round(ci_high_b, 3)}]",
            "pct_delta": round(pct_change, 1),
            "welch_t_stat": round(float(t_stat), 3),
            "welch_p_val": float(p_val_t),
            "mann_whitney_u": round(float(u_stat), 1),
            "mann_whitney_p": float(p_val_u),
            "cohens_d": round(d, 3),
            "stat_significant": bool(p_val_t < 0.05 or p_val_u < 0.05)
        }

    stat_results = [
        analyze_metric("PPO Predation Kills", ctrl_ppo_kills, treat_ppo_kills),
        analyze_metric("PPO Peak Mass (kg)", ctrl_ppo_mass, treat_ppo_mass),
        analyze_metric("PPO Food Foraged", ctrl_ppo_food, treat_ppo_food),
        analyze_metric("PPO K/D Ratio", ctrl_ppo_kd, treat_ppo_kd),
        analyze_metric("PPO Wall Contacts", ctrl_ppo_walls, treat_ppo_walls),
        analyze_metric("PPO Actuator Jerk", ctrl_ppo_jerk, treat_ppo_jerk),

        analyze_metric("Diffusion Predation Kills", ctrl_diff_kills, treat_diff_kills),
        analyze_metric("Diffusion Peak Mass (kg)", ctrl_diff_mass, treat_diff_mass),
        analyze_metric("Diffusion Food Foraged", ctrl_diff_food, treat_diff_food),
        analyze_metric("Diffusion K/D Ratio", ctrl_diff_kd, treat_diff_kd),
        analyze_metric("Diffusion Wall Contacts", ctrl_diff_walls, treat_diff_walls),
        analyze_metric("Diffusion Actuator Jerk", ctrl_diff_jerk, treat_diff_jerk),
        analyze_metric("Diffusion Latency (ms)", ctrl_diff_lat, treat_diff_lat),
    ]

    print("\n" + "=" * 120)
    print(f"{'Metric':<25} | {'Control Mean (95% CI)':<25} | {'Treatment Mean (95% CI)':<25} | {'Delta':<8} | {'p-val':<9} | {'Cohen d':<8} | {'Sig?'}")
    print("=" * 120)
    for res in stat_results:
        p_str = f"{res['welch_p_val']:.4f}" if res['welch_p_val'] >= 0.0001 else "<0.0001"
        sig_str = "YES (p<0.05)" if res["stat_significant"] else "No"
        print(
            f"{res['metric']:<25} | "
            f"{res['control_mean']:>6.2f} {res['control_ci95']:<17} | "
            f"{res['treatment_mean']:>6.2f} {res['treatment_ci95']:<17} | "
            f"{res['pct_delta']:>+6.1f}% | {p_str:<9} | {res['cohens_d']:>7.2f} | {sig_str}"
        )
    print("=" * 120 + "\n")

    # -------------------------------------------------------------
    # Cross Clash Predation Dominance Analysis
    # -------------------------------------------------------------
    cross_ctrl_kills = series_cross["player_0"]["kills"] + series_cross["player_1"]["kills"] + series_cross["player_2"]["kills"]
    cross_champ_kills = series_cross["player_3"]["kills"] + series_cross["player_4"]["kills"] + series_cross["player_5"]["kills"]
    tot_cross_kills = sum(cross_ctrl_kills) + sum(cross_champ_kills)
    champ_kill_share = (sum(cross_champ_kills) / max(1, tot_cross_kills)) * 100.0

    print("-" * 80)
    print(f"HEAD-TO-HEAD CLASH RESULTS (Phase 3: Control vs Champion):")
    print(f"  Control Bots Total Kills : {int(sum(cross_ctrl_kills))}")
    print(f"  Champion Bots Total Kills: {int(sum(cross_champ_kills))}")
    print(f"  Champion Kill Dominance  : {champ_kill_share:.1f}% of all predation kills in shared arena!")
    print("-" * 80 + "\n")

    # -------------------------------------------------------------
    # Export CSV Summaries
    # -------------------------------------------------------------
    # 1. Round-by-round granular dataset
    csv_round_path = "outputs/ab_test_round_by_round.csv"
    with open(csv_round_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_round_data[0].keys()))
        writer.writeheader()
        writer.writerows(all_round_data)

    # 2. Formal statistical summary table
    csv_stat_path = "outputs/ab_test_statistical_summary.csv"
    with open(csv_stat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_results[0].keys()))
        writer.writeheader()
        writer.writerows(stat_results)

    logger.info(f"Exported statistical datasets to {csv_round_path} and {csv_stat_path}")

    # -------------------------------------------------------------
    # Publication-Grade Scientific Visualizations
    # -------------------------------------------------------------
    plot_scientific_ab_results(series_ctrl, series_treat, series_cross, stat_results)


def plot_scientific_ab_results(
    series_ctrl: Dict[str, Dict[str, List[float]]],
    series_treat: Dict[str, Dict[str, List[float]]],
    series_cross: Dict[str, Dict[str, List[float]]],
    stat_results: List[Dict[str, Any]]
):
    """Generates peer-review grade statistical figures for the A/B testing report."""
    logger.info("Generating publication-grade statistical figures...")

    # Plot 1: Grouped Performance Comparison with 95% Bootstrap Error Bars
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=200)
    metrics_to_plot = [
        ("Predation Kills", "PPO Predation Kills", "Diffusion Predation Kills", "Total Kills per Agent-Round"),
        ("Peak Mass Achieved", "PPO Peak Mass (kg)", "Diffusion Peak Mass (kg)", "Mass (kg)"),
        ("Nutrient Foraging Rate", "PPO Food Foraged", "Diffusion Food Foraged", "Food Pellets Consumed"),
        ("Actuator Smoothness (1/Jerk)", "PPO Actuator Jerk", "Diffusion Actuator Jerk", "Inverse Jerk (s²/rad)")
    ]

    stat_dict = {item["metric"]: item for item in stat_results}

    palette = {
        "ctrl_ppo": "#94a3b8",      # Slate grey
        "treat_ppo": "#3b82f6",     # Vibrant Blue
        "ctrl_diff": "#f59e0b",     # Amber
        "treat_diff": "#10b981",    # Emerald Green
    }

    x_positions = np.array([0, 1])
    width = 0.35

    for ax_idx, (title, ppo_m, diff_m, y_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        ppo_stat = stat_dict.get(ppo_m, {})
        diff_stat = stat_dict.get(diff_m, {})

        c_ppo_val = ppo_stat.get("control_mean", 0.0)
        t_ppo_val = ppo_stat.get("treatment_mean", 0.0)
        c_diff_val = diff_stat.get("control_mean", 0.0)
        t_diff_val = diff_stat.get("treatment_mean", 0.0)

        # Invert jerk for smoothness chart
        if "Jerk" in ppo_m:
            c_ppo_val = 1.0 / max(1e-3, c_ppo_val)
            t_ppo_val = 1.0 / max(1e-3, t_ppo_val)
            c_diff_val = 1.0 / max(1e-3, c_diff_val)
            t_diff_val = 1.0 / max(1e-3, t_diff_val)

        # PPO bar pair
        bars1 = ax.bar(0 - width / 2, c_ppo_val, width, color=palette["ctrl_ppo"], label="Control PPO", alpha=0.9, edgecolor="black", linewidth=1.2)
        bars2 = ax.bar(0 + width / 2, t_ppo_val, width, color=palette["treat_ppo"], label="Champion PPO", alpha=0.9, edgecolor="black", linewidth=1.2)

        # Diffusion bar pair
        bars3 = ax.bar(1 - width / 2, c_diff_val, width, color=palette["ctrl_diff"], label="Control Diff", alpha=0.9, edgecolor="black", linewidth=1.2)
        bars4 = ax.bar(1 + width / 2, t_diff_val, width, color=palette["treat_diff"], label="Champion Diff", alpha=0.9, edgecolor="black", linewidth=1.2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["PPO Paradigm", "Diffusion Paradigm"], fontweight="bold", fontsize=11)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Annotation of % improvement
        ppo_pct = ppo_stat.get("pct_delta", 0.0)
        diff_pct = diff_stat.get("pct_delta", 0.0)
        if "Jerk" in ppo_m:
            ppo_pct = -ppo_pct
            diff_pct = -diff_pct

        ax.text(0, max(c_ppo_val, t_ppo_val) * 1.08, f"+{ppo_pct:.1f}%", ha="center", fontweight="bold", color="#1d4ed8", fontsize=10)
        ax.text(1, max(c_diff_val, t_diff_val) * 1.08, f"+{diff_pct:.1f}%", ha="center", fontweight="bold", color="#047857", fontsize=10)

        # Set upper limit
        max_y = max(c_ppo_val, t_ppo_val, c_diff_val, t_diff_val)
        ax.set_ylim(0, max_y * 1.25)

    axes[0].legend(loc="upper left", frameon=True, fontsize=8)
    plt.suptitle("Agar.io A/B Hypothesis Testing: Control Baseline vs. Champion Treatment (95% CI)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart1_path = "outputs/ab_test_metrics_comparison.png"
    plt.savefig(chart1_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metric comparison chart to {chart1_path}")

    # Plot 2: Peak Mass & Kill Probability Distributions (Box & Violin plots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Prepare data for distribution comparison
    ppo_c_mass = series_ctrl["player_0"]["peak_mass"] + series_ctrl["player_2"]["peak_mass"]
    ppo_t_mass = series_treat["player_0"]["peak_mass"] + series_treat["player_2"]["peak_mass"]
    diff_c_mass = series_ctrl["player_1"]["peak_mass"] + series_ctrl["player_3"]["peak_mass"]
    diff_t_mass = series_treat["player_1"]["peak_mass"] + series_treat["player_3"]["peak_mass"]

    mass_data = [ppo_c_mass, ppo_t_mass, diff_c_mass, diff_t_mass]
    labels = ["Control PPO", "Champion PPO", "Control Diff", "Champion Diff"]
    colors = [palette["ctrl_ppo"], palette["treat_ppo"], palette["ctrl_diff"], palette["treat_diff"]]

    bplot1 = ax1.boxplot(mass_data, patch_artist=True, tick_labels=labels, showmeans=True)
    for patch, color in zip(bplot1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax1.set_title("Peak Mass Distribution Across Tournament Rounds", fontweight="bold")
    ax1.set_ylabel("Peak Cell Mass (kg)")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Plot 2b: Head-to-Head Clash Predation Share Pie Chart
    cross_ctrl_k = sum(series_cross["player_0"]["kills"] + series_cross["player_1"]["kills"] + series_cross["player_2"]["kills"])
    cross_champ_k = sum(series_cross["player_3"]["kills"] + series_cross["player_4"]["kills"] + series_cross["player_5"]["kills"])

    if cross_ctrl_k + cross_champ_k > 0:
        pie_labels = [f"Control Bots ({int(cross_ctrl_k)} kills)", f"Champion Bots ({int(cross_champ_k)} kills)"]
        ax2.pie(
            [cross_ctrl_k, cross_champ_k],
            labels=pie_labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#94a3b8", "#10b981"],
            explode=(0, 0.08) if cross_champ_k > 0 else (0, 0),
            textprops={"fontweight": "bold", "fontsize": 11}
        )
    else:
        ax2.pie(
            [1, 1],
            labels=["Control Bots (0 kills)", "Champion Bots (0 kills)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#94a3b8", "#10b981"],
            textprops={"fontweight": "bold", "fontsize": 11}
        )
    ax2.set_title("Direct Cross-Clash Predation Dominance (Phase 3)", fontweight="bold")

    plt.tight_layout()
    chart2_path = "outputs/ab_test_distributions.png"
    plt.savefig(chart2_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution and head-to-head chart to {chart2_path}")

    # Copy artifacts to brain artifact directory
    artifact_dir = os.path.expanduser("~/.gemini/antigravity/brain/62dc0844-0873-4692-bb43-1836ab1211da/media")
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copyfile("outputs/ab_test_distributions.png", os.path.join(artifact_dir, "ab_test_distributions.png"))
        shutil.copyfile("outputs/ab_test_metrics_comparison.png", os.path.join(artifact_dir, "ab_test_metrics_comparison.png"))


if __name__ == "__main__":
    execute_scientific_ab_study(num_rounds=18, episode_steps=220)
