"""
Multi-Agent Agar.io Tournament Benchmark: PPO vs. Diffusion Policy vs. Competitor Bots.
Evaluates:
1. Predation Kills & Death Ratio
2. Peak Mass Attained
3. Nutrient Foraging Rate (food pellets consumed)
4. Actuator Jerk / Motor Chatter (L2 delta norm)
5. Real-Time Inference Latency (ms)
Exports quantitative results to outputs/agar_tournament_results.csv.
"""

import os
import sys
import time
import csv
import logging
from typing import Dict, Any, List
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_agar_parallel_gpu import TorchAgarPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarTournament")


def compute_action_jerk(actions: List[np.ndarray]) -> float:
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


def run_tournament(num_rounds: int = 15, episode_steps: int = 250):
    logger.info("Initializing Multi-Agent Agar.io Competitive Tournament...")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    # 1. Pure Neural Competitor Policies: 3 PPO vs. 3 Diffusion
    ppo_weights = "outputs/agar_ppo_gpu.pt"
    if not os.path.exists(ppo_weights):
        ppo_weights = "projects/sim2real-ppo-navigation/outputs/agar_ppo_gpu.pt"

    ppo_agent_learner = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)
    ppo_agent_clone1 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)
    ppo_agent_clone2 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)

    diff_agent_1 = AgarDiffusionPolicy(action_horizon=16, exec_horizon=2, action_dim=3, obs_dim=38, num_ddim_steps=6, seed=42)
    diff_agent_2 = AgarDiffusionPolicy(action_horizon=16, exec_horizon=2, action_dim=3, obs_dim=38, num_ddim_steps=6, seed=105)
    diff_agent_3 = AgarDiffusionPolicy(action_horizon=16, exec_horizon=2, action_dim=3, obs_dim=38, num_ddim_steps=6, seed=202)

    competitors = {
        "player_0": ("PPO (Active Learner)", ppo_agent_learner),
        "player_1": ("Diffusion Policy (Agent 1)", diff_agent_1),
        "player_2": ("PPO (Clone 1)", ppo_agent_clone1),
        "player_3": ("Diffusion Policy (Agent 2)", diff_agent_2),
        "player_4": ("PPO (Clone 2)", ppo_agent_clone2),
        "player_5": ("Diffusion Policy (Agent 3)", diff_agent_3),
    }

    policy_stats = {name: {
        "kills": 0,
        "deaths": 0,
        "food": 0,
        "peak_mass": [],
        "jerk": [],
        "inference_ms": []
    } for name, _ in competitors.values()}

    env = PartiallyObservableAgarEnv(arena_size=14.0, num_players=6, num_food=120, max_steps=episode_steps)

    for round_idx in range(num_rounds):
        obs_dict, _ = env.reset(seed=200 + round_idx)
        for _, pol in competitors.values():
            if hasattr(pol, "reset"):
                pol.reset()

        round_actions = {name: [] for name, _ in competitors.values()}

        for step in range(episode_steps):
            actions = {}
            for pid, (pol_name, policy) in competitors.items():
                t0 = time.perf_counter()
                act = policy.predict(obs_dict[pid], deterministic=True)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                actions[pid] = act
                round_actions[pol_name].append(act.copy())
                policy_stats[pol_name]["inference_ms"].append(dt_ms)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # Collect round stats
        for pid, (pol_name, _) in competitors.items():
            p_state = env.players[pid]
            policy_stats[pol_name]["kills"] += p_state.kills
            policy_stats[pol_name]["deaths"] += p_state.deaths
            policy_stats[pol_name]["food"] += p_state.food_eaten
            policy_stats[pol_name]["peak_mass"].append(p_state.peak_mass)
            policy_stats[pol_name]["jerk"].append(compute_action_jerk(round_actions[pol_name]))

    # Compile Summary Table
    rows = []
    print("\n" + "=" * 110)
    print(f"{'Competitor Paradigm':<28} | {'Kills':<6} | {'Deaths':<6} | {'K/D':<6} | {'Peak Mass':<10} | {'Jerk':<8} | {'Latency':<8}")
    print("=" * 110)

    ppo_kills, ppo_deaths, ppo_mass, ppo_food, ppo_jerk, ppo_lat = [], [], [], [], [], []
    diff_kills, diff_deaths, diff_mass, diff_food, diff_jerk, diff_lat = [], [], [], [], [], []

    for pol_name in sorted(policy_stats.keys()):
        stats = policy_stats[pol_name]
        kills = stats["kills"]
        deaths = max(1, stats["deaths"])
        kd_ratio = kills / deaths
        mean_peak = float(np.mean(stats["peak_mass"]))
        mean_jerk = float(np.mean(stats["jerk"]))
        mean_lat = float(np.mean(stats["inference_ms"]))
        mean_food = stats["food"] / num_rounds

        if "PPO" in pol_name:
            ppo_kills.append(kills)
            ppo_deaths.append(stats["deaths"])
            ppo_mass.append(mean_peak)
            ppo_food.append(mean_food)
            ppo_jerk.append(mean_jerk)
            ppo_lat.append(mean_lat)
        else:
            diff_kills.append(kills)
            diff_deaths.append(stats["deaths"])
            diff_mass.append(mean_peak)
            diff_food.append(mean_food)
            diff_jerk.append(mean_jerk)
            diff_lat.append(mean_lat)

        rows.append({
            "policy": pol_name,
            "total_kills": kills,
            "total_deaths": stats["deaths"],
            "kd_ratio": round(kd_ratio, 2),
            "mean_peak_mass": round(mean_peak, 1),
            "mean_food_eaten": round(mean_food, 1),
            "actuator_jerk": round(mean_jerk, 4),
            "mean_latency_ms": round(mean_lat, 3)
        })

        print(f"{pol_name:<28} | {kills:<6d} | {stats['deaths']:<6d} | {kd_ratio:<6.2f} | {mean_peak:<7.1f} kg | {mean_jerk:<8.4f} | {mean_lat:<6.3f}ms")

    print("-" * 110)
    print(
        f"{'PPO Collective Average':<28} | {sum(ppo_kills):<6d} | {sum(ppo_deaths):<6d} | "
        f"{sum(ppo_kills) / max(1, sum(ppo_deaths)):<6.2f} | {np.mean(ppo_mass):<7.1f} kg | "
        f"{np.mean(ppo_jerk):<8.4f} | {np.mean(ppo_lat):<6.3f}ms"
    )
    print(
        f"{'Diffusion Collective Average':<28} | {sum(diff_kills):<6d} | {sum(diff_deaths):<6d} | "
        f"{sum(diff_kills) / max(1, sum(diff_deaths)):<6.2f} | {np.mean(diff_mass):<7.1f} kg | "
        f"{np.mean(diff_jerk):<8.4f} | {np.mean(diff_lat):<6.3f}ms"
    )
    print("=" * 110 + "\n")

    # Export to CSV
    keys = list(rows[0].keys())
    for out_path in ["outputs/agar_tournament_results.csv", "projects/sim2real-ppo-navigation/outputs/agar_tournament_results.csv"]:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Tournament results exported to outputs/agar_tournament_results.csv")


if __name__ == "__main__":
    run_tournament(num_rounds=12, episode_steps=200)
