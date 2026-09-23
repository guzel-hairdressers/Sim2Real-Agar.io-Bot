"""
Competitive League Benchmark: PPO Champion vs Master Heuristic Bots vs Diffusion.
Runs a 20-round tournament and records:
1. Win Rate (% 1st place mass)
2. Average Final Mass & Peak Mass
3. Total Kills & Deaths (K/D ratio)
4. Suicidal Collision Rate (closing head-on into a larger predator)
Zero external dependencies (uses standard library csv).
Outputs: outputs/competitive_league_benchmark.csv
"""

import os
import sys
import time
import math
import csv
import logging
from typing import Dict, Any, List
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.heuristic_agent import MasterHeuristicAgarBot
from src.train_champion_ppo import ChampionPPOAgent
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LeagueBenchmark")


def run_benchmark(num_rounds: int = 20, episode_steps: int = 350):
    logger.info(f"Starting Competitive League Tournament ({num_rounds} rounds, {episode_steps} steps/round)...")

    ppo_weights = "outputs/agar_ppo_champion.pt"
    if not os.path.exists(ppo_weights):
        ppo_weights = "projects/sim2real-ppo-navigation/outputs/agar_ppo_champion.pt"

    diff_weights = "outputs/agar_diffusion_champion.pt"
    if not os.path.exists(diff_weights):
        diff_weights = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_champion.pt"

    critic_weights = "outputs/agar_diffusion_critic_champion.pt"
    if not os.path.exists(critic_weights):
        critic_weights = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_critic_champion.pt"

    env = PartiallyObservableAgarEnv(
        arena_size=14.0,
        num_players=6,
        num_food=100,
        num_viruses=5,
        max_steps=episode_steps,
        mass_decay_multiplier=1.35,
        max_pieces=4
    )

    ppo_agent = ChampionPPOAgent(weights_path=ppo_weights)
    diff_agent = AgarDiffusionPolicy(
        model_path=diff_weights if os.path.exists(diff_weights) else None,
        critic_path=critic_weights if os.path.exists(critic_weights) else None,
        action_horizon=16,
        exec_horizon=2,
        action_dim=3,
        obs_dim=38,
        num_ddim_steps=5,
        seed=777
    )

    competitor_registry = {
        "player_0": ("PPO-CHAMPION", ppo_agent),
        "player_1": ("HEURISTIC-APEX", MasterHeuristicAgarBot(pid="player_1", profile="apex", seed=101)),
        "player_2": ("HEURISTIC-HUNTER", MasterHeuristicAgarBot(pid="player_2", profile="hunter", seed=202)),
        "player_3": ("DIFFUSION-CHAMPION", diff_agent),
        "player_4": ("HEURISTIC-SURVIVOR", MasterHeuristicAgarBot(pid="player_4", profile="survivor", seed=303)),
        "player_5": ("PPO-PRODIGY", ChampionPPOAgent(weights_path=ppo_weights)),
    }

    names = {pid: info[0] for pid, info in competitor_registry.items()}

    # Stats tracking
    stats = {
        name: {
            "wins": 0,
            "final_masses": [],
            "peak_masses": [],
            "kills": 0,
            "deaths": 0,
            "suicidal_events": 0,
        }
        for name in names.values()
    }

    for r in range(1, num_rounds + 1):
        obs_dict, _ = env.reset(seed=5000 + r * 37)

        for step in range(episode_steps):
            actions = {}
            for pid, (c_name, agent) in competitor_registry.items():
                act = agent.predict(obs_dict[pid], deterministic=True)
                actions[pid] = act

                # Check suicidal advance: is agent thrusting directly towards a close, larger predator?
                obs = obs_dict[pid]
                t_fwd, t_dist = float(obs[22]), float(obs[24])
                thrust = float(act[0])
                if t_dist > 0.001 and t_dist < 0.35 and t_fwd > 0.20 and thrust > 0.6:
                    stats[c_name]["suicidal_events"] += 1

            obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # End of round evaluation
        round_masses = {names[pid]: infos[pid]["mass"] for pid in env.player_ids}
        round_winner = max(round_masses, key=round_masses.get)
        stats[round_winner]["wins"] += 1

        for pid in env.player_ids:
            c_name = names[pid]
            p_inf = infos[pid]
            stats[c_name]["final_masses"].append(p_inf["mass"])
            stats[c_name]["peak_masses"].append(p_inf["peak_mass"])
            stats[c_name]["kills"] += p_inf["kills"]
            stats[c_name]["deaths"] += p_inf["deaths"]

        if r % 5 == 0 or r == num_rounds:
            logger.info(f"Completed Round {r:02d}/{num_rounds:02d} | Leader: {round_winner} ({round_masses[round_winner]:.1f}kg)")

    # Aggregate tournament summary
    rows = []
    for name, s in stats.items():
        win_rate = (s["wins"] / float(num_rounds)) * 100.0
        avg_mass = float(np.mean(s["final_masses"]))
        avg_peak = float(np.mean(s["peak_masses"]))
        total_k = s["kills"]
        total_d = s["deaths"]
        kd_ratio = total_k / max(1, total_d)
        suicides_per_round = s["suicidal_events"] / float(num_rounds)

        rows.append({
            "Competitor": name,
            "Win Rate (%)": round(win_rate, 1),
            "Wins": s["wins"],
            "Avg Final Mass (kg)": round(avg_mass, 1),
            "Avg Peak Mass (kg)": round(avg_peak, 1),
            "Total Kills": total_k,
            "Total Deaths": total_d,
            "K/D Ratio": round(kd_ratio, 2),
            "Suicide Events / Rnd": round(suicides_per_round, 2)
        })

    # Sort by Win Rate descending, then Peak Mass descending
    rows.sort(key=lambda x: (x["Win Rate (%)"], x["Avg Peak Mass (kg)"]), reverse=True)

    header = f"{'Competitor':<20} | {'Win Rate':<10} | {'Wins':<6} | {'Avg Mass':<10} | {'Peak Mass':<10} | {'Kills':<7} | {'Deaths':<7} | {'K/D':<6} | {'Suicide/Rnd':<12}"
    print("\n" + "=" * 105)
    print("                      COMPETITIVE LEAGUE TOURNAMENT BENCHMARK RESULTS")
    print("=" * 105)
    print(header)
    print("-" * 105)
    for r in rows:
        print(f"{r['Competitor']:<20} | {str(r['Win Rate (%)'])+'%':<10} | {r['Wins']:<6d} | {str(r['Avg Final Mass (kg)'])+'kg':<10} | {str(r['Avg Peak Mass (kg)'])+'kg':<10} | {r['Total Kills']:<7d} | {r['Total Deaths']:<7d} | {r['K/D Ratio']:<6.2f} | {r['Suicide Events / Rnd']:<12.2f}")
    print("=" * 105 + "\n")

    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/competitive_league_benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    alt_csv = "projects/sim2real-ppo-navigation/outputs/competitive_league_benchmark.csv"
    if os.path.exists("projects/sim2real-ppo-navigation/outputs"):
        with open(alt_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    logger.info(f"Saved tournament benchmark results to {out_csv}")
    return rows


if __name__ == "__main__":
    run_benchmark(num_rounds=20, episode_steps=350)
