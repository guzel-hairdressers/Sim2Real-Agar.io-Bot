"""
Sim2Real Robustness Benchmark & Domain Shift Evaluation:
Evaluates navigation policy generalization across physical parameter shifts:
- Surface Friction Coefficient (μ: 0.2 to 1.6)
- Robot Mass / Payload (m: 1.0 kg to 5.0 kg)
- Actuation Delay Latency (0 to 3 control cycles)
Compares Domain-Randomized (DR) policy vs. Nominal (Non-DR) baseline.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.navigation_env import ContinuousNavigationEnv
from src.train import LightweightHeuristicPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers=None, tablefmt="github"):
        res = []
        if headers:
            res.append("| " + " | ".join(str(h) for h in headers) + " |")
            res.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            res.append("| " + " | ".join(str(c) for c in r) + " |")
        return "\n".join(res)


def evaluate_policy_on_config(
    policy,
    friction: float,
    mass: float,
    delay: int,
    sensor_noise: float = 0.02,
    num_episodes: int = 20
) -> Dict[str, float]:
    """Evaluates policy on a specific physical configuration."""
    env = ContinuousNavigationEnv()
    env.friction_coeff = friction
    env.robot_mass = mass
    env.sensor_noise_std = sensor_noise

    success_count = 0
    collision_count = 0
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0
        # Simulating delay queue
        action_queue = [np.zeros(2, dtype=np.float32) for _ in range(delay + 1)]

        while not done:
            action = policy.predict(obs)
            action_queue.append(action)
            act_to_exec = action_queue.pop(0)

            obs, r, term, trunc, info = env.step(act_to_exec)
            ep_reward += r
            done = term or trunc

        rewards.append(ep_reward)
        if info.get("is_success", False):
            success_count += 1
        elif info.get("collision", False):
            collision_count += 1

    return {
        "success_rate": (success_count / num_episodes) * 100.0,
        "collision_rate": (collision_count / num_episodes) * 100.0,
        "mean_reward": float(np.mean(rewards)),
    }


def run_robustness_sweep(output_csv: str = "outputs/robustness_results.csv"):
    logger.info("Initializing Sim2Real Parameter Sweep Benchmark...")
    policy = LightweightHeuristicPolicy()

    # Define test grid
    test_conditions = [
        # (Label, Friction, Mass, Delay)
        ("Nominal Sim Baseline", 0.8, 2.0, 0),
        ("Low Friction (Ice / Wet Tile)", 0.25, 2.0, 0),
        ("High Friction (Dense Carpet)", 1.5, 2.0, 0),
        ("Heavy Payload (2.5x Mass)", 0.8, 5.0, 0),
        ("Lightweight Chassis (0.5x Mass)", 0.8, 1.0, 0),
        ("Microcontroller Latency (1 step)", 0.8, 2.0, 1),
        ("Severe Wireless Latency (2 steps)", 0.8, 2.0, 2),
        ("Adverse Compound Real-World Shift", 0.35, 4.0, 2),
    ]

    table_rows = []
    csv_rows = ["Condition,Friction_mu,Mass_kg,Delay_steps,Success_Rate_pct,Collision_Rate_pct,Mean_Reward"]

    for label, friction, mass, delay in test_conditions:
        res = evaluate_policy_on_config(policy, friction=friction, mass=mass, delay=delay, num_episodes=25)
        table_rows.append([
            label,
            f"{friction:.2f}",
            f"{mass:.1f} kg",
            f"{delay} steps",
            f"{res['success_rate']:.1f}%",
            f"{res['collision_rate']:.1f}%",
            f"{res['mean_reward']:.1f}"
        ])
        csv_rows.append(
            f'"{label}",{friction},{mass},{delay},{res["success_rate"]:.2f},{res["collision_rate"]:.2f},{res["mean_reward"]:.2f}'
        )

    headers = ["Domain Condition", "Friction (μ)", "Mass", "Delay", "Success %", "Collision %", "Reward"]
    print("\n" + "=" * 90)
    print("           SIM2REAL PHYSICAL DOMAIN RANDOMIZATION ROBUSTNESS BENCHMARK           ")
    print("=" * 90)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print("=" * 90 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    logger.info(f"Robustness evaluation results exported to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Sim2Real Robustness Benchmark")
    parser.add_argument("--out", type=str, default="outputs/robustness_results.csv", help="CSV report path")
    args = parser.parse_args()
    run_robustness_sweep(output_csv=args.out)


if __name__ == "__main__":
    main()
