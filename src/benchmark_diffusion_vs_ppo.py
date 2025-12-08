"""
Comparative Robustness Benchmark: Diffusion Policy vs. PPO vs. MSE Behavioral Cloning.
Evaluates:
1. Multimodal Obstacle Clearance Success Rate (%)
2. Actuator Smoothness & Action Jerk (high-frequency chatter)
3. Sim-to-Real Latency Delay Resilience (0-step, 1-step, 2-step lag)
4. Inference Latency (ms)
Exports quantitative results to outputs/diffusion_vs_ppo_benchmark.csv.
"""

import os
import sys
import time
import csv
import logging
from typing import Dict, Any, List
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.navigation_env import ContinuousNavigationEnv
from envs.domain_randomization import DomainRandomizationWrapper
from src.train import ContinuousPPOAgent
from src.diffusion_policy import (
    DiffusionPolicy,
    MSEBehavioralCloningPolicy,
    MultimodalDemonstrationGenerator
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def compute_action_jerk(actions: List[np.ndarray]) -> float:
    """Computes mean action delta L2 norm (jerk / high-frequency chatter)."""
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


def evaluate_policy_on_env(
    policy: Any,
    env_wrapper_fn,
    num_episodes: int = 20,
    seed_offset: int = 100
) -> Dict[str, float]:
    """Runs policy rollouts and computes success rate, collision rate, jerk, and steps."""
    successes = 0
    collisions = 0
    all_jerks = []
    episode_lengths = []
    inference_times = []

    for ep in range(num_episodes):
        env = env_wrapper_fn(seed=seed_offset + ep)
        obs, info = env.reset(seed=seed_offset + ep)
        if hasattr(policy, "reset"):
            policy.reset()

        terminated = False
        truncated = False
        ep_actions = []

        while not (terminated or truncated):
            t0 = time.perf_counter()
            action = policy.predict(obs, deterministic=True)
            inference_times.append((time.perf_counter() - t0) * 1000.0) # ms

            ep_actions.append(action.copy())
            obs, reward, terminated, truncated, step_info = env.step(action)

        if step_info.get("is_success", False):
            successes += 1
        if step_info.get("collision", False):
            collisions += 1

        all_jerks.append(compute_action_jerk(ep_actions))
        episode_lengths.append(len(ep_actions))

    return {
        "success_rate": (successes / num_episodes) * 100.0,
        "collision_rate": (collisions / num_episodes) * 100.0,
        "mean_jerk": float(np.mean(all_jerks)),
        "mean_steps": float(np.mean(episode_lengths)),
        "mean_inference_ms": float(np.mean(inference_times))
    }


def run_comprehensive_benchmark():
    logger.info("Initializing Comparative Benchmark: Diffusion Policy vs. PPO vs. MSE-BC...")

    os.makedirs("outputs", exist_ok=True)
    csv_file = "outputs/diffusion_vs_ppo_benchmark.csv"

    # 1. Initialize policies
    logger.info("Setting up policies...")
    ppo_policy = ContinuousPPOAgent(obs_dim=12, action_dim=2)
    weights_path = "outputs/policy_weights.npz"
    if not os.path.exists(weights_path):
        weights_path = "projects/sim2real-ppo-navigation/outputs/policy_weights.npz"
    if os.path.exists(weights_path):
        ppo_policy.net.load(weights_path)
        logger.info(f"Loaded trained PPO policy weights from {weights_path}")
    else:
        logger.warning("PPO policy weights not found, using initialized weights")

    # Generate multimodal demonstrations and train MSE-BC
    generator = MultimodalDemonstrationGenerator(action_horizon=16)
    demos = generator.generate_demonstrations(num_episodes=40, seed=42)
    flattened_demos = []
    for d in demos:
        for o, a in zip(d["observations"], d["actions"]):
            flattened_demos.append((o, a))

    bc_policy = MSEBehavioralCloningPolicy(obs_dim=12, action_dim=2, hidden_dim=64, seed=42)
    bc_policy.train_on_demos(flattened_demos, epochs=20)

    diffusion_policy = DiffusionPolicy(
        action_horizon=16,
        exec_horizon=8,
        action_dim=2,
        obs_dim=12,
        num_ddim_steps=10,
        seed=42
    )

    policies = {
        "MSE Behavioral Cloning": bc_policy,
        "PPO (Reinforcement Learning)": ppo_policy,
        "Diffusion Policy (Receding Horizon DDIM)": diffusion_policy
    }

    # 2. Benchmark under different Latency Delays
    delay_scenarios = [
        ("Nominal (0-step Delay)", 0),
        ("Moderate Latency (1-step Delay)", 1),
        ("High Latency (2-step Delay)", 2)
    ]

    results = []

    print("\n" + "=" * 95)
    print(f"{'Policy':<42} | {'Scenario':<30} | {'Success':<8} | {'Jerk':<8} | {'Inference':<10}")
    print("=" * 95)

    for scenario_name, delay_steps in delay_scenarios:
        def make_env_wrapper(seed: int):
            base_env = ContinuousNavigationEnv(arena_size=10.0, num_obstacles=4)
            wrapped = DomainRandomizationWrapper(
                base_env,
                friction_range=(0.6, 1.0),
                mass_range=(1.8, 2.5),
                sensor_noise_range=(0.005, 0.02),
                max_action_delay=delay_steps
            )
            # Fix delay for controlled comparison
            wrapped.action_delay_steps = delay_steps
            return wrapped

        for pol_name, pol_instance in policies.items():
            metrics = evaluate_policy_on_env(pol_instance, make_env_wrapper, num_episodes=20, seed_offset=200)

            results.append({
                "policy": pol_name,
                "scenario": scenario_name,
                "delay_steps": delay_steps,
                "success_rate_pct": round(metrics["success_rate"], 1),
                "collision_rate_pct": round(metrics["collision_rate"], 1),
                "actuator_jerk": round(metrics["mean_jerk"], 4),
                "mean_steps": round(metrics["mean_steps"], 1),
                "inference_latency_ms": round(metrics["mean_inference_ms"], 2)
            })

            print(
                f"{pol_name:<42} | {scenario_name:<30} | "
                f"{metrics['success_rate']:>6.1f}% | {metrics['mean_jerk']:>8.4f} | {metrics['mean_inference_ms']:>7.2f} ms"
            )

    print("=" * 95 + "\n")

    # Export to CSV
    keys = list(results[0].keys())
    for target_path in [csv_file, "projects/sim2real-ppo-navigation/outputs/diffusion_vs_ppo_benchmark.csv"]:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Comparative benchmark successfully exported to {csv_file} and projects/sim2real-ppo-navigation/outputs/")


if __name__ == "__main__":
    run_comprehensive_benchmark()
