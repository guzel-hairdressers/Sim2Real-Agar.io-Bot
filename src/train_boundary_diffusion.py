import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.train_champion import (
    train_champion_diffusion_with_smoothness,
    train_champion_critic,
    TorchAgarCritic
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBoundaryDiffusion")

def main():
    dataset_path = "outputs/agar_champion_expert_rollouts.npz"
    obs_npy = "outputs/expert_boundary_obs.npy"
    acts_npy = "outputs/expert_boundary_acts.npy"

    if os.path.exists(obs_npy) and os.path.exists(acts_npy):
        logger.info(f"Loading fresh expert rollouts from: {obs_npy}")
        obs_all = np.load(obs_npy).astype(np.float32)
        acts_all = np.load(acts_npy).astype(np.float32)
        # Package into npz for future runs
        np.savez_compressed(dataset_path, obs=obs_all, acts=acts_all)
    elif os.path.exists(dataset_path):
        logger.info(f"Loading boundary-aware expert dataset from: {dataset_path}")
        data = np.load(dataset_path)
        obs_all = data["obs"].astype(np.float32)
        acts_all = data["acts"].astype(np.float32)
    else:
        dataset_path = "projects/sim2real-ppo-navigation/outputs/agar_champion_expert_rollouts.npz"
        logger.info(f"Loading fallback dataset from: {dataset_path}")
        data = np.load(dataset_path)
        obs_all = data["obs"].astype(np.float32)
        acts_all = data["acts"].astype(np.float32)

    logger.info(f"Loaded {len(obs_all)} expert transitions (obs_dim={obs_all.shape[-1]}, act_horizon={acts_all.shape[1]})")

    # 1. Train Diffusion Denoising Policy with Temporal Smoothness
    logger.info("Training Boundary-Aware Diffusion Policy...")
    diff_save = "outputs/agar_diffusion_champion.pt"
    train_champion_diffusion_with_smoothness(
        expert_dataset_path=dataset_path,
        save_path=diff_save,
        epochs=30,
        batch_size=128,
        lr=1e-3,
        smooth_weight=0.05
    )

    # Synchronize weights across root and project outputs
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dirs = [
        os.path.join(root_dir, "outputs"),
        os.path.join(root_dir, "projects", "sim2real-ppo-navigation", "outputs")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        import shutil
        shutil.copyfile(diff_save, os.path.join(d, "agar_diffusion_champion.pt"))

    # 2. Train Q-Value Critic for Best-of-N Guidance
    logger.info("Training Multi-Objective Value Critic...")
    first_acts = acts_all[:, 0, :].astype(np.float32)

    # Extract body-frame observation features
    mass_val = obs_all[:, 3] * 12.0                # normalized cell mass
    can_split = obs_all[:, 4] > 0.5               # split capability
    threat_dist = obs_all[:, 24]                  # nearest threat distance (norm)
    prey_fwd = obs_all[:, 26]                     # prey forward distance in body frame
    prey_dist = obs_all[:, 28]                    # prey distance
    prey_ratio = obs_all[:, 29]                   # prey mass ratio
    food_density = obs_all[:, 32]                 # visible food density
    wall_front = obs_all[:, 34]                   # forward wall raycast
    virus_dist = obs_all[:, 36]                   # virus obstacle distance
    virus_ahead = obs_all[:, 37] > 0.5

    act_thrust = first_acts[:, 0]
    act_steer = first_acts[:, 1]
    act_split = first_acts[:, 2]

    # Compute physically grounded Monte Carlo Q-value targets:
    # A. Wall penalty: heavily penalize driving towards wall when wall_front is close
    wall_pen = np.where((wall_front < 0.25) & (act_thrust > 0.5), -4.0 * (0.25 - wall_front) / 0.25, 0.0)

    # B. Threat evasion: penalize moving towards close threat
    threat_pen = np.where((threat_dist > 0.01) & (threat_dist < 0.35) & (act_thrust > 0.6), -3.5 * (0.35 - threat_dist), 0.0)

    # C. Predatory split attack: strongly reward split action when prey is in front corridor
    split_reward = np.where(
        can_split & (act_split > 0.5) & (prey_dist > 0.05) & (prey_dist < 0.65) & (prey_fwd > 0.2) & (prey_ratio > 0.2),
        6.0,
        np.where((act_split > 0.5) & ((prey_dist <= 0.05) | (wall_front < 0.20)), -3.0, 0.0)
    )

    # D. Foraging & momentum: reward high speed foraging in open arena
    forage_bonus = np.where((wall_front > 0.35) & (threat_dist > 0.40), act_thrust * (1.5 + food_density * 2.0), 0.0)

    # E. Anti-circling penalty: penalize sharp continuous steering
    circling_pen = -0.8 * np.abs(act_steer)

    # F. Virus penalty: avoid viruses if large
    virus_pen = np.where((mass_val > 3.0) & virus_ahead & (virus_dist < 0.30) & (act_thrust > 0.5), -4.0, 0.0)

    critic_returns = (mass_val + wall_pen + threat_pen + split_reward + forage_bonus + circling_pen + virus_pen).astype(np.float32)

    critic_save = "outputs/agar_diffusion_critic_champion.pt"
    train_champion_critic(
        obs_all=obs_all,
        acts_all=first_acts,
        rewards_all=critic_returns,
        save_path=critic_save,
        epochs=25,
        batch_size=128,
        lr=1e-3
    )

    for d in out_dirs:
        shutil.copyfile(critic_save, os.path.join(d, "agar_diffusion_critic_champion.pt"))

    logger.info("Boundary-Aware Diffusion Policy and Critic Training Complete!")

if __name__ == "__main__":
    main()
