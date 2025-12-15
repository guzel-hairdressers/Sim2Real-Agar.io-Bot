"""
Train Pure Neural Diffusion Policy and Full-Trajectory Value Critic.
Trained on 76,486 clean deterministic expert rollouts (zero exploration noise).
Uses 1D Temporal ResNet Denoising Trunk + Trajectory Critic Q(s, A).
"""

import os
import sys
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agar_diffusion_policy import (
    TorchAgarDiffusionNet,
    TorchAgarCritic,
    AgarDiffusionSchedule
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPureDiffusion")


def main():
    dataset_path = "outputs/agar_clean_expert_rollouts.npz"
    if not os.path.exists(dataset_path):
        dataset_path = "projects/sim2real-ppo-navigation/outputs/agar_clean_expert_rollouts.npz"

    data = np.load(dataset_path)
    obs_all = data["obs"].astype(np.float32)
    acts_all = data["acts"].astype(np.float32)
    N = len(obs_all)
    obs_dim = obs_all.shape[-1]
    act_dim = acts_all.shape[-1]
    horizon = acts_all.shape[1]

    logger.info(f"Loaded clean expert dataset: N={N}, obs_dim={obs_dim}, horizon={horizon}, act_dim={act_dim}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # 1. Train Denoising Policy
    logger.info("--- Step 1: Training 1D Temporal ResNet Denoising Policy ---")
    net = TorchAgarDiffusionNet(action_horizon=horizon, action_dim=act_dim, obs_dim=obs_dim, hidden_dim=256).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 35
    batch_size = 128
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    schedule = AgarDiffusionSchedule(num_timesteps=100)
    alphas_cumprod = torch.tensor(schedule.alphas_cumprod, dtype=torch.float32, device=device)

    dataset_indices = np.arange(N)
    smooth_weight = 0.08

    print("\n" + "=" * 80)
    print(f"{'Epoch':<8} | {'MSE Loss':<14} | {'Smooth Loss':<14} | {'Total Loss':<14} | {'LR':<10}")
    print("=" * 80)

    net.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(dataset_indices)
        epoch_mse = 0.0
        epoch_sm = 0.0
        epoch_tot = 0.0
        num_batches = 0

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            b_idx = dataset_indices[start_idx:end_idx]
            B = len(b_idx)

            b_obs = torch.tensor(obs_all[b_idx], device=device)
            b_act = torch.tensor(acts_all[b_idx], device=device)

            k_steps = torch.randint(0, 100, (B,), device=device)
            t_norm = (k_steps.float() / 100.0).unsqueeze(-1)

            noise = torch.randn_like(b_act)
            alpha_bars = alphas_cumprod[k_steps].view(B, 1, 1)

            noisy_act = torch.sqrt(alpha_bars) * b_act + torch.sqrt(1.0 - alpha_bars) * noise
            pred_noise = net(noisy_act, t_norm, b_obs)

            loss_mse = nn.functional.mse_loss(pred_noise, noise)

            pred_x0 = (noisy_act - torch.sqrt(1.0 - alpha_bars) * pred_noise) / torch.sqrt(alpha_bars)
            loss_smooth = torch.mean((pred_x0[:, 1:] - pred_x0[:, :-1]) ** 2)

            total_loss = loss_mse + smooth_weight * loss_smooth

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_mse += loss_mse.item()
            epoch_sm += loss_smooth.item()
            epoch_tot += total_loss.item()
            num_batches += 1

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"{epoch:<8d} | {epoch_mse/num_batches:<14.5f} | {epoch_sm/num_batches:<14.5f} | {epoch_tot/num_batches:<14.5f} | {cur_lr:<10.1e}")

    print("=" * 80 + "\n")

    diff_save = "outputs/agar_diffusion_champion.pt"
    torch.save(net.state_dict(), diff_save)
    alt_diff = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_champion.pt"
    os.makedirs(os.path.dirname(alt_diff), exist_ok=True)
    torch.save(net.state_dict(), alt_diff)
    logger.info(f"Saved Diffusion Policy model to {diff_save}")

    # 2. Train Trajectory-Level Q-Value Critic
    logger.info("--- Step 2: Training Trajectory-Level Q-Value Critic Q(s, A) ---")
    critic = TorchAgarCritic(obs_dim=obs_dim, action_dim=act_dim, action_horizon=horizon, hidden_dim=128).to(device)
    critic_opt = optim.AdamW(critic.parameters(), lr=1e-3, weight_decay=1e-4)
    critic_epochs = 25

    # Compute Trajectory-Level Returns for each sample
    mass_val = obs_all[:, 3] * 12.0
    can_split = obs_all[:, 4] > 0.5
    threat_dist = obs_all[:, 24]
    prey_fwd = obs_all[:, 26]
    prey_lat = obs_all[:, 27]
    prey_dist = obs_all[:, 28]
    food_density = obs_all[:, 32]
    wall_front = obs_all[:, 34]

    # Trajectory-level summary features
    traj_thrust = acts_all[:, :, 0].mean(axis=1)
    traj_steer = acts_all[:, :, 1]
    traj_split = acts_all[:, :, 2].max(axis=1)
    traj_jerk = np.mean(np.abs(traj_steer[:, 1:] - traj_steer[:, :-1]), axis=1)

    # Physics-grounded Trajectory Evaluation:
    # A. Wall Collision Avoidance
    wall_pen = np.where((wall_front < 0.25) & (traj_thrust > 0.6), -5.0 * (0.25 - wall_front) / 0.25, 0.0)

    # B. Threat Evasion
    threat_pen = np.where((threat_dist > 0.01) & (threat_dist < 0.35) & (traj_thrust > 0.5), -4.0 * (0.35 - threat_dist), 0.0)

    # C. Predatory Split Attack
    split_bonus = np.where(
        can_split & (traj_split > 0.5) & (prey_dist > 0.05) & (prey_dist < 0.65) & (prey_fwd > 0.15) & (np.abs(prey_lat) < 0.35),
        7.0,
        np.where((traj_split > 0.5) & ((prey_dist <= 0.05) | (wall_front < 0.20)), -3.0, 0.0)
    )

    # D. Open Arena Foraging & High Velocity
    forage_bonus = np.where((wall_front > 0.30) & (threat_dist > 0.35), traj_thrust * (2.0 + food_density * 2.5), 0.0)

    # E. Trajectory Smoothness Bonus (penalize steering jerk)
    smooth_bonus = -2.5 * traj_jerk

    critic_returns = (mass_val + wall_pen + threat_pen + split_bonus + forage_bonus + smooth_bonus).astype(np.float32)

    t_obs = torch.tensor(obs_all, dtype=torch.float32, device=device)
    t_acts = torch.tensor(acts_all, dtype=torch.float32, device=device)
    t_rets = torch.tensor(critic_returns, dtype=torch.float32, device=device)

    critic.train()
    for c_epoch in range(1, critic_epochs + 1):
        np.random.shuffle(dataset_indices)
        c_loss = 0.0
        num_b = 0
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            b_idx = dataset_indices[start_idx:end_idx]

            q_pred = critic(t_obs[b_idx], t_acts[b_idx])
            loss = nn.functional.mse_loss(q_pred, t_rets[b_idx])

            critic_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            c_loss += loss.item()
            num_b += 1

        if c_epoch % 5 == 0 or c_epoch == critic_epochs:
            logger.info(f"Critic Epoch {c_epoch:02d}/{critic_epochs:02d} | MSE Loss: {c_loss / max(1, num_b):.5f}")

    critic_save = "outputs/agar_diffusion_critic_champion.pt"
    torch.save(critic.state_dict(), critic_save)
    alt_crit = "projects/sim2real-ppo-navigation/outputs/agar_diffusion_critic_champion.pt"
    torch.save(critic.state_dict(), alt_crit)
    logger.info(f"Saved Trajectory Critic to {critic_save}")
    logger.info("Champion Diffusion Training Complete!")


if __name__ == "__main__":
    main()
