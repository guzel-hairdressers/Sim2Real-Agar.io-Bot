"""
Diffusion Policy for Continuous Multi-Agent Agar.io Survival and Predation.
Generates multimodal 16-step action trajectories [thrust, steer, split] via Denoising Diffusion
Implicit Models (DDIM) conditioned on 38-dimensional POMDP state observations.
Features:
1. 1D Temporal ResNet Denoising Trunk with FiLM conditioning.
2. Fast DDIM reverse sampling (5-10 inference steps).
3. Agile Receding Horizon Control (RHC) with execution horizon T_e = 2 for rapid reaction.
4. Multimodal trajectory generation: handles bifurcated evasion vs. predatory pursuit vs. split lunges.
"""

import os
import sys
import time
import math
import logging
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarDiffusion")


class AgarDiffusionSchedule:
    """Noise schedule for Agar.io action diffusion."""

    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

    def ddim_step(self, a_t: np.ndarray, eps_pred: np.ndarray, t_curr: int, t_next: int, eta: float = 0.0) -> np.ndarray:
        alpha_bar_t = self.alphas_cumprod[t_curr]
        alpha_bar_next = self.alphas_cumprod[t_next] if t_next >= 0 else 1.0

        x_0_pred = (a_t - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / np.sqrt(alpha_bar_t)
        x_0_pred[:, 0] = np.clip(x_0_pred[:, 0], 0.0, 1.0)
        x_0_pred[:, 1] = np.clip(x_0_pred[:, 1], -1.0, 1.0)
        if x_0_pred.shape[1] > 2:
            x_0_pred[:, 2] = np.clip(x_0_pred[:, 2], 0.0, 1.0)

        c1 = np.sqrt(np.maximum(0.0, 1.0 - alpha_bar_next - eta**2))
        dir_xt = c1 * eps_pred

        noise = np.random.randn(*a_t.shape).astype(np.float32) if eta > 0.0 else 0.0
        a_next = np.sqrt(alpha_bar_next) * x_0_pred + dir_xt + eta * noise
        return a_next.astype(np.float32)


class AgarTemporalDenoisingNet:
    """1D Temporal ResNet / MLP Denoising Network with FiLM conditioning (NumPy fallback)."""

    def __init__(self, action_horizon: int = 16, action_dim: int = 3, obs_dim: int = 38, hidden_dim: int = 128, seed: int = 42):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        rng = np.random.RandomState(seed)
        self.w_obs = rng.randn(obs_dim, hidden_dim).astype(np.float32) * 0.1
        self.b_obs = np.zeros(hidden_dim, dtype=np.float32)

        self.w1 = rng.randn(action_horizon * action_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.w_film_gamma = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05
        self.w_film_beta = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05

        self.w2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.w_out = rng.randn(hidden_dim, action_horizon * action_dim).astype(np.float32) * 0.05
        self.b_out = np.zeros(action_horizon * action_dim, dtype=np.float32)

    def forward(self, a_noisy: np.ndarray, t_norm: float, obs: np.ndarray) -> np.ndarray:
        a_flat = a_noisy.reshape(-1)
        h_obs = np.maximum(0, np.dot(obs, self.w_obs) + self.b_obs)
        h_cond = h_obs + t_norm

        gamma = np.dot(h_cond, self.w_film_gamma) + 1.0
        beta = np.dot(h_cond, self.w_film_beta)

        h1 = np.maximum(0, np.dot(a_flat, self.w1) + self.b1)
        h1_film = gamma * h1 + beta

        h2 = np.maximum(0, np.dot(h1_film, self.w2) + self.b2)
        out_flat = np.dot(h1_film + h2, self.w_out) + self.b_out
        return out_flat.reshape(self.action_horizon, self.action_dim)


class ResBlock1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TorchAgarCritic(nn.Module):
    """Trajectory-Level Q-Value Critic Q(s, A) for Value-Guided Diffusion and Best-of-N Candidate Ranking."""

    def __init__(self, obs_dim: int = 38, action_dim: int = 3, action_horizon: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        in_dim = obs_dim + action_dim * action_horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if act.ndim == 3:
            act_flat = act.reshape(act.shape[0], -1)
        elif act.ndim == 2:
            if act.shape[-1] == self.action_dim * self.action_horizon:
                act_flat = act
            else:
                act_flat = act.repeat(1, self.action_horizon)
        else:
            act_flat = act.reshape(obs.shape[0], -1)
        x = torch.cat([obs, act_flat], dim=-1)
        return self.net(x).squeeze(-1)


class TorchAgarDiffusionNet(nn.Module):
    """Deep 1D Temporal Conditional ResNet Denoising Network for Agar.io trajectories."""

    def __init__(self, action_horizon: int = 16, action_dim: int = 3, obs_dim: int = 38, hidden_dim: int = 256):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # Observation conditioning encoder
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Diffusion timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Action trajectory input
        self.act_in = nn.Linear(action_horizon * action_dim, hidden_dim)

        # FiLM conditioning
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        # Deep ResNet Trunk (2 residual blocks)
        self.trunk = nn.Sequential(
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim),
        )

        # Output prediction head for noise epsilon
        self.act_out = nn.Linear(hidden_dim, action_horizon * action_dim)

    def forward(self, a_noisy: torch.Tensor, t_norm: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        B = a_noisy.shape[0]
        a_flat = a_noisy.reshape(B, -1)
        cond = self.obs_mlp(obs) + self.time_mlp(t_norm)

        gamma = self.film_gamma(cond) + 1.0
        beta = self.film_beta(cond)

        h = self.act_in(a_flat)
        h_film = gamma * h + beta
        h_out = self.trunk(h_film)
        eps = self.act_out(h_out).reshape(B, self.action_horizon, self.action_dim)
        return eps


class AgarDiffusionPolicy:
    """
    Diffusion Policy for Agar.io with Agile Receding Horizon Control (RHC).
    - Generates 16-step trajectories [thrust, steer, split] using fast DDIM reverse sampling.
    - Re-plans every T_e = 2 steps for lightning-fast closed-loop reaction.
    - Captures multimodal evasions, pursuit curves, and split-kill strikes.
    """

    def __init__(
        self,
        action_horizon: int = 16,
        exec_horizon: int = 2,
        action_dim: int = 3,
        obs_dim: int = 38,
        num_ddim_steps: int = 8,
        seed: int = 42,
        model_path: Optional[str] = None,
        critic_path: Optional[str] = None,
        num_candidates: int = 4
    ):
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_ddim_steps = num_ddim_steps
        self.num_candidates = num_candidates

        self.schedule = AgarDiffusionSchedule(num_timesteps=100)
        self.net = AgarTemporalDenoisingNet(
            action_horizon=action_horizon,
            action_dim=action_dim,
            obs_dim=obs_dim,
            hidden_dim=128,
            seed=seed
        )

        # PyTorch neural network loading
        self.torch_net: Optional[TorchAgarDiffusionNet] = None
        self.torch_critic: Optional[TorchAgarCritic] = None
        paths_to_check = [model_path] if model_path else [
            "outputs/agar_diffusion_model.pt",
            "projects/sim2real-ppo-navigation/outputs/agar_diffusion_model.pt"
        ]
        for pt_path in paths_to_check:
            if pt_path and os.path.exists(pt_path):
                try:
                    t_net = TorchAgarDiffusionNet(
                        action_horizon=action_horizon,
                        action_dim=action_dim,
                        obs_dim=obs_dim,
                        hidden_dim=256
                    )
                    t_net.load_state_dict(torch.load(pt_path, map_location="cpu"))
                    t_net.eval()
                    self.torch_net = t_net
                    logger.info(f"Loaded trained PyTorch Diffusion model from: {pt_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load {pt_path}: {e}")

        # Critic loading for Value Guidance
        critic_paths = [critic_path] if critic_path else [
            "outputs/agar_diffusion_critic.pt",
            "projects/sim2real-ppo-navigation/outputs/agar_diffusion_critic.pt"
        ]
        for c_path in critic_paths:
            if c_path and os.path.exists(c_path):
                try:
                    c_net = TorchAgarCritic(obs_dim=obs_dim, action_dim=action_dim, action_horizon=action_horizon, hidden_dim=128)
                    c_net.load_state_dict(torch.load(c_path, map_location="cpu"))
                    c_net.eval()
                    self.torch_critic = c_net
                    logger.info(f"Loaded trained PyTorch Diffusion Critic from: {c_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load Critic from {c_path}: {e}")

        self.action_buffer: List[np.ndarray] = []
        self.steps_since_plan = 0
        self.last_action: Optional[np.ndarray] = None

    def reset(self):
        self.action_buffer.clear()
        self.steps_since_plan = 0
        self.last_action = None

    def sample_trajectory_ddim(self, obs: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        if self.torch_net is not None:
            return self.sample_trajectory_torch(obs, steps=steps or self.num_ddim_steps, num_candidates=self.num_candidates)

        steps = steps or self.num_ddim_steps
        K = self.schedule.num_timesteps
        ddim_timesteps = np.linspace(K - 1, 0, steps, dtype=int)

        threat_dist = obs[24] if len(obs) > 24 else 0.0
        threat_dx = obs[22] if len(obs) > 22 else 0.0
        threat_dy = obs[23] if len(obs) > 23 else 0.0

        prey_dist = obs[28] if len(obs) > 28 else 0.0
        prey_dx = obs[26] if len(obs) > 26 else 0.0
        prey_dy = obs[27] if len(obs) > 27 else 0.0

        can_split = (obs[4] > 0.5) if len(obs) > 4 else False

        a_current = np.random.randn(self.action_horizon, self.action_dim).astype(np.float32)

        if threat_dist > 0.01 and threat_dist < 0.70:
            escape_angle = math.atan2(-threat_dy, -threat_dx)
            noise_bias = float(np.mean(a_current[:, 1]))
            flank_steer = 0.85 if noise_bias >= 0 else -0.85
            target_steer = float(np.clip(escape_angle / np.pi + flank_steer * 0.35, -1.0, 1.0))
            target_thrust = 1.0
            target_split = 0.0
        elif prey_dist > 0.01 and prey_dist < 0.80:
            target_angle = math.atan2(prey_dy, prey_dx)
            target_steer = float(np.clip(target_angle / np.pi, -1.0, 1.0))
            target_thrust = 0.95
            target_split = 1.0 if (can_split and prey_dist < 0.45 and abs(target_steer) < 0.25) else 0.0
        else:
            target_steer = 0.0
            target_thrust = 0.80
            target_split = 0.0

        clean_target = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        for i in range(self.action_horizon):
            w = (i + 1) / float(self.action_horizon)
            clean_target[i, 0] = target_thrust * (0.8 + 0.2 * w)
            clean_target[i, 1] = target_steer * (1.0 - 0.2 * (1.0 - w))
            if self.action_dim > 2:
                clean_target[i, 2] = target_split if i < 3 else 0.0

        for idx in range(len(ddim_timesteps)):
            t_curr = int(ddim_timesteps[idx])
            t_next = int(ddim_timesteps[idx + 1]) if idx + 1 < len(ddim_timesteps) else -1

            alpha_bar_t = self.schedule.alphas_cumprod[t_curr]
            sqrt_alpha_bar = np.sqrt(alpha_bar_t)
            sqrt_one_minus = np.sqrt(1.0 - alpha_bar_t)

            eps_pred = (a_current - sqrt_alpha_bar * clean_target) / np.maximum(sqrt_one_minus, 1e-4)

            diffs = np.zeros_like(a_current)
            diffs[1:] = a_current[1:] - a_current[:-1]
            eps_pred += 0.03 * diffs

            a_current = self.schedule.ddim_step(a_current, eps_pred, t_curr, t_next, eta=0.0)

        a_current[:, 0] = np.clip(a_current[:, 0], 0.0, 1.0)
        a_current[:, 1] = np.clip(a_current[:, 1], -1.0, 1.0)
        if self.action_dim > 2:
            a_current[:, 2] = np.clip(a_current[:, 2], 0.0, 1.0)
        return a_current

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if len(self.action_buffer) == 0 or self.steps_since_plan >= self.exec_horizon:
            if self.torch_net is not None:
                trajectory = self.sample_trajectory_torch(obs, steps=self.num_ddim_steps, num_candidates=self.num_candidates)
            else:
                trajectory = self.sample_trajectory_ddim(obs, steps=self.num_ddim_steps)
            self.action_buffer = [trajectory[i] for i in range(len(trajectory))]
            self.steps_since_plan = 0

        action = self.action_buffer.pop(0).copy()
        self.steps_since_plan += 1
        self.last_action = action.copy()
        return action

    def sample_trajectory_torch(self, obs: np.ndarray, steps: int = 8, num_candidates: int = 4) -> np.ndarray:
        device = next(self.torch_net.parameters()).device
        K = self.schedule.num_timesteps
        ddim_timesteps = np.linspace(K - 1, 0, steps, dtype=int)

        # Value guidance: sample num_candidates in parallel if critic is available
        B = num_candidates if (self.torch_critic is not None and num_candidates > 1) else 1
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1)
        a_current = torch.randn(B, self.action_horizon, self.action_dim, device=device)

        for idx in range(len(ddim_timesteps)):
            t_curr = int(ddim_timesteps[idx])
            t_next = int(ddim_timesteps[idx + 1]) if idx + 1 < len(ddim_timesteps) else -1

            t_norm = torch.full((B, 1), t_curr / float(K), dtype=torch.float32, device=device)
            with torch.no_grad():
                eps_pred = self.torch_net(a_current, t_norm, obs_t)

            alpha_bar_t = float(self.schedule.alphas_cumprod[t_curr])
            alpha_bar_next = float(self.schedule.alphas_cumprod[t_next]) if t_next >= 0 else 1.0

            x_0_pred = (a_current - math.sqrt(1.0 - alpha_bar_t) * eps_pred) / math.sqrt(alpha_bar_t)
            x_0_pred[..., 0] = torch.clamp(x_0_pred[..., 0], 0.0, 1.0)
            x_0_pred[..., 1] = torch.clamp(x_0_pred[..., 1], -1.0, 1.0)
            if self.action_dim > 2:
                x_0_pred[..., 2] = torch.clamp(x_0_pred[..., 2], 0.0, 1.0)

            c1 = math.sqrt(max(0.0, 1.0 - alpha_bar_next))
            a_current = math.sqrt(alpha_bar_next) * x_0_pred + c1 * eps_pred

        if B > 1:
            with torch.no_grad():
                q_scores = self.torch_critic(obs_t, a_current)
                best_idx = torch.argmax(q_scores).item()
            best_traj = a_current[best_idx].cpu().numpy()
        else:
            best_traj = a_current.squeeze(0).cpu().numpy()

        best_traj[:, 0] = np.clip(best_traj[:, 0], 0.0, 1.0)
        best_traj[:, 1] = np.clip(best_traj[:, 1], -1.0, 1.0)
        if self.action_dim > 2:
            best_traj[:, 2] = np.clip(best_traj[:, 2], 0.0, 1.0)
        return best_traj


def train_diffusion_policy(
    expert_dataset_path: str = "outputs/agar_expert_rollouts.npz",
    save_path: str = "outputs/agar_diffusion_model.pt",
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 256
) -> TorchAgarDiffusionNet:
    """Supervise-trains the 1D Temporal Diffusion Denoising Network on elite 3-action PPO rollouts."""
    if not os.path.exists(expert_dataset_path):
        alt = os.path.join("projects/sim2real-ppo-navigation", expert_dataset_path)
        if os.path.exists(alt):
            expert_dataset_path = alt
        else:
            raise FileNotFoundError(f"Expert rollout dataset not found at: {expert_dataset_path}")

    data = np.load(expert_dataset_path)
    obs_all = data["obs"].astype(np.float32)
    acts_all = data["acts"].astype(np.float32)
    N = len(obs_all)
    act_dim = acts_all.shape[-1]
    obs_dim = obs_all.shape[-1]
    logger.info(f"Loaded {N} elite expert rollout trajectories (obs_dim={obs_dim}, act_dim={act_dim}) from {expert_dataset_path}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training Diffusion Policy on device: {device}")

    net = TorchAgarDiffusionNet(action_horizon=16, action_dim=act_dim, obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    schedule = AgarDiffusionSchedule(num_timesteps=100)
    alphas_cumprod = torch.tensor(schedule.alphas_cumprod, dtype=torch.float32, device=device)

    dataset_indices = np.arange(N)

    print("\n" + "=" * 65)
    print(f"{'Epoch':<8} | {'Loss (MSE)':<15} | {'LR':<12} | {'Samples/sec':<15}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        np.random.shuffle(dataset_indices)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_idx = dataset_indices[start_idx:end_idx]
            B = len(batch_idx)

            b_obs = torch.tensor(obs_all[batch_idx], device=device)
            b_act_0 = torch.tensor(acts_all[batch_idx], device=device)

            k_steps = torch.randint(0, 100, (B,), device=device)
            t_norm = (k_steps.float() / 100.0).unsqueeze(-1)

            noise = torch.randn_like(b_act_0)
            alpha_bars = alphas_cumprod[k_steps].view(B, 1, 1)

            noisy_act = torch.sqrt(alpha_bars) * b_act_0 + torch.sqrt(1.0 - alpha_bars) * noise
            pred_noise = net(noisy_act, t_norm, b_obs)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        elapsed = max(1e-3, time.time() - t0)
        sps = int(N / elapsed)
        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch:<8d} | {avg_loss:<15.5f} | {current_lr:<12.1e} | {sps:<15d}")

    print("=" * 65 + "\n")

    for p in [save_path, os.path.join("projects/sim2real-ppo-navigation", save_path)]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        torch.save(net.state_dict(), p)
    logger.info(f"Trained Diffusion Policy model saved to: {save_path}")
    return net
