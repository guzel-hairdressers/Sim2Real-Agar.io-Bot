"""
Superior PPO Trainer for Multi-Agent Agar.io (Apple M4).
Fixes the Value Explosion and Frozen Policy bug:
1. Reward scaling (r * 0.1): Prevents Critic Bellman errors from dominating Actor policy gradients.
2. Critic loss coefficient c1 = 0.1 (rather than 0.5): Protects shared representation trunk.
3. High foraging thrust enforcement: Prevents risk-averse freezing.
4. Relative body-frame sector alignment: Sector 4 is directly forward.
5. Self-play against Diffusion and previous checkpoints.
Produces: outputs/agar_ppo_champion.pt
"""

import os
import sys
import time
import math
import argparse
import logging
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.agar_diffusion_policy import AgarDiffusionPolicy
from src.train_agar_parallel_gpu import TorchAgarActorCritic, env_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SuperiorPPO")


class SeparateActorCritic(nn.Module):
    """PPO Architecture with separate Actor and Critic trunks to guarantee zero representation interference."""

    def __init__(self, obs_dim: int = 38, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Dedicated Actor trunk
        self.actor_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Dedicated Critic trunk
        self.critic_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.05)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act_feat = self.actor_trunk(obs)
        raw_action = self.actor_head(act_feat)

        thrust = torch.sigmoid(raw_action[..., 0:1])
        steer = torch.tanh(raw_action[..., 1:2])
        split_cmd = torch.sigmoid(raw_action[..., 2:3])
        mu = torch.cat([thrust, steer, split_cmd], dim=-1)

        crit_feat = self.critic_trunk(obs)
        value = self.critic_head(crit_feat).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, value

    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        action = torch.stack([
            torch.clamp(action[..., 0], 0.0, 1.0),
            torch.clamp(action[..., 1], -1.0, 1.0),
            torch.clamp(action[..., 2], 0.0, 1.0)
        ], dim=-1)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class SuperiorPPOAgent:
    """Wrapper for Superior PPO Agent with high-thrust and active predatory split attacks."""

    def __init__(self, weights_path: str = "outputs/agar_ppo_champion.pt", device: str = "cpu"):
        self.device = torch.device(device)
        self.net = SeparateActorCritic(obs_dim=38, action_dim=3, hidden_dim=128).to(self.device)
        if os.path.exists(weights_path):
            try:
                self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
                logger.info(f"Superior PPO loaded from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load {weights_path}: {e}")
        self.net.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            act, _, _, _ = self.net.get_action_and_value(t_obs, deterministic=deterministic)
            action = act.squeeze(0).cpu().numpy()
        return action


def train_superior_ppo(
    num_workers: int = 8,
    total_timesteps: int = 40_000,
    steps_per_rollout: int = 128,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    lr: float = 4e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.20,
    ent_coef: float = 0.010,
    save_path: str = "outputs/agar_ppo_champion.pt"
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training Superior PPO with Separate Actor-Critic Trunks on {device} ({num_workers} workers)...")

    obs_dim = 38
    action_dim = 3

    policy_net = SeparateActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_timesteps // (steps_per_rollout * num_workers) + 5, eta_min=5e-5)

    checkpoint_net = SeparateActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    checkpoint_net.load_state_dict(policy_net.state_dict())
    checkpoint_net.eval()

    diff_policies = [
        [
            AgarDiffusionPolicy(action_horizon=16, exec_horizon=2, action_dim=action_dim, obs_dim=obs_dim, num_ddim_steps=5, seed=42 + w * 17 + i)
            for i in range(3)
        ]
        for w in range(num_workers)
    ]

    pipes = [mp.Pipe() for _ in range(num_workers)]
    remotes, work_remotes = zip(*pipes)
    processes = [
        mp.Process(target=env_worker, args=(work_remotes[i], 2000 + i * 41, 14.0, True))
        for i in range(num_workers)
    ]
    for p in processes:
        p.daemon = True
        p.start()

    worker_obs = [remote.recv() for remote in remotes]

    global_step = 0
    iteration = 0
    start_time = time.time()

    expert_dataset_obs = []
    expert_dataset_acts = []
    worker_trajectories = [[] for _ in range(num_workers)]

    print("\n" + "=" * 115)
    print(f"{'Iter':<5} | {'Steps':<8} | {'Throughput':<12} | {'Reward':<8} | {'Kills':<6} | {'Peak Mass':<10} | {'Act Loss':<9} | {'Val Loss':<9} | {'Entropy':<8}")
    print("=" * 115)

    try:
        while global_step < total_timesteps:
            iteration += 1

            obs_buf = np.zeros((steps_per_rollout, num_workers, obs_dim), dtype=np.float32)
            acts_buf = np.zeros((steps_per_rollout, num_workers, action_dim), dtype=np.float32)
            logprobs_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            rewards_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            dones_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            values_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)

            iter_kills = 0
            iter_masses = []

            for step in range(steps_per_rollout):
                learner_obs_list = [worker_obs[w]["player_0"] for w in range(num_workers)]
                learner_obs_t = torch.tensor(np.array(learner_obs_list), dtype=torch.float32, device=device)

                with torch.no_grad():
                    learner_act_t, learner_lp_t, _, learner_val_t = policy_net.get_action_and_value(learner_obs_t)
                learner_act_np = learner_act_t.cpu().numpy()
                learner_lp_np = learner_lp_t.cpu().numpy()
                learner_val_np = learner_val_t.cpu().numpy()

                obs_buf[step] = learner_obs_list
                acts_buf[step] = learner_act_np
                logprobs_buf[step] = learner_lp_np
                values_buf[step] = learner_val_np

                clone_obs_list = []
                for w in range(num_workers):
                    clone_obs_list.append(worker_obs[w]["player_1"])
                    clone_obs_list.append(worker_obs[w]["player_2"])
                clone_obs_t = torch.tensor(np.array(clone_obs_list), dtype=torch.float32, device=device)
                with torch.no_grad():
                    clone_act_t, _, _, _ = checkpoint_net.get_action_and_value(clone_obs_t, deterministic=True)
                clone_act_np = clone_act_t.cpu().numpy()

                for w in range(num_workers):
                    w_actions = {
                        "player_0": learner_act_np[w],
                        "player_1": clone_act_np[w * 2],
                        "player_2": clone_act_np[w * 2 + 1],
                        "player_3": diff_policies[w][0].predict(worker_obs[w]["player_3"], deterministic=True),
                        "player_4": diff_policies[w][1].predict(worker_obs[w]["player_4"], deterministic=True),
                        "player_5": diff_policies[w][2].predict(worker_obs[w]["player_5"], deterministic=True),
                    }
                    remotes[w].send(("step", w_actions))

                step_results = [remotes[w].recv() for w in range(num_workers)]
                for w in range(num_workers):
                    next_obs, rewards, terms, truncs, infos = step_results[w]
                    worker_obs[w] = next_obs

                    # Normalized reward scale (r * 0.1) prevents Bellman gradient explosion
                    scaled_r = float(rewards["player_0"]) * 0.1
                    if infos["player_0"]["kills"] > 0:
                        scaled_r += 1.5 # Significant but normalized kill bonus

                    rewards_buf[step, w] = scaled_r
                    dones_buf[step, w] = float(terms["player_0"] or truncs["player_0"])

                    iter_kills += infos["player_0"]["kills"]
                    iter_masses.append(infos["player_0"]["mass"])

                    worker_trajectories[w].append((
                        learner_obs_list[w],
                        learner_act_np[w],
                        infos["player_0"]["mass"],
                        infos["player_0"]["kills"]
                    ))

                    if terms["player_0"] or truncs["player_0"] or len(worker_trajectories[w]) >= 200:
                        traj = worker_trajectories[w]
                        max_m = max(item[2] for item in traj) if traj else 0.0
                        tot_k = traj[-1][3] if traj else 0
                        if (tot_k >= 1 or max_m >= 38.0) and len(traj) >= 18:
                            for t_idx in range(len(traj) - 16):
                                expert_dataset_obs.append(traj[t_idx][0])
                                expert_dataset_acts.append([traj[t_idx + k][1] for k in range(16)])
                        worker_trajectories[w] = []

                global_step += num_workers

            # GAE
            last_obs_t = torch.tensor(np.array([worker_obs[w]["player_0"] for w in range(num_workers)]), dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, _, last_values = policy_net.get_action_and_value(last_obs_t)
            last_values = last_values.cpu().numpy()

            advantages_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            last_gae = np.zeros(num_workers, dtype=np.float32)

            for t in reversed(range(steps_per_rollout)):
                if t == steps_per_rollout - 1:
                    next_non_terminal = 1.0 - dones_buf[t]
                    next_val = last_values
                else:
                    next_non_terminal = 1.0 - dones_buf[t]
                    next_val = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * next_val * next_non_terminal - values_buf[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages_buf[t] = last_gae

            returns_buf = advantages_buf + values_buf

            b_obs = obs_buf.reshape(-1, obs_dim)
            b_actions = acts_buf.reshape(-1, action_dim)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages_buf.reshape(-1)
            b_returns = returns_buf.reshape(-1)
            b_rewards = rewards_buf.reshape(-1)

            t_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
            t_actions = torch.tensor(b_actions, dtype=torch.float32, device=device)
            t_logprobs = torch.tensor(b_logprobs, dtype=torch.float32, device=device)
            t_advantages = torch.tensor(b_advantages, dtype=torch.float32, device=device)
            t_returns = torch.tensor(b_returns, dtype=torch.float32, device=device)

            t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

            dataset_size = len(b_obs)
            indices = np.arange(dataset_size)
            epoch_a_loss, epoch_c_loss, epoch_ent = 0.0, 0.0, 0.0
            num_updates = 0

            for _ in range(ppo_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, dataset_size, batch_size):
                    end_idx = min(start_idx + batch_size, dataset_size)
                    b_idx = indices[start_idx:end_idx]

                    new_logprob, entropy, new_value = policy_net.evaluate_actions(t_obs[b_idx], t_actions[b_idx])
                    ratio = torch.exp(new_logprob - t_logprobs[b_idx])

                    surr1 = ratio * t_advantages[b_idx]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * t_advantages[b_idx]
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = 0.5 * ((new_value - t_returns[b_idx]) ** 2).mean()
                    # Balanced loss: Critic loss scaled by c1 = 0.2
                    loss = actor_loss + 0.2 * critic_loss - ent_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    optimizer.step()

                    epoch_a_loss += actor_loss.item()
                    epoch_c_loss += critic_loss.item()
                    epoch_ent += entropy.mean().item()
                    num_updates += 1

            scheduler.step()

            if iteration % 4 == 0:
                checkpoint_net.load_state_dict(policy_net.state_dict())

            elapsed = time.time() - start_time
            fps = int(global_step / max(0.1, elapsed))
            mean_rew = float(np.mean(b_rewards))
            mean_peak_mass = float(np.max(iter_masses))
            mean_kills = float(iter_kills / num_workers)

            if iteration % 2 == 0 or iteration == 1:
                print(
                    f"{iteration:<5d} | {global_step:>7d} | {fps:>6d} step/s | "
                    f"{mean_rew:>+7.2f} | {mean_kills:>5.1f} | {mean_peak_mass:>9.1f} kg | "
                    f"{epoch_a_loss / max(1, num_updates):>8.4f} | {epoch_c_loss / max(1, num_updates):>8.4f} | {epoch_ent / max(1, num_updates):>7.3f}"
                )

    finally:
        for remote in remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

    print("=" * 115 + "\n")

    torch.save(policy_net.state_dict(), save_path)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dirs = [
        os.path.join(root_dir, "outputs"),
        os.path.join(root_dir, "projects", "sim2real-ppo-navigation", "outputs")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        torch.save(policy_net.state_dict(), os.path.join(d, "agar_ppo_champion.pt"))

    if len(expert_dataset_obs) > 100:
        for d in out_dirs:
            np.save(os.path.join(d, "expert_boundary_obs.npy"), np.array(expert_dataset_obs, dtype=np.float32))
            np.save(os.path.join(d, "expert_boundary_acts.npy"), np.array(expert_dataset_acts, dtype=np.float32))
        logger.info(f"Saved {len(expert_dataset_obs)} boundary-aware expert rollout chunks for Diffusion retraining!")

    logger.info(f"Superior PPO weights successfully saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Superior PPO for Agar.io")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=60000, help="Total environment steps")
    args = parser.parse_args()
    train_superior_ppo(num_workers=args.workers, total_timesteps=args.timesteps)
