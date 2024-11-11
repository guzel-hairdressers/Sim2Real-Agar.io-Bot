"""
High-Performance Parallel Vectorized GPU Trainer for Multi-Agent Agar.io (PPO + Diffusion Self-Play).
Scaled for Apple M4 Multi-Core Architecture (8 parallel arenas, 48 concurrent agents):
1. Vectorized GAE per worker column (proper temporal credit assignment)
2. 38-dimensional continuous POMDP observations
3. 3-dimensional continuous actions [thrust, steer, split_trigger]
4. Authentic Agar.io Splitting Mechanic & Speed Inversion Physics
5. Automated Elite Expert Rollout Harvesting (kills >= 1, mass >= 38kg)
6. Integrated Supervised Training of Diffusion Policy on elite PPO demonstrations
"""

import os
import sys
import time
import math
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.agar_diffusion_policy import AgarDiffusionPolicy, train_diffusion_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ParallelGPUAgar")


# ---------------------------------------------------------------------------
# 1. PyTorch GPU Neural Network Architecture (38-dim obs -> 3D continuous actions)
# ---------------------------------------------------------------------------

class TorchAgarActorCritic(nn.Module):
    """Deep Actor-Critic Network for 38-dim Agar.io POMDP state space with Splitting."""

    def __init__(self, obs_dim: int = 38, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head: thrust in [0, 1], steer in [-1, 1], split in [0, 1]
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Critic value head
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
        features = self.trunk(obs)
        raw_action = self.actor_head(features)

        # Bounding: Dim 0 thrust in [0, 1], Dim 1 steer in [-1, 1], Dim 2 split in [0, 1]
        thrust = torch.sigmoid(raw_action[..., 0:1])
        steer = torch.tanh(raw_action[..., 1:2])
        if self.action_dim > 2:
            split_cmd = torch.sigmoid(raw_action[..., 2:3])
            mu = torch.cat([thrust, steer, split_cmd], dim=-1)
        else:
            mu = torch.cat([thrust, steer], dim=-1)

        value = self.critic_head(features).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, value

    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        if self.action_dim > 2:
            action = torch.stack([
                torch.clamp(action[..., 0], 0.0, 1.0),
                torch.clamp(action[..., 1], -1.0, 1.0),
                torch.clamp(action[..., 2], 0.0, 1.0)
            ], dim=-1)
        else:
            action = torch.stack([
                torch.clamp(action[..., 0], 0.0, 1.0),
                torch.clamp(action[..., 1], -1.0, 1.0)
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


class TorchAgarPPOAgent:
    """Wrapper for real-time inference during tournaments and video rendering."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu", obs_dim: int = 38, action_dim: int = 3):
        self.device = torch.device(device)
        self.net = TorchAgarActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(self.device)
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state_dict)
                logger.info(f"Loaded PyTorch PPO weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
        self.net.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            act, _, _, _ = self.net.get_action_and_value(t_obs, deterministic=deterministic)
            action = act.squeeze(0).cpu().numpy()

        # Tactical Split Lunge Trigger
        can_split = (obs[4] > 0.5) if len(obs) > 4 else False
        if can_split and len(obs) >= 30:
            has_prey = (obs[29] > 0.05)
            prey_dist = obs[28]
            has_threat = (obs[25] > 0.05 and obs[24] < 0.35)
            if has_prey and not has_threat and (0.05 <= prey_dist <= 0.60):
                prey_angle = math.atan2(obs[27], obs[26])
                if abs(prey_angle) < 0.45:
                    action[0] = 1.0
                    action[1] = float(np.clip(prey_angle * 1.5, -1.0, 1.0))
                    action[2] = 1.0
        elif not can_split:
            action[2] = 0.0

        return action


# ---------------------------------------------------------------------------
# 2. Parallel Multiprocessing Environment Worker
# ---------------------------------------------------------------------------

def env_worker(remote, env_seed: int, arena_size: float = 14.0, domain_randomize: bool = False):
    env = PartiallyObservableAgarEnv(
        arena_size=arena_size,
        num_players=6,
        num_food=60,
        max_steps=400,
        dt=0.12,
        domain_randomize_arena=domain_randomize,
        total_world_mass=200.0,
        pellet_mass=0.5
    )
    obs_dict, _ = env.reset(seed=env_seed)
    remote.send(obs_dict)

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs_dict, rewards, terms, truncs, infos = env.step(data)
            if any(truncs.values()):
                obs_dict, _ = env.reset()
            remote.send((obs_dict, rewards, terms, truncs, infos))
        elif cmd == "reset":
            obs_dict, _ = env.reset(seed=data)
            remote.send(obs_dict)
        elif cmd == "close":
            remote.close()
            break


# ---------------------------------------------------------------------------
# 3. High-Throughput Parallel GPU PPO Trainer (Apple M4 Multi-Core)
# ---------------------------------------------------------------------------

def train_parallel_gpu_agar(
    num_workers: int = 8,
    total_timesteps: int = 120_000,
    steps_per_rollout: int = 128,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    lr: float = 4e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.15,
    ent_coef: float = 0.005,
    save_path: str = "outputs/agar_ppo_gpu.pt"
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device} | Parallel Arena Workers: {num_workers}")
    logger.info(f"Total Concurrent Agents per Step: {num_workers * 6} (3 PPO Learner/Clones vs. 3 Diffusion Policies)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    obs_dim = 38
    action_dim = 3

    # 1. Initialize Neural Network & Optimizer on GPU
    policy_net = TorchAgarActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)

    # Pre-populate checkpoint clone for self-play
    checkpoint_net = TorchAgarActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    checkpoint_net.load_state_dict(policy_net.state_dict())
    checkpoint_net.eval()

    # Dedicated Diffusion policy instances per worker (prevents buffer cross-talk)
    diff_policies = [
        [
            AgarDiffusionPolicy(action_horizon=16, exec_horizon=2, action_dim=action_dim, obs_dim=obs_dim, num_ddim_steps=5, seed=42 + w * 13 + i)
            for i in range(3)
        ]
        for w in range(num_workers)
    ]

    # 2. Launch Parallel Worker Processes across M4 cores
    pipes = [mp.Pipe() for _ in range(num_workers)]
    remotes, work_remotes = zip(*pipes)
    processes = [
        mp.Process(target=env_worker, args=(work_remotes[i], 100 + i * 17, 14.0))
        for i in range(num_workers)
    ]
    for p in processes:
        p.daemon = True
        p.start()

    worker_obs = [remote.recv() for remote in remotes]

    global_step = 0
    iteration = 0
    start_time = time.time()

    history_kills = []
    history_mass = []
    history_rewards = []

    # Elite expert rollout harvesting for Diffusion Policy imitation learning
    worker_trajectories = [[] for _ in range(num_workers)]
    expert_dataset_obs = []
    expert_dataset_acts = []

    print("\n" + "=" * 115)
    print(f"{'Iter':<5} | {'Steps':<8} | {'Throughput':<12} | {'Reward':<8} | {'Kills':<6} | {'Peak Mass':<10} | {'Act Loss':<9} | {'Val Loss':<9} | {'Entropy':<8}")
    print("=" * 115)

    try:
        while global_step < total_timesteps:
            iteration += 1

            # Vectorized rollout buffers: (steps_per_rollout, num_workers, ...)
            obs_buf = np.zeros((steps_per_rollout, num_workers, obs_dim), dtype=np.float32)
            acts_buf = np.zeros((steps_per_rollout, num_workers, action_dim), dtype=np.float32)
            logprobs_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            rewards_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            dones_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            values_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)

            iter_kills = 0
            iter_masses = []

            # 3. Rollout Collection Phase across 8 parallel arenas
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

                # Self-Play Clones (player_1, player_2)
                clone_obs_list = []
                for w in range(num_workers):
                    clone_obs_list.append(worker_obs[w]["player_1"])
                    clone_obs_list.append(worker_obs[w]["player_2"])
                clone_obs_t = torch.tensor(np.array(clone_obs_list), dtype=torch.float32, device=device)
                with torch.no_grad():
                    clone_act_t, _, _, _ = checkpoint_net.get_action_and_value(clone_obs_t, deterministic=True)
                clone_act_np = clone_act_t.cpu().numpy()

                # Dispatch step to each worker
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

                # Gather step results from all 8 arenas
                step_results = [remotes[w].recv() for w in range(num_workers)]
                for w in range(num_workers):
                    next_obs, rewards, terms, truncs, infos = step_results[w]
                    worker_obs[w] = next_obs

                    rewards_buf[step, w] = rewards["player_0"]
                    dones_buf[step, w] = float(terms["player_0"] or truncs["player_0"])

                    iter_kills += infos["player_0"]["kills"]
                    iter_masses.append(infos["player_0"]["mass"])

                    # Expert trajectory tracking for Diffusion Policy training
                    worker_trajectories[w].append((learner_obs_list[w], learner_act_np[w], infos["player_0"]["mass"], infos["player_0"]["kills"]))
                    if terms["player_0"] or truncs["player_0"] or len(worker_trajectories[w]) >= 200:
                        traj = worker_trajectories[w]
                        max_m = max(item[2] for item in traj) if traj else 0.0
                        tot_k = traj[-1][3] if traj else 0
                        # ELITE FILTER: Confirmed kills or dominant predator mass >= 38.0kg
                        if (tot_k >= 1 or max_m >= 38.0) and len(traj) >= 18:
                            for t_idx in range(len(traj) - 16):
                                expert_dataset_obs.append(traj[t_idx][0])
                                expert_dataset_acts.append([traj[t_idx + k][1] for k in range(16)])
                        worker_trajectories[w] = []

                global_step += num_workers

            # 4. Correct Vectorized GAE Advantage Calculation per Worker Column
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

            # Flatten across (steps * workers)
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

            # 5. PPO Optimization Epochs on GPU
            dataset_size = len(b_obs)
            indices = np.arange(dataset_size)

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_entropy = 0.0
            num_updates = 0

            for _ in range(ppo_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, dataset_size, batch_size):
                    end_idx = min(start_idx + batch_size, dataset_size)
                    batch_idx = indices[start_idx:end_idx]

                    new_logprob, entropy, new_value = policy_net.evaluate_actions(t_obs[batch_idx], t_actions[batch_idx])
                    ratio = torch.exp(new_logprob - t_logprobs[batch_idx])

                    surr1 = ratio * t_advantages[batch_idx]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * t_advantages[batch_idx]
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = 0.5 * ((new_value - t_returns[batch_idx]) ** 2).mean()
                    loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    optimizer.step()

                    epoch_actor_loss += actor_loss.item()
                    epoch_critic_loss += critic_loss.item()
                    epoch_entropy += entropy.mean().item()
                    num_updates += 1

            # Update Self-Play Checkpoint clone periodically
            if iteration % 4 == 0:
                checkpoint_net.load_state_dict(policy_net.state_dict())

            # 6. Live Learning Metric Logging
            elapsed = time.time() - start_time
            fps = int(global_step / max(0.1, elapsed))
            mean_rew = float(np.mean(b_rewards))
            mean_peak_mass = float(np.max(iter_masses))
            mean_kills = float(iter_kills / num_workers)
            avg_a_loss = epoch_actor_loss / max(1, num_updates)
            avg_c_loss = epoch_critic_loss / max(1, num_updates)
            avg_ent = epoch_entropy / max(1, num_updates)

            history_rewards.append(mean_rew)
            history_kills.append(mean_kills)
            history_mass.append(mean_peak_mass)

            print(
                f"{iteration:<5d} | {global_step:>7d} | {fps:>6d} step/s | "
                f"{mean_rew:>+7.2f} | {mean_kills:>5.1f} | {mean_peak_mass:>9.1f} kg | "
                f"{avg_a_loss:>8.4f} | {avg_c_loss:>8.4f} | {avg_ent:>7.3f}"
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

    # 7. Save Model Checkpoints
    torch.save(policy_net.state_dict(), save_path)
    torch.save(policy_net.state_dict(), "projects/sim2real-ppo-navigation/outputs/agar_ppo_gpu.pt")
    logger.info(f"PyTorch GPU Model Checkpoint saved to: {save_path}")

    # 8. Export Harvested Elite Expert Rollouts & Supervise-Train Diffusion Policy
    for w in range(num_workers):
        traj = worker_trajectories[w]
        if len(traj) >= 18:
            max_m = max(item[2] for item in traj) if traj else 0.0
            tot_k = traj[-1][3] if traj else 0
            if tot_k >= 1 or max_m >= 38.0:
                for t_idx in range(len(traj) - 16):
                    expert_dataset_obs.append(traj[t_idx][0])
                    expert_dataset_acts.append([traj[t_idx + k][1] for k in range(16)])

    if len(expert_dataset_obs) > 50:
        exp_obs = np.array(expert_dataset_obs, dtype=np.float32)
        exp_acts = np.array(expert_dataset_acts, dtype=np.float32)
        logger.info(f"Exporting {len(exp_obs)} elite expert trajectory chunks for Diffusion Policy training...")
        for p in ["outputs/agar_expert_rollouts.npz", "projects/sim2real-ppo-navigation/outputs/agar_expert_rollouts.npz"]:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            np.savez(p, obs=exp_obs, acts=exp_acts)

        logger.info("Initiating Genuine Diffusion Policy Training on Harvested Elite PPO Rollouts...")
        train_diffusion_policy(
            expert_dataset_path="outputs/agar_expert_rollouts.npz",
            save_path="outputs/agar_diffusion_model.pt",
            epochs=35,
            batch_size=64,
            lr=1e-3,
            hidden_dim=256
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel GPU Multi-Agent Agar PPO on Apple M4")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel environments (8 on M4)")
    parser.add_argument("--timesteps", type=int, default=120_000, help="Total training timesteps")
    args = parser.parse_args()

    train_parallel_gpu_agar(num_workers=args.workers, total_timesteps=args.timesteps)
