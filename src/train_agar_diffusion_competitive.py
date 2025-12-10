"""
Competitive Multi-Agent Arena Training: Diffusion Policy vs. PPO on Apple M4 GPU.
Conducts fair empirical training over 120,000 environment interaction steps
matching PPO's exact training budget step-for-step across 8 parallel arena environments.

Architecture:
1. 8 Parallel Arena Environments (48 concurrent agents: 24 Diffusion vs. 24 PPO).
2. Online Temporal Denoising Network (TorchAgarDiffusionNet) trained on high-performing trajectories.
3. Online Action-Value Critic (TorchAgarCritic) trained on Bellman TD errors.
4. Value-Guided DDIM Sampling (Best-of-N Candidate Generation) + Temporal Ensembling.
5. Saves outputs/agar_diffusion_model.pt and outputs/agar_diffusion_critic.pt.
"""

import os
import sys
import time
import math
import argparse
import logging
from collections import deque
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_agar_parallel_gpu import TorchAgarActorCritic
from src.agar_diffusion_policy import (
    TorchAgarDiffusionNet,
    TorchAgarCritic,
    AgarDiffusionSchedule,
    AgarDiffusionPolicy
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarDiffusionCompetitive")


def env_worker(remote, seed: int, arena_size: float = 14.0):
    env = PartiallyObservableAgarEnv(arena_size=arena_size, num_players=6, num_food=120, max_steps=250)
    obs, _ = env.reset(seed=seed)
    remote.send(obs)

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                next_obs, rewards, terms, truncs, infos = env.step(data)
                remote.send((next_obs, rewards, terms, truncs, infos))
            elif cmd == "reset":
                obs, _ = env.reset(seed=data)
                remote.send(obs)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
        except EOFError:
            break


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.obs = np.zeros((capacity, 38), dtype=np.float32)
        self.acts = np.zeros((capacity, 3), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, 38), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.acts[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.dones[idx]
        )


def train_diffusion_competitive(
    num_workers: int = 8,
    total_timesteps: int = 120_000,
    steps_per_rollout: int = 256,
    batch_size: int = 128,
    save_path_diff: str = "outputs/agar_diffusion_model.pt",
    save_path_critic: str = "outputs/agar_diffusion_critic.pt",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Initializing Competitive Diffusion vs PPO Training on: {device}")
    logger.info(f"Hardware scaling: {num_workers} parallel arenas ({num_workers * 6} total agents, 24 Diffusion vs 24 PPO)")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    obs_dim = 38
    act_dim = 3
    action_horizon = 16

    # 1. Initialize PPO Opponents (eval mode with trained weights)
    ppo_weights = "outputs/agar_ppo_gpu.pt"
    if not os.path.exists(ppo_weights):
        ppo_weights = "projects/sim2real-ppo-navigation/outputs/agar_ppo_gpu.pt"

    ppo_net = TorchAgarActorCritic(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=128).to(device)
    if os.path.exists(ppo_weights):
        ppo_net.load_state_dict(torch.load(ppo_weights, map_location=device))
        logger.info(f"Loaded trained PPO opponent weights from: {ppo_weights}")
    ppo_net.eval()

    # 2. Initialize Diffusion Policy & Critic Networks
    diff_net = TorchAgarDiffusionNet(action_horizon=action_horizon, action_dim=act_dim, obs_dim=obs_dim, hidden_dim=256).to(device)
    critic_net = TorchAgarCritic(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=128).to(device)
    target_critic = TorchAgarCritic(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=128).to(device)
    target_critic.load_state_dict(critic_net.state_dict())
    target_critic.eval()

    optimizer_diff = optim.AdamW(diff_net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_critic = optim.AdamW(critic_net.parameters(), lr=1e-3, weight_decay=1e-4)

    total_iters = total_timesteps // (num_workers * steps_per_rollout)
    scheduler_diff = optim.lr_scheduler.CosineAnnealingLR(optimizer_diff, T_max=max(1, total_iters), eta_min=5e-5)

    schedule = AgarDiffusionSchedule(num_timesteps=100)
    alphas_cumprod = torch.tensor(schedule.alphas_cumprod, dtype=torch.float32, device=device)

    # Replay buffer and trajectory storage
    replay_buffer = ReplayBuffer(capacity=60_000)
    elite_trajectories: List[Tuple[np.ndarray, np.ndarray]] = []

    # Dedicated Diffusion policy wrappers per worker
    diff_wrappers = [
        [
            AgarDiffusionPolicy(action_horizon=action_horizon, exec_horizon=2, action_dim=act_dim, obs_dim=obs_dim, num_ddim_steps=5, seed=42 + w * 17 + i)
            for i in range(3)
        ]
        for w in range(num_workers)
    ]
    # Link live networks
    for w in range(num_workers):
        for i in range(3):
            diff_wrappers[w][i].torch_net = diff_net
            diff_wrappers[w][i].torch_critic = critic_net

    # 3. Launch Parallel Arena Processes
    pipes = [mp.Pipe() for _ in range(num_workers)]
    remotes, work_remotes = zip(*pipes)
    processes = [
        mp.Process(target=env_worker, args=(work_remotes[i], 300 + i * 23, 14.0))
        for i in range(num_workers)
    ]
    for p in processes:
        p.daemon = True
        p.start()

    worker_obs = [remote.recv() for remote in remotes]

    global_step = 0
    iteration = 0
    start_time = time.time()

    # Track episode stats per player in each arena
    diff_player_ids = ["player_1", "player_3", "player_5"]
    ppo_player_ids = ["player_0", "player_2", "player_4"]

    worker_ep_trajs = [[[] for _ in range(3)] for _ in range(num_workers)]

    print("\n" + "=" * 120)
    print(f"{'Iter':<5} | {'Step':<8} | {'Throughput':<12} | {'Diff Rew':<9} | {'Diff Kills':<11} | {'Diff Mass':<10} | {'PPO Kills':<10} | {'PPO Mass':<9} | {'Q Loss':<8} | {'Diff Loss':<9}")
    print("=" * 120)

    try:
        while global_step < total_timesteps:
            iteration += 1
            iter_diff_rewards = []
            iter_diff_kills = 0
            iter_diff_masses = []
            iter_ppo_kills = 0
            iter_ppo_masses = []

            # A. Rollout Collection Phase across 8 parallel arenas
            for step in range(steps_per_rollout):
                # 1. PPO Opponent Actions (batched across 8 arenas * 3 PPO players = 24)
                ppo_obs_list = []
                for w in range(num_workers):
                    for pid in ppo_player_ids:
                        ppo_obs_list.append(worker_obs[w][pid])
                ppo_obs_t = torch.tensor(np.array(ppo_obs_list), dtype=torch.float32, device=device)
                with torch.no_grad():
                    ppo_act_t, _, _, _ = ppo_net.get_action_and_value(ppo_obs_t, deterministic=True)
                ppo_act_np = ppo_act_t.cpu().numpy()

                # 2. Diffusion Actions
                all_actions = []
                for w in range(num_workers):
                    w_actions = {}
                    # PPO actions
                    for idx, pid in enumerate(ppo_player_ids):
                        w_actions[pid] = ppo_act_np[w * 3 + idx]

                    # Diffusion actions
                    for idx, pid in enumerate(diff_player_ids):
                        obs_i = worker_obs[w][pid]
                        # Epsilon exploration: early on, explore alternative directions
                        explore_prob = max(0.05, 0.35 * (1.0 - global_step / total_timesteps))
                        if np.random.rand() < explore_prob:
                            act_i = diff_wrappers[w][idx].predict(obs_i, deterministic=False)
                            # add small exploration jitter
                            act_i[0] = np.clip(act_i[0] + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
                            act_i[1] = np.clip(act_i[1] + np.random.uniform(-0.15, 0.15), -1.0, 1.0)
                        else:
                            act_i = diff_wrappers[w][idx].predict(obs_i, deterministic=True)

                        w_actions[pid] = act_i
                    remotes[w].send(("step", w_actions))

                # Gather step results from arenas
                step_results = [remotes[w].recv() for w in range(num_workers)]
                for w in range(num_workers):
                    next_obs, rewards, terms, truncs, infos = step_results[w]

                    # Process Diffusion agents
                    for idx, pid in enumerate(diff_player_ids):
                        r = rewards[pid]
                        d = float(terms[pid] or truncs[pid])
                        iter_diff_rewards.append(r)
                        iter_diff_kills += infos[pid]["kills"]
                        iter_diff_masses.append(infos[pid]["mass"])

                        # Store transition
                        replay_buffer.add(
                            worker_obs[w][pid],
                            w_actions[pid],
                            r,
                            next_obs[pid],
                            d
                        )

                        # Track trajectory chunk
                        worker_ep_trajs[w][idx].append((
                            worker_obs[w][pid],
                            w_actions[pid],
                            infos[pid]["mass"],
                            infos[pid]["kills"]
                        ))

                        if d > 0.5 or len(worker_ep_trajs[w][idx]) >= 200:
                            traj = worker_ep_trajs[w][idx]
                            max_m = max(item[2] for item in traj) if traj else 0.0
                            tot_k = traj[-1][3] if traj else 0
                            # High-performance threshold: mass >= 26 kg or kills >= 1
                            if (tot_k >= 1 or max_m >= 26.0) and len(traj) >= 18:
                                for t_idx in range(len(traj) - 16):
                                    elite_trajectories.append((
                                        traj[t_idx][0],
                                        np.array([traj[t_idx + k][1] for k in range(16)], dtype=np.float32)
                                    ))
                                if len(elite_trajectories) > 30_000:
                                    elite_trajectories = elite_trajectories[-20_000:]
                            worker_ep_trajs[w][idx] = []

                    # Track PPO stats
                    for pid in ppo_player_ids:
                        iter_ppo_kills += infos[pid]["kills"]
                        iter_ppo_masses.append(infos[pid]["mass"])

                    worker_obs[w] = next_obs

                global_step += num_workers * len(diff_player_ids)

            # B. Critic TD-Learning Optimization Phase
            epoch_q_loss = 0.0
            num_q_updates = 0
            if replay_buffer.size >= batch_size:
                for _ in range(8):
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    t_s = torch.tensor(b_s, dtype=torch.float32, device=device)
                    t_a = torch.tensor(b_a, dtype=torch.float32, device=device)
                    t_r = torch.tensor(b_r, dtype=torch.float32, device=device)
                    t_ns = torch.tensor(b_ns, dtype=torch.float32, device=device)
                    t_d = torch.tensor(b_d, dtype=torch.float32, device=device)

                    q_pred = critic_net(t_s, t_a)
                    with torch.no_grad():
                        target_act = torch.stack([
                            torch.full((batch_size,), 0.95, device=device),
                            torch.clamp(t_ns[:, 26] * 1.5, -1.0, 1.0),
                            torch.where(t_ns[:, 4] > 0.5, 1.0, 0.0)
                        ], dim=-1)
                        next_q = target_critic(t_ns, target_act)
                        q_target = t_r + 0.99 * (1.0 - t_d) * next_q

                    loss_q = nn.functional.mse_loss(q_pred, q_target)
                    optimizer_critic.zero_grad()
                    loss_q.backward()
                    nn.utils.clip_grad_norm_(critic_net.parameters(), 1.0)
                    optimizer_critic.step()

                    # Polyak averaging target critic
                    for param, target_param in zip(critic_net.parameters(), target_critic.parameters()):
                        target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

                    epoch_q_loss += loss_q.item()
                    num_q_updates += 1

            # C. Diffusion Policy Optimization Phase
            epoch_diff_loss = 0.0
            num_diff_updates = 0
            if len(elite_trajectories) >= batch_size:
                for _ in range(8):
                    batch_idx = np.random.randint(0, len(elite_trajectories), size=batch_size)
                    b_obs = torch.tensor(np.array([elite_trajectories[i][0] for i in batch_idx]), dtype=torch.float32, device=device)
                    b_acts = torch.tensor(np.array([elite_trajectories[i][1] for i in batch_idx]), dtype=torch.float32, device=device)

                    k_steps = torch.randint(0, 100, (batch_size,), device=device)
                    t_norm = (k_steps.float() / 100.0).unsqueeze(-1)

                    noise = torch.randn_like(b_acts)
                    alpha_bars = alphas_cumprod[k_steps].view(batch_size, 1, 1)

                    noisy_act = torch.sqrt(alpha_bars) * b_acts + torch.sqrt(1.0 - alpha_bars) * noise
                    pred_noise = diff_net(noisy_act, t_norm, b_obs)

                    # MSE Denoising Loss + Temporal Smoothness Regularization
                    mse_loss = nn.functional.mse_loss(pred_noise, noise)
                    smoothness_loss = torch.mean((pred_noise[:, 1:] - pred_noise[:, :-1]) ** 2)
                    loss_diff = mse_loss + 0.08 * smoothness_loss

                    optimizer_diff.zero_grad()
                    loss_diff.backward()
                    nn.utils.clip_grad_norm_(diff_net.parameters(), 1.0)
                    optimizer_diff.step()

                    epoch_diff_loss += loss_diff.item()
                    num_diff_updates += 1

                scheduler_diff.step()

            # D. Live Telemetry
            elapsed = time.time() - start_time
            fps = int(global_step / max(0.1, elapsed))
            mean_d_rew = float(np.mean(iter_diff_rewards)) if iter_diff_rewards else 0.0
            d_kills = float(iter_diff_kills / (num_workers * len(diff_player_ids)))
            d_peak_m = float(np.max(iter_diff_masses)) if iter_diff_masses else 0.0
            p_kills = float(iter_ppo_kills / (num_workers * len(ppo_player_ids)))
            p_peak_m = float(np.max(iter_ppo_masses)) if iter_ppo_masses else 0.0
            avg_q = epoch_q_loss / max(1, num_q_updates)
            avg_diff = epoch_diff_loss / max(1, num_diff_updates)

            print(
                f"{iteration:<5d} | {global_step:>7d} | {fps:>6d} step/s | "
                f"{mean_d_rew:>+8.2f} | {d_kills:>10.2f} | {d_peak_m:>9.1f} kg | "
                f"{p_kills:>9.2f} | {p_peak_m:>8.1f} kg | {avg_q:>7.3f} | {avg_diff:>8.4f}"
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

    print("=" * 120 + "\n")

    # 4. Save Final Weights
    for p_d, p_c in [
        (save_path_diff, save_path_critic),
        ("projects/sim2real-ppo-navigation/" + save_path_diff, "projects/sim2real-ppo-navigation/" + save_path_critic)
    ]:
        os.makedirs(os.path.dirname(os.path.abspath(p_d)), exist_ok=True)
        torch.save(diff_net.state_dict(), p_d)
        torch.save(critic_net.state_dict(), p_c)

    logger.info(f"Successfully saved competitive Diffusion Model to: {save_path_diff}")
    logger.info(f"Successfully saved competitive Diffusion Critic to: {save_path_critic}")
    return diff_net, critic_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competitive Arena Training: Diffusion vs PPO on Apple M4")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel environments (8 on M4)")
    parser.add_argument("--timesteps", type=int, default=120_000, help="Total environment timesteps")
    args = parser.parse_args()

    train_diffusion_competitive(num_workers=args.workers, total_timesteps=args.timesteps)
