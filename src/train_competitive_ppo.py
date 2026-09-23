"""
Competitive PPO Trainer with Master Heuristic Opponents in Arena.
Features:
1. Multi-Agent Competitive League: PPO Learner vs Master Heuristics (Apex, Hunter, Survivor) + Self-Play Clones.
2. Anti-Suicide & Tactical Predation Reward Shaping:
   - Severe penalty for moving towards heavier predators (punishing suicidal forward drift).
   - High reward for eating opponents (+35.0) and penalty for death (-25.0).
   - Reward for successful split attacks against prey.
3. Harvests elite competitive rollouts for Diffusion Policy training.
4. Outputs: outputs/agar_ppo_champion.pt and outputs/agar_clean_expert_rollouts.npz
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
from src.heuristic_agent import MasterHeuristicAgarBot
from src.train_superior_ppo import SeparateActorCritic, SuperiorPPOAgent
from src.train_agar_parallel_gpu import env_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CompetitivePPO")


def train_competitive_ppo(
    num_workers: int = 8,
    total_timesteps: int = 50_000,
    steps_per_rollout: int = 128,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    lr: float = 3.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.20,
    ent_coef: float = 0.012,
    save_path: str = "outputs/agar_ppo_champion.pt"
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training Competitive PPO against Master Heuristics on {device} ({num_workers} parallel arenas)...")

    obs_dim = 38
    action_dim = 3

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    # Initialize Policy and Optimizer
    policy_net = SeparateActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)

    # Load existing champion as warm start if available
    if os.path.exists(save_path):
        try:
            policy_net.load_state_dict(torch.load(save_path, map_location=device))
            logger.info(f"Loaded existing weights from {save_path} for warm start.")
        except Exception as e:
            logger.warning(f"Could not warm start from {save_path}: {e}")

    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)
    total_iters = total_timesteps // (steps_per_rollout * num_workers) + 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=5e-5)

    # Self-play clone checkpoint
    checkpoint_net = SeparateActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    checkpoint_net.load_state_dict(policy_net.state_dict())
    checkpoint_net.eval()

    # Master Heuristic opponents per worker arena
    heuristic_apex = [MasterHeuristicAgarBot(pid="player_1", profile="apex", seed=100 + w) for w in range(num_workers)]
    heuristic_hunter = [MasterHeuristicAgarBot(pid="player_2", profile="hunter", seed=200 + w) for w in range(num_workers)]
    heuristic_survivor = [MasterHeuristicAgarBot(pid="player_4", profile="survivor", seed=400 + w) for w in range(num_workers)]

    # Start parallel workers
    pipes = [mp.Pipe() for _ in range(num_workers)]
    remotes, work_remotes = zip(*pipes)
    processes = [
        mp.Process(target=env_worker, args=(work_remotes[i], 3000 + i * 47, 14.0, True))
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

                # Clone decisions for player_3 and player_5
                clone_obs_list = []
                for w in range(num_workers):
                    clone_obs_list.append(worker_obs[w]["player_3"])
                    clone_obs_list.append(worker_obs[w]["player_5"])
                clone_obs_t = torch.tensor(np.array(clone_obs_list), dtype=torch.float32, device=device)
                with torch.no_grad():
                    clone_act_t, _, _, _ = checkpoint_net.get_action_and_value(clone_obs_t, deterministic=True)
                clone_act_np = clone_act_t.cpu().numpy()

                # Dispatch actions: 3 Heuristic Bots + 2 Clones + 1 Learner
                for w in range(num_workers):
                    w_actions = {
                        "player_0": learner_act_np[w],
                        "player_1": heuristic_apex[w].predict(worker_obs[w]["player_1"]),
                        "player_2": heuristic_hunter[w].predict(worker_obs[w]["player_2"]),
                        "player_3": clone_act_np[w * 2],
                        "player_4": heuristic_survivor[w].predict(worker_obs[w]["player_4"]),
                        "player_5": clone_act_np[w * 2 + 1],
                    }
                    remotes[w].send(("step", w_actions))

                step_results = [remotes[w].recv() for w in range(num_workers)]
                for w in range(num_workers):
                    next_obs, rewards, terms, truncs, infos = step_results[w]
                    worker_obs[w] = next_obs

                    # Tactical Reward Shaping against Heuristic Opponents
                    raw_r = float(rewards["player_0"]) * 0.1
                    p0_obs = learner_obs_list[w]
                    t_fwd, t_dist = float(p0_obs[22]), float(p0_obs[24])

                    # 1. Anti-Suicide Penalty: Heavily penalize moving forward towards an approaching predator
                    if t_dist > 0.001 and t_dist < 0.60 and t_fwd > 0.05:
                        raw_r -= 0.35 * (0.60 - t_dist) * t_fwd

                    # 2. Kill / Death feedback
                    if infos["player_0"]["kills"] > 0:
                        raw_r += 2.0  # Decisive reward for killing competent heuristic bots
                    if infos["player_0"]["deaths"] > 0:
                        raw_r -= 2.0  # Decisive penalty for getting caught

                    rewards_buf[step, w] = raw_r
                    dones_buf[step, w] = float(terms["player_0"] or truncs["player_0"])

                    iter_kills += infos["player_0"]["kills"]
                    iter_masses.append(infos["player_0"]["mass"])

                    worker_trajectories[w].append((
                        learner_obs_list[w],
                        learner_act_np[w],
                        infos["player_0"]["mass"],
                        infos["player_0"]["kills"]
                    ))

                    # Trajectory harvesting from high-performing rounds
                    if terms["player_0"] or truncs["player_0"] or len(worker_trajectories[w]) >= 200:
                        traj = worker_trajectories[w]
                        max_m = max(item[2] for item in traj) if traj else 0.0
                        tot_k = traj[-1][3] if traj else 0
                        if (tot_k >= 1 or max_m >= 32.0) and len(traj) >= 18:
                            for t_idx in range(len(traj) - 16):
                                expert_dataset_obs.append(traj[t_idx][0])
                                expert_dataset_acts.append([traj[t_idx + k][1] for k in range(16)])
                        worker_trajectories[w] = []

                global_step += num_workers

            # Compute GAE
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
                    loss = actor_loss + 0.15 * critic_loss - ent_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    optimizer.step()

                    epoch_a_loss += actor_loss.item()
                    epoch_c_loss += critic_loss.item()
                    epoch_ent += entropy.mean().item()
                    num_updates += 1

            scheduler.step()

            # Update checkpoint clone every 8 iterations
            if iteration % 8 == 0:
                checkpoint_net.load_state_dict(policy_net.state_dict())
                checkpoint_net.eval()

            # Telemetry
            elapsed = time.time() - start_time
            fps = global_step / max(0.1, elapsed)
            avg_rew = float(b_rewards.mean())
            avg_mass = float(np.mean(iter_masses)) if iter_masses else 0.0
            mean_a_loss = epoch_a_loss / max(1, num_updates)
            mean_c_loss = epoch_c_loss / max(1, num_updates)
            mean_ent = epoch_ent / max(1, num_updates)

            if iteration % 2 == 0 or global_step >= total_timesteps:
                print(f"{iteration:<5d} | {global_step:<8d} | {fps:<9.0f} step/s | {avg_rew:<8.3f} | {iter_kills:<6d} | {avg_mass:<10.1f} | {mean_a_loss:<9.4f} | {mean_c_loss:<9.4f} | {mean_ent:<8.4f}")

    finally:
        for r in remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in processes:
            p.join(timeout=1.0)

    # Save trained champion weights
    torch.save(policy_net.state_dict(), save_path)
    logger.info(f"Saved competitive PPO Champion to {save_path}")

    # Also save to projects folder path if different
    alt_path = "projects/sim2real-ppo-navigation/outputs/agar_ppo_champion.pt"
    if os.path.abspath(save_path) != os.path.abspath(alt_path):
        torch.save(policy_net.state_dict(), alt_path)

    # Save harvested expert rollouts
    if len(expert_dataset_obs) > 0:
        obs_arr = np.array(expert_dataset_obs, dtype=np.float32)
        act_arr = np.array(expert_dataset_acts, dtype=np.float32)
        rollout_path = "outputs/agar_clean_expert_rollouts.npz"
        np.savez_compressed(rollout_path, obs=obs_arr, acts=act_arr)
        np.savez_compressed("projects/sim2real-ppo-navigation/outputs/agar_clean_expert_rollouts.npz", obs=obs_arr, acts=act_arr)
        logger.info(f"Saved {len(obs_arr)} competitive expert trajectory chunks to {rollout_path}")


if __name__ == "__main__":
    train_competitive_ppo(total_timesteps=45_000)
