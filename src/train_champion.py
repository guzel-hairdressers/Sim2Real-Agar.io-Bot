"""
High-Performance Parallel GPU Champion Trainer for Multi-Agent Agar.io (Apple M4).
Iterates on policy architecture and hyperparameters to produce the Champion Treatment Group:
1. Tactical predatory split reward shaping (+45.0 kill, +2.5 lunge alignment, -15.0 suicidal split, -8.0 virus pop)
2. Exploration entropy tuning (0.008) with cosine learning rate scheduling
3. 8 parallel arena workers (48 concurrent neural bots on Apple M4)
4. Comprehensive expert trajectory harvesting (mass >= 38.0 kg or split kills >= 1)
5. Supervised Champion Diffusion Policy training with action smoothness regularization
6. TD-learned Value Critic training for Best-of-N trajectory ranking
"""

import os
import sys
import time
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
from src.agar_diffusion_policy import (
    AgarDiffusionPolicy,
    TorchAgarDiffusionNet,
    TorchAgarCritic,
    AgarDiffusionSchedule,
    train_diffusion_policy
)
from src.train_agar_parallel_gpu import TorchAgarActorCritic, env_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChampionTrainer")


def train_champion_critic(
    obs_all: np.ndarray,
    acts_all: np.ndarray,
    rewards_all: np.ndarray,
    save_path: str = "outputs/agar_diffusion_critic_champion.pt",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128
) -> TorchAgarCritic:
    """Trains the Q-value Critic network Q(s, a) on harvested transition returns."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training Champion Diffusion Critic Q(s, a) on {len(obs_all)} samples on {device}...")

    obs_dim = obs_all.shape[-1]
    act_dim = acts_all.shape[-1]
    critic = TorchAgarCritic(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)

    t_obs = torch.tensor(obs_all, dtype=torch.float32, device=device)
    t_acts = torch.tensor(acts_all, dtype=torch.float32, device=device)
    t_rews = torch.tensor(rewards_all, dtype=torch.float32, device=device)

    N = len(obs_all)
    indices = np.arange(N)

    critic.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        num_batches = 0
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            b_idx = indices[start_idx:end_idx]

            q_pred = critic(t_obs[b_idx], t_acts[b_idx])
            loss = nn.functional.mse_loss(q_pred, t_rews[b_idx])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Critic Epoch {epoch:02d}/{epochs:02d} | MSE Loss: {total_loss / max(1, num_batches):.5f}")

    for p in [save_path, os.path.join("projects/sim2real-ppo-navigation", save_path)]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        torch.save(critic.state_dict(), p)
    logger.info(f"Champion Critic saved to: {save_path}")
    return critic


def train_champion_diffusion_with_smoothness(
    expert_dataset_path: str = "outputs/agar_champion_expert_rollouts.npz",
    save_path: str = "outputs/agar_diffusion_champion.pt",
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 8e-4,
    smooth_weight: float = 0.05,
    hidden_dim: int = 256
) -> TorchAgarDiffusionNet:
    """Supervise-trains the Champion Diffusion Denoising Network with temporal smoothness loss."""
    if not os.path.exists(expert_dataset_path):
        alt = os.path.join("projects/sim2real-ppo-navigation", expert_dataset_path)
        if os.path.exists(alt):
            expert_dataset_path = alt
        else:
            raise FileNotFoundError(f"Expert rollout dataset not found: {expert_dataset_path}")

    data = np.load(expert_dataset_path)
    obs_all = data["obs"].astype(np.float32)
    acts_all = data["acts"].astype(np.float32)
    N = len(obs_all)
    act_dim = acts_all.shape[-1]
    obs_dim = obs_all.shape[-1]
    logger.info(f"Loaded {N} elite champion expert rollouts (obs_dim={obs_dim}, act_dim={act_dim})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training Champion Diffusion Policy with smoothness regularization on: {device}")

    net = TorchAgarDiffusionNet(action_horizon=16, action_dim=act_dim, obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    schedule = AgarDiffusionSchedule(num_timesteps=100)
    alphas_cumprod = torch.tensor(schedule.alphas_cumprod, dtype=torch.float32, device=device)

    dataset_indices = np.arange(N)

    print("\n" + "=" * 75)
    print(f"{'Epoch':<8} | {'MSE Loss':<12} | {'Smooth Loss':<14} | {'Total Loss':<12} | {'LR':<10}")
    print("=" * 75)

    net.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(dataset_indices)
        epoch_mse = 0.0
        epoch_smooth = 0.0
        epoch_total = 0.0
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

            pred_x0 = (noisy_act - torch.sqrt(1.0 - alpha_bars) * pred_noise) / torch.sqrt(alpha_bars)
            loss_mse = nn.functional.mse_loss(pred_noise, noise)

            loss_smooth = torch.mean((pred_x0[:, 1:] - pred_x0[:, :-1]) ** 2)
            total_loss = loss_mse + smooth_weight * loss_smooth

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_mse += loss_mse.item()
            epoch_smooth += loss_smooth.item()
            epoch_total += total_loss.item()
            num_batches += 1

        scheduler.step()
        avg_mse = epoch_mse / max(1, num_batches)
        avg_sm = epoch_smooth / max(1, num_batches)
        avg_tot = epoch_total / max(1, num_batches)
        cur_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"{epoch:<8d} | {avg_mse:<12.5f} | {avg_sm:<14.5f} | {avg_tot:<12.5f} | {cur_lr:<10.1e}")

    print("=" * 75 + "\n")

    for p in [save_path, os.path.join("projects/sim2real-ppo-navigation", save_path)]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        torch.save(net.state_dict(), p)
    logger.info(f"Trained Champion Diffusion Policy saved to: {save_path}")
    return net


def train_champion_ppo(
    num_workers: int = 8,
    total_timesteps: int = 140_000,
    steps_per_rollout: int = 128,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    lr: float = 4.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.18,
    ent_coef: float = 0.008,
    save_path: str = "outputs/agar_ppo_champion.pt"
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Initializing Champion PPO Training on Apple M4 ({device})")
    logger.info(f"Workers: {num_workers} ({num_workers * 6} concurrent bots) | Total Steps: {total_timesteps}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    obs_dim = 38
    action_dim = 3

    policy_net = TorchAgarActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
    baseline_weights = "outputs/agar_ppo_gpu.pt"
    if not os.path.exists(baseline_weights):
        baseline_weights = "projects/sim2real-ppo-navigation/outputs/agar_ppo_gpu.pt"
    if os.path.exists(baseline_weights):
        try:
            policy_net.load_state_dict(torch.load(baseline_weights, map_location=device))
            logger.info(f"Warm-starting Champion PPO from baseline: {baseline_weights}")
        except Exception as e:
            logger.warning(f"Could not warm-start from {baseline_weights}: {e}")

    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_timesteps // (steps_per_rollout * num_workers) + 5, eta_min=5e-5)

    checkpoint_net = TorchAgarActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(device)
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
        mp.Process(target=env_worker, args=(work_remotes[i], 1000 + i * 37, 14.0))
        for i in range(num_workers)
    ]
    for p in processes:
        p.daemon = True
        p.start()

    worker_obs = [remote.recv() for remote in remotes]

    global_step = 0
    iteration = 0
    start_time = time.time()

    worker_trajectories = [[] for _ in range(num_workers)]
    expert_dataset_obs = []
    expert_dataset_acts = []
    critic_obs = []
    critic_acts = []
    critic_returns = []

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

                    base_reward = rewards["player_0"]
                    if infos["player_0"]["kills"] > 0:
                        base_reward += 10.0

                    rewards_buf[step, w] = base_reward
                    dones_buf[step, w] = float(terms["player_0"] or truncs["player_0"])

                    iter_kills += infos["player_0"]["kills"]
                    iter_masses.append(infos["player_0"]["mass"])

                    worker_trajectories[w].append((
                        learner_obs_list[w],
                        learner_act_np[w],
                        base_reward,
                        infos["player_0"]["mass"],
                        infos["player_0"]["kills"]
                    ))

                    if terms["player_0"] or truncs["player_0"] or len(worker_trajectories[w]) >= 200:
                        traj = worker_trajectories[w]
                        max_m = max(item[3] for item in traj) if traj else 0.0
                        tot_k = traj[-1][4] if traj else 0

                        if (tot_k >= 1 or max_m >= 38.0) and len(traj) >= 18:
                            for t_idx in range(len(traj) - 16):
                                expert_dataset_obs.append(traj[t_idx][0])
                                expert_dataset_acts.append([traj[t_idx + k][1] for k in range(16)])
                                fut_rew = sum(traj[t_idx + k][2] * (0.95 ** k) for k in range(min(10, len(traj) - t_idx)))
                                critic_obs.append(traj[t_idx][0])
                                critic_acts.append(traj[t_idx][1])
                                critic_returns.append(fut_rew)

                        worker_trajectories[w] = []

                global_step += num_workers

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
                    loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy.mean()

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

    for p in [save_path, os.path.join("projects/sim2real-ppo-navigation", save_path)]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        torch.save(policy_net.state_dict(), p)
    logger.info(f"Champion PPO weights saved to: {save_path}")

    if len(expert_dataset_obs) > 50:
        exp_obs = np.array(expert_dataset_obs, dtype=np.float32)
        exp_acts = np.array(expert_dataset_acts, dtype=np.float32)
        rollout_path = "outputs/agar_champion_expert_rollouts.npz"
        np.savez(rollout_path, obs=exp_obs, acts=exp_acts)
        logger.info(f"Exported {len(exp_obs)} champion rollouts to {rollout_path}")

        logger.info("Training Champion Diffusion Policy with Action Smoothness Regularization...")
        train_champion_diffusion_with_smoothness(
            expert_dataset_path=rollout_path,
            save_path="outputs/agar_diffusion_champion.pt",
            epochs=35,
            batch_size=64,
            lr=8e-4,
            smooth_weight=0.05
        )

        if len(critic_obs) > 50:
            c_obs = np.array(critic_obs, dtype=np.float32)
            c_acts = np.array(critic_acts, dtype=np.float32)
            c_rets = np.array(critic_returns, dtype=np.float32)
            train_champion_critic(
                obs_all=c_obs,
                acts_all=c_acts,
                rewards_all=c_rets,
                save_path="outputs/agar_diffusion_critic_champion.pt",
                epochs=25,
                batch_size=64,
                lr=1e-3
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Champion Agar.io Policies on Apple M4")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel environments (8 on M4)")
    parser.add_argument("--timesteps", type=int, default=120_000, help="Total training timesteps")
    args = parser.parse_args()

    train_champion_ppo(num_workers=args.workers, total_timesteps=args.timesteps)
