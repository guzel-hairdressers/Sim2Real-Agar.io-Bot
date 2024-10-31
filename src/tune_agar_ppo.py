"""
Automated Hyperparameter Sweep for Vectorized Agar.io PPO Self-Play.
Sweeps across:
- Learning rates: [2e-4, 4e-4, 8e-4]
- Entropy bonus coefficients: [0.005, 0.015, 0.035]
- Clip epsilon: [0.15, 0.20]
Evaluates on Apple Silicon GPU / MPS with vectorized GAE across parallel arenas.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_agar_parallel_gpu import TorchAgarActorCritic, env_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PPOTune")


def evaluate_hparam_trial(
    lr: float,
    entropy_coef: float,
    clip_eps: float,
    total_steps: int = 6000,
    steps_per_rollout: int = 150,
    num_workers: int = 4,
    device_name: str = "mps"
) -> Dict[str, float]:
    device = torch.device(device_name if (device_name == "mps" and torch.backends.mps.is_available()) else "cpu")

    policy_net = TorchAgarActorCritic(obs_dim=36, action_dim=2, hidden_dim=128).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)

    # Launch worker processes
    pipes = [mp.Pipe() for _ in range(num_workers)]
    remotes, work_remotes = zip(*pipes)
    processes = [
        mp.Process(target=env_worker, args=(work_remotes[i], 500 + i * 23, 14.0))
        for i in range(num_workers)
    ]
    for p in processes:
        p.daemon = True
        p.start()

    worker_obs = [remote.recv() for remote in remotes]

    global_step = 0
    gamma = 0.99
    gae_lambda = 0.95
    ppo_epochs = 4
    batch_size = 64

    total_kills = 0
    peak_mass_achieved = 15.0
    total_rewards = 0.0
    reward_samples = 0

    try:
        while global_step < total_steps:
            obs_buf = np.zeros((steps_per_rollout, num_workers, 36), dtype=np.float32)
            acts_buf = np.zeros((steps_per_rollout, num_workers, 2), dtype=np.float32)
            logprobs_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            rewards_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            dones_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)
            values_buf = np.zeros((steps_per_rollout, num_workers), dtype=np.float32)

            for step in range(steps_per_rollout):
                learner_obs = np.array([worker_obs[w]["player_0"] for w in range(num_workers)], dtype=np.float32)
                learner_obs_t = torch.tensor(learner_obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    learner_act_t, learner_lp_t, _, learner_val_t = policy_net.get_action_and_value(learner_obs_t)

                learner_act_np = learner_act_t.cpu().numpy()
                learner_lp_np = learner_lp_t.cpu().numpy()
                learner_val_np = learner_val_t.cpu().numpy()

                obs_buf[step] = learner_obs
                acts_buf[step] = learner_act_np
                logprobs_buf[step] = learner_lp_np
                values_buf[step] = learner_val_np

                for w in range(num_workers):
                    # Simplified self-play clones
                    w_actions = {
                        "player_0": learner_act_np[w],
                        "player_1": np.random.uniform(-0.5, 0.5, size=2).astype(np.float32),
                        "player_2": np.random.uniform(-0.5, 0.5, size=2).astype(np.float32),
                        "player_3": np.array([0.7, 0.0], dtype=np.float32),
                        "player_4": np.array([0.7, 0.0], dtype=np.float32),
                        "player_5": np.array([0.7, 0.0], dtype=np.float32),
                    }
                    remotes[w].send(("step", w_actions))

                step_results = [remotes[w].recv() for w in range(num_workers)]
                for w in range(num_workers):
                    next_obs, rewards, terms, truncs, infos = step_results[w]
                    worker_obs[w] = next_obs
                    rewards_buf[step, w] = rewards["player_0"]
                    dones_buf[step, w] = float(terms["player_0"] or truncs["player_0"])

                    total_kills += infos["player_0"]["kills"]
                    m = infos["player_0"]["mass"]
                    if m > peak_mass_achieved:
                        peak_mass_achieved = m
                    total_rewards += rewards["player_0"]
                    reward_samples += 1

                global_step += num_workers

            # Vectorized GAE per worker column
            last_obs = np.array([worker_obs[w]["player_0"] for w in range(num_workers)], dtype=np.float32)
            last_obs_t = torch.tensor(last_obs, dtype=torch.float32, device=device)
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
            b_obs = obs_buf.reshape(-1, 36)
            b_actions = acts_buf.reshape(-1, 2)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages_buf.reshape(-1)
            b_returns = returns_buf.reshape(-1)

            t_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
            t_actions = torch.tensor(b_actions, dtype=torch.float32, device=device)
            t_logprobs = torch.tensor(b_logprobs, dtype=torch.float32, device=device)
            t_advantages = torch.tensor(b_advantages, dtype=torch.float32, device=device)
            t_returns = torch.tensor(b_returns, dtype=torch.float32, device=device)

            t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

            dataset_size = len(b_obs)
            indices = np.arange(dataset_size)

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

                    loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                    optimizer.step()

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

    avg_reward = total_rewards / max(1, reward_samples)
    return {
        "lr": lr,
        "entropy_coef": entropy_coef,
        "clip_eps": clip_eps,
        "avg_reward": float(avg_reward),
        "total_kills": int(total_kills),
        "peak_mass": float(peak_mass_achieved)
    }


def run_hyperparameter_sweep() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING AGAR.IO PPO HYPERPARAMETER SWEEP (PARALLEL VECTORIZED GAE)")
    print("=" * 80)

    trials = [
        {"lr": 2e-4, "entropy_coef": 0.010, "clip_eps": 0.15},
        {"lr": 4e-4, "entropy_coef": 0.015, "clip_eps": 0.20},
        {"lr": 4e-4, "entropy_coef": 0.005, "clip_eps": 0.15},
        {"lr": 8e-4, "entropy_coef": 0.025, "clip_eps": 0.20},
    ]

    results = []
    print(f"{'Trial':<6} | {'LR':<8} | {'Entropy':<8} | {'Clip':<6} | {'Avg Reward':<12} | {'Kills':<6} | {'Peak Mass':<10}")
    print("-" * 80)

    for i, t in enumerate(trials):
        t0 = time.time()
        res = evaluate_hparam_trial(
            lr=t["lr"],
            entropy_coef=t["entropy_coef"],
            clip_eps=t["clip_eps"],
            total_steps=5000,
            num_workers=4
        )
        elapsed = time.time() - t0
        results.append(res)
        print(f"{i+1:<6d} | {res['lr']:<8.1e} | {res['entropy_coef']:<8.3f} | {res['clip_eps']:<6.2f} | {res['avg_reward']:<12.3f} | {res['total_kills']:<6d} | {res['peak_mass']:<10.1f} (in {elapsed:.1f}s)")

    # Rank by composite score
    best_trial = max(results, key=lambda r: r["peak_mass"] * 0.5 + r["total_kills"] * 5.0 + r["avg_reward"] * 20.0)

    print("=" * 80)
    print(f"BEST HYPERPARAMETERS FOUND: LR={best_trial['lr']}, Entropy={best_trial['entropy_coef']}, Clip={best_trial['clip_eps']}")
    print(f"Performance: Peak Mass={best_trial['peak_mass']} kg, Kills={best_trial['total_kills']}, Avg Reward={best_trial['avg_reward']:.3f}")
    print("=" * 80)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ppo_hparam_results.json", "w") as f:
        json.dump({"best": best_trial, "all_trials": results}, f, indent=2)

    return best_trial


if __name__ == "__main__":
    run_hyperparameter_sweep()
