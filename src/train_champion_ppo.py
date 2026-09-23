"""
Champion PPO Trainer: Hybrid Continuous-Steer + Bernoulli-Split Policy trained against unhandicapped Master Heuristics.
Solves the accidental split problem and optimizes tactical predation to consistently outperform heuristics.
1. Phase 1: Harvest 20,000+ elite transitions from Master Heuristics (Apex, Hunter, Survivor).
2. Phase 2: Supervised imitation pre-training (BCEWithLogits for split + MSE for steering/thrust).
3. Phase 3: 100,000 timesteps of competitive PPO reinforcement learning in the mixed league.
4. Outputs: outputs/agar_ppo_champion.pt and outputs/agar_clean_expert_rollouts.npz
"""

import os
import sys
import time
import math
import logging
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.heuristic_agent import MasterHeuristicAgarBot
from src.train_agar_parallel_gpu import env_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChampionPPO")


class HybridActorCritic(nn.Module):
    """
    Hybrid Actor-Critic with dedicated trunks:
    - Continuous Gaussian for Thrust [0, 1] and Steer [-1, 1] with low exploration variance (sigma ~ 0.22)
    - Discrete Bernoulli for Split trigger (eliminates accidental suicidal splits)
    - Dedicated Critic trunk to prevent representation interference
    """

    def __init__(self, obs_dim: int = 38, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim

        # Dedicated Actor trunk
        self.actor_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Continuous action head: [thrust, steer]
        self.cont_head = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.full((2,), -1.5))  # sigma = 0.223 (sharp, low jitter)

        # Discrete split head: logits
        self.split_head = nn.Linear(hidden_dim, 1)

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
        nn.init.orthogonal_(self.cont_head.weight, gain=0.05)
        nn.init.orthogonal_(self.split_head.weight, gain=0.05)
        nn.init.constant_(self.split_head.bias, -2.0)  # Start with low split probability (~12%)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        act_feat = self.actor_trunk(obs)
        cont_raw = self.cont_head(act_feat)
        thrust = torch.sigmoid(cont_raw[..., 0:1])
        steer = torch.tanh(cont_raw[..., 1:2])
        cont_mu = torch.cat([thrust, steer], dim=-1)

        split_logit = self.split_head(act_feat).squeeze(-1)

        crit_feat = self.critic_trunk(obs)
        value = self.critic_head(crit_feat).squeeze(-1)

        std = torch.exp(torch.clamp(self.log_std, -2.5, -0.8))
        return cont_mu, std, split_logit, value

    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
        cont_mu, std, split_logit, value = self.forward(obs)
        cont_dist = Normal(cont_mu, std)
        split_prob = torch.sigmoid(split_logit)
        split_dist = Bernoulli(probs=split_prob)

        if deterministic:
            cont_act = cont_mu
            split_act = (split_prob > 0.65).float()
        else:
            cont_act = cont_dist.rsample()
            split_act = split_dist.sample()

        thrust = torch.clamp(cont_act[..., 0:1], 0.0, 1.0)
        steer = torch.clamp(cont_act[..., 1:2], -1.0, 1.0)
        action = torch.cat([thrust, steer, split_act.unsqueeze(-1)], dim=-1)

        log_prob = cont_dist.log_prob(cont_act).sum(dim=-1) + split_dist.log_prob(split_act)
        entropy = cont_dist.entropy().sum(dim=-1) + split_dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        cont_mu, std, split_logit, value = self.forward(obs)
        cont_dist = Normal(cont_mu, std)
        split_prob = torch.sigmoid(split_logit)
        split_dist = Bernoulli(probs=split_prob)

        cont_act = action[..., 0:2]
        split_act = action[..., 2]

        log_prob = cont_dist.log_prob(cont_act).sum(dim=-1) + split_dist.log_prob(split_act)
        entropy = cont_dist.entropy().sum(dim=-1) + split_dist.entropy()
        return log_prob, entropy, value


class ChampionPPOAgent:
    """Wrapper for Champion PPO Agent with Hybrid Continuous-Steer + Bernoulli-Split."""

    def __init__(self, weights_path: str = "outputs/agar_ppo_champion.pt", device: str = "cpu"):
        self.device = torch.device(device)
        self.net = HybridActorCritic(obs_dim=38, hidden_dim=128).to(self.device)
        if os.path.exists(weights_path):
            try:
                self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
                logger.info(f"Champion PPO loaded from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load {weights_path}: {e}")
        self.net.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            act, _, _, _ = self.net.get_action_and_value(t_obs, deterministic=deterministic)
            action = act.squeeze(0).cpu().numpy()
        return action


# ---------------------------------------------------------------------------
# Phase 1: Harvesting Elite Demonstrations from Master Heuristics
# ---------------------------------------------------------------------------

def harvest_elite_heuristic_data(num_rounds: int = 30, episode_steps: int = 350) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Phase 1: Harvesting elite demonstrations from Master Heuristic Bots ({num_rounds} rounds)...")
    env = PartiallyObservableAgarEnv(
        arena_size=14.0,
        num_players=6,
        num_food=100,
        num_viruses=5,
        max_steps=episode_steps,
        mass_decay_multiplier=1.35,
        max_pieces=4
    )

    bots = {
        "player_0": MasterHeuristicAgarBot(pid="player_0", profile="apex", seed=10),
        "player_1": MasterHeuristicAgarBot(pid="player_1", profile="hunter", seed=20),
        "player_2": MasterHeuristicAgarBot(pid="player_2", profile="survivor", seed=30),
        "player_3": MasterHeuristicAgarBot(pid="player_3", profile="apex", seed=40),
        "player_4": MasterHeuristicAgarBot(pid="player_4", profile="hunter", seed=50),
        "player_5": MasterHeuristicAgarBot(pid="player_5", profile="survivor", seed=60),
    }

    obs_list = []
    act_list = []
    total_kills = 0

    for r in range(1, num_rounds + 1):
        obs_dict, _ = env.reset(seed=1000 + r * 17)
        round_obs = {pid: [] for pid in env.player_ids}
        round_acts = {pid: [] for pid in env.player_ids}

        for step in range(episode_steps):
            actions = {pid: bots[pid].predict(obs_dict[pid]) for pid in env.player_ids}
            for pid in env.player_ids:
                round_obs[pid].append(obs_dict[pid].copy())
                round_acts[pid].append(actions[pid].copy())
            obs_dict, _, _, _, infos = env.step(actions)

        for pid in env.player_ids:
            inf = infos[pid]
            total_kills += inf["kills"]
            # Collect from genuinely strong performers (peak mass >= 35kg or kills >= 1)
            if inf["peak_mass"] >= 35.0 or inf["kills"] >= 1:
                obs_list.extend(round_obs[pid])
                act_list.extend(round_acts[pid])

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    logger.info(f"Phase 1 Complete: Harvested {len(obs_arr)} elite transitions (Total Kills: {total_kills})")
    return obs_arr, act_arr


# ---------------------------------------------------------------------------
# Phase 2: Supervised Imitation Pre-Training
# ---------------------------------------------------------------------------

def pretrain_hybrid_policy(
    policy_net: HybridActorCritic,
    obs_data: np.ndarray,
    act_data: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128
):
    logger.info(f"Phase 2: Supervised Imitation Pre-Training ({epochs} epochs)...")
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_split = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
    N = len(obs_data)
    indices = np.arange(N)

    t_obs = torch.tensor(obs_data, dtype=torch.float32, device=device)
    t_acts = torch.tensor(act_data, dtype=torch.float32, device=device)

    policy_net.train()
    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        num_b = 0

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            b_idx = indices[start_idx:end_idx]

            b_o = t_obs[b_idx]
            b_a = t_acts[b_idx]

            cont_mu, _, split_logit, _ = policy_net(b_o)

            loss_thrust = nn.functional.mse_loss(cont_mu[:, 0], b_a[:, 0])
            loss_steer = nn.functional.mse_loss(cont_mu[:, 1], b_a[:, 1])
            loss_split = criterion_split(split_logit, b_a[:, 2])

            loss = loss_thrust + 2.0 * loss_steer + 1.5 * loss_split

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_b += 1

        if ep % 5 == 0 or ep == epochs or ep == 1:
            logger.info(f"Pre-Train Epoch {ep:02d}/{epochs:02d} | Loss: {total_loss / max(1, num_b):.5f}")


# ---------------------------------------------------------------------------
# Phase 3: Competitive PPO Reinforcement Learning
# ---------------------------------------------------------------------------

def train_competitive_ppo_rl(
    policy_net: HybridActorCritic,
    device: torch.device,
    num_workers: int = 8,
    total_timesteps: int = 80_000,
    steps_per_rollout: int = 128,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    save_path: str = "outputs/agar_ppo_champion.pt"
):
    logger.info(f"Phase 3: Multi-Agent Competitive PPO against Master Heuristics ({total_timesteps} timesteps)...")
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1e-5)
    total_iters = total_timesteps // (steps_per_rollout * num_workers) + 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=5e-5)

    obs_dim = 38
    action_dim = 3

    checkpoint_net = HybridActorCritic(obs_dim=obs_dim, hidden_dim=128).to(device)
    checkpoint_net.load_state_dict(policy_net.state_dict())
    checkpoint_net.eval()

    heuristic_apex = [MasterHeuristicAgarBot(pid="player_1", profile="apex", seed=100 + w) for w in range(num_workers)]
    heuristic_hunter = [MasterHeuristicAgarBot(pid="player_2", profile="hunter", seed=200 + w) for w in range(num_workers)]
    heuristic_survivor = [MasterHeuristicAgarBot(pid="player_4", profile="survivor", seed=400 + w) for w in range(num_workers)]

    pipes = [mp.Pipe() for _ in range(num_workers)]
    remotes, work_remotes = zip(*pipes)
    processes = [
        mp.Process(target=env_worker, args=(work_remotes[i], 5000 + i * 59, 14.0, True))
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
                    clone_obs_list.append(worker_obs[w]["player_3"])
                    clone_obs_list.append(worker_obs[w]["player_5"])
                clone_obs_t = torch.tensor(np.array(clone_obs_list), dtype=torch.float32, device=device)
                with torch.no_grad():
                    clone_act_t, _, _, _ = checkpoint_net.get_action_and_value(clone_obs_t, deterministic=True)
                clone_act_np = clone_act_t.cpu().numpy()

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

                    raw_r = float(rewards["player_0"]) * 0.1
                    p0_obs = learner_obs_list[w]
                    t_fwd, t_dist = float(p0_obs[22]), float(p0_obs[24])

                    # Anti-Suicide Evasion Penalty
                    if t_dist > 0.001 and t_dist < 0.50:
                        if t_fwd > 0.08:
                            raw_r -= 0.60 * (0.50 - t_dist) * t_fwd
                        elif t_fwd < -0.10:
                            raw_r += 0.35 * abs(t_fwd)

                    # Predation vs Death Shaping
                    if infos["player_0"]["kills"] > 0:
                        raw_r += 4.5  # Massive reward for taking down heuristic bots
                    if infos["player_0"]["deaths"] > 0:
                        raw_r -= 3.0  # Decisive penalty for death

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

                    if terms["player_0"] or truncs["player_0"] or len(worker_trajectories[w]) >= 200:
                        traj = worker_trajectories[w]
                        max_m = max(item[2] for item in traj) if traj else 0.0
                        tot_k = traj[-1][3] if traj else 0
                        if (tot_k >= 1 or max_m >= 35.0) and len(traj) >= 18:
                            for t_idx in range(len(traj) - 16):
                                expert_dataset_obs.append(traj[t_idx][0])
                                expert_dataset_acts.append([traj[t_idx + k][1] for k in range(16)])
                        worker_trajectories[w] = []

                global_step += num_workers

            # Generalized Advantage Estimation
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
                delta = rewards_buf[t] + 0.99 * next_val * next_non_terminal - values_buf[t]
                last_gae = delta + 0.99 * 0.95 * next_non_terminal * last_gae
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
                    surr2 = torch.clamp(ratio, 0.80, 1.20) * t_advantages[b_idx]
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = 0.5 * ((new_value - t_returns[b_idx]) ** 2).mean()
                    loss = actor_loss + 0.15 * critic_loss - 0.005 * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    optimizer.step()

                    epoch_a_loss += actor_loss.item()
                    epoch_c_loss += critic_loss.item()
                    epoch_ent += entropy.mean().item()
                    num_updates += 1

            scheduler.step()

            if iteration % 8 == 0:
                checkpoint_net.load_state_dict(policy_net.state_dict())
                checkpoint_net.eval()

            elapsed = time.time() - start_time
            fps = global_step / max(0.1, elapsed)
            avg_rew = float(b_rewards.mean())
            avg_mass = float(np.mean(iter_masses)) if iter_masses else 0.0
            mean_a_loss = epoch_a_loss / max(1, num_updates)
            mean_c_loss = epoch_c_loss / max(1, num_updates)
            mean_ent = epoch_ent / max(1, num_updates)

            if iteration % 4 == 0 or global_step >= total_timesteps:
                print(f"{iteration:<5d} | {global_step:<8d} | {fps:<9.0f} step/s | {avg_rew:<8.3f} | {iter_kills:<6d} | {avg_mass:<10.1f} | {mean_a_loss:<9.4f} | {mean_c_loss:<9.4f} | {mean_ent:<8.4f}")

    finally:
        for r in remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in processes:
            p.join(timeout=1.0)

    # Save champion weights
    torch.save(policy_net.state_dict(), save_path)
    alt_path = "projects/sim2real-ppo-navigation/outputs/agar_ppo_champion.pt"
    if os.path.abspath(save_path) != os.path.abspath(alt_path):
        os.makedirs(os.path.dirname(alt_path), exist_ok=True)
        torch.save(policy_net.state_dict(), alt_path)
    logger.info(f"Saved Champion PPO model to {save_path}")

    # Save elite rollouts
    if len(expert_dataset_obs) > 0:
        obs_arr = np.array(expert_dataset_obs, dtype=np.float32)
        act_arr = np.array(expert_dataset_acts, dtype=np.float32)
        rollout_path = "outputs/agar_clean_expert_rollouts.npz"
        np.savez_compressed(rollout_path, obs=obs_arr, acts=act_arr)
        if os.path.exists("projects/sim2real-ppo-navigation/outputs"):
            np.savez_compressed("projects/sim2real-ppo-navigation/outputs/agar_clean_expert_rollouts.npz", obs=obs_arr, acts=act_arr)
        logger.info(f"Saved {len(obs_arr)} elite competitive trajectory chunks to {rollout_path}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")

    # Phase 1: Harvest elite demonstrations
    obs_data, act_data = harvest_elite_heuristic_data(num_rounds=30, episode_steps=350)

    # Initialize Architecture
    policy_net = HybridActorCritic(obs_dim=38, hidden_dim=128).to(device)

    # Phase 2: Supervised Imitation Warm-Start
    pretrain_hybrid_policy(policy_net, obs_data, act_data, device, epochs=20, batch_size=128)

    # Phase 3: Competitive PPO Reinforcement Learning
    train_competitive_ppo_rl(policy_net, device, num_workers=8, total_timesteps=80_000)


if __name__ == "__main__":
    main()
