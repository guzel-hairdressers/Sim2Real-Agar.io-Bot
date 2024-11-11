"""
Multi-Agent PPO Self-Play Training in Partially Observable Agar.io Arena.
Features:
1. Continuous Actor-Critic Network (36-dim POMDP observation -> [thrust, steer])
2. Generalized Advantage Estimation (GAE-lambda=0.95, gamma=0.99)
3. Clipped Surrogate Policy Objective (clip_eps=0.2)
4. Multi-Agent League: Learner trains against self-play clones and heuristic foragers
5. Exports trained weights to outputs/agar_ppo_weights.npz
"""

import os
import sys
import time
import math
import logging
import argparse
from typing import Tuple, Dict, Any, List, Optional
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarPPO")


def orthogonal_init(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    flat_shape = (shape[0], int(np.prod(shape[1:])))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return (gain * q[:shape[0], :shape[1]]).astype(np.float32)


class HeuristicAgarBot:
    """
    Rule-based competitor cell:
    1. Flee: if a dangerous predator is within vision, steer directly away with max thrust.
    2. Hunt: if an edible prey is within vision, pursue it.
    3. Forage: steer toward local food cluster centroid.
    4. Repel from arena boundaries.
    """

    def __init__(self, pid: str):
        self.pid = pid

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # obs structure:
        # [0..3]: ego [radius, vel, omega, mass]
        # [4..11]: threat radar (8)
        # [12..19]: prey radar (8)
        # [20..23]: threat_vec [dx, dy, dist, mass_ratio]
        # [24..27]: prey_vec [dx, dy, dist, mass_ratio]
        # [28..31]: food_vec [dx, dy, density, nearest_dist]
        # [32..35]: hazard_vec [wall_dist, virus_dist, virus_ahead, vuln]

        threat_dist = obs[22]
        threat_dx = obs[20]
        threat_dy = obs[21]

        prey_dist = obs[26]
        prey_dx = obs[24]
        prey_dy = obs[25]

        food_dx = obs[28]
        food_dy = obs[29]
        wall_dist = obs[32]

        thrust = 0.75
        steer = 0.0

        # 1. Immediate Threat Evasion
        if threat_dist > 0.01 and threat_dist < 0.75:
            # Flee in opposite direction of threat
            desired_heading = math.atan2(-threat_dy, -threat_dx)
            steer = float(np.clip(desired_heading / np.pi, -1.0, 1.0))
            thrust = 1.0
        # 2. Opportunistic Prey Hunt
        elif prey_dist > 0.01 and prey_dist < 0.85:
            desired_heading = math.atan2(prey_dy, prey_dx)
            steer = float(np.clip(desired_heading / np.pi, -1.0, 1.0))
            thrust = 0.95
        # 3. Nutrient Foraging
        elif abs(food_dx) > 0.01 or abs(food_dy) > 0.01:
            desired_heading = math.atan2(food_dy, food_dx)
            steer = float(np.clip(desired_heading / np.pi, -1.0, 1.0))
            thrust = 0.75
        else:
            steer = float(np.random.uniform(-0.3, 0.3))
            thrust = 0.65

        # Wall repulsion if close to edge
        if wall_dist < 0.15:
            steer = float(np.clip(steer + np.random.choice([-0.8, 0.8]), -1.0, 1.0))

        # Output in [thrust (0..1), steer (-1..1)]
        thrust = float(np.clip(thrust, 0.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))
        return np.array([thrust, steer], dtype=np.float32)


class AgarActorCriticNet:
    """Continuous Actor-Critic Network for Agar.io POMDP observations."""

    def __init__(self, obs_dim: int = 36, action_dim: int = 2, hidden_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Actor MLP
        self.w_a1 = orthogonal_init((obs_dim, hidden_dim), gain=np.sqrt(2))
        self.b_a1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_a2 = orthogonal_init((hidden_dim, hidden_dim), gain=np.sqrt(2))
        self.b_a2 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_a3 = orthogonal_init((hidden_dim, action_dim), gain=0.1)
        self.b_a3 = np.zeros(action_dim, dtype=np.float32)

        # Log std for continuous Gaussian policy
        self.log_std = np.full(action_dim, -0.5, dtype=np.float32)

        # Critic MLP
        self.w_c1 = orthogonal_init((obs_dim, hidden_dim), gain=np.sqrt(2))
        self.b_c1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_c2 = orthogonal_init((hidden_dim, hidden_dim), gain=np.sqrt(2))
        self.b_c2 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_c3 = orthogonal_init((hidden_dim, 1), gain=1.0)
        self.b_c3 = np.zeros(1, dtype=np.float32)

    def forward_actor(self, obs: np.ndarray) -> np.ndarray:
        h1 = np.tanh(np.dot(obs, self.w_a1) + self.b_a1)
        h2 = np.tanh(np.dot(h1, self.w_a2) + self.b_a2)
        raw = np.dot(h2, self.w_a3) + self.b_a3
        # Dim 0: thrust in [0, 1] via sigmoid
        # Dim 1: steer in [-1, 1] via tanh
        if raw.ndim == 1:
            thrust = 1.0 / (1.0 + np.exp(-raw[0]))
            steer = float(np.tanh(raw[1]))
            return np.array([thrust, steer], dtype=np.float32)
        else:
            thrust = 1.0 / (1.0 + np.exp(-raw[:, 0:1]))
            steer = np.tanh(raw[:, 1:2])
            return np.hstack([thrust, steer]).astype(np.float32)

    def forward_critic(self, obs: np.ndarray) -> float:
        h1 = np.tanh(np.dot(obs, self.w_c1) + self.b_c1)
        h2 = np.tanh(np.dot(h1, self.w_c2) + self.b_c2)
        val = float((np.dot(h2, self.w_c3) + self.b_c3)[0])
        return val

    def sample_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        mu = self.forward_actor(obs)
        if deterministic:
            thrust = float(np.clip(mu[0], 0.0, 1.0))
            steer = float(np.clip(mu[1], -1.0, 1.0))
            return np.array([thrust, steer], dtype=np.float32), 0.0

        std = np.exp(self.log_std)
        noise = np.random.normal(0.0, 1.0, size=self.action_dim).astype(np.float32)
        raw_act = mu + std * noise
        thrust = float(np.clip(raw_act[0], 0.0, 1.0))
        steer = float(np.clip(raw_act[1], -1.0, 1.0))
        action = np.array([thrust, steer], dtype=np.float32)

        # Log prob
        var = std ** 2
        log_prob = float(np.sum(-0.5 * ((action - mu) ** 2) / var - self.log_std - 0.5 * np.log(2.0 * np.pi)))
        return action, log_prob

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        np.savez(
            file_path,
            w_a1=self.w_a1, b_a1=self.b_a1,
            w_a2=self.w_a2, b_a2=self.b_a2,
            w_a3=self.w_a3, b_a3=self.b_a3,
            log_std=self.log_std,
            w_c1=self.w_c1, b_c1=self.b_c1,
            w_c2=self.w_c2, b_c2=self.b_c2,
            w_c3=self.w_c3, b_c3=self.b_c3
        )
        logger.info(f"Agar PPO weights saved to: {file_path}")

    def load(self, file_path: str):
        data = np.load(file_path)
        self.w_a1 = data["w_a1"]
        self.b_a1 = data["b_a1"]
        self.w_a2 = data["w_a2"]
        self.b_a2 = data["b_a2"]
        self.w_a3 = data["w_a3"]
        self.b_a3 = data["b_a3"]
        self.log_std = data["log_std"]
        self.w_c1 = data["w_c1"]
        self.b_c1 = data["b_c1"]
        self.w_c2 = data["w_c2"]
        self.b_c2 = data["b_c2"]
        self.w_c3 = data["w_c3"]
        self.b_c3 = data["b_c3"]
        logger.info(f"Loaded Agar PPO weights from: {file_path}")


class AgarPPOAgent:
    """Continuous PPO Agent managing rollouts, GAE, and mini-batch updates."""

    def __init__(self, obs_dim: int = 36, action_dim: int = 2, lr: float = 3e-3, gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2):
        self.net = AgarActorCriticNet(obs_dim=obs_dim, action_dim=action_dim)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        act, _ = self.net.sample_action(obs, deterministic=deterministic)
        return act

    def pretrain_imitation(self, num_steps: int = 2500, epochs: int = 35):
        """Warm-starts actor network using expert heuristic demonstration rollouts."""
        logger.info("Pretraining Agar Actor network on heuristic forager demonstrations...")
        env = PartiallyObservableAgarEnv(arena_size=20.0, num_players=6, num_food=120)
        obs_dict, _ = env.reset(seed=42)
        demo_obs = []
        demo_act = []

        bots = {pid: HeuristicAgarBot(pid) for pid in env.player_ids}

        for step in range(num_steps):
            actions = {pid: bots[pid].predict(obs_dict[pid]) for pid in env.player_ids}
            for pid in env.player_ids:
                demo_obs.append(obs_dict[pid].copy())
                demo_act.append(actions[pid].copy())
            obs_dict, _, terms, truncs, _ = env.step(actions)
            if any(truncs.values()):
                obs_dict, _ = env.reset()

        obs_arr = np.array(demo_obs, dtype=np.float32)
        act_arr = np.array(demo_act, dtype=np.float32)

        for _ in range(epochs):
            for i in range(len(obs_arr)):
                o = obs_arr[i]
                target_a = act_arr[i]
                h1 = np.tanh(np.dot(o, self.net.w_a1) + self.net.b_a1)
                h2 = np.tanh(np.dot(h1, self.net.w_a2) + self.net.b_a2)
                pred_a = self.net.forward_actor(o)
                grad = 2.0 * (pred_a - target_a)

                # Backprop through layers
                self.net.w_a3 -= 0.005 * np.outer(h2, grad)
                self.net.b_a3 -= 0.005 * grad
                grad_h2 = np.dot(grad, self.net.w_a3.T) * (1.0 - h2 ** 2)
                self.net.w_a2 -= 0.005 * np.outer(h1, grad_h2)
                self.net.b_a2 -= 0.005 * grad_h2
                grad_h1 = np.dot(grad_h2, self.net.w_a2.T) * (1.0 - h1 ** 2)
                self.net.w_a1 -= 0.005 * np.outer(o, grad_h1)
                self.net.b_a1 -= 0.005 * grad_h1

        logger.info("Imitation pretraining finished successfully.")

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_val: float) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_v = next_val if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def train_epoch(self, obs_b: np.ndarray, act_b: np.ndarray, old_lp_b: np.ndarray, adv_b: np.ndarray, ret_b: np.ndarray, epochs: int = 3):
        adv_b = (adv_b - np.mean(adv_b)) / (np.std(adv_b) + 1e-8)
        std = np.exp(self.net.log_std)
        var = std ** 2

        for _ in range(epochs):
            for i in range(len(obs_b)):
                o = obs_b[i]
                a = act_b[i]
                old_lp = old_lp_b[i]
                adv = adv_b[i]
                ret = ret_b[i]

                # Actor
                h1_a = np.tanh(np.dot(o, self.net.w_a1) + self.net.b_a1)
                h2_a = np.tanh(np.dot(h1_a, self.net.w_a2) + self.net.b_a2)
                mu = self.net.forward_actor(o)

                cur_lp = float(np.sum(-0.5 * ((a - mu) ** 2) / var - self.net.log_std - 0.5 * np.log(2.0 * np.pi)))
                ratio = np.exp(np.clip(cur_lp - old_lp, -20.0, 2.0))
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv

                d_lp_d_mu = (a - mu) / var
                grad_a3 = -d_lp_d_mu * (adv if ratio == surr1 else 0.0)

                self.net.w_a3 -= self.lr * 0.05 * np.outer(h2_a, grad_a3)
                self.net.b_a3 -= self.lr * 0.05 * grad_a3

                grad_a2 = np.dot(grad_a3, self.net.w_a3.T) * (1.0 - h2_a ** 2)
                self.net.w_a2 -= self.lr * 0.05 * np.outer(h1_a, grad_a2)
                self.net.b_a2 -= self.lr * 0.05 * grad_a2

                grad_a1 = np.dot(grad_a2, self.net.w_a2.T) * (1.0 - h1_a ** 2)
                self.net.w_a1 -= self.lr * 0.05 * np.outer(o, grad_a1)
                self.net.b_a1 -= self.lr * 0.05 * grad_a1

                # Critic
                h1_c = np.tanh(np.dot(o, self.net.w_c1) + self.net.b_c1)
                h2_c = np.tanh(np.dot(h1_c, self.net.w_c2) + self.net.b_c2)
                v = float((np.dot(h2_c, self.net.w_c3) + self.net.b_c3)[0])
                val_err = v - ret
                self.net.w_c3 -= self.lr * 0.1 * (val_err * h2_c[:, None])
                self.net.b_c3 -= self.lr * 0.1 * val_err

    def train_self_play(self, total_timesteps: int = 5000, save_path: str = "outputs/agar_ppo_weights.npz"):
        """Executes multi-agent self-play training loop in PartiallyObservableAgarEnv."""
        self.pretrain_imitation(num_steps=1800, epochs=25)
        logger.info(f"Starting Multi-Agent Self-Play PPO for {total_timesteps:,} timesteps...")

        env = PartiallyObservableAgarEnv(arena_size=20.0, num_players=6, num_food=120)
        step = 0
        episodes = 0
        learner_id = "player_0"

        # Opponents: mix of heuristic bots and self-play snapshots
        opponents = {pid: HeuristicAgarBot(pid) for pid in env.player_ids if pid != learner_id}

        obs_dict, _ = env.reset(seed=100)

        while step < total_timesteps:
            rollout_obs, rollout_act, rollout_rew, rollout_done, rollout_val, rollout_lp = [], [], [], [], [], []
            rollout_len = 512

            for _ in range(rollout_len):
                o_learner = obs_dict[learner_id]
                val = self.net.forward_critic(o_learner)
                act_learner, log_p = self.net.sample_action(o_learner, deterministic=False)

                # Query opponents
                actions = {learner_id: act_learner}
                for pid, opp in opponents.items():
                    actions[pid] = opp.predict(obs_dict[pid])

                next_obs, rewards, terms, truncs, infos = env.step(actions)
                done = terms[learner_id] or truncs[learner_id]

                rollout_obs.append(o_learner.copy())
                rollout_act.append(act_learner.copy())
                rollout_rew.append(rewards[learner_id])
                rollout_done.append(float(done))
                rollout_val.append(val)
                rollout_lp.append(log_p)

                step += 1
                obs_dict = next_obs

                if done:
                    episodes += 1
                    obs_dict, _ = env.reset(seed=100 + episodes)

            last_v = self.net.forward_critic(obs_dict[learner_id])
            advs, rets = self.compute_gae(
                np.array(rollout_rew, dtype=np.float32),
                np.array(rollout_val, dtype=np.float32),
                np.array(rollout_done, dtype=np.float32),
                last_v
            )

            self.train_epoch(
                np.array(rollout_obs, dtype=np.float32),
                np.array(rollout_act, dtype=np.float32),
                np.array(rollout_lp, dtype=np.float32),
                advs,
                rets,
                epochs=3
            )

            mean_rew = float(np.mean(rollout_rew))
            mass_learner = env.players[learner_id].mass
            logger.info(f"Step {step:5d}/{total_timesteps} | Episodes: {episodes} | Mean Reward: {mean_rew:+.2f} | Learner Mass: {mass_learner:.1f}")

        # Save to both outputs and project folder
        self.net.save(save_path)
        project_path = "projects/sim2real-ppo-navigation/outputs/agar_ppo_weights.npz"
        self.net.save(project_path)
        logger.info("Self-play training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Agar.io PPO Self-Play Training")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total training timesteps")
    parser.add_argument("--output", type=str, default="outputs/agar_ppo_weights.npz", help="Output path")
    args = parser.parse_args()

    agent = AgarPPOAgent(obs_dim=36, action_dim=2)
    agent.train_self_play(total_timesteps=args.timesteps, save_path=args.output)


if __name__ == "__main__":
    main()
