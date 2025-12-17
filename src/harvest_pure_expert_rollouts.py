"""
Harvest clean, deterministic expert rollouts from trained PPO champion for Diffusion training.
Ensures zero imitation of exploration noise (uses deterministic=True).
Collects 16-step action chunks [thrust, steer, split] conditioned on 38-dim body-frame state.
"""

import os
import sys
import logging
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_superior_ppo import SuperiorPPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExpertHarvest")


def harvest(num_rounds: int = 50, episode_steps: int = 350, horizon: int = 16):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("projects/sim2real-ppo-navigation/outputs", exist_ok=True)

    ppo_weights = "outputs/agar_ppo_champion.pt"
    agent = SuperiorPPOAgent(weights_path=ppo_weights)

    env = PartiallyObservableAgarEnv(
        arena_size=14.0,
        num_players=6,
        num_food=100,
        max_steps=episode_steps,
        mass_decay_multiplier=1.35,
        max_pieces=4
    )

    all_obs_chunks = []
    all_act_chunks = []
    total_kills_seen = 0

    for r in range(1, num_rounds + 1):
        obs_dict, _ = env.reset(seed=1000 + r * 19)
        agent.net.eval()

        player_trajectories = {pid: [] for pid in env.player_ids}

        for step in range(episode_steps):
            actions = {}
            for pid in env.player_ids:
                act = agent.predict(obs_dict[pid], deterministic=True)
                actions[pid] = act
                player_trajectories[pid].append((obs_dict[pid].copy(), act.copy()))

            obs_dict, rew, term, trunc, info = env.step(actions)

        # Harvest high-performing trajectories (mass >= 32kg or kills >= 1)
        for pid in env.player_ids:
            p_state = env.players[pid]
            total_kills_seen += p_state.kills
            if p_state.peak_mass >= 32.0 or p_state.kills >= 1:
                traj = player_trajectories[pid]
                L = len(traj)
                if L >= horizon + 2:
                    for t in range(L - horizon):
                        obs_t = traj[t][0]
                        act_chunk = np.array([traj[t + k][1] for k in range(horizon)], dtype=np.float32)
                        all_obs_chunks.append(obs_t)
                        all_act_chunks.append(act_chunk)

        if r % 10 == 0:
            logger.info(f"Round {r:02d}/{num_rounds:02d} | Collected {len(all_obs_chunks)} chunks | Kills: {total_kills_seen}")

    obs_arr = np.array(all_obs_chunks, dtype=np.float32)
    act_arr = np.array(all_act_chunks, dtype=np.float32)

    logger.info(f"Harvest complete! Total chunks: {len(obs_arr)} (obs_dim={obs_arr.shape[-1]}, horizon={act_arr.shape[1]})")

    out_paths = [
        "outputs/agar_clean_expert_rollouts.npz",
        "projects/sim2real-ppo-navigation/outputs/agar_clean_expert_rollouts.npz"
    ]
    for p in out_paths:
        np.savez_compressed(p, obs=obs_arr, acts=act_arr)
        logger.info(f"Saved dataset to {p}")


if __name__ == "__main__":
    harvest(num_rounds=50, episode_steps=350)
