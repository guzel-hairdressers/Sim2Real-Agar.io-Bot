"""Observable scripted exploit regressions, distinct from tournament seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from envs.agar_env import PartiallyObservableAgarEnv
from src.arena_policies import ArenaPPO, ArenaDiffusion, LegacyPolicy
from src.arena_tactics import teacher_action


def evaluate(algorithm, checkpoint, critic=None):
    candidates = {'new': ArenaPPO(checkpoint) if algorithm == 'ppo' else ArenaDiffusion(checkpoint, critic),
                  'legacy': LegacyPolicy(algorithm)}
    cases = [
        ('head_on_predator', 20, 60, [0, 0], [2.5, 0], 0),
        ('flanking_predator', 20, 60, [0, 0], [1.8, 1.8], 0),
        ('split_bait', 45, 30, [0, 0], [1.5, 0], 0),
        ('safe_prey', 90, 20, [0, 0], [1.5, 0], 0),
        ('corner_escape', 20, 60, [7.3, 7.3], [4.8, 7.3], np.pi),
    ]
    rows = []
    for name, mass, opponent_mass, pos, other, yaw in cases:
        for seed in (711, 733, 751, 769):
            for group, policy in candidates.items():
                policy.reset(seed)
                env = PartiallyObservableAgarEnv(arena_size=18, num_players=2, num_food=0,
                    num_viruses=0, dt=1/30, max_steps=300, observation_version=2)
                env.reset(seed=seed)
                a, b = env.players.values()
                a.mass, b.mass = mass, opponent_mass
                a.pos, b.pos = np.array(pos, np.float32), np.array(other, np.float32)
                a.yaw, b.yaw = yaw, yaw + np.pi
                # Small, predeclared variations, unrelated to training/test seeds.
                b.pos = b.pos + np.random.default_rng(seed).uniform(-.1, .1, 2).astype(np.float32)
                env.total_world_mass_cap = mass + opponent_mass
                env.reserve_mass = 0
                obs = {p: env._get_obs(p) for p in env.player_ids}
                for _ in range(300):
                    opponent = np.array([0, 0, 0], np.float32) if name == 'safe_prey' else teacher_action(obs['player_1'])
                    obs, _, _, _, _ = env.step({'player_0': policy.predict(obs['player_0']), 'player_1': opponent})
                    if a.deaths or b.deaths:
                        break
                rows.append(dict(case=name, seed=seed, group=group, deaths=a.deaths,
                    opponent_deaths=b.deaths, piece_kills=a.kills, mass=a.mass,
                    wall_contacts=a.wall_contacts, seconds=env.step_count/30))
    return dict(algorithm=algorithm, rows=rows,
        totals={g: {k: sum(row[k] for row in rows if row['group']==g)
                    for k in ('deaths', 'opponent_deaths', 'piece_kills', 'wall_contacts')}
                for g in candidates})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', choices=['ppo', 'diffusion'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--critic')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    result = evaluate(args.algorithm, args.checkpoint, args.critic)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
