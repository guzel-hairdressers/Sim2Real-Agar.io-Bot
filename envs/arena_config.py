"""Shared physical configuration for training, evaluation and play."""
from dataclasses import asdict, dataclass

OBSERVATION_VERSION = 2
OBSERVATION_DIM = 70


@dataclass(frozen=True)
class ArenaConfig:
    arena_size: float = 18.0
    num_players: int = 7
    num_food: int = 180
    num_viruses: int = 7
    total_world_mass: float = 550.0
    dt: float = 1.0 / 30.0
    max_steps: int = 1800
    max_pieces: int = 4
    observation_version: int = OBSERVATION_VERSION

    def make(self):
        from envs.agar_env import PartiallyObservableAgarEnv
        return PartiallyObservableAgarEnv(**asdict(self))


def event_reward(previous, current, base_reward):
    """Pay once per event, never once per cumulative counter per step."""
    kills = max(0, current['kills'] - previous['kills'])
    deaths = max(0, current['deaths'] - previous['deaths'])
    return float(base_reward) * 0.1 + 4.5 * kills - 3.0 * deaths
