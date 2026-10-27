"""Versioned deployment policies; incompatible checkpoints are fatal."""
from pathlib import Path
import numpy as np
import torch

from envs.arena_config import OBSERVATION_DIM, OBSERVATION_VERSION, ArenaConfig
from src.train_champion_ppo import HybridActorCritic
from src.agar_diffusion_policy import AgarDiffusionPolicy
from src.arena_tactics import safety_filter, teacher_action


def load_package(path, algorithm):
    package = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'schema_version': 2, 'observation_version': OBSERVATION_VERSION,
                'obs_dim': OBSERVATION_DIM, 'algorithm': algorithm}
    for key, value in expected.items():
        if package.get(key) != value:
            raise ValueError(f'{path}: incompatible {key}: expected {value!r}')
    if package['arena_config']['dt'] != ArenaConfig().dt:
        raise ValueError('Checkpoint control time differs from deployment')
    return package


class TacticalTeacher:
    def predict(self, obs, deterministic=True):
        return teacher_action(obs)

    def reset(self, seed=None):
        pass


class ArenaPPO:
    def __init__(self, path, assisted=True):
        package = load_package(path, 'ppo')
        self.net = HybridActorCritic(OBSERVATION_DIM)
        self.net.load_state_dict(package['state_dict'], strict=True)
        self.net.eval()
        self.assisted = assisted
        self.interventions = self.observations = 0

    def predict(self, obs, deterministic=True):
        if np.shape(obs) != (OBSERVATION_DIM,):
            raise ValueError('PPO requires the v2 observation schema')
        with torch.inference_mode():
            raw = self.net.get_action_and_value(torch.as_tensor(obs).unsqueeze(0), deterministic)[0][0].numpy()
        raw = np.clip(raw, [0, -1, 0], [1, 1, 1]).astype(np.float32)
        action = safety_filter(raw, obs) if self.assisted else raw
        self.observations += 1
        self.interventions += int(np.max(np.abs(action - raw)) > 1e-5)
        return action

    def reset(self, seed=None):
        self.interventions = self.observations = 0


class ArenaDiffusion:
    def __init__(self, path, critic_path=None, seed=42, assisted=True):
        package = load_package(path, 'diffusion')
        if critic_path:
            load_package(critic_path, 'return_critic')
        self.policy = AgarDiffusionPolicy(
            obs_dim=OBSERVATION_DIM, action_horizon=16, exec_horizon=3,
            num_ddim_steps=5, num_candidates=4 if critic_path else 1,
            model_path=str(path), critic_path=str(critic_path) if critic_path else '', seed=seed)
        if self.policy.torch_net is None or (critic_path and self.policy.torch_critic is None):
            raise ValueError('A deployment checkpoint failed to load')
        self.assisted = assisted
        self.interventions = self.observations = 0

    def predict(self, obs, deterministic=True):
        if np.shape(obs) != (OBSERVATION_DIM,):
            raise ValueError('Diffusion requires the v2 observation schema')
        # Replan immediately when an observed emergency makes the old plan stale.
        if obs[47] > 0 and obs[47] < .35:
            self.policy.reset()
        raw = self.policy.predict(obs, deterministic, apply_reflex=False)
        action = safety_filter(raw, obs) if self.assisted else raw
        self.observations += 1
        self.interventions += int(np.max(np.abs(action - raw)) > 1e-5)
        return action

    def reset(self, seed=None):
        self.policy.reset(seed)
        self.interventions = self.observations = 0


class LegacyPolicy:
    """Frozen ca3f24d reflex + unchanged original checkpoint, for evaluation."""
    def __init__(self, algorithm, root=None, seed=42):
        from benchmarks.legacy_reflex_ca3f24d import apply_tactical_combat_reflex
        from src.train_champion_ppo import ChampionPPOAgent
        root = Path(root or Path(__file__).resolve().parents[1]) / 'outputs'
        self.reflex = apply_tactical_combat_reflex
        if algorithm == 'ppo':
            self.policy = ChampionPPOAgent(str(root / 'agar_ppo_champion.pt'))
        else:
            self.policy = AgarDiffusionPolicy(model_path=str(root / 'agar_diffusion_champion.pt'),
                critic_path=str(root / 'agar_diffusion_critic_champion.pt'),
                action_horizon=16, exec_horizon=3, num_ddim_steps=5, seed=seed)

    def predict(self, obs, deterministic=True):
        raw = self.policy.predict(obs[:38], deterministic, apply_reflex=False)
        return self.reflex(raw, obs[:38])

    def reset(self, seed=None):
        if hasattr(self.policy, 'reset'):
            self.policy.reset(seed)
