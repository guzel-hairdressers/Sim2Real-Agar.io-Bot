"""
Domain Randomization Wrapper for Gymnasium Mobile Robot Navigation.
Introduces stochastic variations across physical properties during each episode reset:
- Friction coefficient (surface grip: ice, linoleum, carpet)
- Robot mass (payload variations)
- Sensor Gaussian noise (simulating cheap time-of-flight LiDAR)
- Motor command latency / delay buffer (simulating I2C / UART microcontroller transmission delay)
"""

from collections import deque
from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper injecting Epistemic and Aleatoric uncertainty for Sim2Real transfer.
    """

    def __init__(
        self,
        env: gym.Env,
        randomize_friction: bool = True,
        friction_range: Tuple[float, float] = (0.3, 1.4),
        randomize_mass: bool = True,
        mass_range: Tuple[float, float] = (1.2, 4.5),
        randomize_sensor_noise: bool = True,
        sensor_noise_range: Tuple[float, float] = (0.005, 0.06),
        max_action_delay: int = 2,
    ):
        super().__init__(env)
        self.randomize_friction = randomize_friction
        self.friction_range = friction_range
        self.randomize_mass = randomize_mass
        self.mass_range = mass_range
        self.randomize_sensor_noise = randomize_sensor_noise
        self.sensor_noise_range = sensor_noise_range
        self.max_action_delay = max_action_delay

        self.current_delay = 0
        self.action_queue: deque = deque()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Randomize physics parameters on the base environment
        base_env = self.unwrapped

        if self.randomize_friction:
            base_env.friction_coeff = float(
                np.random.uniform(self.friction_range[0], self.friction_range[1])
            )

        if self.randomize_mass:
            base_env.robot_mass = float(
                np.random.uniform(self.mass_range[0], self.mass_range[1])
            )

        if self.randomize_sensor_noise:
            base_env.sensor_noise_std = float(
                np.random.uniform(self.sensor_noise_range[0], self.sensor_noise_range[1])
            )

        if self.max_action_delay > 0:
            self.current_delay = int(np.random.randint(0, self.max_action_delay + 1))
        else:
            self.current_delay = 0

        self.action_queue.clear()
        obs, info = self.env.reset(seed=seed, options=options)

        # Populate initial queue with neutral actions
        for _ in range(self.current_delay + 1):
            self.action_queue.append(np.zeros_like(self.action_space.low))

        info["domain_params"] = {
            "friction_coeff": base_env.friction_coeff,
            "robot_mass": base_env.robot_mass,
            "sensor_noise_std": base_env.sensor_noise_std,
            "action_delay_steps": self.current_delay,
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Motor latency: push current action, pop delayed action
        self.action_queue.append(action)
        delayed_action = self.action_queue.popleft()

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        return obs, reward, terminated, truncated, info
