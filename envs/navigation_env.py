"""
Continuous Navigation Environment implementing Farama Gymnasium Standard:
Differential-drive mobile robot equipped with rangefinder LiDAR beams
navigating dynamic obstacle environments to reach target coordinate waypoints.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousNavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        arena_size: float = 10.0,
        num_obstacles: int = 6,
        num_lidar_rays: int = 8,
        max_lidar_range: float = 5.0,
        max_steps: int = 300
    ):
        super().__init__()
        self.arena_size = arena_size
        self.num_obstacles = num_obstacles
        self.num_lidar_rays = num_lidar_rays
        self.max_lidar_range = max_lidar_range
        self.max_steps = max_steps

        # Action space: [linear_accel (-1 to 1 m/s^2), angular_velocity (-1 to 1 rad/s)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space:
        # [0..7]: 8 normalized lidar range readings in [0, 1]
        # [8]: Normalized distance to target
        # [9]: Relative angle to target in [-pi, pi] / pi
        # [10]: Current linear velocity
        # [11]: Current angular velocity
        obs_dim = num_lidar_rays + 4
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )

        # Dynamic physics parameters (modifiable by Domain Randomization)
        self.robot_mass = 2.0         # kg
        self.friction_coeff = 0.8     # mu
        self.motor_delay_steps = 0    # action latency buffer
        self.sensor_noise_std = 0.0   # Gaussian noise std

        # Internal state
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.obstacles: np.ndarray = np.empty((0, 3), dtype=np.float32) # [x, y, radius]
        self.step_count = 0
        self.prev_dist_to_goal = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Spawn robot at center
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_yaw = self.np_random.uniform(-np.pi, np.pi)
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        # Spawn target at random perimeter location
        target_dist = self.np_random.uniform(3.0, self.arena_size / 2.0 - 0.5)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        self.target_pos = np.array([
            target_dist * np.cos(target_angle),
            target_dist * np.sin(target_angle)
        ], dtype=np.float32)

        # Spawn obstacles
        obs_list = []
        for _ in range(self.num_obstacles):
            ox = self.np_random.uniform(-self.arena_size / 2.5, self.arena_size / 2.5)
            oy = self.np_random.uniform(-self.arena_size / 2.5, self.arena_size / 2.5)
            # Avoid spawning on top of robot
            if np.linalg.norm([ox, oy]) > 1.2:
                radius = self.np_random.uniform(0.3, 0.6)
                obs_list.append([ox, oy, radius])
        self.obstacles = np.array(obs_list, dtype=np.float32) if obs_list else np.empty((0, 3), dtype=np.float32)

        self.prev_dist_to_goal = np.linalg.norm(self.target_pos - self.robot_pos)
        obs = self._get_obs()
        return obs, {}

    def _compute_lidar_readings(self) -> np.ndarray:
        """Simulates 8-ray radial LiDAR distance scans."""
        angles = np.linspace(-np.pi, np.pi, self.num_lidar_rays, endpoint=False) + self.robot_yaw
        ranges = np.full(self.num_lidar_rays, self.max_lidar_range, dtype=np.float32)

        for i, ray_angle in enumerate(angles):
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            # Check arena boundary walls
            for dim, bound in enumerate([self.arena_size / 2.0, -self.arena_size / 2.0]):
                if abs(ray_dir[dim]) > 1e-4:
                    t = (bound - self.robot_pos[dim]) / ray_dir[dim]
                    if 0 < t < ranges[i]:
                        ranges[i] = t

            # Check obstacle intersections
            for ox, oy, r in self.obstacles:
                d = np.array([ox, oy]) - self.robot_pos
                proj = np.dot(d, ray_dir)
                if proj > 0:
                    perp_dist_sq = np.dot(d, d) - proj ** 2
                    if perp_dist_sq < r ** 2:
                        t = proj - np.sqrt(r ** 2 - perp_dist_sq)
                        if 0 < t < ranges[i]:
                            ranges[i] = t

        # Normalize to [0, 1]
        norm_ranges = ranges / self.max_lidar_range
        if self.sensor_noise_std > 0:
            noise = self.np_random.normal(0, self.sensor_noise_std, size=norm_ranges.shape)
            norm_ranges = np.clip(norm_ranges + noise, 0.0, 1.0)
        return norm_ranges.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        lidar = self._compute_lidar_readings()
        diff = self.target_pos - self.robot_pos
        dist = np.linalg.norm(diff)
        norm_dist = np.clip(dist / self.arena_size, 0.0, 1.0)

        angle_to_target = np.arctan2(diff[1], diff[0])
        rel_angle = (angle_to_target - self.robot_yaw + np.pi) % (2 * np.pi) - np.pi
        norm_rel_angle = rel_angle / np.pi

        return np.concatenate([
            lidar,
            [norm_dist, norm_rel_angle, self.linear_vel / 2.0, self.angular_vel / np.pi]
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        dt = 0.1  # 10 Hz control loop

        linear_acc = float(action[0])
        angular_vel_cmd = float(action[1])

        # Integrate differential-drive kinematics with physical friction and damping
        effective_acc = linear_acc * (self.friction_coeff / 0.8) / (self.robot_mass / 2.0)
        self.linear_vel = np.clip(self.linear_vel + effective_acc * dt, -0.5, 1.5)
        self.angular_vel = angular_vel_cmd

        self.robot_yaw += self.angular_vel * dt
        self.robot_yaw = (self.robot_yaw + np.pi) % (2 * np.pi) - np.pi

        self.robot_pos[0] += self.linear_vel * np.cos(self.robot_yaw) * dt
        self.robot_pos[1] += self.linear_vel * np.sin(self.robot_yaw) * dt

        # Collision detection
        dist_to_goal = np.linalg.norm(self.target_pos - self.robot_pos)
        terminated = False
        collision = False

        # Check boundary collision
        if np.any(np.abs(self.robot_pos) > (self.arena_size / 2.0 - 0.2)):
            collision = True

        # Check obstacle collision
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(self.robot_pos - np.array([ox, oy])) < (r + 0.2):
                collision = True
                break

        # Compute rewards
        progress = self.prev_dist_to_goal - dist_to_goal
        reward = progress * 10.0 - 0.05  # Step time penalty

        if dist_to_goal < 0.4:
            reward += 100.0
            terminated = True
        elif collision:
            reward -= 50.0
            terminated = True

        self.prev_dist_to_goal = dist_to_goal
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()

        return obs, float(reward), terminated, truncated, {"is_success": dist_to_goal < 0.4, "collision": collision}
