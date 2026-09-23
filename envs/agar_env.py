"""
Partially Observable Agar.io Multi-Agent Continuous Environment with Splitting Mechanics.
Farama Gymnasium-compliant multi-agent survival arena with:
1. Dynamic mass scaling & mass-dependent speed physics (v_max ~ M^-0.35)
2. Authentic Agar.io Splitting Mechanic: Spacebar split projectile launch (v_burst = 4.8 m/s)
3. Nutrient food foraging (150 pellets with immediate replenishing)
4. Spiked virus obstacles (repels or splits massive cells)
5. Multi-agent predation consumption across multi-cell pieces
6. Circular Field of View (FoV) Fog of War: agents only observe entities within vision radius R_vis
7. 38-dimensional continuous observation vectors and 3-dimensional continuous actions [thrust, steer, split]
"""

from typing import Tuple, Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SubPiece:
    pos: np.ndarray
    mass: float
    vel: np.ndarray
    merge_cooldown: float = 0.0
    prev_pos: np.ndarray = None

    def __post_init__(self):
        if self.prev_pos is None:
            self.prev_pos = self.pos.copy()

    @property
    def radius(self) -> float:
        return float(0.35 * math.sqrt(max(1.0, self.mass) / 15.0))

    @property
    def max_speed(self) -> float:
        base_speed = 1.7
        return float(np.clip(base_speed * (15.0 / max(1.0, self.mass)) ** 0.35, 0.45, 2.2))


class PlayerState:
    """State of an individual player cell (or split sub-cells) in the Agar arena."""

    def __init__(self, player_id: str, pos: np.ndarray, color: Tuple[int, int, int], initial_mass: float = 15.0, max_pieces: int = 4):
        self.player_id = player_id
        self.color = color
        self.yaw = float(np.random.uniform(-np.pi, np.pi))
        self.angular_vel = 0.0
        self.kills = 0
        self.deaths = 0
        self.food_eaten = 0
        self.peak_mass = float(initial_mass)
        self.alive_steps = 0
        self.split_cooldown = 0.0
        self.max_pieces = max_pieces
        self.action_history = deque(maxlen=5)
        self.wall_contacts = 0

        # List of sub-cell pieces (starts with 1 single main cell)
        self.pieces: List[SubPiece] = [
            SubPiece(pos=pos.astype(np.float32), mass=float(initial_mass), vel=np.zeros(2, dtype=np.float32))
        ]

    @property
    def mass(self) -> float:
        return float(sum(p.mass for p in self.pieces))

    @mass.setter
    def mass(self, val: float):
        if not self.pieces:
            self.pieces = [SubPiece(pos=np.zeros(2, dtype=np.float32), mass=val, vel=np.zeros(2, dtype=np.float32))]
        else:
            scale = val / max(1e-4, self.mass)
            for p in self.pieces:
                p.mass *= scale

    @property
    def pos(self) -> np.ndarray:
        if not self.pieces:
            return np.zeros(2, dtype=np.float32)
        total_m = sum(p.mass for p in self.pieces)
        if total_m < 1e-4:
            return self.pieces[0].pos.copy()
        return (sum(p.pos * p.mass for p in self.pieces) / total_m).astype(np.float32)

    @pos.setter
    def pos(self, new_pos: np.ndarray):
        if not self.pieces:
            self.pieces = [SubPiece(pos=new_pos.astype(np.float32), mass=15.0, vel=np.zeros(2, dtype=np.float32))]
        else:
            delta = new_pos.astype(np.float32) - self.pos
            for p in self.pieces:
                p.pos += delta

    @property
    def linear_vel(self) -> float:
        if not self.pieces:
            return 0.0
        return float(np.linalg.norm(self.pieces[0].vel))

    @linear_vel.setter
    def linear_vel(self, val: float):
        heading = np.array([np.cos(self.yaw), np.sin(self.yaw)], dtype=np.float32)
        for p in self.pieces:
            p.vel = heading * val

    @property
    def radius(self) -> float:
        return float(0.35 * math.sqrt(max(1.0, self.mass) / 15.0))

    @property
    def max_speed(self) -> float:
        base_speed = 1.7
        return float(np.clip(base_speed * (15.0 / max(1.0, self.mass)) ** 0.35, 0.45, 2.2))

    @property
    def vision_radius(self) -> float:
        return float(np.clip(5.5 * (self.mass / 15.0) ** 0.22, 5.0, 9.5))

    @property
    def can_split(self) -> bool:
        return any(p.mass >= 36.0 for p in self.pieces) and (self.split_cooldown <= 0.0) and (len(self.pieces) < self.max_pieces)

    def split(self) -> bool:
        """Executes authentic Agar.io splitting: divides eligible pieces in half (capped at max_pieces)."""
        if not self.can_split:
            return False

        heading = np.array([math.cos(self.yaw), math.sin(self.yaw)], dtype=np.float32)
        new_projectiles = []

        eligible = [p for p in self.pieces if p.mass >= 36.0]
        eligible.sort(key=lambda p: p.mass, reverse=True)

        for p in eligible:
            if len(self.pieces) + len(new_projectiles) >= self.max_pieces:
                break
            half_m = float(p.mass * 0.5)
            p.mass = half_m
            p.merge_cooldown = 8.0
            p.vel = -heading * 0.4  # Slight recoil impulse

            launch_pos = p.pos + heading * (p.radius * 1.6)
            launch_vel = heading * 4.8  # Explosive launch impulse: 4.8 m/s burst
            projectile = SubPiece(
                pos=launch_pos,
                mass=half_m,
                vel=launch_vel,
                merge_cooldown=8.0  # Can re-merge after 8 seconds
            )
            new_projectiles.append(projectile)

        if len(new_projectiles) == 0:
            return False

        self.pieces.extend(new_projectiles)
        self.split_cooldown = 0.8  # Fast cooldown allows rapid multi-splitting
        return True


class PartiallyObservableAgarEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        arena_size: float = 20.0,
        num_players: int = 6,
        num_food: int = 100,
        num_viruses: int = 6,
        max_steps: int = 400,
        dt: float = 0.1,
        domain_randomize_arena: bool = False,
        total_world_mass: float = 400.0,
        pellet_mass: float = 0.5,
        mass_decay_multiplier: float = 1.35,
        max_pieces: int = 4
    ):
        super().__init__()
        self.base_arena_size = float(arena_size)
        self.domain_randomize_arena = domain_randomize_arena
        self.arena_size = arena_size
        self.half_arena = arena_size / 2.0
        self.num_players = num_players
        self.num_food = num_food
        self.max_food = max(num_food, 180)
        self.min_food = min(num_food, 40)
        self.num_viruses = num_viruses
        self.max_steps = max_steps
        self.dt = dt
        self.mass_decay_multiplier = float(mass_decay_multiplier)
        self.max_pieces = int(max_pieces)

        # Closed Thermodynamic Mass Ecosystem Invariants
        self.total_world_mass_cap = float(total_world_mass)
        self.pellet_mass = float(pellet_mass)
        self.reserve_mass = 0.0

        self.player_ids = [f"player_{i}" for i in range(num_players)]

        # Distinct neon colors for players (BGR format for OpenCV)
        self.player_colors = [
            (243, 156, 18),   # Cyan / Teal
            (236, 72, 153),   # Rose / Magenta
            (16, 185, 129),   # Emerald Green
            (239, 68, 68),    # Crimson Red
            (168, 85, 247),   # Purple / Violet
            (245, 158, 11),   # Amber / Orange
        ]

        # Action space per agent: [thrust (0.0 to 1.0), steer (-1.0 to 1.0), split_trigger (0.0 to 1.0)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: 38 continuous normalized features (including split availability and state)
        self.obs_dim = 38
        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_dim, dtype=np.float32),
            high=np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32
        )

        self.players: Dict[str, PlayerState] = {}
        self.food_positions: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.viruses: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.step_count = 0
        self.predation_events: List[Dict[str, Any]] = []

    @property
    def total_world_mass(self) -> float:
        """Returns exact sum of all player mass + all active food pellet mass + reserve buffer."""
        player_m = sum(p.mass for p in self.players.values())
        food_m = len(self.food_positions) * self.pellet_mass
        return float(player_m + food_m + self.reserve_mass)

    def _compute_spawn_density_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes high-resolution (50x50+) 2D potential field grid:
        1. S(x): Clearance shadow across all sub-pieces (zero food inside any player).
        2. F_corridors(x): Inverse-mass inter-agent bridges (shifted toward lighter competitors).
        3. F_center(x): Central oasis Gaussian potential.
        4. Ambient baseline floor.
        """
        grid_res = max(50, int(self.arena_size * 3.5))
        xs = np.linspace(-self.half_arena + 0.5, self.half_arena - 0.5, grid_res, dtype=np.float32)
        ys = np.linspace(-self.half_arena + 0.5, self.half_arena - 0.5, grid_res, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx, gy], axis=-1)  # (grid_res, grid_res, 2)

        # 1. Player Clearance Shadow S(x) across ALL sub-pieces of all active players
        shadow = np.ones((grid_res, grid_res), dtype=np.float32)
        for player in self.players.values():
            for piece in player.pieces:
                diff = grid_pts - piece.pos
                dist_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
                sig = piece.radius + 0.65
                shadow *= np.clip(1.0 - np.exp(-dist_sq / (2.0 * sig * sig)), 0.0, 1.0)

        # 2. Pairwise inverse-mass inter-agent corridor potential
        # c_ij = (M_j * p_i + M_i * p_j) / (M_i + M_j) heavily shifted toward the lighter player!
        corridor_field = np.zeros((grid_res, grid_res), dtype=np.float32)
        active_players = [p for p in self.players.values() if p.mass > 0]
        n_active = len(active_players)
        if n_active >= 2:
            pair_count = 0
            for i in range(n_active):
                for j in range(i + 1, n_active):
                    p_i = active_players[i]
                    p_j = active_players[j]
                    m_i = max(1.0, p_i.mass)
                    m_j = max(1.0, p_j.mass)
                    # Biased toward smaller player
                    c_ij = (m_j * p_i.pos + m_i * p_j.pos) / (m_i + m_j)
                    d_ij = float(np.linalg.norm(p_i.pos - p_j.pos))
                    sigma_ij = max(1.5, 0.35 * d_ij)
                    diff_c = grid_pts - c_ij
                    dist_sq_c = diff_c[..., 0] ** 2 + diff_c[..., 1] ** 2
                    corridor_field += np.exp(-dist_sq_c / (2.0 * sigma_ij * sigma_ij))
                    pair_count += 1
            if pair_count > 0:
                corridor_field /= pair_count

        # 3. Central Oasis Field
        center_dist_sq = grid_pts[..., 0] ** 2 + grid_pts[..., 1] ** 2
        sigma_center = 0.35 * self.half_arena
        center_field = np.exp(-center_dist_sq / (2.0 * sigma_center * sigma_center))

        # 4. Ambient floor
        ambient_floor = 0.08

        # Superposition & Normalization
        weight = shadow * (0.55 * corridor_field + 0.35 * center_field + ambient_floor)
        tot_w = float(np.sum(weight))
        if tot_w < 1e-6:
            prob = np.ones((grid_res, grid_res), dtype=np.float32) / (grid_res * grid_res)
        else:
            prob = weight / tot_w
        return prob, grid_pts

    def _sample_food_from_grid(self, count: int) -> np.ndarray:
        """Samples food coordinates from normalized density grid with continuous 2D jitter."""
        if count <= 0:
            return np.empty((0, 2), dtype=np.float32)
        prob, grid_pts = self._compute_spawn_density_grid()
        flat_prob = prob.ravel()
        grid_res = prob.shape[0]
        delta = self.arena_size / grid_res

        chosen_indices = self.np_random.choice(len(flat_prob), size=count, p=flat_prob)
        sampled_pts = []
        for idx in chosen_indices:
            u = idx // grid_res
            v = idx % grid_res
            center = grid_pts[u, v]
            # Continuous uniform jitter within cell box to eliminate grid-locking
            jitter = self.np_random.uniform(-delta * 0.48, delta * 0.48, size=2).astype(np.float32)
            pt = np.clip(center + jitter, -self.half_arena + 0.4, self.half_arena - 0.4)
            sampled_pts.append(pt)
        return np.array(sampled_pts, dtype=np.float32)

    def _sample_safe_virus_pos(self, min_clearance: float = 4.5, existing_viruses: Optional[List[Any]] = None) -> np.ndarray:
        """Samples a fresh virus position with guaranteed clearance from all player cells and existing viruses."""
        best_cand = None
        max_min_dist = -1.0

        for _ in range(60):
            vx = float(self.np_random.uniform(-self.half_arena + 2.5, self.half_arena - 2.5))
            vy = float(self.np_random.uniform(-self.half_arena + 2.5, self.half_arena - 2.5))
            cand = np.array([vx, vy], dtype=np.float32)

            min_player_dist = float("inf")
            for p in self.players.values():
                for pc in p.pieces:
                    d = float(np.linalg.norm(cand - pc.pos))
                    if d < min_player_dist:
                        min_player_dist = d

            if min_player_dist > max_min_dist:
                max_min_dist = min_player_dist
                best_cand = np.array([vx, vy, 0.65], dtype=np.float32)

            # Check clearance from existing viruses
            v_pool = existing_viruses if existing_viruses is not None else self.viruses
            too_close_v = False
            if len(v_pool) > 0:
                for v in v_pool:
                    v_arr = np.array(v[:2], dtype=np.float32)
                    if np.linalg.norm(cand - v_arr) < 2.2:
                        too_close_v = True
                        break

            if min_player_dist >= min_clearance and not too_close_v:
                return np.array([vx, vy, 0.65], dtype=np.float32)

        if best_cand is not None:
            return best_cand
        return np.array([0.0, 0.0, 0.65], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.predation_events.clear()

        # Domain Randomization: vary cage diameter between 11.5m and 17.0m
        if self.domain_randomize_arena:
            self.arena_size = float(self.np_random.uniform(11.5, 17.0))
            self.half_arena = self.arena_size / 2.0
        else:
            self.arena_size = self.base_arena_size
            self.half_arena = self.arena_size / 2.0

        # Initialize players in balanced ring formation first
        self.players.clear()
        total_init_player_mass = 0.0
        for i, pid in enumerate(self.player_ids):
            angle = (2.0 * np.pi * i) / self.num_players + self.np_random.uniform(-0.2, 0.2)
            spawn_dist = self.np_random.uniform(4.0, 7.5)
            pos = np.array([spawn_dist * np.cos(angle), spawn_dist * np.sin(angle)], dtype=np.float32)
            init_m = float(self.np_random.uniform(13.0, 15.0))
            total_init_player_mass += init_m
            color = self.player_colors[i % len(self.player_colors)]
            self.players[pid] = PlayerState(pid, pos, color, initial_mass=init_m, max_pieces=self.max_pieces)

        # Initialize static green virus cells with guaranteed player clearance (minimum 4.5m)
        virus_list = []
        for _ in range(self.num_viruses):
            v_cand = self._sample_safe_virus_pos(min_clearance=4.5, existing_viruses=virus_list)
            virus_list.append(v_cand)
        self.viruses = np.array(virus_list, dtype=np.float32)

        # Allocate food pellets and reserve mass under exact closed-loop mass conservation
        initial_food_count = min(self.num_food, int((self.total_world_mass_cap - total_init_player_mass) / self.pellet_mass))
        active_food_mass = initial_food_count * self.pellet_mass
        self.reserve_mass = float(self.total_world_mass_cap - total_init_player_mass - active_food_mass)

        # Sample initial food positions using the high-resolution field grid
        self.food_positions = self._sample_food_from_grid(initial_food_count)

        obs_dict = {pid: self._get_obs(pid) for pid in self.player_ids}
        return obs_dict, {}

    def _respawn_player(self, pid: str):
        safe_pos = None
        for _ in range(15):
            candidate = self.np_random.uniform(
                -self.half_arena + 2.0,
                self.half_arena - 2.0,
                size=2
            ).astype(np.float32)
            too_close = False
            for other_id, other in self.players.items():
                if other_id != pid and np.linalg.norm(candidate - other.pos) < 3.5:
                    too_close = True
                    break
            if not too_close:
                safe_pos = candidate
                break

        if safe_pos is None:
            safe_pos = self.np_random.uniform(-self.half_arena + 2.0, self.half_arena - 2.0, size=2).astype(np.float32)

        p = self.players[pid]
        p.max_pieces = self.max_pieces
        p.yaw = float(self.np_random.uniform(-np.pi, np.pi))
        p.angular_vel = 0.0
        p.split_cooldown = 0.0
        p.action_history.clear()

        # Deduct respawn mass from reserve_mass to maintain world mass invariant
        spawn_m = min(14.0, max(10.0, self.reserve_mass))
        self.reserve_mass = max(0.0, self.reserve_mass - spawn_m)
        p.pieces = [
            SubPiece(pos=safe_pos, mass=spawn_m, vel=np.zeros(2, dtype=np.float32))
        ]

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self.step_count += 1
        rewards = {pid: 0.0 for pid in self.player_ids}
        self.predation_events.clear()

        # 1. Action parsing & kinematic updates
        for pid, player in self.players.items():
            act = actions.get(pid, np.array([0.5, 0.0, 0.0], dtype=np.float32))
            thrust = float(np.clip(act[0], 0.0, 1.0))
            steer = float(np.clip(act[1], -1.0, 1.0))
            split_cmd = float(act[2]) if len(act) > 2 else 0.0
            player.action_history.append(np.array([thrust, steer, split_cmd], dtype=np.float32))

            # Trigger split
            if split_cmd > 0.5:
                did_split = player.split()
                if did_split:
                    rewards[pid] += 0.2

            player.split_cooldown = max(0.0, player.split_cooldown - self.dt)

            # Anti-circling & winding penalty
            rewards[pid] -= 0.03 * abs(steer)
            if len(player.action_history) >= 20:
                recent_steers = [a[1] for a in player.action_history[-20:]]
                mean_steer = float(np.mean(recent_steers))
                if abs(mean_steer) > 0.45:
                    winding_factor = (abs(mean_steer) - 0.45) / 0.55
                    rewards[pid] -= 0.08 * winding_factor

            # Angular kinematics (steers heading)
            max_omega = 3.0
            player.angular_vel = steer * max_omega
            player.yaw = (player.yaw + player.angular_vel * self.dt + np.pi) % (2.0 * np.pi) - np.pi
            heading_unit = np.array([np.cos(player.yaw), np.sin(player.yaw)], dtype=np.float32)

            # Kinematics for all sub-pieces of this player
            wall_margin = 0.1
            for piece in player.pieces:
                piece.merge_cooldown = max(0.0, piece.merge_cooldown - self.dt)

                # Linear kinematics (Agar.io speed inversion: giant cells slower, small cells faster)
                is_sprinting = (thrust > 0.88) and (piece.mass > 18.0)
                speed_multiplier = 1.35 if is_sprinting else 1.0
                if is_sprinting:
                    sprint_burn = min(piece.mass - 10.0, 0.15 * self.dt)
                    if sprint_burn > 0:
                        piece.mass -= sprint_burn
                        self.reserve_mass += sprint_burn

                target_speed = thrust * piece.max_speed * speed_multiplier
                target_vel = heading_unit * target_speed
                curr_speed = float(np.linalg.norm(piece.vel))

                if curr_speed > piece.max_speed:
                    # Fluid drag on projectile launch burst
                    drag_rate = min(1.0, 3.5 * self.dt)
                    piece.vel += (target_vel - piece.vel) * drag_rate
                else:
                    piece.vel += (target_vel - piece.vel) * 0.35

                piece.prev_pos = piece.pos.copy()
                piece.pos += piece.vel * self.dt

                # Wall bounce & boundary collision
                p_radius = piece.radius
                hit_wall = False
                for dim in range(2):
                    if piece.pos[dim] > self.half_arena - (p_radius + wall_margin):
                        piece.pos[dim] = self.half_arena - (p_radius + wall_margin)
                        piece.vel[dim] *= -0.5
                        hit_wall = True
                    elif piece.pos[dim] < -self.half_arena + (p_radius + wall_margin):
                        piece.pos[dim] = -self.half_arena + (p_radius + wall_margin)
                        piece.vel[dim] *= -0.5
                        hit_wall = True

                if hit_wall:
                    player.wall_contacts += 1
                    impact_spd = float(np.linalg.norm(piece.vel))
                    # Scaled impact penalty (-0.10 to -0.22) penalizes high-speed wall grinding
                    rewards[pid] -= 0.10 + 0.12 * min(1.0, impact_spd / max(0.1, piece.max_speed))

            # Multi-piece kinematics: gravitational attraction after cooldown + soft-body repulsion during cooldown + clean re-merge upon expiration
            if len(player.pieces) > 1:
                # Inward gravitational attraction for sub-pieces whose merge cooldown has expired
                ready_pieces = [p for p in player.pieces if p.merge_cooldown <= 0.0]
                if len(ready_pieces) > 1:
                    total_ready_m = sum(p.mass for p in ready_pieces)
                    com = sum(p.pos * p.mass for p in ready_pieces) / max(1e-3, total_ready_m)
                    for p in ready_pieces:
                        diff_com = com - p.pos
                        d_com = float(np.linalg.norm(diff_com))
                        if d_com > 0.02:
                            pull_dir = diff_com / d_com
                            pull_spd = 1.2 * min(1.0, d_com / 1.5)
                            p.pos += pull_dir * (pull_spd * self.dt)
                            p.vel += pull_dir * (1.5 * self.dt)

                merged_any = False
                i = 0
                while i < len(player.pieces):
                    j = i + 1
                    while j < len(player.pieces):
                        p_i = player.pieces[i]
                        p_j = player.pieces[j]
                        d_ij = float(np.linalg.norm(p_i.pos - p_j.pos))
                        r_sum = p_i.radius + p_j.radius

                        if d_ij < r_sum:
                            if p_i.merge_cooldown <= 0.0 and p_j.merge_cooldown <= 0.0:
                                # Both cooldowns expired: cleanly re-merge with exact mass & momentum conservation
                                total_m = p_i.mass + p_j.mass
                                v_merged = (p_i.mass * p_i.vel + p_j.mass * p_j.vel) / max(1e-4, total_m)
                                if p_i.mass >= p_j.mass:
                                    p_i.mass = total_m
                                    p_i.vel = v_merged
                                    player.pieces.pop(j)
                                else:
                                    p_j.mass = total_m
                                    p_j.vel = v_merged
                                    player.pieces.pop(i)
                                    i -= 1
                                    merged_any = True
                                    break
                                merged_any = True
                                continue
                            else:
                                # On merge cooldown: soft-body contact repulsion so pieces fan out tactically
                                overlap = r_sum - d_ij
                                if d_ij > 1e-4:
                                    normal = (p_i.pos - p_j.pos) / d_ij
                                else:
                                    ang = float(self.np_random.uniform(0, 2 * np.pi))
                                    normal = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
                                p_i.pos += normal * (overlap * 0.35)
                                p_j.pos -= normal * (overlap * 0.35)
                                p_i.vel += normal * (overlap * 1.5 * self.dt)
                                p_j.vel -= normal * (overlap * 1.5 * self.dt)
                        j += 1
                    i += 1
                if merged_any:
                    rewards[pid] += 0.5

            # Anti-circling angular winding penalty & forward exploration drive
            rewards[pid] -= 0.025 * (player.angular_vel ** 2)
            rewards[pid] += 0.02 * thrust

            # Metabolic mass decay with superlinear burn for giant cells
            if player.mass > 18.0:
                excess = player.mass - 18.0
                decay_rate = 0.015 * self.mass_decay_multiplier * ((excess / 12.0) ** 1.15)
                decay_amount = min(excess, decay_rate * self.dt)
                if decay_amount > 0:
                    largest = max(player.pieces, key=lambda p: p.mass)
                    largest.mass = max(10.0, largest.mass - decay_amount)
                    self.reserve_mass += decay_amount

            player.alive_steps += 1
            if player.mass > player.peak_mass:
                player.peak_mass = player.mass

        # 2. Food pellet foraging across all pieces (strict conservation)
        if len(self.food_positions) > 0:
            eaten_food_indices = set()
            for pid, player in self.players.items():
                for piece in player.pieces:
                    if len(self.food_positions) == 0:
                        break
                    diff = self.food_positions - piece.pos
                    dists_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
                    thresh = (piece.radius + 0.08) ** 2
                    eaten_mask = np.where(dists_sq < thresh)[0]
                    valid_eaten = [int(idx) for idx in eaten_mask if idx not in eaten_food_indices]
                    if len(valid_eaten) > 0:
                        eaten_food_indices.update(valid_eaten)
                        gain = len(valid_eaten) * self.pellet_mass
                        piece.mass += gain
                        player.food_eaten += len(valid_eaten)
                        rewards[pid] += gain * 1.5

            if len(eaten_food_indices) > 0:
                remaining_mask = np.ones(len(self.food_positions), dtype=bool)
                remaining_mask[list(eaten_food_indices)] = False
                self.food_positions = self.food_positions[remaining_mask]

        # 3. Authentic Agar.io Spiked Virus Popping Mechanics (Explosive Multi-Piece Fragmentation)
        consumed_virus_indices = set()
        for pid, player in self.players.items():
            new_pieces = []
            for piece_idx, piece in enumerate(player.pieces):
                for v_idx, (vx, vy, vr) in enumerate(self.viruses):
                    if v_idx in consumed_virus_indices:
                        continue
                    v_pos = np.array([vx, vy], dtype=np.float32)
                    dist_v = float(np.linalg.norm(piece.pos - v_pos))
                    if dist_v < (piece.radius + vr):
                        # Case A: Cell >= 40kg hits virus -> POPS INTO MULTIPLE SUB-PIECES!
                        if piece.mass >= 40.0:
                            curr_count = len(player.pieces) + len(new_pieces)
                            num_frags = min(player.max_pieces, player.max_pieces - curr_count + 1)
                            if num_frags >= 2:
                                frag_m = float(piece.mass / num_frags)
                                piece.mass = frag_m
                                piece.merge_cooldown = 12.0

                                # Explode radial burst outward
                                base_ang = player.yaw
                                for f_idx in range(1, num_frags):
                                    theta = base_ang + (2.0 * np.pi * f_idx / num_frags)
                                    dir_u = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
                                    b_pos = v_pos + dir_u * (vr + 0.3)
                                    b_vel = dir_u * float(np.random.uniform(3.5, 4.8))
                                    frag_pc = SubPiece(
                                        pos=b_pos,
                                        mass=frag_m,
                                        vel=b_vel,
                                        merge_cooldown=12.0
                                    )
                                    new_pieces.append(frag_pc)

                                rewards[pid] -= 6.0  # Virus popping is a tactical vulnerability
                                consumed_virus_indices.add(v_idx)
                                break
                        else:
                            # Case B: Small cell (< 40kg) safely bounces or uses virus as shield
                            normal = (piece.pos - v_pos) / max(1e-4, dist_v)
                            piece.pos = v_pos + normal * (piece.radius + vr + 0.02)
            if new_pieces:
                player.pieces.extend(new_pieces)

        # Respawn popped viruses with guaranteed clearance away from all players (minimum 4.5m)
        for v_idx in consumed_virus_indices:
            self.viruses[v_idx] = self._sample_safe_virus_pos(min_clearance=4.5)

        # 4. Multi-piece Predation Consumption (70% absorbed, 30% blasted as radial shrapnel)
        eaten_player_ids = set()
        all_pieces = []
        for pid, player in self.players.items():
            for idx, pc in enumerate(player.pieces):
                all_pieces.append((pid, idx, pc))

        all_pieces.sort(key=lambda x: x[2].mass, reverse=True)
        eaten_piece_keys = set()

        for i in range(len(all_pieces)):
            pred_pid, pred_idx, pred_piece = all_pieces[i]
            if (pred_pid, pred_idx) in eaten_piece_keys or pred_pid in eaten_player_ids:
                continue

            for j in range(i + 1, len(all_pieces)):
                prey_pid, prey_idx, prey_piece = all_pieces[j]
                if prey_pid == pred_pid:
                    continue
                if (prey_pid, prey_idx) in eaten_piece_keys or prey_pid in eaten_player_ids:
                    continue

                mass_gap = pred_piece.mass - prey_piece.mass
                mass_ratio = pred_piece.mass / max(1.0, prey_piece.mass)
                dist = float(np.linalg.norm(pred_piece.pos - prey_piece.pos))
                overlap_req = pred_piece.radius + 0.20 * prey_piece.radius

                # Continuous Collision Detection: check trajectory sweep to prevent tunneling during 4.8m/s split lunges
                seg = pred_piece.pos - pred_piece.prev_pos
                seg_len_sq = float(np.dot(seg, seg))
                if seg_len_sq > 1e-4:
                    t_proj = max(0.0, min(1.0, float(np.dot(prey_piece.pos - pred_piece.prev_pos, seg)) / seg_len_sq))
                    closest_pt = pred_piece.prev_pos + t_proj * seg
                    swept_dist = float(np.linalg.norm(prey_piece.pos - closest_pt))
                else:
                    swept_dist = dist
                effective_dist = min(dist, swept_dist)

                # SUBSTANTIAL KILL CRITERIA: Delta mass >= 6.0 kg AND ratio >= 1.20 AND overlap
                if mass_gap >= 6.0 and mass_ratio >= 1.20 and effective_dist < overlap_req:
                    eaten_piece_keys.add((prey_pid, prey_idx))
                    prey_m = prey_piece.mass
                    absorbed_mass = prey_m * 0.70  # Killer absorbs 70%
                    splatter_mass = prey_m * 0.30  # 30% explodes outward as debris

                    pred_piece.mass += absorbed_mass
                    self.players[pred_pid].kills += 1
                    rewards[pred_pid] += 30.0
                    rewards[prey_pid] -= 20.0

                    self.predation_events.append({
                        "predator": pred_pid,
                        "prey": prey_pid,
                        "pos": prey_piece.pos.copy(),
                        "mass_gained": absorbed_mass
                    })

                    # Blast 30% splatter outward as high-velocity shrapnel beyond predator radius
                    num_shrapnel = max(2, int(splatter_mass / self.pellet_mass))
                    pellet_splatter_mass = num_shrapnel * self.pellet_mass
                    leftover_to_reserve = splatter_mass - pellet_splatter_mass
                    self.reserve_mass += leftover_to_reserve

                    blast_r = pred_piece.radius + 1.8
                    angles = np.linspace(0, 2 * np.pi, num_shrapnel, endpoint=False) + float(self.np_random.uniform(0, 0.5))
                    carcass_pts = []
                    for ang in angles:
                        u = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
                        pt = np.clip(
                            pred_piece.pos + u * float(blast_r + self.np_random.uniform(0.3, 1.2)),
                            -self.half_arena + 0.4,
                            self.half_arena - 0.4
                        )
                        carcass_pts.append(pt)
                    if len(carcass_pts) > 0:
                        c_arr = np.array(carcass_pts, dtype=np.float32)
                        if len(self.food_positions) == 0:
                            self.food_positions = c_arr
                        else:
                            self.food_positions = np.vstack([self.food_positions, c_arr])

        # Remove eaten pieces
        for pid, player in self.players.items():
            player.pieces = [pc for idx, pc in enumerate(player.pieces) if (pid, idx) not in eaten_piece_keys]
            if len(player.pieces) == 0:
                eaten_player_ids.add(pid)
                player.deaths += 1

        for pid in eaten_player_ids:
            self._respawn_player(pid)

        # 5. Food replenishment from reserve mass using the high-resolution field grid
        curr_food_count = len(self.food_positions)
        needed_pellets = min(self.max_food - curr_food_count, int(self.reserve_mass / self.pellet_mass))
        if needed_pellets > 0 and (curr_food_count < self.min_food or self.reserve_mass > 25.0):
            spawn_batch = min(needed_pellets, 8)
            new_pts = self._sample_food_from_grid(spawn_batch)
            if len(new_pts) > 0:
                if len(self.food_positions) == 0:
                    self.food_positions = new_pts
                else:
                    self.food_positions = np.vstack([self.food_positions, new_pts])
                self.reserve_mass -= spawn_batch * self.pellet_mass

        # 6. Threat Evasion & Prey Pursuit Velocity Shaping
        for pid, player in self.players.items():
            heading_unit = np.array([np.cos(player.yaw), np.sin(player.yaw)], dtype=np.float32)
            speed_ratio = player.linear_vel / max(0.1, player.max_speed)

            for other_id, other in self.players.items():
                if other_id == pid:
                    continue
                dist = float(np.linalg.norm(player.pos - other.pos))
                mass_gap = other.mass - player.mass

                # Case A: Danger - Other is heavier predator
                if mass_gap >= 6.0 and other.mass >= 1.20 * player.mass:
                    strike_zone = other.radius + player.radius + 1.8
                    if dist < strike_zone:
                        danger_factor = (strike_zone - dist) / strike_zone
                        rewards[pid] -= 0.40 * danger_factor

                        away_unit = (player.pos - other.pos) / max(1e-3, dist)
                        evasion_dot = float(np.dot(heading_unit, away_unit)) * speed_ratio
                        rewards[pid] += 0.25 * evasion_dot

                # Case B: Opportunity - Other is edible prey
                elif (player.mass - other.mass) >= 6.0 and player.mass >= 1.20 * other.mass:
                    if dist < player.vision_radius:
                        hunt_factor = (player.vision_radius - dist) / player.vision_radius
                        rewards[pid] += 0.25 * hunt_factor

                        towards_unit = (other.pos - player.pos) / max(1e-3, dist)
                        pursuit_dot = float(np.dot(heading_unit, towards_unit)) * speed_ratio
                        rewards[pid] += 0.30 * pursuit_dot

                        # Reward aiming and triggering split attack when prey is directly in front!
                        if pursuit_dot > 0.85 and player.can_split:
                            rewards[pid] += 0.35

            # Boundary Proximity Repulsion (steer away from approaching perimeter walls)
            d_to_walls = [
                player.pos[0] - (-self.half_arena), # left
                self.half_arena - player.pos[0],    # right
                player.pos[1] - (-self.half_arena), # bottom
                self.half_arena - player.pos[1]     # top
            ]
            min_d_w = min(d_to_walls)
            if min_d_w < 1.5:
                wall_idx = int(np.argmin(d_to_walls))
                inward_normals = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])]
                inward_n = inward_normals[wall_idx]
                heading_dot = float(np.dot(heading_unit, inward_n))
                proximity_factor = (1.5 - min_d_w) / 1.5
                if heading_dot < 0.0:
                    # Heading towards wall
                    rewards[pid] -= 0.12 * proximity_factor * abs(heading_dot) * speed_ratio
                else:
                    # Turning away from wall
                    rewards[pid] += 0.06 * proximity_factor * heading_dot * speed_ratio

        is_done = (self.step_count >= self.max_steps)
        obs_dict = {pid: self._get_obs(pid) for pid in self.player_ids}
        terms = {pid: False for pid in self.player_ids}
        truncs = {pid: is_done for pid in self.player_ids}
        infos = {pid: {
            "mass": self.players[pid].mass,
            "kills": self.players[pid].kills,
            "deaths": self.players[pid].deaths,
            "food_eaten": self.players[pid].food_eaten,
            "peak_mass": self.players[pid].peak_mass,
            "pieces_count": len(self.players[pid].pieces),
            "wall_contacts": self.players[pid].wall_contacts,
            "total_world_mass": self.total_world_mass,
            "reserve_mass": self.reserve_mass
        } for pid in self.player_ids}

        return obs_dict, rewards, terms, truncs, infos

    def _get_obs(self, pid: str) -> np.ndarray:
        player = self.players[pid]
        r_vis = player.vision_radius

        # 1. Ego state (6 dims: radius, linear_vel, angular_vel, mass, can_split, is_split)
        ego_state = np.array([
            np.clip(player.radius / 3.0, 0.0, 1.0),
            np.clip(player.linear_vel / 2.2, 0.0, 1.0),
            np.clip(player.angular_vel / 3.0, -1.0, 1.0),
            np.clip(player.mass / 150.0, 0.0, 1.0),
            1.0 if player.can_split else 0.0,
            1.0 if len(player.pieces) > 1 else 0.0
        ], dtype=np.float32)

        # 2. Radial Threat & Prey Radar (8 sectors each = 16 dims)
        num_sectors = 8
        sector_width = 2.0 * np.pi / num_sectors

        threat_radar = np.ones(num_sectors, dtype=np.float32)
        prey_radar = np.ones(num_sectors, dtype=np.float32)

        nearest_threat = None
        min_threat_dist = float("inf")
        nearest_prey = None
        min_prey_dist = float("inf")

        for other_id, other in self.players.items():
            if other_id == pid:
                continue
            delta = other.pos - player.pos
            dist = float(np.linalg.norm(delta))

            if dist > r_vis:
                continue

            angle_to_other = math.atan2(delta[1], delta[0])
            rel_angle = (angle_to_other - player.yaw + np.pi) % (2.0 * np.pi) - np.pi
            sec_idx = int(math.floor((rel_angle + np.pi) / sector_width)) % num_sectors
            norm_dist = float(np.clip(dist / r_vis, 0.0, 1.0))

            mass_gap = other.mass - player.mass
            mass_ratio = other.mass / max(1.0, player.mass)
            inv_ratio = player.mass / max(1.0, other.mass)

            if mass_gap >= 6.0 and mass_ratio >= 1.20:
                if norm_dist < threat_radar[sec_idx]:
                    threat_radar[sec_idx] = norm_dist
                if dist < min_threat_dist:
                    min_threat_dist = dist
                    nearest_threat = (delta, dist, other.mass)
            elif -mass_gap >= 6.0 and inv_ratio >= 1.20:
                if norm_dist < prey_radar[sec_idx]:
                    prey_radar[sec_idx] = norm_dist
                if dist < min_prey_dist:
                    min_prey_dist = dist
                    nearest_prey = (delta, dist, other.mass)

        # Visible food pellets for food_vec
        visible_food = []
        for fx, fy in self.food_positions:
            delta = np.array([fx, fy], dtype=np.float32) - player.pos
            dist = float(np.linalg.norm(delta))
            if dist <= r_vis:
                visible_food.append((delta, dist))

        # Body-frame rotation basis (along heading = +x_b, lateral left = +y_b)
        cos_yaw = math.cos(player.yaw)
        sin_yaw = math.sin(player.yaw)

        # 3. Nearest Threat Vector in Ego Body Frame (4 dims: fwd, lat, dist, mass_ratio)
        if nearest_threat is not None:
            t_delta, t_dist, t_mass = nearest_threat
            t_fwd = float(t_delta[0] * cos_yaw + t_delta[1] * sin_yaw) / r_vis
            t_lat = float(-t_delta[0] * sin_yaw + t_delta[1] * cos_yaw) / r_vis
            threat_vec = np.array([
                t_fwd,
                t_lat,
                t_dist / r_vis,
                np.clip(t_mass / player.mass, 1.0, 5.0) / 5.0
            ], dtype=np.float32)
        else:
            threat_vec = np.zeros(4, dtype=np.float32)

        # 4. Nearest Prey Vector in Ego Body Frame (4 dims: fwd, lat, dist, mass_ratio)
        if nearest_prey is not None:
            p_delta, p_dist, p_mass = nearest_prey
            p_fwd = float(p_delta[0] * cos_yaw + p_delta[1] * sin_yaw) / r_vis
            p_lat = float(-p_delta[0] * sin_yaw + p_delta[1] * cos_yaw) / r_vis
            prey_vec = np.array([
                p_fwd,
                p_lat,
                p_dist / r_vis,
                np.clip(player.mass / max(1.0, p_mass), 1.0, 5.0) / 5.0
            ], dtype=np.float32)
        else:
            prey_vec = np.zeros(4, dtype=np.float32)

        # 5. Local Food Centroid & Density in Ego Body Frame (4 dims: fwd, lat, density, nearest_dist)
        if len(visible_food) > 0:
            food_deltas = np.array([f[0] for f in visible_food], dtype=np.float32)
            centroid = np.mean(food_deltas, axis=0)
            nearest_food_d = min(f[1] for f in visible_food)
            c_fwd = float(centroid[0] * cos_yaw + centroid[1] * sin_yaw) / r_vis
            c_lat = float(-centroid[0] * sin_yaw + centroid[1] * cos_yaw) / r_vis
            food_vec = np.array([
                c_fwd,
                c_lat,
                np.clip(len(visible_food) / 30.0, 0.0, 1.0),
                nearest_food_d / r_vis
            ], dtype=np.float32)
        else:
            food_vec = np.zeros(4, dtype=np.float32)

        # 6. Arena Wall & Virus Obstacle Proximity (4 dims)
        # A. Forward boundary raycast along current heading
        hx = math.cos(player.yaw)
        hy = math.sin(player.yaw)
        ray_dists = []
        if hx > 1e-4:
            ray_dists.append((self.half_arena - player.pos[0]) / hx)
        elif hx < -1e-4:
            ray_dists.append((-self.half_arena - player.pos[0]) / hx)
        if hy > 1e-4:
            ray_dists.append((self.half_arena - player.pos[1]) / hy)
        elif hy < -1e-4:
            ray_dists.append((-self.half_arena - player.pos[1]) / hy)

        forward_wall_dist = min(ray_dists) if len(ray_dists) > 0 else self.half_arena
        wall_front_norm = float(np.clip(forward_wall_dist / r_vis, 0.0, 1.0))

        # B. Closest wall distance and relative bearing in body frame
        d_left = player.pos[0] - (-self.half_arena)
        d_right = self.half_arena - player.pos[0]
        d_bottom = player.pos[1] - (-self.half_arena)
        d_top = self.half_arena - player.pos[1]
        all_dists = [d_left, d_right, d_bottom, d_top]
        min_wall_dist = float(min(all_dists))
        closest_wall_idx = int(np.argmin(all_dists))
        wall_angles = [np.pi, 0.0, -np.pi / 2.0, np.pi / 2.0]
        wall_angle = wall_angles[closest_wall_idx]
        rel_wall_bearing = float((wall_angle - player.yaw + np.pi) % (2.0 * np.pi) - np.pi) / np.pi

        min_virus_d = 1.0
        virus_ahead = 0.0
        for vx, vy, vr in self.viruses:
            delta = np.array([vx, vy], dtype=np.float32) - player.pos
            dist = float(np.linalg.norm(delta))
            if dist <= r_vis:
                norm_d = dist / r_vis
                if norm_d < min_virus_d:
                    min_virus_d = norm_d
                    angle = math.atan2(delta[1], delta[0])
                    rel_angle = abs((angle - player.yaw + np.pi) % (2.0 * np.pi) - np.pi)
                    if rel_angle < np.pi / 4.0:
                        virus_ahead = 1.0

        hazard_vec = np.array([
            wall_front_norm,
            rel_wall_bearing,
            min_virus_d,
            virus_ahead
        ], dtype=np.float32)

        obs = np.concatenate([
            ego_state,      # 6
            threat_radar,   # 8
            prey_radar,     # 8
            threat_vec,     # 4
            prey_vec,       # 4
            food_vec,       # 4
            hazard_vec      # 4
        ], dtype=np.float32)

        return obs.astype(np.float32)
