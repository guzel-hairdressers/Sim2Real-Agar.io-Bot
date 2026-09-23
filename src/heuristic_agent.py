"""
Master Heuristic Agar.io Bot: High-Performance Deterministic Tactical Policy.
Features:
1. Multi-Sector Potential Field Predator Evasion (threat_radar + threat_vec) with inverse-square repulsion.
2. Opportunistic Tactical Split Attack (4.8 m/s burst lunge when prey is aligned in crosshairs).
3. Dual-Regime Virus Logic:
   - Heavy cells (>= 40kg): Treat viruses as fatal hazards with strong obstacle avoidance.
   - Light cells (< 35kg): Use viruses as protective shields/bunkers against large predators.
4. Wall Tangent Deflection: Prevents being cornered or driving perpendicularly into perimeter boundaries.
5. High-Efficiency Nutrient Foraging: Vector-directed navigation toward dense food clusters.
6. Profiles: 'apex' (balanced master), 'hunter' (aggressive split predator), 'survivor' (ultra-evasive).
"""

from typing import Optional, Tuple
import math
import numpy as np


class MasterHeuristicAgarBot:
    """
    Unhandicapped, deterministic Agar.io tactical agent.
    Consumes the 38-dim POMDP observation vector and produces [thrust, steer, split].
    """

    def __init__(self, pid: str = "heuristic_bot", profile: str = "apex", seed: Optional[int] = None):
        self.pid = pid
        self.profile = profile.lower()
        self.rng = np.random.RandomState(seed)

        # Profile parameters
        if self.profile == "hunter":
            self.threat_threshold = 0.65       # Focuses on hunting until threat is close
            self.threat_gain = 3.5
            self.hunt_gain = 2.8
            self.split_dist_max = 0.62        # Splits from further away
            self.split_angle_max = 0.26       # Wider split tolerance (~15 deg)
            self.food_gain = 0.70
            self.base_thrust = 0.95
        elif self.profile == "survivor":
            self.threat_threshold = 0.95       # Ultra-vigilant: flees early
            self.threat_gain = 5.0
            self.hunt_gain = 1.6
            self.split_dist_max = 0.40        # Only splits when very close
            self.split_angle_max = 0.15       # Tight split angle
            self.food_gain = 1.10
            self.base_thrust = 0.85
        else:  # "apex" (balanced master)
            self.threat_threshold = 0.80
            self.threat_gain = 4.2
            self.hunt_gain = 2.2
            self.split_dist_max = 0.55
            self.split_angle_max = 0.20       # ~11.5 degrees
            self.food_gain = 0.90
            self.base_thrust = 0.90

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Computes the deterministic action [thrust, steer, split] from 38-dim observation.
        """
        # 1. Parse observation vector
        ego_radius = float(obs[0]) * 3.0
        ego_vel = float(obs[1]) * 2.2
        ego_mass = float(obs[3]) * 150.0
        can_split = bool(obs[4] > 0.5)
        is_split = bool(obs[5] > 0.5)

        threat_radar = obs[6:14]    # 8 sectors
        prey_radar = obs[14:22]     # 8 sectors

        # Threat vector in ego body frame: [fwd, lat, dist, mass_ratio]
        t_fwd, t_lat, t_dist, t_mratio = float(obs[22]), float(obs[23]), float(obs[24]), float(obs[25])
        # Prey vector in ego body frame: [fwd, lat, dist, mass_ratio]
        p_fwd, p_lat, p_dist, p_mratio = float(obs[26]), float(obs[27]), float(obs[28]), float(obs[29])
        # Food vector in ego body frame: [c_fwd, c_lat, density, nearest_dist]
        f_fwd, f_lat, f_density, f_ndist = float(obs[30]), float(obs[31]), float(obs[32]), float(obs[33])
        # Hazard vector: [wall_front_norm, rel_wall_bearing, min_virus_d, virus_ahead]
        wall_front = float(obs[34])
        wall_bearing = float(obs[35]) * np.pi  # Convert from [-1, 1] to radians [-pi, pi]
        virus_dist = float(obs[36])
        virus_ahead = float(obs[37]) > 0.5

        # Initialize cumulative steering force in body frame: [F_fwd, F_lat]
        f_vec = np.zeros(2, dtype=np.float32)
        thrust = self.base_thrust
        split_cmd = 0.0

        has_threat = (t_dist > 0.001 and t_dist < self.threat_threshold)
        has_prey = (p_dist > 0.001 and p_dist < 0.90)

        # -------------------------------------------------------------
        # STEP 1: PREDATOR EVASION (Highest Priority)
        # -------------------------------------------------------------
        if has_threat:
            thrust = 1.0  # Maximum sprint when endangered

            # Primary threat repulsion (opposite of [t_fwd, t_lat])
            t_dir = np.array([t_fwd, t_lat], dtype=np.float32)
            t_norm = float(np.linalg.norm(t_dir))
            if t_norm > 1e-4:
                # Force scaled inversely with distance and proportionally with predator mass
                repel_mag = self.threat_gain * (1.0 / max(0.08, t_dist) ** 2) * (1.0 + t_mratio * 2.0)
                f_vec -= (t_dir / t_norm) * repel_mag

            # Multi-sector radar integration (repel from all sectors containing threats)
            sector_width = np.pi / 4.0
            for k in range(8):
                sec_d = float(threat_radar[k])
                if sec_d < 0.85:
                    # Sector center angle relative to ego yaw
                    sec_theta = -np.pi + (k + 0.5) * sector_width
                    u_sec = np.array([math.cos(sec_theta), math.sin(sec_theta)], dtype=np.float32)
                    sec_mag = (self.threat_gain * 0.45) * (1.0 / max(0.10, sec_d) ** 2)
                    f_vec -= u_sec * sec_mag

            # Small cell viral shielding: if small (< 35kg), virus is a safe shield!
            if ego_mass < 35.0 and virus_dist < 0.70:
                f_vec[0] += 1.2

        # -------------------------------------------------------------
        # STEP 2: PREY PURSUIT & TACTICAL SPLIT ATTACK
        # -------------------------------------------------------------
        elif has_prey:
            p_dir = np.array([p_fwd, p_lat], dtype=np.float32)
            p_norm = float(np.linalg.norm(p_dir))
            if p_norm > 1e-4:
                attract_mag = self.hunt_gain * (1.0 / max(0.15, p_dist))
                f_vec += (p_dir / p_norm) * attract_mag

            thrust = 0.95
            angle_to_prey = math.atan2(p_lat, p_fwd)

            # Evaluate Tactical Split Attack
            # Requirements:
            # 1. Can split (eligible piece >= 36kg)
            # 2. Prey within split strike envelope (p_dist < split_dist_max)
            # 3. Prey centered in crosshairs (|angle| < split_angle_max)
            # 4. No nearby predator lurking to consume split half
            # 5. Not splitting straight into a virus if ego is poppable
            safe_from_predator = (t_dist < 0.001 or t_dist > 0.75)
            safe_from_virus = not (ego_mass >= 40.0 and virus_ahead)

            if can_split and (p_dist < self.split_dist_max) and (abs(angle_to_prey) < self.split_angle_max) and safe_from_predator and safe_from_virus:
                split_cmd = 1.0
                thrust = 1.0
                f_vec = p_dir * 10.0  # Lock onto target during split

        # -------------------------------------------------------------
        # STEP 3: NUTRIENT FORAGING (When no immediate combat)
        # -------------------------------------------------------------
        else:
            if abs(f_fwd) > 1e-3 or abs(f_lat) > 1e-3:
                food_dir = np.array([f_fwd, f_lat], dtype=np.float32)
                f_mag = float(np.linalg.norm(food_dir))
                if f_mag > 1e-4:
                    f_vec += (food_dir / f_mag) * (self.food_gain * (1.0 + f_density * 2.0))
            else:
                # Default exploration drift
                f_vec += np.array([1.0, 0.0], dtype=np.float32)
            thrust = self.base_thrust

        # -------------------------------------------------------------
        # STEP 4: HAZARD AVOIDANCE (Viruses & Boundaries)
        # -------------------------------------------------------------
        # A. Virus obstacle avoidance for heavy cells (>= 40kg)
        if ego_mass >= 40.0:
            if virus_ahead or virus_dist < 0.45:
                v_repel = 4.5 / max(0.10, virus_dist)
                # Deflect sharply laterally
                f_vec[0] -= v_repel * 0.7
                f_vec[1] += v_repel * (1.0 if f_vec[1] >= 0 else -1.0)
                # Never split when virus is nearby
                split_cmd = 0.0

        # B. Wall Proximity Deflection & Tangent Sliding
        if wall_front < 0.30:
            # Wall directly ahead: strong pushback + tangent slide
            w_repel = 5.0 * (0.30 - wall_front) / 0.30
            f_vec[0] -= w_repel * 2.0
            # Slide along the wall away from the closest perpendicular direction
            slide_dir = 1.0 if wall_bearing < 0 else -1.0
            f_vec[1] += w_repel * slide_dir * 2.5

        # -------------------------------------------------------------
        # STEP 5: COMPUTE FINAL STEER & THRUST
        # -------------------------------------------------------------
        if np.linalg.norm(f_vec) < 1e-4:
            target_angle = 0.0
        else:
            target_angle = math.atan2(f_vec[1], f_vec[0])

        steer = float(np.clip(target_angle / 0.40, -1.0, 1.0))
        thrust = float(np.clip(thrust, 0.0, 1.0))

        return np.array([thrust, steer, split_cmd], dtype=np.float32)
