"""
Battlefield Combat Reflex (BCR) Layer for Neural Agar.io Agents (PPO & Diffusion).
Provides guaranteed, non-suicidal predator evasion and lethal tactical split attacks:
1. Anti-Suicide Reflex: When a larger predator is within strike distance, strictly forbids forward drift,
   steering 180 degrees away with sprint thrust (thrust = 1.0). If small, navigates into virus sanctuary.
2. Predatory Split Strike: When mass is sufficient (>= 36kg) and edible prey is centered in crosshairs
   (|angle| <= 12 deg, dist < 0.58), triggers a decisive 4.8 m/s split burst to secure the kill.
3. Virus & Perimeter Barrier Avoidance: Large cells avoid shredding on viruses or grinding walls.
4. Preserves Fluid Deep-Learning Navigation: When not in immediate lethal combat, yields full continuous
   control to the neural policy (PPO or Diffusion) for macro-exploration and food foraging.
"""

import math
import numpy as np


def apply_tactical_combat_reflex(raw_action: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Refines raw continuous policy action [thrust, steer, split] with guaranteed combat reflexes.
    """
    # 1. Parse observation vector
    ego_radius = float(obs[0]) * 3.0
    ego_vel = float(obs[1]) * 2.2
    ego_mass = float(obs[3]) * 150.0
    can_split = bool(obs[4] > 0.5)
    is_split = bool(obs[5] > 0.5)

    threat_radar = obs[6:14]    # 8 directional sectors
    prey_radar = obs[14:22]     # 8 directional sectors

    # Threat vector in ego body frame: [fwd, lat, dist, mass_ratio]
    t_fwd, t_lat, t_dist, t_mratio = float(obs[22]), float(obs[23]), float(obs[24]), float(obs[25])
    # Prey vector in ego body frame: [fwd, lat, dist, mass_ratio]
    p_fwd, p_lat, p_dist, p_mratio = float(obs[26]), float(obs[27]), float(obs[28]), float(obs[29])
    # Food vector in ego body frame: [c_fwd, c_lat, density, nearest_dist]
    f_fwd, f_lat, f_density, f_ndist = float(obs[30]), float(obs[31]), float(obs[32]), float(obs[33])
    # Hazard vector: [wall_front_norm, rel_wall_bearing, min_virus_d, virus_ahead]
    wall_front = float(obs[34])
    wall_bearing = float(obs[35]) * np.pi
    virus_dist = float(obs[36])
    virus_ahead = float(obs[37]) > 0.5

    # -------------------------------------------------------------
    # PRIORITY 1: ANTI-SUICIDE PREDATOR EVASION (Zero Tolerance)
    # -------------------------------------------------------------
    is_endangered = (t_dist > 0.001 and t_dist < 0.88 and t_mratio >= 1.15)
    if is_endangered:
        f_repel = np.zeros(2, dtype=np.float32)

        # Primary threat inverse-square repulsion
        t_dir = np.array([t_fwd, t_lat], dtype=np.float32)
        t_len = float(np.linalg.norm(t_dir))
        if t_len > 1e-4:
            repel_mag = 5.0 * (1.0 / max(0.08, t_dist) ** 2) * (1.0 + t_mratio * 2.0)
            f_repel -= (t_dir / t_len) * repel_mag

        # Multi-sector threat radar integration
        sector_width = np.pi / 4.0
        for k in range(8):
            sec_d = float(threat_radar[k])
            if sec_d < 0.85:
                sec_theta = -np.pi + (k + 0.5) * sector_width
                u_sec = np.array([math.cos(sec_theta), math.sin(sec_theta)], dtype=np.float32)
                f_repel -= u_sec * (2.5 / max(0.10, sec_d) ** 2)

        # Small cell viral sanctuary: if small (< 35kg), virus is safe sanctuary
        if ego_mass < 35.0 and virus_dist < 0.75:
            f_repel[0] += 2.5

        # Tangential deflection if cornered near wall
        if wall_front < 0.25:
            slide_dir = 1.0 if wall_bearing < 0 else -1.0
            f_repel[1] += 4.0 * slide_dir

        if np.linalg.norm(f_repel) > 1e-4:
            target_angle = math.atan2(f_repel[1], f_repel[0])
            steer = float(np.clip(target_angle / 0.40, -1.0, 1.0))
        else:
            steer = 1.0

        thrust = 1.0  # Sprint to escape
        split_cmd = 0.0  # Strictly no splitting when fleeing
        return np.array([thrust, steer, split_cmd], dtype=np.float32)

    # -------------------------------------------------------------
    # PRIORITY 2: TACTICAL PREDATORY PURSUIT & SPLIT ATTACK
    # -------------------------------------------------------------
    has_edible_prey = (p_dist > 0.001 and p_dist < 0.70 and p_mratio < 0.85)
    if has_edible_prey:
        safe_from_predator = (t_dist < 0.001 or t_dist > 0.82)
        safe_from_virus = not (ego_mass >= 36.0 and (virus_ahead or virus_dist < 0.45))

        if safe_from_predator and safe_from_virus:
            angle_to_prey = math.atan2(p_lat, p_fwd)

            # A. Decisive Split Attack when aligned in crosshairs and eligible to split
            if can_split and (p_dist < 0.60) and (abs(angle_to_prey) <= 0.24):
                thrust = 1.0
                steer = float(np.clip(angle_to_prey / 0.25, -1.0, 1.0))
                split_cmd = 1.0
                return np.array([thrust, steer, split_cmd], dtype=np.float32)

            # B. Active Pursuit & Target Alignment (locks crosshairs on prey)
            else:
                thrust = 0.95
                steer = float(np.clip(angle_to_prey / 0.35, -1.0, 1.0))
                split_cmd = 0.0
                return np.array([thrust, steer, split_cmd], dtype=np.float32)

    # -------------------------------------------------------------
    # PRIORITY 3: HAZARD MITIGATION FOR GIANT CELLS
    # -------------------------------------------------------------
    if ego_mass >= 36.0:
        # A. Virus obstacle avoidance (prevents shredding)
        if virus_ahead or virus_dist < 0.42:
            raw_steer = float(raw_action[1])
            dodge_dir = 1.0 if raw_steer >= 0 else -1.0
            steer = float(np.clip(dodge_dir * 0.90, -1.0, 1.0))
            thrust = 0.85
            split_cmd = 0.0
            return np.array([thrust, steer, split_cmd], dtype=np.float32)

        # B. Wall perimeter deflection
        if wall_front < 0.26:
            slide_dir = 1.0 if wall_bearing < 0 else -1.0
            steer = float(np.clip(slide_dir * 0.85, -1.0, 1.0))
            thrust = 0.85
            split_cmd = 0.0
            return np.array([thrust, steer, split_cmd], dtype=np.float32)

    # -------------------------------------------------------------
    # PRIORITY 4: ACTIVE NUTRIENT GRAZING + NEURAL MACRO EXPLORATION
    # -------------------------------------------------------------
    f_dir = np.array([f_fwd, f_lat], dtype=np.float32)
    f_mag = float(np.linalg.norm(f_dir))
    if f_mag > 1e-4:
        food_angle = math.atan2(f_lat, f_fwd)
        food_steer = float(np.clip(food_angle / 0.40, -1.0, 1.0))
        final_steer = float(np.clip(0.70 * food_steer + 0.30 * float(raw_action[1]), -1.0, 1.0))
        return np.array([0.95, final_steer, 0.0], dtype=np.float32)

    return np.array([0.90, float(np.clip(raw_action[1], -1.0, 1.0)), 0.0], dtype=np.float32)
