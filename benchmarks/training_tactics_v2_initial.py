# Frozen teacher/filter used to produce the accepted training data and PPO.
# Reproduction only; deployment uses src.arena_tactics with revised clearance.
"""Observable tactical teacher and bounded interventions for learned policies.

The teacher supplies demonstrations. The safety filter only changes actions
near threats/hazards or rejects an unsafe split; it does not replace ordinary
neural foraging. Neither method sees the simulator or hidden opponents.
"""
import math
import numpy as np


def context(obs):
    mass = float(obs[3]) * 150
    radius = float(np.clip(5.5 * (max(1, mass) / 15) ** .22, 5, 9.5))
    if len(obs) >= 70:
        return mass, radius, obs[42:48], obs[48:54], obs[64:70], float(obs[38]) * 150
    # Legacy ratios have opposite numerators, and both are scaled by five.
    t = np.array([obs[22], obs[23], 0, 0, mass * obs[25] * 5 / 150, obs[24]])
    prey_mass = mass / max(1e-5, obs[29] * 5) if obs[28] > 0 else 0
    p = np.array([obs[26], obs[27], 0, 0, prey_mass / 150, obs[28]])
    return mass, radius, t, p, t, mass


def split_is_safe(obs):
    mass, radius, threat, prey, post_threat, largest = context(obs)
    if obs[4] < .5 or prey[5] <= 0:
        return False
    # Legacy observations cannot establish the sizes of already split pieces.
    if len(obs) < 70 and obs[5] > .5:
        return False
    half, target = largest / 2, float(prey[4]) * 150
    if half < target + 7 or half < 1.22 * target:
        return False
    if post_threat[5] > 0 and post_threat[5] * radius < 3.0:
        return False
    if threat[5] > 0 and threat[5] * radius < 3.5:
        return False
    if obs[34] * radius < 2 or (obs[37] > .5 and obs[36] * radius < 2.5):
        return False
    if abs(math.atan2(prey[1], prey[0])) > .16:
        return False
    # Conservative constant-velocity interception, including projectile drag.
    # Actual collision and outcome remain simulator measurements, not a claim
    # that this approximation guarantees a kill.
    pos = np.array([.35 * math.sqrt(half / 15) * 1.6, 0.])
    target_pos = np.asarray(prey[:2], dtype=float) * radius
    target_v = np.asarray(prey[2:4], dtype=float) * 6
    if len(obs) >= 70:
        target_v = target_v + obs[60:62] * 6
    velocity = 4.8
    terminal = min(2.2, 1.7 * (15 / half) ** .4) * 1.35
    capture = .35 * math.sqrt(half / 15) + .07 * math.sqrt(target / 15)
    for _ in range(18):
        velocity += (terminal - velocity) * (3.5 / 30)
        pos[0] += velocity / 30
        target_pos += target_v / 30
        if np.linalg.norm(pos - target_pos) < capture * .9:
            return True
    return False


def _wall_force(obs, radius):
    bearing = float(obs[35]) * math.pi
    if len(obs) < 70:
        distance = float(obs[34]) * radius
        return -np.array([math.cos(bearing), math.sin(bearing)]) * max(0, 1.3 - distance) * 3
    distances = obs[56:60] * radius
    wall_angles = [math.pi, 0, -math.pi / 2, math.pi / 2]
    yaw = wall_angles[int(np.argmin(distances))] - bearing
    force = np.zeros(2)
    for d, angle in zip(distances, wall_angles):
        if d < 1.4:
            inward = angle - yaw + math.pi
            force += np.array([math.cos(inward), math.sin(inward)]) * (1.4 - d) * 4
    return force


def teacher_action(obs):
    """Deterministic teacher based exclusively on the observation contract."""
    mass, radius, threat, prey, _, _ = context(obs)
    danger = threat[5] > 0 and threat[5] * radius < 4
    force = np.zeros(2)
    if danger:
        displacement = threat[:2] * radius
        predicted = displacement + threat[2:4] * 6 * .35
        force -= predicted / max(.2, np.linalg.norm(predicted)) * 4
        # Incorporate multiple directions, without treating total player mass
        # as piece mass in the new observation contract.
        if len(obs) < 70:
            for k, d in enumerate(obs[6:14]):
                if d < .75:
                    angle = -math.pi + (k + .5) * math.pi / 4
                    force -= np.array([math.cos(angle), math.sin(angle)]) / max(.1, float(d))
    elif prey[5] > 0 and prey[5] * radius < 4:
        lead = prey[:2] * radius + prey[2:4] * 6 * .3
        force += lead / max(.1, np.linalg.norm(lead)) * 2
    else:
        food = obs[62:64] if len(obs) >= 70 else obs[30:32]
        force += food / max(.01, np.linalg.norm(food)) * 1.5
        if np.linalg.norm(force) < .01:
            force[0] = 1
    force += _wall_force(obs, radius)
    if mass >= 36 and obs[36] * radius < 2.2:
        virus = obs[54:56] if len(obs) >= 70 else np.array([obs[36], 0])
        distance = np.linalg.norm(virus) * radius
        force -= virus / max(.01, np.linalg.norm(virus)) * max(0, 2.2 - distance) * 5
    angle = math.atan2(force[1], force[0])
    # Brake while facing a nearby predator/obstacle instead of sprinting at it
    # throughout a 180-degree turn. Existing velocity still has inertia.
    thrust = .18 if abs(angle) > 1.3 else (.65 if abs(angle) > .7 else .95)
    if danger and abs(angle) < .7:
        thrust = 1.
    return np.array([thrust, np.clip(angle / .45, -1, 1), float(split_is_safe(obs) and not danger)], np.float32)


def safety_filter(raw_action, obs):
    """Preserve learned actions except observable imminent risk/invalid split."""
    action = np.clip(np.asarray(raw_action), [0, -1, 0], [1, 1, 1]).astype(np.float32)
    mass, radius, threat, _, _, _ = context(obs)
    danger = threat[5] > 0 and threat[5] * radius < 3.2
    wall = obs[34] * radius < .9
    virus = mass >= 36 and obs[37] > .5 and obs[36] * radius < 1.8
    if danger or wall or virus:
        action = teacher_action(obs)
    if action[2] > .5 and not split_is_safe(obs):
        action[2] = 0
    return action
