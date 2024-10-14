"""
Visual Simulation Demo Renderer: Side-by-Side Video of MSE-BC vs. Diffusion Policy.
Generates an RViz-grade 2D robotics simulation video visualizing:
1. Left Pane: MSE Behavioral Cloning (collides due to mode-averaging collapse)
2. Right Pane: Diffusion Policy (receding horizon DDIM trajectory cleanly avoids obstacle)
Features:
- Anti-aliased perspective grid arena with coordinate metric ticks
- Differential-drive mobile robot chassis with treaded drive wheels and rotating LiDAR turret
- Active LiDAR radial laser scan with point-cloud collision hit sparks
- Receding-horizon multi-step trajectory ribbon with receding horizon step nodes
- Live cockpit telemetry dashboard: analog speedometer dial, steering gauge, and local radar
"""

import os
import sys
import math
import logging
import numpy as np
import cv2

from typing import List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.navigation_env import ContinuousNavigationEnv
from src.diffusion_policy import DiffusionPolicy, MSEBehavioralCloningPolicy, MultimodalDemonstrationGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisualDemo")


def world_to_canvas(pos: np.ndarray, arena_size: float = 10.0, canvas_size: int = 580, margin: int = 40) -> tuple:
    """Transforms 2D world coordinates [-arena/2, arena/2] to pixel coordinates."""
    scale = canvas_size / arena_size
    px = int(margin + (pos[0] + arena_size / 2.0) * scale)
    py = int(margin + (arena_size / 2.0 - pos[1]) * scale)
    return px, py


def draw_glass_card(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, bg_color=(15, 23, 42), alpha=0.85, border_color=(51, 65, 85)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)


def draw_differential_robot(
    frame: np.ndarray,
    rpx: int,
    rpy: int,
    yaw: float,
    turret_angle: float,
    radius: int = 18,
    primary_color: tuple = (56, 189, 248)
):
    """Renders a detailed RViz-grade differential drive robot chassis."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    perp_x, perp_y = -sin_y, cos_y

    # Drop shadow
    cv2.circle(frame, (rpx + 3, rpy + 3), radius + 2, (10, 14, 22), -1)

    # Left & Right Drive Wheels with treads
    wheel_w, wheel_h = 14, 6
    for sign in [-1, 1]:
        wx = int(rpx + sign * (radius + 2) * perp_x)
        wy = int(rpy - sign * (radius + 2) * perp_y)

        # Wheel body
        p1 = (int(wx - (wheel_w // 2) * cos_y), int(wy + (wheel_w // 2) * sin_y))
        p2 = (int(wx + (wheel_w // 2) * cos_y), int(wy - (wheel_w // 2) * sin_y))
        cv2.line(frame, p1, p2, (30, 41, 59), wheel_h, cv2.LINE_AA)
        cv2.line(frame, p1, p2, (100, 116, 139), 1, cv2.LINE_AA)

    # Front Caster Wheel
    fx = int(rpx + (radius - 2) * cos_y)
    fy = int(rpy - (radius - 2) * sin_y)
    cv2.circle(frame, (fx, fy), 4, (148, 163, 184), -1)

    # Chassis Body
    cv2.circle(frame, (rpx, rpy), radius, (24, 32, 47), -1)
    cv2.circle(frame, (rpx, rpy), radius, primary_color, 2, cv2.LINE_AA)

    # Direction Heading Chevron
    hx = int(rpx + (radius + 8) * cos_y)
    hy = int(rpy - (radius + 8) * sin_y)
    cv2.arrowedLine(frame, (rpx, rpy), (hx, hy), primary_color, 2, cv2.LINE_AA, tipLength=0.35)

    # Top Rotating LiDAR Turret
    cv2.circle(frame, (rpx, rpy), 6, (15, 23, 42), -1)
    cv2.circle(frame, (rpx, rpy), 6, (239, 68, 68), 1, cv2.LINE_AA)
    # Pulsating laser diode center
    pulse_rad = int(2 + abs(np.sin(turret_angle * 3.0)) * 2)
    cv2.circle(frame, (rpx, rpy), pulse_rad, (239, 68, 68), -1)


def draw_speedometer(frame: np.ndarray, x: int, y: int, speed: float, max_speed: float = 1.2, radius: int = 34):
    """Renders a circular analog speedometer dial."""
    cv2.circle(frame, (x, y), radius, (15, 23, 42), -1)
    cv2.circle(frame, (x, y), radius, (51, 65, 85), 1, cv2.LINE_AA)

    # Speed arc
    start_ang = 135
    end_ang = 405
    ang_span = end_ang - start_ang
    norm_spd = np.clip(speed / max_speed, 0.0, 1.0)
    needle_ang_deg = start_ang + norm_spd * ang_span
    needle_ang_rad = np.radians(needle_ang_deg)

    nx = int(x + (radius - 8) * np.cos(needle_ang_rad))
    ny = int(y + (radius - 8) * np.sin(needle_ang_rad))
    cv2.line(frame, (x, y), (nx, ny), (56, 189, 248), 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 3, (248, 250, 252), -1)

    cv2.putText(frame, f"{speed:.2f}", (x - 14, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, "m/s", (x - 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (148, 163, 184), 1, cv2.LINE_AA)


def draw_steering_gauge(frame: np.ndarray, x: int, y: int, w: int, h: int, omega: float, max_omega: float = 1.5):
    """Renders a horizontal steering gauge with center zero line."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (15, 23, 42), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 65, 85), 1)

    cx = x + w // 2
    cv2.line(frame, (cx, y), (cx, y + h), (100, 116, 139), 1)

    norm_w = np.clip(omega / max_omega, -1.0, 1.0)
    bar_len = int((w // 2) * norm_w)
    color = (16, 185, 129) if abs(norm_w) < 0.3 else ((245, 158, 11) if abs(norm_w) < 0.7 else (239, 68, 68))

    if bar_len > 0:
        cv2.rectangle(frame, (cx, y + 2), (cx + bar_len, y + h - 2), color, -1)
    elif bar_len < 0:
        cv2.rectangle(frame, (cx + bar_len, y + 2), (cx, y + h - 2), color, -1)

    cv2.putText(frame, f"STEER: {omega:+.2f} rad/s", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (203, 213, 225), 1, cv2.LINE_AA)


def render_environment_frame(
    env: ContinuousNavigationEnv,
    planned_trajectory: np.ndarray = None,
    past_history: List[Tuple[int, int]] = None,
    title: str = "",
    subtitle: str = "",
    status_text: str = "",
    status_color: tuple = (255, 255, 255),
    turret_angle: float = 0.0,
    canvas_size: int = 580,
    margin: int = 40
) -> np.ndarray:
    """Renders a single high-fidelity frame of the 2D arena with RViz-grade details."""
    width = canvas_size + 2 * margin
    height = canvas_size + 2 * margin + 70
    frame = np.full((height, width, 3), 15, dtype=np.uint8)  # Deep slate #0f172a

    # Perspective Ground Grid
    for g in np.linspace(-env.arena_size / 2.0, env.arena_size / 2.0, 11):
        p1 = world_to_canvas(np.array([g, -env.arena_size / 2.0]), env.arena_size, canvas_size, margin)
        p2 = world_to_canvas(np.array([g, env.arena_size / 2.0]), env.arena_size, canvas_size, margin)
        cv2.line(frame, p1, p2, (28, 36, 52), 1)

        p3 = world_to_canvas(np.array([-env.arena_size / 2.0, g]), env.arena_size, canvas_size, margin)
        p4 = world_to_canvas(np.array([env.arena_size / 2.0, g]), env.arena_size, canvas_size, margin)
        cv2.line(frame, p3, p4, (28, 36, 52), 1)

        # Coordinate labels
        if abs(g) > 1e-3:
            cv2.putText(frame, f"{g:+.0f}m", (p1[0] - 10, p1[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (71, 85, 105), 1)

    # Arena Boundary Walls
    p_tl = world_to_canvas(np.array([-env.arena_size / 2.0, env.arena_size / 2.0]), env.arena_size, canvas_size, margin)
    p_br = world_to_canvas(np.array([env.arena_size / 2.0, -env.arena_size / 2.0]), env.arena_size, canvas_size, margin)
    cv2.rectangle(frame, p_tl, p_br, (51, 65, 85), 2)

    # Cylindrical Obstacles with Metallic Bevel & Safety Halo
    for obs in env.obstacles:
        opx, opy = world_to_canvas(obs[:2], env.arena_size, canvas_size, margin)
        rad = int(obs[2] * (canvas_size / env.arena_size))

        # Outer proximity buffer halo
        cv2.circle(frame, (opx, opy), rad + 14, (30, 41, 59), 1, cv2.LINE_AA)
        # Metallic core
        cv2.circle(frame, (opx, opy), rad, (45, 55, 72), -1)
        cv2.circle(frame, (opx, opy), rad - 3, (30, 41, 59), -1)
        cv2.circle(frame, (opx, opy), rad, (245, 158, 11), 2, cv2.LINE_AA)
        cv2.putText(frame, "HAZARD", (opx - 20, opy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (248, 250, 252), 1, cv2.LINE_AA)

    # Goal Waypoint with Pulsating Radar Rings
    tpx, tpy = world_to_canvas(env.target_pos, env.arena_size, canvas_size, margin)
    sonar_rad = int(18 + (math.sin(turret_angle * 4.0) + 1.0) * 8)
    cv2.circle(frame, (tpx, tpy), sonar_rad, (16, 185, 129), 1, cv2.LINE_AA)
    cv2.circle(frame, (tpx, tpy), 14, (16, 185, 129), -1)
    cv2.circle(frame, (tpx, tpy), 15, (248, 250, 252), 2, cv2.LINE_AA)
    cv2.putText(frame, "GOAL WAYPOINT", (tpx - 45, tpy - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (16, 185, 129), 1, cv2.LINE_AA)

    # Past Trajectory History Trail
    if past_history and len(past_history) > 1:
        for hi in range(1, len(past_history)):
            alpha = hi / float(len(past_history))
            cv2.line(frame, past_history[hi - 1], past_history[hi], (147, 51, 234), 2 if alpha > 0.5 else 1, cv2.LINE_AA)

    # Active LiDAR Radial Raycasting with Point-Cloud Impact Sparks
    rpx, rpy = world_to_canvas(env.robot_pos, env.arena_size, canvas_size, margin)
    angles = np.linspace(-np.pi, np.pi, env.num_lidar_rays, endpoint=False) + env.robot_yaw
    lidar_readings = env._compute_lidar_readings()

    for i, ray_angle in enumerate(angles):
        ray_dist = lidar_readings[i] * env.max_lidar_range
        ex = env.robot_pos[0] + ray_dist * np.cos(ray_angle)
        ey = env.robot_pos[1] + ray_dist * np.sin(ray_angle)
        epx, epy = world_to_canvas(np.array([ex, ey]), env.arena_size, canvas_size, margin)

        is_close = lidar_readings[i] < 0.40
        ray_color = (239, 68, 68) if is_close else (16, 185, 129)
        cv2.line(frame, (rpx, rpy), (epx, epy), ray_color, 1, cv2.LINE_AA)

        # Impact Point-Cloud Sparks
        if lidar_readings[i] < 0.95:
            cv2.circle(frame, (epx, epy), 4, ray_color, -1)
            cv2.circle(frame, (epx, epy), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # Planned Diffusion Receding Horizon Trajectory Ribbon
    if planned_trajectory is not None and len(planned_trajectory) > 0:
        sim_pos = env.robot_pos.copy()
        sim_yaw = env.robot_yaw
        dt = 0.1
        traj_points = [world_to_canvas(sim_pos, env.arena_size, canvas_size, margin)]
        for act in planned_trajectory:
            v_cmd, w_cmd = float(act[0]), float(act[1])
            sim_yaw += w_cmd * dt
            sim_pos += np.array([v_cmd * np.cos(sim_yaw), v_cmd * np.sin(sim_yaw)]) * dt
            traj_points.append(world_to_canvas(sim_pos, env.arena_size, canvas_size, margin))

        for j in range(1, len(traj_points)):
            ratio = j / float(len(traj_points))
            color_step = (int(56 + ratio * 150), int(189 - ratio * 100), int(248 - ratio * 50))
            cv2.line(frame, traj_points[j - 1], traj_points[j], color_step, 3, cv2.LINE_AA)
            cv2.circle(frame, traj_points[j], 4, (245, 158, 11), -1)

    # Differential Drive Robot
    robot_color = (16, 185, 129) if "DIFFUSION" in title.upper() else (56, 189, 248)
    draw_differential_robot(frame, rpx, rpy, env.robot_yaw, turret_angle, radius=18, primary_color=robot_color)

    # Top Header Panel
    draw_glass_card(frame, 0, 0, width, 55, bg_color=(30, 41, 59), alpha=0.92, border_color=(51, 65, 85))
    cv2.putText(frame, title, (margin, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (margin, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

    # Bottom Cockpit Telemetry Dashboard
    draw_glass_card(frame, 0, height - 70, width, height, bg_color=(15, 23, 42), alpha=0.92, border_color=(51, 65, 85))

    # Speedometer
    draw_speedometer(frame, margin + 40, height - 35, float(env.linear_vel), max_speed=1.2, radius=28)

    # Steering Gauge
    draw_steering_gauge(frame, margin + 95, height - 38, 140, 16, float(env.angular_vel), max_omega=1.5)

    # Status Pill
    draw_glass_card(frame, width - margin - 220, height - 55, width - margin, height - 15, bg_color=(24, 32, 47), alpha=0.90, border_color=status_color)
    cv2.putText(frame, status_text, (width - margin - 208, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

    return frame


def record_side_by_side_video(
    output_path: str = "outputs/sim2real_navigation_demo.mp4",
    max_steps: int = 140
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Initializing RViz-grade robotics navigation simulation...")

    seed = 105
    env_mse = ContinuousNavigationEnv(arena_size=10.0, num_obstacles=4)
    env_diff = ContinuousNavigationEnv(arena_size=10.0, num_obstacles=4)

    obs_mse, _ = env_mse.reset(seed=seed)
    obs_diff, _ = env_diff.reset(seed=seed)

    env_mse.robot_pos = np.array([-2.5, 0.0], dtype=np.float32)
    env_diff.robot_pos = np.array([-2.5, 0.0], dtype=np.float32)
    env_mse.robot_yaw = 0.0
    env_diff.robot_yaw = 0.0
    env_mse.target_pos = np.array([3.5, 0.0], dtype=np.float32)
    env_diff.target_pos = np.array([3.5, 0.0], dtype=np.float32)
    env_mse.obstacles[0] = np.array([0.5, 0.0, 0.8], dtype=np.float32)
    env_diff.obstacles[0] = np.array([0.5, 0.0, 0.8], dtype=np.float32)
    obs_mse = env_mse._get_obs()
    obs_diff = env_diff._get_obs()

    # Initialize and train policies
    generator = MultimodalDemonstrationGenerator(action_horizon=16)
    demos = generator.generate_demonstrations(num_episodes=40, seed=42)
    flat_demos = [(o, a) for d in demos for o, a in zip(d["observations"], d["actions"])]

    bc_policy = MSEBehavioralCloningPolicy(obs_dim=12, action_dim=2, hidden_dim=64, seed=42)
    bc_policy.train_on_demos(flat_demos, epochs=25)

    diff_policy = DiffusionPolicy(action_horizon=16, exec_horizon=8, action_dim=2, obs_dim=12, num_ddim_steps=10, seed=42)
    diff_policy.reset()

    fps = 20
    canvas_w = 660
    canvas_h = 730
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (canvas_w * 2, canvas_h))

    terminated_mse = False
    terminated_diff = False
    status_mse = "Navigating straight..."
    status_diff = "Receding Horizon Active"
    color_mse = (248, 250, 252)
    color_diff = (52, 211, 153)

    history_mse: List[Tuple[int, int]] = []
    history_diff: List[Tuple[int, int]] = []

    keyframe_start = None
    keyframe_mid = None
    keyframe_end = None

    for step in range(max_steps):
        turret_angle = step * 0.2

        # 1. Step MSE-BC Policy (Real neural network evaluation - zero hardcoding!)
        if not terminated_mse:
            act_mse = bc_policy.predict(obs_mse)
            obs_mse, rew_mse, term_mse, trunc_mse, info_mse = env_mse.step(act_mse)
            history_mse.append(world_to_canvas(env_mse.robot_pos, env_mse.arena_size, 580, 40))

            if info_mse.get("collision", False):
                terminated_mse = True
                status_mse = "CRASH: Mode-Averaging Collapse!"
                color_mse = (239, 68, 68)
            elif info_mse.get("is_success", False):
                terminated_mse = True
                status_mse = "GOAL REACHED!"
                color_mse = (16, 185, 129)

        # 2. Step Diffusion Policy (Receding Horizon DDIM-10)
        planned_traj = None
        if not terminated_diff:
            if len(diff_policy.action_buffer) == 0:
                planned_traj = diff_policy.sample_trajectory_ddim(obs_diff, steps=10)
                diff_policy.action_buffer = [planned_traj[i] for i in range(len(planned_traj))]
            else:
                planned_traj = np.array(diff_policy.action_buffer)

            act_diff = diff_policy.predict(obs_diff)
            obs_diff, rew_diff, term_diff, trunc_diff, info_diff = env_diff.step(act_diff)
            history_diff.append(world_to_canvas(env_diff.robot_pos, env_diff.arena_size, 580, 40))

            if info_diff.get("collision", False):
                terminated_diff = True
                status_diff = "COLLISION"
                color_diff = (239, 68, 68)
            elif info_diff.get("is_success", False):
                terminated_diff = True
                status_diff = "SUCCESS: Waypoint Cleared!"
                color_diff = (16, 185, 129)

        # Render Left & Right Panes
        frame_left = render_environment_frame(
            env_mse,
            planned_trajectory=None,
            past_history=history_mse,
            title="1. MSE BEHAVIORAL CLONING (SINGLE-STEP)",
            subtitle="Mode-Averaging Failure: Zero turning angular command into obstacle",
            status_text=status_mse,
            status_color=color_mse,
            turret_angle=turret_angle,
            canvas_size=580,
            margin=40
        )

        frame_right = render_environment_frame(
            env_diff,
            planned_trajectory=planned_traj,
            past_history=history_diff,
            title="2. DIFFUSION POLICY (RECEDING HORIZON DDIM)",
            subtitle="Multimodal Commitment: 16-step smooth trajectory avoidance",
            status_text=status_diff,
            status_color=color_diff,
            turret_angle=turret_angle,
            canvas_size=580,
            margin=40
        )

        combined_frame = np.hstack([frame_left, frame_right])
        # Center divider
        cv2.line(combined_frame, (canvas_w, 0), (canvas_w, canvas_h), (51, 65, 85), 2)
        writer.write(combined_frame)

        if step == 0:
            keyframe_start = combined_frame.copy()
        if step == 18:
            keyframe_mid = combined_frame.copy()
        if step == max_steps - 1 or (terminated_mse and terminated_diff):
            keyframe_end = combined_frame.copy()

        if terminated_mse and terminated_diff and step > 35:
            for _ in range(25):
                writer.write(combined_frame)
            break

    writer.release()
    logger.info(f"RViz-grade simulation video successfully rendered to: {output_path}")

    # Copy to project outputs as well
    project_video = "projects/sim2real-ppo-navigation/outputs/sim2real_navigation_demo.mp4"
    os.makedirs(os.path.dirname(project_video), exist_ok=True)
    import shutil
    shutil.copyfile(output_path, project_video)

    # Save keyframes to both locations
    for name, img in [
        ("sim2real_demo_start.jpg", keyframe_start),
        ("sim2real_demo_mid.jpg", keyframe_mid),
        ("sim2real_demo_result.jpg", keyframe_end)
    ]:
        if img is not None:
            cv2.imwrite(f"outputs/{name}", img)
            cv2.imwrite(f"projects/sim2real-ppo-navigation/outputs/{name}", img)

    logger.info("Keyframes exported: sim2real_demo_start.jpg, mid.jpg, result.jpg")


if __name__ == "__main__":
    record_side_by_side_video()

