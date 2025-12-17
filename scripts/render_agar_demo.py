"""
Arcade-Grade Multi-Agent Agar.io Visual Demo Renderer.
Renders an authentic, production-grade 30-FPS video demonstrating:
1. Dynamic Player Follow-Cam centered on primary agent
2. Realistic Fog of War / Partial Observability (illuminated vision circle R_vis surrounded by deep shadow)
3. Glowing cellular membranes with pulse effects, mass labels, and heading vectors
4. Diffusion Policy receding-horizon multi-step trajectory ribbon (16-step planned escape/hunt path)
5. Predation collision shockwave particle rings
6. Minimap Radar Inset showing global field and vision cone
7. Live Agar Leaderboard (#1..#5) and Cockpit Telemetry HUD
"""

import os
import sys
import math
import logging
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_agar_parallel_gpu import TorchAgarPPOAgent
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarRenderer")


def draw_spiked_virus(img: np.ndarray, center: tuple, radius: int, spikes: int = 14, angle_offset: float = 0.0):
    """Draws a classic Agar.io spiked green virus obstacle."""
    pts = []
    inner_r = radius * 0.82
    outer_r = radius * 1.18
    for i in range(spikes * 2):
        r = outer_r if i % 2 == 0 else inner_r
        theta = angle_offset + i * np.pi / spikes
        px = int(center[0] + r * math.cos(theta))
        py = int(center[1] + r * math.sin(theta))
        pts.append([px, py])
    pts = np.array(pts, np.int32)
    # Spiked neon green fill and darker border
    cv2.fillPoly(img, [pts], (46, 204, 113)) # Emerald/Neon green BGR
    cv2.polylines(img, [pts], True, (39, 174, 96), 2, cv2.LINE_AA)
    # Inner nucleus circle
    cv2.circle(img, center, int(radius * 0.4), (39, 174, 96), -1)


def draw_glowing_cell(
    img: np.ndarray,
    center: tuple,
    radius: int,
    color_bgr: tuple,
    name: str,
    mass: float,
    yaw: float,
    is_primary: bool = False
):
    """Renders a translucent glowing cellular membrane with border halo and text labels."""
    cx, cy = center
    if radius < 3:
        return

    # Outer glow halo
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius + 6, color_bgr, 3)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # Main cell body
    cv2.circle(img, (cx, cy), radius, color_bgr, -1, cv2.LINE_AA)
    # Inner gradient / dark translucent core
    darker_color = tuple(int(c * 0.65) for c in color_bgr)
    cv2.circle(img, (cx, cy), int(radius * 0.88), darker_color, -1, cv2.LINE_AA)

    # Cell border ring
    bright_border = tuple(min(255, int(c * 1.3)) for c in color_bgr)
    cv2.circle(img, (cx, cy), radius, bright_border, 2, cv2.LINE_AA)

    # Heading directional chevron (aim / split direction)
    hx = int(cx + (radius * 0.65) * math.cos(yaw))
    hy = int(cy - (radius * 0.65) * math.sin(yaw))  # Screen Y is inverted relative to Cartesian world Y
    cv2.line(img, (cx, cy), (hx, hy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (hx, hy), 3, (255, 255, 255), -1)

    # Name and Mass label
    if radius > 12:
        font_scale = 0.38 if radius < 25 else 0.46
        # Name
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(img, name, (cx - tw // 2, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        # Mass
        mass_text = f"{int(mass)}"
        (mw, mh), _ = cv2.getTextSize(mass_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.05, 1)
        cv2.putText(img, mass_text, (cx - mw // 2, cy + mh + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.05, (226, 232, 240), 1, cv2.LINE_AA)


def render_agar_demo_video(
    output_path: str = "outputs/agar_arena_demo.mp4",
    num_frames: int = 1500,
    fps: int = 30,
    steps_per_frame: int = 3,
    sim_dt: float = 0.0833333
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_sim_time = num_frames * steps_per_frame * sim_dt
    logger.info(f"Initializing Arcade-Grade Agar.io Multi-Agent Renderer: {num_frames} frames ({num_frames/fps:.1f}s video) at 7.5x simulation speed ({total_sim_time:.1f}s match)...")

    arena_size = 14.0
    env = PartiallyObservableAgarEnv(
        arena_size=arena_size,
        num_players=6,
        num_food=100,
        num_viruses=5,
        max_steps=num_frames * steps_per_frame + 500,
        dt=sim_dt,
        total_world_mass=400.0,
        pellet_mass=0.5,
        mass_decay_multiplier=1.3
    )
    obs_dict, _ = env.reset(seed=670)

    # Competitor Agents (Champion Tier preferred):
    diff_weights = "outputs/agar_diffusion_champion.pt" if os.path.exists("outputs/agar_diffusion_champion.pt") else "outputs/agar_diffusion_model.pt"
    critic_weights = "outputs/agar_diffusion_critic_champion.pt" if os.path.exists("outputs/agar_diffusion_critic_champion.pt") else "outputs/agar_diffusion_critic.pt"
    ppo_weights = "outputs/agar_ppo_champion.pt" if os.path.exists("outputs/agar_ppo_champion.pt") else "outputs/agar_ppo_gpu.pt"
    logger.info(f"Rendering demo with PPO weights: {ppo_weights} | Diffusion weights: {diff_weights}")

    diff_agent_1 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=4, action_dim=3, obs_dim=38, num_ddim_steps=10, num_candidates=8, seed=42)
    diff_agent_2 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=4, action_dim=3, obs_dim=38, num_ddim_steps=10, num_candidates=8, seed=105)
    diff_agent_3 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=4, action_dim=3, obs_dim=38, num_ddim_steps=10, num_candidates=8, seed=202)

    if "champion" in ppo_weights:
        ppo_agent_1 = SuperiorPPOAgent(weights_path=ppo_weights)
        ppo_agent_2 = SuperiorPPOAgent(weights_path=ppo_weights)
        ppo_agent_3 = SuperiorPPOAgent(weights_path=ppo_weights)
    else:
        ppo_agent_1 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)
        ppo_agent_2 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)
        ppo_agent_3 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)

    competitors = {
        "player_0": ("DIFFUSION-ALPHA", diff_agent_1, (16, 185, 129)),  # Emerald Neon (Follow Cam)
        "player_1": ("PPO-ALPHA", ppo_agent_1, (243, 156, 18)),         # Cyber Cyan
        "player_2": ("DIFFUSION-BETA", diff_agent_2, (236, 72, 153)),   # Rose Magenta
        "player_3": ("PPO-BETA", ppo_agent_2, (168, 85, 247)),          # Violet Purple
        "player_4": ("DIFFUSION-GAMMA", diff_agent_3, (245, 158, 11)),  # Amber Gold
        "player_5": ("PPO-GAMMA", ppo_agent_3, (239, 68, 68)),          # Crimson Red
    }

    # Food pellet colors (pastel palette)
    palette = [
        (255, 179, 71), (144, 238, 144), (255, 105, 180),
        (135, 206, 250), (221, 160, 221), (255, 215, 0)
    ]
    food_colors = [palette[i % len(palette)] for i in range(env.num_food)]

    # Layout dimensions
    total_w = 1280
    total_h = 720
    main_w = 890
    main_h = 720
    hud_x = main_w
    hud_w = total_w - main_w # 390 px

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (total_w, total_h))

    # Shockwave ripple particles: list of dict(x, y, radius, alpha, color)
    shockwaves = []

    keyframe_start = None
    keyframe_clash = None
    keyframe_end = None

    cam_x, cam_y = 0.0, 0.0
    current_ppm = 48.0
    focus_pid = "player_0"

    for frame_idx in range(num_frames):
        # 1. Step environment multiple times for 5.0x simulation speed
        planned_trajectory = None
        for _ in range(steps_per_frame):
            # Dynamic Best-Player Selection (highest combat score: mass + 25.0 * kills)
            best_pid = max(env.players.keys(), key=lambda p: env.players[p].mass + env.players[p].kills * 25.0)
            focus_pid = best_pid

            actions = {}
            for pid, (name, agent, _) in competitors.items():
                if pid == focus_pid and isinstance(agent, AgarDiffusionPolicy):
                    if len(agent.action_buffer) == 0:
                        planned_trajectory = agent.sample_trajectory_ddim(obs_dict[pid], steps=10)
                        agent.action_buffer = [planned_trajectory[i] for i in range(len(planned_trajectory))]
                    else:
                        planned_trajectory = np.array(agent.action_buffer)
                    actions[pid] = agent.predict(obs_dict[pid], deterministic=True)
                else:
                    actions[pid] = agent.predict(obs_dict[pid], deterministic=True)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)

            # Trigger shockwave ripples on predation events
            for event in env.predation_events:
                shockwaves.append({
                    "pos": event["pos"].copy(),
                    "r": 10.0,
                    "max_r": 90.0,
                    "alpha": 1.0,
                    "color": (255, 255, 255)
                })

        # 2. Main Follow-Cam Viewport: Centered on the BEST player
        frame = np.full((total_h, total_w, 3), 12, dtype=np.uint8) # Deep slate base #0c0f1d
        main_view = np.full((main_h, main_w, 3), 14, dtype=np.uint8)

        # Smooth camera tracking locked on best player
        focus_player = env.players[focus_pid]
        target_cam_x, target_cam_y = focus_player.pos
        if frame_idx == 0:
            cam_x, cam_y = float(target_cam_x), float(target_cam_y)
        else:
            cam_x = 0.88 * cam_x + 0.12 * float(target_cam_x)
            cam_y = 0.88 * cam_y + 0.12 * float(target_cam_y)

        # Dynamic zoom: zooms out as the best cell grows and fragments
        target_ppm = float(np.clip(48.0 * (16.0 / max(16.0, focus_player.mass)) ** 0.35, 22.0, 52.0))
        current_ppm = 0.90 * current_ppm + 0.10 * target_ppm
        pixels_per_meter = current_ppm

        def world_to_screen(wx: float, wy: float) -> tuple:
            sx = int(main_w / 2.0 + (wx - cam_x) * pixels_per_meter)
            sy = int(main_h / 2.0 - (wy - cam_y) * pixels_per_meter)
            return sx, sy

        # Draw Grid lines (relative to camera)
        grid_step = 2.0 # 2 meters per grid line
        grid_start_x = math.floor((-env.half_arena - cam_x) / grid_step) * grid_step
        grid_end_x = math.ceil((env.half_arena - cam_x) / grid_step) * grid_step

        for gx in np.arange(-env.half_arena, env.half_arena + 0.1, grid_step):
            p1 = world_to_screen(gx, -env.half_arena)
            p2 = world_to_screen(gx, env.half_arena)
            cv2.line(main_view, p1, p2, (26, 36, 56), 1)

        for gy in np.arange(-env.half_arena, env.half_arena + 0.1, grid_step):
            p1 = world_to_screen(-env.half_arena, gy)
            p2 = world_to_screen(env.half_arena, gy)
            cv2.line(main_view, p1, p2, (26, 36, 56), 1)

        # Draw Arena Boundary Walls
        wall_tl = world_to_screen(-env.half_arena, env.half_arena)
        wall_br = world_to_screen(env.half_arena, -env.half_arena)
        cv2.rectangle(main_view, wall_tl, wall_br, (239, 68, 68), 3)

        # Draw Food Pellets
        for i, (fx, fy) in enumerate(env.food_positions):
            sx, sy = world_to_screen(fx, fy)
            if -10 <= sx < main_w + 10 and -10 <= sy < main_h + 10:
                cv2.circle(main_view, (sx, sy), 4, food_colors[i % len(food_colors)], -1, cv2.LINE_AA)
                cv2.circle(main_view, (sx, sy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Spiked Viruses
        virus_rotation = frame_idx * 0.04
        for vx, vy, vr in env.viruses:
            vsx, vsy = world_to_screen(vx, vy)
            v_radius_px = int(vr * pixels_per_meter)
            if -50 <= vsx < main_w + 50 and -50 <= vsy < main_h + 50:
                draw_spiked_virus(main_view, (vsx, vsy), v_radius_px, spikes=14, angle_offset=virus_rotation)

        # Draw Active Polar Raycaster Sensor Beams for Focus Agent
        focus_obs = obs_dict[focus_pid]
        threat_rays = focus_obs[6:14]  # 8 threat sector normalized distances
        prey_rays = focus_obs[14:22]   # 8 prey sector normalized distances

        sec_w = 2.0 * math.pi / 8.0
        f_pos = focus_player.pos
        f_r = focus_player.radius
        v_r = focus_player.vision_radius

        for s_i in range(8):
            # Center angle of sector relative to world
            sec_ang = focus_player.yaw + (s_i + 0.5) * sec_w - math.pi
            c_a, s_a = math.cos(sec_ang), math.sin(sec_ang)

            start_w = f_pos + f_r * np.array([c_a, s_a])
            p_dist = float(prey_rays[s_i])
            t_dist = float(threat_rays[s_i])
            start_px = world_to_screen(start_w[0], start_w[1])

            if t_dist < 0.92:
                # Threat detected: warning red ray with impact indicator
                end_w = f_pos + (f_r + t_dist * max(0.5, v_r - f_r)) * np.array([c_a, s_a])
                end_px = world_to_screen(end_w[0], end_w[1])
                cv2.line(main_view, start_px, end_px, (239, 68, 68), 2, cv2.LINE_AA)
                cv2.circle(main_view, end_px, 6, (239, 68, 68), -1)
                cv2.circle(main_view, end_px, 8, (255, 255, 255), 1, cv2.LINE_AA)
            elif p_dist < 0.92:
                # Prey detected: emerald green ray with target lock pip
                end_w = f_pos + (f_r + p_dist * max(0.5, v_r - f_r)) * np.array([c_a, s_a])
                end_px = world_to_screen(end_w[0], end_w[1])
                cv2.line(main_view, start_px, end_px, (52, 211, 153), 2, cv2.LINE_AA)
                cv2.circle(main_view, end_px, 5, (52, 211, 153), -1)
                cv2.circle(main_view, end_px, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Ambient scan ray: subtle sensor beam to fog horizon
                end_w = f_pos + v_r * np.array([c_a, s_a])
                end_px = world_to_screen(end_w[0], end_w[1])
                cv2.line(main_view, start_px, end_px, (35, 55, 80), 1, cv2.LINE_AA)
                cv2.circle(main_view, end_px, 2, (56, 189, 248), -1)

        # Draw Planned Receding-Horizon Trajectory Ribbon for Diffusion Policy
        if planned_trajectory is not None and len(planned_trajectory) > 1:
            traj_pts = []
            sim_pos = focus_player.pos.copy()
            sim_yaw = focus_player.yaw
            traj_pts.append(world_to_screen(sim_pos[0], sim_pos[1]))

            for step_act in planned_trajectory[:12]:
                thrust_val, steer_val = step_act[0], step_act[1]
                sim_yaw = (sim_yaw + steer_val * 3.0 * env.dt + np.pi) % (2.0 * np.pi) - np.pi
                speed = thrust_val * focus_player.max_speed
                sim_pos += np.array([speed * math.cos(sim_yaw) * env.dt, speed * math.sin(sim_yaw) * env.dt])
                traj_pts.append(world_to_screen(sim_pos[0], sim_pos[1]))

            for i in range(len(traj_pts) - 1):
                cv2.line(main_view, traj_pts[i], traj_pts[i + 1], (16, 185, 129), 2, cv2.LINE_AA)
                cv2.circle(main_view, traj_pts[i + 1], 3, (52, 211, 153), -1)

        # Draw Players (all pieces of each player)
        # Sort by mass ascending so larger players draw on top
        sorted_players = sorted(competitors.keys(), key=lambda p: env.players[p].mass)
        for pid in sorted_players:
            p_state = env.players[pid]
            name, _, col = competitors[pid]
            for pc_idx, piece in enumerate(p_state.pieces):
                psx, psy = world_to_screen(piece.pos[0], piece.pos[1])
                prad_px = max(6, int(piece.radius * pixels_per_meter))
                piece_label = name if len(p_state.pieces) == 1 else f"{name} #{pc_idx+1}"
                draw_glowing_cell(
                    main_view,
                    (psx, psy),
                    prad_px,
                    col,
                    piece_label,
                    piece.mass,
                    p_state.yaw,
                    is_primary=(pid == focus_pid and pc_idx == 0)
                )

        # Draw Shockwave Ripples
        active_shockwaves = []
        for sw in shockwaves:
            sw_sx, sw_sy = world_to_screen(sw["pos"][0], sw["pos"][1])
            sw_r = int(sw["r"])
            if sw_r < sw["max_r"]:
                overlay = main_view.copy()
                cv2.circle(overlay, (sw_sx, sw_sy), sw_r, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(overlay, (sw_sx, sw_sy), int(sw_r * 0.7), (243, 156, 18), 2, cv2.LINE_AA)
                alpha = max(0.0, 1.0 - (sw_r / sw["max_r"]))
                cv2.addWeighted(overlay, alpha * 0.8, main_view, 1.0 - alpha * 0.8, 0, main_view)
                sw["r"] += 4.0
                active_shockwaves.append(sw)
        shockwaves = active_shockwaves

        # 3. ATMOSPHERIC FOG OF WAR (Partial Observability Mask)
        # Circular illumination mask centered on screen (where focus player is)
        focus_center = (int(main_w / 2.0), int(main_h / 2.0))
        r_vis_px = int(focus_player.vision_radius * pixels_per_meter)

        # Create Fog-of-War darkness overlay
        fog_mask = np.zeros((main_h, main_w), dtype=np.uint8)
        cv2.circle(fog_mask, focus_center, r_vis_px, 255, -1)
        # Soft blur boundary around vision perimeter
        fog_mask = cv2.GaussianBlur(fog_mask, (31, 31), 15)

        dark_fog = np.full_like(main_view, 6) # Dark void #060913
        alpha_channel = (fog_mask.astype(np.float32) / 255.0)[:, :, None]
        # Blend: clear inside vision, 85% fog darkness outside
        main_view = (main_view * (0.15 + 0.85 * alpha_channel) + dark_fog * (0.85 * (1.0 - alpha_channel))).astype(np.uint8)

        # Vision perimeter ring (pulsing turquoise dashed rim)
        cv2.circle(main_view, focus_center, r_vis_px, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(
            main_view,
            f"FOG OF WAR: VISION LIMIT ({focus_player.vision_radius:.1f}m)",
            (focus_center[0] - 110, focus_center[1] - r_vis_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (125, 211, 252), 1, cv2.LINE_AA
        )

        # 4. APEX LEADER PERSPECTIVE HUD BANNER
        banner_text = f"APEX LEADER CAM: {competitors[focus_pid][0]} | MASS: {focus_player.mass:.1f}kg | KILLS: {focus_player.kills} | PIECES: {len(focus_player.pieces)}"
        cv2.rectangle(main_view, (15, 15), (690, 52), (15, 23, 42), -1)
        cv2.rectangle(main_view, (15, 15), (690, 52), (0, 215, 255), 2)
        cv2.circle(main_view, (35, 34), 8, (0, 215, 255), -1)
        cv2.putText(main_view, banner_text, (52, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Place Main View into Frame
        frame[0:main_h, 0:main_w] = main_view

        # Center separator
        cv2.line(frame, (main_w, 0), (main_w, total_h), (51, 65, 85), 2)

        # 4. HUD Sidebar (Right Pane: 390px wide)
        hud = frame[:, hud_x:total_w]

        # Top Header Card
        cv2.rectangle(hud, (15, 15), (hud_w - 15, 65), (15, 23, 42), -1)
        cv2.rectangle(hud, (15, 15), (hud_w - 15, 65), (51, 65, 85), 1)
        cv2.putText(hud, "AGAR.IO ARENA (5.0x SPEED)", (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (248, 250, 252), 1, cv2.LINE_AA)
        sec_elapsed = frame_idx / float(fps)
        total_sec = num_frames / float(fps)
        sim_elapsed = frame_idx * steps_per_frame * env.dt
        cv2.putText(hud, f"POMDP | 1m VIDEO ({sec_elapsed:.1f}s/60s) | 5m MATCH: {sim_elapsed:.1f}s/300s", (26, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (52, 211, 153), 1, cv2.LINE_AA)

        # Global Radar Minimap (320x320)
        minimap_y = 80
        minimap_size = 320
        minimap_x = (hud_w - minimap_size) // 2

        cv2.putText(hud, "GLOBAL RADAR (GOD'S-EYE VIEW)", (minimap_x + 8, minimap_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 116, 139), 1, cv2.LINE_AA)

        minimap_canvas = np.full((minimap_size, minimap_size, 3), (10, 15, 28), dtype=np.uint8)
        cv2.rectangle(minimap_canvas, (0, 0), (minimap_size - 1, minimap_size - 1), (51, 65, 85), 1)

        def world_to_minimap(wx: float, wy: float) -> tuple:
            mx = int(minimap_size / 2.0 + (wx / env.half_arena) * (minimap_size / 2.0 - 8))
            my = int(minimap_size / 2.0 - (wy / env.half_arena) * (minimap_size / 2.0 - 8))
            return mx, my

        # Food density spots on minimap
        for fx, fy in env.food_positions[::4]: # Sample every 4th pellet
            mx, my = world_to_minimap(fx, fy)
            if 0 <= mx < minimap_size and 0 <= my < minimap_size:
                cv2.circle(minimap_canvas, (mx, my), 1, (100, 116, 139), -1)

        # Draw Viruses on minimap
        for vx, vy, vr in env.viruses:
            mx, my = world_to_minimap(vx, vy)
            cv2.circle(minimap_canvas, (mx, my), 3, (46, 204, 113), -1)

        # Draw Vision Circle of primary player on minimap (clipped)
        f_mx, f_my = world_to_minimap(focus_player.pos[0], focus_player.pos[1])
        vis_mini_r = int((focus_player.vision_radius / env.half_arena) * (minimap_size / 2.0 - 8))
        cv2.circle(minimap_canvas, (f_mx, f_my), vis_mini_r, (56, 189, 248), 1, cv2.LINE_AA)

        # Draw all players on minimap
        for pid in competitors.keys():
            p_st = env.players[pid]
            mx, my = world_to_minimap(p_st.pos[0], p_st.pos[1])
            _, _, col = competitors[pid]
            p_mini_r = max(2, int(p_st.radius * 3.5))
            cv2.circle(minimap_canvas, (mx, my), p_mini_r, col, -1)

        hud[minimap_y:minimap_y + minimap_size, minimap_x:minimap_x + minimap_size] = minimap_canvas

        # Live Agar Leaderboard
        lb_y = minimap_y + minimap_size + 20
        cv2.putText(hud, "LIVE AGAR LEADERBOARD", (minimap_x, lb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (241, 245, 249), 1, cv2.LINE_AA)

        ranked = sorted(competitors.keys(), key=lambda p: env.players[p].mass, reverse=True)
        for rank_idx, pid in enumerate(ranked[:5]):
            p_st = env.players[pid]
            name, _, col = competitors[pid]
            row_y = lb_y + 16 + rank_idx * 26

            # Rank card
            is_focus = (pid == focus_pid)
            bg_col = (30, 41, 59) if is_focus else (15, 23, 42)
            cv2.rectangle(hud, (minimap_x, row_y), (minimap_x + minimap_size, row_y + 22), bg_col, -1)
            if is_focus:
                cv2.rectangle(hud, (minimap_x, row_y), (minimap_x + minimap_size, row_y + 22), (16, 185, 129), 1)

            # Color pip
            cv2.circle(hud, (minimap_x + 12, row_y + 11), 5, col, -1)
            # Rank #
            cv2.putText(hud, f"#{rank_idx + 1}", (minimap_x + 24, row_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (148, 163, 184), 1, cv2.LINE_AA)
            # Name
            cv2.putText(hud, name, (minimap_x + 48, row_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (248, 250, 252), 1, cv2.LINE_AA)
            # Mass & Kills
            stats_str = f"M: {int(p_st.mass)}  (K: {p_st.kills})"
            cv2.putText(hud, stats_str, (minimap_x + minimap_size - 85, row_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (52, 211, 153), 1, cv2.LINE_AA)

        # Cockpit Telemetry & Threat Warning Card (Bottom)
        card_y = lb_y + 16 + 5 * 26 + 12
        card_h = total_h - card_y - 15
        cv2.rectangle(hud, (minimap_x, card_y), (minimap_x + minimap_size, card_y + card_h), (15, 23, 42), -1)
        cv2.rectangle(hud, (minimap_x, card_y), (minimap_x + minimap_size, card_y + card_h), (51, 65, 85), 1)

        # Check if threat is nearby for focus player
        obs_focus = obs_dict[focus_pid]
        threat_dist_norm = obs_focus[22]
        is_threat_alert = (threat_dist_norm > 0.01 and threat_dist_norm < 0.65)

        # Alert Banner
        alert_bg = (239, 68, 68) if is_threat_alert else (16, 185, 129)
        alert_text = "WARNING: PREDATOR IN FOV! EVADING" if is_threat_alert else "STATUS: CLEAR - ACTIVE FORAGING"
        cv2.rectangle(hud, (minimap_x + 8, card_y + 8), (minimap_x + minimap_size - 8, card_y + 32), alert_bg, -1)
        cv2.putText(hud, alert_text, (minimap_x + 16, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry metrics
        cv2.putText(hud, f"Primary Cell: {competitors[focus_pid][0]}", (minimap_x + 12, card_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Mass: {focus_player.mass:.1f} kg  |  Radius: {focus_player.radius:.2f} m", (minimap_x + 12, card_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (203, 213, 225), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Speed: {focus_player.linear_vel:.2f} m/s (Max: {focus_player.max_speed:.2f})", (minimap_x + 12, card_y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (203, 213, 225), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Diffusion Horizon: Ta=16, Te=6 | DDIM: 10 steps", (minimap_x + 12, card_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (52, 211, 153), 1, cv2.LINE_AA)

        writer.write(frame)

        if frame_idx == 15:
            keyframe_start = frame.copy()
        if frame_idx == int(num_frames * 0.45):
            keyframe_clash = frame.copy()
        if frame_idx == num_frames - 1:
            keyframe_end = frame.copy()

    writer.release()
    logger.info(f"Agar arena video rendered to: {output_path}")

    # Copy to project output folder and dedicated best player video file
    import shutil
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dirs = [
        os.path.join(root_dir, "outputs"),
        os.path.join(root_dir, "projects", "sim2real-ppo-navigation", "outputs")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        if os.path.abspath(output_path) != os.path.abspath(os.path.join(d, "agar_arena_demo.mp4")):
            shutil.copyfile(output_path, os.path.join(d, "agar_arena_demo.mp4"))
        if os.path.abspath(output_path) != os.path.abspath(os.path.join(d, "best_player_demo.mp4")):
            shutil.copyfile(output_path, os.path.join(d, "best_player_demo.mp4"))

    # Save keyframes to outputs
    for name, img in [
        ("agar_demo_start.jpg", keyframe_start),
        ("agar_demo_clash.jpg", keyframe_clash),
        ("agar_demo_leaderboard.jpg", keyframe_end),
        ("best_player_start.jpg", keyframe_start),
        ("best_player_clash.jpg", keyframe_clash),
        ("best_player_leaderboard.jpg", keyframe_end)
    ]:
        if img is not None:
            for d in out_dirs:
                cv2.imwrite(os.path.join(d, name), img)

    logger.info("Keyframes exported: agar_demo_start.jpg, agar_demo_clash.jpg, agar_demo_leaderboard.jpg, best_player_clash.jpg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render Agar.io Arcade Demo Video")
    parser.add_argument("--frames", type=int, default=900, help="Number of video frames (default 900 = 30s at 30fps)")
    args = parser.parse_args()

    render_agar_demo_video(num_frames=args.frames, fps=30, steps_per_frame=2, sim_dt=0.0833333)
