"""
Interactive Real-Time Playable Agar.io: Human vs. AI Arena (PPO & Diffusion Policy).
Compete live against trained AI models at regular 1.0x simulation speed (30 FPS).

Controls:
- Mouse Move: Steer heading and regulate speed (distance from screen center controls thrust).
- Spacebar: Trigger multi-split attack (splits cell in half, launching projectile forward).
- W key: Turbo sprint forward.
- R key: Respawn / Restart match.
- ESC / Q key: Exit game.
"""

import os
import sys
import time
import math
import argparse
import logging
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.agar_env import PartiallyObservableAgarEnv
from src.train_agar_parallel_gpu import TorchAgarPPOAgent
from src.train_superior_ppo import SuperiorPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy
from scripts.render_agar_demo import draw_glowing_cell, draw_spiked_virus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarPlay")


class HumanPlayerController:
    """Manages mouse and keyboard interaction for the human player."""

    def __init__(self, screen_center: tuple):
        self.center_x, self.center_y = screen_center
        self.mouse_x = self.center_x
        self.mouse_y = self.center_y
        self.split_triggered = False
        self.sprint_active = False

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

    def get_action(self, current_yaw: float, can_split: bool) -> np.ndarray:
        dx = self.mouse_x - self.center_x
        dy = -(self.mouse_y - self.center_y) # Screen Y inverted

        dist = math.hypot(dx, dy)
        if dist > 8.0:
            target_angle = math.atan2(dy, dx)
            angle_diff = (target_angle - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
            steer = float(np.clip(angle_diff / (np.pi * 0.5), -1.0, 1.0))
            thrust = float(np.clip(dist / 140.0, 0.4, 1.0))
        else:
            steer = 0.0
            thrust = 0.2

        if self.sprint_active:
            thrust = 1.0

        split_cmd = 1.0 if self.split_triggered else 0.0
        self.split_triggered = False # Consume one-shot trigger
        return np.array([thrust, steer, split_cmd], dtype=np.float32)


def play_agar(headless: bool = False, max_frames: int = 3000, ai_tier: str = "champion"):
    fps = 30
    sim_dt = 1.0 / float(fps)
    arena_size = 14.0

    env = PartiallyObservableAgarEnv(
        arena_size=arena_size,
        num_players=6,
        num_food=150,
        num_viruses=5,
        max_steps=max_frames,
        dt=sim_dt
    )
    obs_dict, _ = env.reset(seed=int(time.time()) % 10000)

    # 1. Load Trained Neural Opponents (Champion Treatment or Baseline Control)
    if ai_tier == "champion":
        ppo_weights = "outputs/agar_ppo_champion.pt"
        if not os.path.exists(ppo_weights):
            ppo_weights = "outputs/agar_ppo_gpu.pt"
        diff_weights = "outputs/agar_diffusion_champion.pt"
        if not os.path.exists(diff_weights):
            diff_weights = "outputs/agar_diffusion_model.pt"
        critic_weights = "outputs/agar_diffusion_critic_champion.pt"
        if not os.path.exists(critic_weights):
            critic_weights = "outputs/agar_diffusion_critic.pt"
        logger.info(f"Loaded CHAMPION Tier AI bots (PPO: {ppo_weights}, Diffusion: {diff_weights})")
    else:
        ppo_weights = "outputs/agar_ppo_gpu.pt"
        diff_weights = "outputs/agar_diffusion_model.pt"
        critic_weights = "outputs/agar_diffusion_critic.pt"
        logger.info(f"Loaded BASELINE Tier AI bots (PPO: {ppo_weights}, Diffusion: {diff_weights})")

    if "champion" in ppo_weights:
        ppo_agent_1 = SuperiorPPOAgent(weights_path=ppo_weights)
        ppo_agent_2 = SuperiorPPOAgent(weights_path=ppo_weights)
    else:
        ppo_agent_1 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)
        ppo_agent_2 = TorchAgarPPOAgent(weights_path=ppo_weights if os.path.exists(ppo_weights) else None)

    diff_agent_1 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=2, action_dim=3, obs_dim=38, num_ddim_steps=5, seed=42)
    diff_agent_2 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=2, action_dim=3, obs_dim=38, num_ddim_steps=5, seed=105)
    diff_agent_3 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=2, action_dim=3, obs_dim=38, num_ddim_steps=5, seed=202)

    # Player Roster (Human on player_0)
    tier_tag = "CHAMP" if ai_tier == "champion" else "BASE"
    competitors = {
        "player_0": ("HUMAN (YOU)", None, (0, 215, 255)),                     # Electric Gold / Yellow
        "player_1": (f"DIFF-{tier_tag}-1", diff_agent_1, (16, 185, 129)),      # Emerald Neon
        "player_2": (f"PPO-{tier_tag}-1", ppo_agent_1, (243, 156, 18)),         # Cyber Cyan
        "player_3": (f"DIFF-{tier_tag}-2", diff_agent_2, (236, 72, 153)),      # Rose Magenta
        "player_4": (f"PPO-{tier_tag}-2", ppo_agent_2, (168, 85, 247)),        # Violet Purple
        "player_5": (f"DIFF-{tier_tag}-3", diff_agent_3, (245, 158, 11)),      # Amber Gold
    }

    # Food pellet palette
    palette = [
        (255, 179, 71), (144, 238, 144), (255, 105, 180),
        (135, 206, 250), (221, 160, 221), (255, 215, 0)
    ]
    food_colors = [palette[i % len(palette)] for i in range(env.num_food)]

    # Display Dimensions
    total_w = 1280
    total_h = 720
    main_w = 900
    main_h = 720
    hud_x = main_w
    hud_w = total_w - main_w

    window_name = "Agar.io AI Arena - Human vs AI (PPO & Diffusion)"
    controller = HumanPlayerController(screen_center=(main_w // 2, main_h // 2))

    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, total_w, total_h)
            cv2.setMouseCallback(window_name, controller.on_mouse)
            logger.info("Playable Agar.io Window Initialized. Click inside window and use Mouse + Spacebar.")
        except Exception as e:
            logger.warning(f"Could not open GUI window (headless environment?): {e}")
            headless = True

    frame_count = 0
    shockwaves = []
    logger.info("Match started at 1.0x regular speed (30 FPS). Compete with PPO and Diffusion bots!")

    while frame_count < max_frames:
        t_start = time.perf_counter()
        frame_count += 1

        human_player = env.players["player_0"]
        human_yaw = human_player.yaw
        can_split = human_player.can_split

        # 1. Gather Actions
        actions = {}
        # Human Action
        actions["player_0"] = controller.get_action(human_yaw, can_split)

        # AI Actions
        for pid in ["player_1", "player_2", "player_3", "player_4", "player_5"]:
            name, agent, _ = competitors[pid]
            actions[pid] = agent.predict(obs_dict[pid], deterministic=True)

        # Step Environment at 1.0x Real-Time Speed (1 step per frame)
        obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # Record predation shockwaves
        for event in env.predation_events:
            shockwaves.append({
                "x": event["pos"][0],
                "y": event["pos"][1],
                "radius": 15,
                "alpha": 1.0,
                "color": (255, 255, 255) if event["predator"] == "player_0" else (239, 68, 68)
            })

        # 2. Render Frame
        frame = np.full((total_h, total_w, 3), (11, 15, 25), dtype=np.uint8)
        main_view = frame[:, :main_w]

        # Camera centered on human player
        cam_x, cam_y = human_player.pos
        pixels_per_meter = 48.0

        def world_to_screen(wx: float, wy: float) -> tuple:
            sx = int(main_w / 2.0 + (wx - cam_x) * pixels_per_meter)
            sy = int(main_h / 2.0 - (wy - cam_y) * pixels_per_meter)
            return sx, sy

        # Arena Grid
        grid_step = 2.0
        for gx in np.arange(-env.half_arena, env.half_arena + 0.1, grid_step):
            p1 = world_to_screen(gx, -env.half_arena)
            p2 = world_to_screen(gx, env.half_arena)
            cv2.line(main_view, p1, p2, (22, 30, 48), 1)
        for gy in np.arange(-env.half_arena, env.half_arena + 0.1, grid_step):
            p1 = world_to_screen(-env.half_arena, gy)
            p2 = world_to_screen(env.half_arena, gy)
            cv2.line(main_view, p1, p2, (22, 30, 48), 1)

        # Arena Red Boundary Walls
        wall_tl = world_to_screen(-env.half_arena, env.half_arena)
        wall_br = world_to_screen(env.half_arena, -env.half_arena)
        cv2.rectangle(main_view, wall_tl, wall_br, (239, 68, 68), 3)

        # Food Pellets
        for i, (fx, fy) in enumerate(env.food_positions):
            sx, sy = world_to_screen(fx, fy)
            if -10 <= sx < main_w + 10 and -10 <= sy < main_h + 10:
                cv2.circle(main_view, (sx, sy), 4, food_colors[i % len(food_colors)], -1, cv2.LINE_AA)

        # Spiked Green Viruses
        v_rot = frame_count * 0.04
        for vx, vy, vr in env.viruses:
            vsx, vsy = world_to_screen(vx, vy)
            v_radius_px = int(vr * pixels_per_meter)
            if -50 <= vsx < main_w + 50 and -50 <= vsy < main_h + 50:
                draw_spiked_virus(main_view, (vsx, vsy), v_radius_px, spikes=14, angle_offset=v_rot)

        # Draw Cells (Opponents first, Human on top)
        draw_order = sorted(competitors.keys(), key=lambda p: (0 if p == "player_0" else 1, env.players[p].mass))
        for pid in draw_order:
            player = env.players[pid]
            name, _, col = competitors[pid]
            is_human = (pid == "player_0")

            for pc in player.pieces:
                sx, sy = world_to_screen(pc.pos[0], pc.pos[1])
                r_px = max(6, int(pc.radius * pixels_per_meter))
                if -150 <= sx < main_w + 150 and -150 <= sy < main_h + 150:
                    draw_glowing_cell(
                        main_view,
                        center=(sx, sy),
                        radius=r_px,
                        color_bgr=col,
                        name=name if pc == player.pieces[0] else "",
                        mass=pc.mass,
                        yaw=player.yaw,
                        is_primary=is_human
                    )

        # Shockwave particle rings
        active_shockwaves = []
        for sw in shockwaves:
            sx, sy = world_to_screen(sw["x"], sw["y"])
            r_px = int(sw["radius"])
            alpha = sw["alpha"]
            if alpha > 0.05 and 0 <= sx < main_w and 0 <= sy < main_h:
                overlay = main_view.copy()
                cv2.circle(overlay, (sx, sy), r_px, sw["color"], max(1, int(4 * alpha)), cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, main_view, 1.0 - alpha, 0, main_view)
                sw["radius"] += 4.0
                sw["alpha"] -= 0.05
                active_shockwaves.append(sw)
        shockwaves = active_shockwaves

        # Fog of War dark vignette around vision circle
        fog_mask = np.zeros((main_h, main_w), dtype=np.uint8)
        cam_screen_center = (main_w // 2, main_h // 2)
        r_vis_px = int(human_player.vision_radius * pixels_per_meter)
        cv2.circle(fog_mask, cam_screen_center, r_vis_px, 255, -1)
        cv2.circle(fog_mask, cam_screen_center, int(r_vis_px * 0.85), 255, -1)
        fog_mask = cv2.GaussianBlur(fog_mask, (45, 45), 0)
        norm_mask = fog_mask.astype(np.float32) / 255.0
        main_view[:] = (main_view.astype(np.float32) * norm_mask[:, :, None] * 0.88 + 12 * (1.0 - norm_mask[:, :, None])).astype(np.uint8)

        # Mouse Aim Reticle
        cv2.circle(main_view, (controller.mouse_x, controller.mouse_y), 8, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(main_view, (controller.mouse_x - 12, controller.mouse_y), (controller.mouse_x + 12, controller.mouse_y), (0, 255, 255), 1)
        cv2.line(main_view, (controller.mouse_x, controller.mouse_y - 12), (controller.mouse_x, controller.mouse_y + 12), (0, 255, 255), 1)

        # 3. Render Glassmorphic Telemetry HUD (Right Panel)
        hud = frame[:, hud_x:]
        cv2.rectangle(hud, (0, 0), (hud_w, total_h), (15, 23, 42), -1)
        cv2.line(hud, (0, 0), (0, total_h), (51, 65, 85), 1)

        # Title
        cv2.putText(hud, "AGAR.IO ARENA [HUMAN vs AI]", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (248, 250, 252), 1, cv2.LINE_AA)
        cv2.putText(hud, "Speed: 1.0x Real-Time (30 FPS)", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (148, 163, 184), 1, cv2.LINE_AA)

        # Minimap Radar Inset
        mm_size = 180
        mm_x = (hud_w - mm_size) // 2
        mm_y = 60
        cv2.rectangle(hud, (mm_x, mm_y), (mm_x + mm_size, mm_y + mm_size), (10, 14, 26), -1)
        cv2.rectangle(hud, (mm_x, mm_y), (mm_x + mm_size, mm_y + mm_size), (71, 85, 105), 1)

        def world_to_mm(wx: float, wy: float) -> tuple:
            mx = int(mm_x + mm_size / 2.0 + (wx / env.arena_size) * (mm_size - 10))
            my = int(mm_y + mm_size / 2.0 - (wy / env.arena_size) * (mm_size - 10))
            return mx, my

        # Draw virus dots on radar
        for vx, vy, _ in env.viruses:
            cv2.circle(hud, world_to_mm(vx, vy), 3, (39, 174, 96), -1)

        # Draw players on radar
        for pid, (pname, _, pcol) in competitors.items():
            pst = env.players[pid]
            pmx, pmy = world_to_mm(pst.pos[0], pst.pos[1])
            is_h = (pid == "player_0")
            r_dot = 5 if is_h else 3
            cv2.circle(hud, (pmx, pmy), r_dot, pcol, -1)
            if is_h:
                cv2.circle(hud, (pmx, pmy), r_dot + 3, (255, 255, 255), 1)

        # Live Leaderboard
        lb_y = mm_y + mm_size + 20
        cv2.putText(hud, "LIVE ARENA LEADERBOARD", (16, lb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

        ranked = sorted(competitors.keys(), key=lambda p: env.players[p].mass, reverse=True)
        for rank_idx, pid in enumerate(ranked):
            p_st = env.players[pid]
            pname, _, col = competitors[pid]
            row_y = lb_y + 12 + rank_idx * 28
            is_h = (pid == "player_0")
            bg_col = (30, 41, 59) if is_h else (15, 23, 42)
            cv2.rectangle(hud, (12, row_y), (hud_w - 12, row_y + 24), bg_col, -1)
            if is_h:
                cv2.rectangle(hud, (12, row_y), (hud_w - 12, row_y + 24), (0, 215, 255), 1)

            cv2.circle(hud, (24, row_y + 12), 5, col, -1)
            cv2.putText(hud, f"#{rank_idx + 1}", (36, row_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (148, 163, 184), 1, cv2.LINE_AA)
            cv2.putText(hud, pname, (64, row_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (248, 250, 252), 1, cv2.LINE_AA)
            cv2.putText(hud, f"{int(p_st.mass)} kg", (hud_w - 65, row_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (52, 211, 153), 1, cv2.LINE_AA)

        # Human Status & Controls Card
        card_y = lb_y + 12 + 6 * 28 + 15
        card_h = total_h - card_y - 12
        cv2.rectangle(hud, (12, card_y), (hud_w - 12, card_y + card_h), (15, 23, 42), -1)
        cv2.rectangle(hud, (12, card_y), (hud_w - 12, card_y + card_h), (51, 65, 85), 1)

        cv2.putText(hud, "HUMAN TELEMETRY", (24, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Total Mass: {human_player.mass:.1f} kg (Pieces: {len(human_player.pieces)})", (24, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Kills: {human_player.kills}  |  Deaths: {human_player.deaths}", (24, card_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (241, 245, 249), 1, cv2.LINE_AA)

        split_status = "READY (PRESS SPACE)" if can_split else (f"COOLDOWN ({human_player.split_cooldown:.1f}s)" if human_player.split_cooldown > 0 else "NEED 36.0 KG")
        split_col = (16, 185, 129) if can_split else (148, 163, 184)
        cv2.putText(hud, f"Split: {split_status}", (24, card_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.32, split_col, 1, cv2.LINE_AA)

        cv2.putText(hud, "CONTROLS:", (24, card_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(hud, "- Mouse: Steer & Throttle", (24, card_y + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (203, 213, 225), 1, cv2.LINE_AA)
        cv2.putText(hud, "- SPACE: Multi-Split Lunge", (24, card_y + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (203, 213, 225), 1, cv2.LINE_AA)
        cv2.putText(hud, "- W: Turbo Sprint | R: Reset", (24, card_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (203, 213, 225), 1, cv2.LINE_AA)

        # 4. Show Window & Handle Keys
        if not headless:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # SPACE
                controller.split_triggered = True
            elif key == ord('w') or key == ord('W'):
                controller.sprint_active = not controller.sprint_active
            elif key == ord('r') or key == ord('R'):
                obs_dict, _ = env.reset(seed=int(time.time()) % 10000)
                logger.info("Game match reset by user.")
            elif key == 27 or key == ord('q') or key == ord('Q'): # ESC or Q
                logger.info("Exiting interactive game.")
                break

        # Regulate 30 FPS wall-clock time
        elapsed = time.perf_counter() - t_start
        sleep_time = max(0.0, sim_dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    if not headless:
        cv2.destroyAllWindows()
    logger.info("Interactive game session concluded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playable Agar.io: Human vs AI Arena")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode for automated validation")
    parser.add_argument("--frames", type=int, default=3000, help="Maximum gameplay frames (default: 3000 = ~100 seconds)")
    parser.add_argument("--ai-tier", type=str, default="champion", choices=["champion", "baseline"], help="AI opponent skill tier")
    args = parser.parse_args()

    play_agar(headless=args.headless, max_frames=args.frames, ai_tier=args.ai_tier)
