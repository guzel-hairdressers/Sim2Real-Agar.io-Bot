"""
Web-Based Interactive Playable Agar.io: Human vs. AI Arena (FastAPI + HTML5 Canvas).
Play live in your browser against trained PPO and Diffusion models at 30 FPS.

Controls:
- Mouse: Steer heading and regulate speed (distance from screen center regulates thrust).
- Spacebar: Tactical split attack (launches projectile cell forward).
- W key: Turbo sprint.
- R key: Respawn / Restart match.
"""

import os
import sys
import time
import math
import argparse
import logging
import threading
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.agar_env import PartiallyObservableAgarEnv
from src.heuristic_agent import MasterHeuristicAgarBot
from src.train_champion_ppo import ChampionPPOAgent
from src.agar_diffusion_policy import AgarDiffusionPolicy
from scripts.render_agar_demo import draw_glowing_cell, draw_spiked_virus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgarWebArena")

app = FastAPI(title="Playable Agar.io AI Arena", description="Human vs Trained Neural Bots")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HumanActionState:
    def __init__(self):
        self.dx = 0.0
        self.dy = 0.0
        self.split_triggered = False
        self.sprint_active = False
        self.reset_requested = False
        self.lock = threading.Lock()

    def update(self, dx: float, dy: float, split: bool, sprint: bool):
        with self.lock:
            self.dx = dx
            self.dy = dy
            if split:
                self.split_triggered = True
            self.sprint_active = sprint

    def get_action(self, current_yaw: float) -> np.ndarray:
        with self.lock:
            dist = math.hypot(self.dx, self.dy)
            if dist > 10.0:
                target_angle = math.atan2(self.dy, self.dx)
                angle_diff = (target_angle - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
                steer = float(np.clip(angle_diff / (np.pi * 0.5), -1.0, 1.0))
                thrust = float(np.clip(dist / 140.0, 0.35, 1.0))
            else:
                steer = 0.0
                thrust = 0.2

            if self.sprint_active:
                thrust = 1.0

            split_val = 1.0 if self.split_triggered else 0.0
            self.split_triggered = False
            return np.array([thrust, steer, split_val], dtype=np.float32)


human_state = HumanActionState()


class ArenaGameManager:
    """Manages continuous environment stepping and rendering for the web stream."""

    def __init__(self):
        self.fps = 30
        self.dt = 1.0 / float(self.fps)
        self.arena_size = 18.0
        self.env = None
        self.competitors = {}
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.is_running = True
        self._init_arena()

    def _init_arena(self):
        self.env = PartiallyObservableAgarEnv(
            arena_size=self.arena_size,
            num_players=7,
            num_food=180,
            num_viruses=7,
            total_world_mass=550.0,
            dt=self.dt
        )
        self.obs_dict, _ = self.env.reset(seed=int(time.time()) % 10000)

        # Load models
        ppo_weights = os.path.join(PROJECT_ROOT, "outputs/agar_ppo_champion.pt")
        diff_weights = os.path.join(PROJECT_ROOT, "outputs/agar_diffusion_champion.pt")
        critic_weights = os.path.join(PROJECT_ROOT, "outputs/agar_diffusion_critic_champion.pt")

        self.ppo_champ_1 = ChampionPPOAgent(weights_path=ppo_weights)
        self.ppo_champ_2 = ChampionPPOAgent(weights_path=ppo_weights)
        self.diff_champ_1 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=3, action_dim=3, obs_dim=38, num_ddim_steps=5, seed=42)
        self.diff_champ_2 = AgarDiffusionPolicy(model_path=diff_weights, critic_path=critic_weights, action_horizon=16, exec_horizon=3, action_dim=3, obs_dim=38, num_ddim_steps=5, seed=108)
        self.heuristic_apex = MasterHeuristicAgarBot(pid="player_5", profile="apex", seed=101)
        self.heuristic_hunter = MasterHeuristicAgarBot(pid="player_6", profile="hunter", seed=202)

        self.competitors = {
            "player_0": ("HUMAN (YOU)", None, (0, 215, 255)),
            "player_1": ("PPO-ALPHA", self.ppo_champ_1, (243, 156, 18)),
            "player_2": ("PPO-BETA", self.ppo_champ_2, (255, 191, 0)),
            "player_3": ("DIFFUSION-1", self.diff_champ_1, (16, 185, 129)),
            "player_4": ("DIFFUSION-2", self.diff_champ_2, (52, 211, 153)),
            "player_5": ("HEURISTIC-APEX", self.heuristic_apex, (239, 68, 68)),
            "player_6": ("HEURISTIC-HUNTER", self.heuristic_hunter, (236, 72, 153)),
        }

        # Pastel colors for food pellets
        palette = [
            (255, 179, 71), (144, 238, 144), (255, 105, 180),
            (135, 206, 250), (221, 160, 221), (255, 215, 0)
        ]
        self.food_colors = [palette[i % len(palette)] for i in range(self.env.num_food)]

    def run_loop(self):
        logger.info("Web Arena Simulation Thread started at 30 FPS.")
        frame_idx = 0
        total_w, total_h = 1200, 675
        main_w = 880
        pixels_per_meter = 38.0

        while self.is_running:
            t0 = time.perf_counter()
            try:
                if human_state.reset_requested:
                    human_state.reset_requested = False
                    self.obs_dict, _ = self.env.reset(seed=int(time.time()) % 10000)

                human_player = self.env.players["player_0"]
                h_action = human_state.get_action(human_player.yaw)

                actions = {"player_0": h_action}
                for pid in ["player_1", "player_2", "player_3", "player_4", "player_5", "player_6"]:
                    _, agent, _ = self.competitors[pid]
                    actions[pid] = agent.predict(self.obs_dict[pid], deterministic=True)

                self.obs_dict, _, _, _, _ = self.env.step(actions)

                # Render Frame
                frame = np.full((total_h, total_w, 3), (11, 15, 25), dtype=np.uint8)
                main_view = frame[:, :main_w]
                cam_x, cam_y = human_player.pos

                def w2s(wx, wy):
                    return int(main_w / 2.0 + (wx - cam_x) * pixels_per_meter), int(total_h / 2.0 - (wy - cam_y) * pixels_per_meter)

                # Arena Grid
                for gx in np.arange(-self.env.half_arena, self.env.half_arena + 0.1, 2.0):
                    cv2.line(main_view, w2s(gx, -self.env.half_arena), w2s(gx, self.env.half_arena), (22, 30, 48), 1)
                for gy in np.arange(-self.env.half_arena, self.env.half_arena + 0.1, 2.0):
                    cv2.line(main_view, w2s(-self.env.half_arena, gy), w2s(self.env.half_arena, gy), (22, 30, 48), 1)

                # Arena Wall
                cv2.rectangle(main_view, w2s(-self.env.half_arena, self.env.half_arena), w2s(self.env.half_arena, -self.env.half_arena), (239, 68, 68), 3)

                # Food Pellets
                for i, (fx, fy) in enumerate(self.env.food_positions):
                    sx, sy = w2s(fx, fy)
                    if -10 <= sx < main_w + 10 and -10 <= sy < total_h + 10:
                        cv2.circle(main_view, (sx, sy), 4, self.food_colors[i % len(self.food_colors)], -1, cv2.LINE_AA)

                draw_order = sorted(self.competitors.keys(), key=lambda p: (0 if p == "player_0" else 1, self.env.players[p].mass))

                # 1. Draw Small Cells (< 36.0kg) that can shelter inside/under viruses
                for pid in draw_order:
                    player = self.env.players[pid]
                    name, _, col = self.competitors[pid]
                    for pc_idx, pc in enumerate(player.pieces):
                        if pc.mass < 36.0:
                            sx, sy = w2s(pc.pos[0], pc.pos[1])
                            r_px = max(6, int(pc.radius * pixels_per_meter))
                            if -150 <= sx < main_w + 150 and -150 <= sy < total_h + 150:
                                draw_glowing_cell(
                                    main_view,
                                    center=(sx, sy),
                                    radius=r_px,
                                    color_bgr=col,
                                    name=name if pc_idx == 0 else "",
                                    mass=pc.mass,
                                    yaw=player.yaw,
                                    is_primary=(pid == "player_0")
                                )

                # 2. Draw Spiked Viruses (drawn over small sheltering cells as camouflage/shield)
                v_rot = frame_idx * 0.04
                for vx, vy, vr in self.env.viruses:
                    vsx, vsy = w2s(vx, vy)
                    v_radius_px = int(vr * pixels_per_meter)
                    if -50 <= vsx < main_w + 50 and -50 <= vsy < total_h + 50:
                        draw_spiked_virus(main_view, (vsx, vsy), v_radius_px, spikes=14, angle_offset=v_rot)

                # 3. Draw Large Cells (>= 36.0kg)
                for pid in draw_order:
                    player = self.env.players[pid]
                    name, _, col = self.competitors[pid]
                    for pc_idx, pc in enumerate(player.pieces):
                        if pc.mass >= 36.0:
                            sx, sy = w2s(pc.pos[0], pc.pos[1])
                            r_px = max(6, int(pc.radius * pixels_per_meter))
                            if -150 <= sx < main_w + 150 and -150 <= sy < total_h + 150:
                                draw_glowing_cell(
                                    main_view,
                                    center=(sx, sy),
                                    radius=r_px,
                                    color_bgr=col,
                                    name=name if pc_idx == 0 else "",
                                    mass=pc.mass,
                                    yaw=player.yaw,
                                    is_primary=(pid == "player_0")
                                )

                # HUD Panel (Right Side)
                hud_view = frame[:, main_w:]
                cv2.line(frame, (main_w, 0), (main_w, total_h), (30, 41, 59), 2)
                cv2.putText(hud_view, "AGAR.IO BATTLEGROUND", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (56, 189, 248), 2, cv2.LINE_AA)
                cv2.putText(hud_view, "2 PPO + 2 DIFF + 2 HEURISTIC + YOU", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (148, 163, 184), 1, cv2.LINE_AA)

                # Human Stats Card
                cv2.rectangle(hud_view, (15, 72), (total_w - main_w - 15, 186), (15, 23, 42), -1)
                cv2.rectangle(hud_view, (15, 72), (total_w - main_w - 15, 186), (56, 189, 248), 1)
                cv2.putText(hud_view, "PILOT TELEMETRY", (25, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (56, 189, 248), 1, cv2.LINE_AA)
                cv2.putText(hud_view, f"MASS: {human_player.mass:.1f} kg", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(hud_view, f"PIECES: {len(human_player.pieces)}/4", (25, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (203, 213, 225), 1, cv2.LINE_AA)
                cv2.putText(hud_view, f"KILLS: {human_player.kills} | DEATHS: {human_player.deaths}", (25, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (52, 211, 153), 1, cv2.LINE_AA)

                # Leaderboard
                cv2.putText(hud_view, "LIVE LEADERBOARD", (20, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (248, 250, 252), 1, cv2.LINE_AA)
                ranked_pids = sorted(self.competitors.keys(), key=lambda p: self.env.players[p].mass, reverse=True)
                for idx, pid in enumerate(ranked_pids):
                    p_obj = self.env.players[pid]
                    p_name, _, p_col = self.competitors[pid]
                    y_pos = 242 + idx * 30
                    bar_w = int(min(1.0, p_obj.mass / 140.0) * (total_w - main_w - 60))
                    cv2.rectangle(hud_view, (20, y_pos - 14), (20 + bar_w, y_pos - 2), (26, 36, 56), -1)
                    cv2.circle(hud_view, (28, y_pos - 8), 4, p_col, -1)
                    text = f"#{idx+1} {p_name}: {p_obj.mass:.1f}kg"
                    cv2.putText(hud_view, text, (40, y_pos - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (241, 245, 249), 1, cv2.LINE_AA)

                # Controls Help
                cv2.putText(hud_view, "CONTROLS:", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
                cv2.putText(hud_view, "- Mouse Move : Steer & Speed", (20, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)
                cv2.putText(hud_view, "- SPACEBAR   : Tactical Split", (20, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)
                cv2.putText(hud_view, "- W key      : Sprint", (20, 551), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)
                cv2.putText(hud_view, "- R key      : Respawn", (20, 574), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)

                # Encode JPEG
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()

                frame_idx += 1
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}", exc_info=True)

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, (1.0 / float(self.fps)) - elapsed)
            time.sleep(sleep_time)


arena_mgr = ArenaGameManager()
sim_thread = threading.Thread(target=arena_mgr.run_loop, daemon=True)
sim_thread.start()


class ControlInput(BaseModel):
    dx: float
    dy: float
    split: bool = False
    sprint: bool = False


@app.post("/action")
async def update_action(ctrl: ControlInput):
    human_state.update(dx=ctrl.dx, dy=-ctrl.dy, split=ctrl.split, sprint=ctrl.sprint)
    return {"status": "ok"}


@app.post("/reset")
async def reset_match():
    human_state.reset_requested = True
    return {"status": "resetting"}


@app.get("/stream")
async def video_feed():
    def frame_generator():
        while True:
            with arena_mgr.lock:
                jpg = arena_mgr.latest_jpeg
            if jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.033)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Playable Agar.io // Human vs Trained AI Bots</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Inter:wght@500;700&display=swap" rel="stylesheet">
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#030712; color:#f8fafc; font-family:'Inter', sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:100vh; overflow:hidden; }
    header { width:100%; max-width:1200px; padding:12px 16px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e293b; font-family:'JetBrains Mono', monospace; font-size:13px; }
    #arenaContainer { position:relative; width:1200px; height:675px; background:#000; border:1px solid #334155; border-radius:8px; overflow:hidden; box-shadow:0 20px 40px rgba(0,0,0,0.8); cursor:crosshair; }
    #arenaImg { width:100%; height:100%; object-fit:contain; }
    footer { margin-top:12px; font-family:'JetBrains Mono', monospace; font-size:12px; color:#94a3b8; display:flex; gap:20px; }
    .badge { background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.3); color:#38bdf8; padding:2px 8px; border-radius:4px; }
  </style>
</head>
<body>
  <header>
    <div><strong style="color:#38bdf8;">AGAR.IO BATTLEGROUND</strong> // Human vs. 2 PPO + 2 Diffusion + 2 Heuristics</div>
    <div>STATUS: <span class="badge">LIVE 30 FPS</span></div>
  </header>

  <div id="arenaContainer">
    <img id="arenaImg" src="/stream" alt="Live Arena Stream" />
  </div>

  <footer>
    <span>Steer: <strong>Move Mouse</strong></span>
    <span>Split Attack: <strong>SPACEBAR</strong></span>
    <span>Turbo Sprint: <strong>Hold W</strong></span>
    <button id="btnRespawn" style="background:#0284c7; color:#fff; border:none; padding:3px 10px; border-radius:4px; cursor:pointer; font-family:'JetBrains Mono', monospace; font-size:12px; transition:background 0.15s;">RESPAWN (R)</button>
  </footer>

  <script>
    const container = document.getElementById('arenaContainer');
    const btnRespawn = document.getElementById('btnRespawn');
    let isSprint = false;
    let isSplitPending = false;
    let lastDx = 0;
    let lastDy = 0;

    // Center of main game viewport (880px of 1200px)
    const mainViewW = 880;
    const mainViewH = 675;

    function sendAction(dx, dy, split = false, sprint = false) {
      fetch('/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dx: dx, dy: dy, split: split, sprint: sprint })
      }).catch(() => {});
    }

    container.addEventListener('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      const scaleX = 1200 / rect.width;
      const scaleY = 675 / rect.height;
      const mouseX = (e.clientX - rect.left) * scaleX;
      const mouseY = (e.clientY - rect.top) * scaleY;

      lastDx = mouseX - (mainViewW / 2.0);
      lastDy = mouseY - (mainViewH / 2.0);
    });

    // Steady 30 FPS client heartbeat loop
    setInterval(() => {
      sendAction(lastDx, lastDy, isSplitPending, isSprint);
      isSplitPending = false;
    }, 33);

    if (btnRespawn) {
      btnRespawn.addEventListener('click', () => {
        fetch('/reset', { method: 'POST' }).catch(() => {});
      });
    }

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        isSplitPending = true;
        sendAction(lastDx, lastDy, true, isSprint);
      } else if (e.code === 'KeyW') {
        isSprint = true;
      } else if (e.code === 'KeyR') {
        fetch('/reset', { method: 'POST' }).catch(() => {});
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.code === 'KeyW') {
        isSprint = false;
      }
    });
  </script>
</body>
</html>
""")


def main():
    parser = argparse.ArgumentParser(description="Agar.io Interactive Web Arena")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to run web game on")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  AGAR.IO AI ARENA // PLAYABLE HUMAN VS NEURAL CHAMPIONS WEB SERVER")
    print("=" * 70)
    print(f"  • Play in Browser : http://{args.host}:{args.port}")
    print("  • Controls        : Mouse (Steer & Thrust), SPACEBAR (Split), W (Sprint)")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
