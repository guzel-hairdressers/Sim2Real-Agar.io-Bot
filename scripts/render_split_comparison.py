"""Render a controlled safe-attack example; this is not the acceptance set."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.agar_env import PartiallyObservableAgarEnv
from src.arena_policies import ArenaPPO, ArenaDiffusion


def scene():
    env = PartiallyObservableAgarEnv(arena_size=18, num_players=2, num_food=0,
        num_viruses=0, observation_version=2, dt=1/30)
    env.reset(seed=79)
    a, b = env.players.values()
    a.mass, b.mass = 90., 20.
    a.pos = np.array([0., 0.], np.float32); a.yaw = 0
    b.pos = np.array([1.5, 0.], np.float32)
    env.reserve_mass = 0
    return env


def main():
    torch.set_num_threads(1)
    critic = ROOT/'outputs/arena_v2_cosine/critic.pt'
    paths = [ROOT/'outputs/arena_v2/ppo.pt', ROOT/'outputs/split_combat_v3/ppo.pt',
             ROOT/'outputs/arena_v2_cosine/diffusion.pt', ROOT/'outputs/split_combat_v3/diffusion.pt']
    names = ['Previous PPO', 'Split-head PPO', 'Previous diffusion', 'Split-head diffusion']
    canvas = np.full((4*220, 4*290, 3), 250, np.uint8)
    results = []
    for row, (name, path) in enumerate(zip(names, paths)):
        policy = ArenaPPO(path) if row < 2 else ArenaDiffusion(path, critic, seed=73)
        env = scene(); split_count = 0; capture_step = None
        for step in range(19):
            if step in (0, 3, 9, 18):
                col = (0, 3, 9, 18).index(step)
                x0, y0 = col*290, row*220
                cv2.rectangle(canvas, (x0+2,y0+2), (x0+287,y0+217), (220,225,230), 1)
                cv2.putText(canvas, name, (x0+12,y0+25), cv2.FONT_HERSHEY_SIMPLEX,.5,(35,40,50),1,cv2.LINE_AA)
                cv2.putText(canvas, f'{step/30:.2f}s | splits {split_count}', (x0+12,y0+48),cv2.FONT_HERSHEY_SIMPLEX,.4,(65,70,80),1,cv2.LINE_AA)
                for pid, player in env.players.items():
                    for piece in player.pieces:
                        px, py = int(x0+76+piece.pos[0]*52), int(y0+132-piece.pos[1]*52)
                        if not (x0+4 < px < x0+286 and y0+65 < py < y0+210):
                            continue
                        color = (195,110,35) if pid=='player_0' else (50,145,225)
                        cv2.circle(canvas,(px,py),max(3,int(piece.radius*52)),color,-1,cv2.LINE_AA)
                        cv2.putText(canvas,str(round(piece.mass)),(px-12,py+5),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
                if capture_step is not None:
                    cv2.putText(canvas,f'Target captured at {capture_step/30:.2f}s',(x0+12,y0+202),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,105,45),1,cv2.LINE_AA)
            if step == 18:
                break
            action = policy.predict(env._get_obs('player_0'))
            split_count += int(action[2]>.5 and env.players['player_0'].can_split)
            env.step({'player_0':action,'player_1':np.zeros(3,np.float32)})
            if capture_step is None and any(e['predator']=='player_0' and e['prey']=='player_1' for e in env.predation_events):
                capture_step = step+1
        results.append(dict(policy=name,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            splits=split_count,capture_step=capture_step))
    out = ROOT/'docs/media/split_comparison.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    (ROOT/'outputs/split_combat_v3/illustrative_scene.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
