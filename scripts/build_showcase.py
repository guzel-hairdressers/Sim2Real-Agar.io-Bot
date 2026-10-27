"""Render real simulator traces, current-policy decisions and benchmark figures.

No generated gameplay, hand-scripted split commands or historical checkpoints.
Requires Pillow and ffmpeg for export; these are optional media-build tools.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.benchmark_showcase import model_paths, snapshot
from src.arena_policies import ArenaPPO, ArenaDiffusion
from envs.agar_env import PartiallyObservableAgarEnv

OUT = ROOT/'docs/media/showcase'
DATA = ROOT/'outputs/showcase_v1'
COLORS = dict(ppo='#3569D4',diffusion='#00836F',heuristic='#B47620')
INK, MUTED, BORDER = '#17243B','#64748B','#DCE4EE'
FONT = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
BOLD = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
FONTS = {}


def font(size,bold=False):
    key = size,bold
    if key not in FONTS:
        path = BOLD if bold else FONT
        if not path.exists():
            from matplotlib.font_manager import findfont, FontProperties
            path = findfont(FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal'))
        FONTS[key] = ImageFont.truetype(str(path),size)
    return FONTS[key]


def text(d,xy,s,size=16,fill=INK,bold=False,anchor=None):
    d.text(xy,str(s),font=font(size,bold),fill=fill,anchor=anchor)


def card(d,box):
    d.rounded_rectangle(box,radius=14,fill='white',outline=BORDER,width=1)


def world(state,size,center=(0,0),scale=38,observer=None):
    w,h = size
    image = Image.new('RGB',size,'#FCFDFE'); d = ImageDraw.Draw(image)
    def point(pos):
        return w/2+(pos[0]-center[0])*scale,h/2-(pos[1]-center[1])*scale
    def visible(pos):
        return observer is None or np.linalg.norm(np.asarray(pos)-observer['pos'])<=observer['vision_radius']
    for coord in range(-8,9,2):
        d.line([point((coord,-9)),point((coord,9))],fill='#EDF1F6')
        d.line([point((-9,coord)),point((9,coord))],fill='#EDF1F6')
    d.rectangle([point((-9,9)),point((9,-9))],outline='#A6B4C6',width=2)
    for pos in state['food']:
        if visible(pos):
            x,y=point(pos);d.ellipse((x-2,y-2,x+2,y+2),fill='#A5B8A7')
    for vx,vy,radius in state['viruses']:
        if not visible((vx,vy)): continue
        x,y=point((vx,vy));r=radius*scale
        pts=[(x+r*(1 if i%2==0 else .8)*math.cos(i*math.pi/12),
              y+r*(1 if i%2==0 else .8)*math.sin(i*math.pi/12)) for i in range(24)]
        d.polygon(pts,fill='#D6A5AD',outline='#B97985')
        d.ellipse((x-r*.35,y-r*.35,x+r*.35,y+r*.35),fill='#B97985')
    for player in sorted(state['players'],key=lambda p:p['mass']):
        for i,piece in enumerate(player['pieces']):
            if observer is not None and player['id']!=observer['id'] and not visible(piece['pos']): continue
            x,y=point(piece['pos']);r=max(4,piece['radius']*scale)
            d.ellipse((x-r,y-r,x+r,y+r),fill=COLORS[player['group']],outline='white',width=2)
            hx,hy=x+math.cos(player['yaw'])*r*.8,y-math.sin(player['yaw'])*r*.8
            d.line([(x+math.cos(player['yaw'])*r*.58,y-math.sin(player['yaw'])*r*.58),(hx,hy)],fill='white',width=2)
            if observer is None and r>13:
                short=player['name'].replace('PPO-','P').replace('DIFF-','D').replace('RULE-','H')
                if short.startswith('H'): short={'RULE-APEX':'HA','RULE-HUNTER':'HH','RULE-SURVIVOR':'HS'}.get(player['name'],'T')
                text(d,(x,y-6),short,11,'white',True,'mm')
                text(d,(x,y+8),round(piece['mass']),10,'white',False,'mm')
    if observer is not None:
        x,y=point(observer['pos']);r=observer['vision_radius']*scale
        mask=Image.new('L',size,0);ImageDraw.Draw(mask).ellipse((x-r,y-r,x+r,y+r),fill=255)
        image=Image.composite(image,Image.new('RGB',size,'#E9EEF5'),mask)
        ImageDraw.Draw(image).ellipse((x-r,y-r,x+r,y+r),outline=COLORS[observer['group']],width=2)
    return image


def gameplay_frame(state,duration):
    image=Image.new('RGB',(1440,900),'#F3F6FA');d=ImageDraw.Draw(image)
    text(d,(30,20),'SIM2REAL / AGAR CONTROL',12,MUTED,True)
    text(d,(30,43),'Learning to navigate and compete',32,INK,True)
    text(d,(31,87),'PPO and diffusion agents in a shared, partially observed world.',17,MUTED)
    card(d,(1190,22,1410,105))
    text(d,(1210,35),'RECORDED SIMULATION',11,MUTED,True)
    text(d,(1210,57),f"{state['seconds']:05.1f}s / {duration:.0f}s",25,INK,True)
    card(d,(28,125,810,850));card(d,(834,125,1412,528));card(d,(834,546,1412,850))
    focus=next(p for p in state['players'] if p['name']=='PPO-1')
    image.paste(world(state,(692,692),scale=38),(73,143))
    # Dashed sensing-radius outline on the spectator map, clipped to the map.
    overlay=Image.new('RGBA',(692,692),(0,0,0,0));od=ImageDraw.Draw(overlay)
    x,y=346+focus['pos'][0]*38,346-focus['pos'][1]*38;r=focus['vision_radius']*38
    for a in range(0,360,12): od.arc((x-r,y-r,x+r,y+r),a,a+6,fill=COLORS['ppo'],width=2)
    image.paste(overlay,(73,143),overlay)
    text(d,(854,144),'Local sensing footprint',21,INK,True)
    text(d,(855,177),f"PPO-1 · {focus['vision_radius']:.1f} m range · 70 numeric inputs",15,MUTED)
    image.paste(world(state,(530,274),center=focus['pos'],scale=118/focus['vision_radius'],observer=focus),(858,214))
    text(d,(855,500),'Illustration of visibility; policies do not receive images.',13,MUTED)
    text(d,(854,565),'Live agent state',21,INK,True)
    for x,label in [(1135,'MASS'),(1220,'CAPTURES'),(1302,'DEATHS'),(1384,'SPLITS')]:
        text(d,(x,600),label,10,MUTED,True,'ra')
    for i,p in enumerate(sorted(state['players'],key=lambda p:p['mass'],reverse=True)):
        y=630+i*28
        d.ellipse((855,y+2,864,y+11),fill=COLORS[p['group']])
        text(d,(876,y),p['name'],14,INK,True)
        for x,v in [(1135,f"{p['mass']:.1f}"),(1220,p['kills']),(1302,p['deaths']),(1384,p['splits'])]:
            text(d,(x,y),v,14,INK,False,'ra')
    x=35
    for label,color in [('Food','#A5B8A7'),('PPO ×2',COLORS['ppo']),('Diffusion ×2',COLORS['diffusion']),('Rules ×3',COLORS['heuristic']),('Virus hazard','#B97985')]:
        d.ellipse((x,866,x+9,875),fill=color);text(d,(x+16,862),label,12,MUTED);x+=140
    text(d,(1410,866),'30 Hz decisions · 15 fps recording · 1× speed',13,MUTED,False,'ra')
    return image


def encode(path,frames,fps=15):
    with open('/tmp/arena_showcase_ffmpeg.log','ab') as log:
        p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1440x900','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in frames: p.stdin.write(frame.tobytes())
        finally:
            p.stdin.close()
        if p.wait()!=0: raise RuntimeError('ffmpeg failed; see /tmp/arena_showcase_ffmpeg.log')


def gif(video,path,start,duration):
    filters='fps=8,scale=864:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(start),'-t',str(duration),'-i',str(video),
                    '-filter_complex',filters,'-loop','0',str(path)],check=True)


def encounter_states():
    paths,_=model_paths();torch.set_num_threads(1)
    scenes=[]
    for algorithm in ('ppo','diffusion'):
        for own,prey in [(90,20),(45,30)]:
            env=PartiallyObservableAgarEnv(arena_size=18,num_players=2,num_food=0,num_viruses=0,
                                         observation_version=2,dt=1/30,max_steps=90)
            env.reset(seed=79);a,b=env.players.values()
            a.mass,b.mass=own,prey;a.pos=np.array([0.,0.],np.float32);a.yaw=0.
            b.pos=np.array([1.5,0.],np.float32);b.yaw=math.pi;env.reserve_mass=0
            policy=ArenaPPO(paths['ppo']) if algorithm=='ppo' else ArenaDiffusion(paths['diffusion'],paths['critic'],seed=79)
            seats={'player_0':dict(group=algorithm,name='PPO-1' if algorithm=='ppo' else 'DIFF-1'),
                   'player_1':dict(group='heuristic',name='TARGET')}
            splits=dict.fromkeys(env.player_ids,0);states=[snapshot(env,seats,splits,{})];capture=None
            for step in range(36):
                action=policy.predict(env._get_obs('player_0'))
                splits['player_0']+=int(action[2]>.5 and a.can_split)
                env.step({'player_0':action,'player_1':np.zeros(3,np.float32)})
                if capture is None and any(e['predator']=='player_0' and e['prey']=='player_1' for e in env.predation_events): capture=step+1
                states.append(snapshot(env,seats,splits,{'player_0':action}))
            scenes.append(dict(algorithm=algorithm,own=own,target=prey,states=states,capture_step=capture,
                               executed_splits=splits['player_0']))
    return scenes


def decision_frame(scenes,index):
    image=Image.new('RGB',(1440,900),'#F3F6FA');d=ImageDraw.Draw(image)
    text(d,(30,20),'CONTROLLED CURRENT-POLICY DEMONSTRATION',12,MUTED,True)
    text(d,(30,44),'When should an agent split?',34,INK,True)
    text(d,(31,94),'Actual policy outputs and simulator physics. Stationary targets. Playback slowed to 1/6 speed.',17,MUTED)
    for k,scene in enumerate(scenes):
        x,y=28+(k%2)*706,143+(k//2)*342
        card(d,(x,y,x+678,y+320))
        text(d,(x+20,y+17),scene['algorithm'].upper(),17,COLORS[scene['algorithm']],True)
        text(d,(x+20,y+47),'Safe split opportunity' if k%2==0 else 'Keep enough mass: hold the split',20,INK,True)
        text(d,(x+20,y+80),f"Actor {scene['own']} · target {scene['target']} · half-mass {scene['own']/2:g}",14,MUTED)
        state=scene['states'][index]
        image.paste(world(state,(636,170),center=(1.,state['players'][0]['pos'][1]),scale=62),(x+20,y+110))
        actor=state['players'][0]
        note=f"Splits executed: {actor['splits']}"
        if scene['capture_step'] and index>=scene['capture_step']:
            note+=f"   ·   Target captured at {scene['capture_step']/30:.2f}s"
        text(d,(x+20,y+291),note,15,INK,True)
    text(d,(30,864),'Splitting halves a piece: a smaller target can still be too large for the launched half.',15,MUTED)
    text(d,(1410,864),f'Simulator time {index/30:.2f}s',15,MUTED,False,'ra')
    return image


def benchmark_figure(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,
                         'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':BORDER})
    fig,axes=plt.subplots(2,2,figsize=(14,8.8),layout='constrained')
    fig.suptitle('Current agents: measured capabilities',fontsize=23,fontweight='bold',x=.03,ha='left')
    groups=['ppo','diffusion','heuristic'];labels=['PPO','Diffusion','Heuristic'];colors=[COLORS[g] for g in groups]
    ax=axes[0,0]
    for group,label,color in zip(groups,labels,colors):
        values=np.array([[t[group] for t in row['timeline']] for row in report['rows']])
        x=np.array([t['seconds'] for t in report['rows'][0]['timeline']])
        samples=np.random.default_rng(417).integers(0,len(values),(10000,len(values)))
        lo,hi=np.percentile(values[samples].mean(axis=1),[2.5,97.5],axis=0)
        ax.plot(x,values.mean(axis=0),label=label,color=color,lw=2.5)
        ax.fill_between(x,lo,hi,color=color,alpha=.1)
    ax.set(title='Growth during a 60-second match',xlabel='Simulator time (seconds)',ylabel='Mean mass per agent (simulation units)')
    ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15)
    ax=axes[0,1];counts=[report['summary'][g]['zero_death_runs'] for g in groups]
    n=[report['summary'][g]['agent_runs'] for g in groups]
    bars=ax.bar(labels,np.array(counts)/n*100,color=colors,width=.55)
    for b,c,total in zip(bars,counts,n): ax.text(b.get_x()+b.get_width()/2,b.get_height()+3,f'{c}/{total}',ha='center',fontweight='bold')
    ax.set(title='Agent runs with no full death',ylabel='Share of 60-second agent runs (%)',ylim=(0,115));ax.grid(axis='y',alpha=.15)
    ax=axes[1,0];minutes=[report['summary'][g]['agent_minutes'] for g in groups]
    rates=[report['summary'][g]['piece_captures']/m for g,m in zip(groups,minutes)]
    bars=ax.bar(labels,rates,color=colors,width=.55)
    for b,v in zip(bars,rates): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.06,f'{v:.2f}',ha='center',fontweight='bold')
    ax.set(title='Captures in mixed-agent matches',ylabel='Opponent pieces captured / agent-minute',ylim=(0,max(rates)*1.3+.1));ax.grid(axis='y',alpha=.15)
    combat=json.loads((ROOT/'outputs/split_combat_v3/combat_acceptance.json').read_text())
    for name in ('ppo','diffusion'):
        if combat['checkpoint_hashes'][name+'.pt']!=report['models'][name]['sha256']:
            raise ValueError('Split benchmark does not describe these models')
    ax=axes[1,1];cs=[combat['summary'][g] for g in groups[:2]]
    rates=[c['positive_projectile_capture_rate']*100 for c in cs]
    bars=ax.bar(labels[:2],rates,color=colors[:2],width=.45)
    for b,c in zip(bars,cs):
        hit=round(c['positive_projectile_capture_rate']*c['positive_opportunities'])
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+3,f"{hit}/{c['positive_opportunities']}",ha='center',fontweight='bold')
    ax.set(title='Separate split-opportunity test',ylabel='Beneficial opportunities captured (%)',ylim=(0,115));ax.grid(axis='y',alpha=.15)
    fig.supxlabel('12 independent matches · 2 PPO + 2 diffusion + 3 heuristics · shaded bands: match-bootstrap 95% CI\nSplit panel: 996 valid encounters per policy; denominator is known beneficial opportunities, not all combat.',fontsize=10,color=MUTED)
    fig.savefig(OUT/'benchmarks.png',dpi=150,facecolor='white');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gameplay-only',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with gzip.open(DATA/'demo_trace.json.gz','rt') as f:trace=json.load(f)
    duration=trace['config']['max_steps']/30
    states=trace['frames']
    # The final state is a still, not an additional video-time interval.
    encode(OUT/'gameplay.mp4',(gameplay_frame(s,duration) for s in states[:-1]))
    gameplay_frame(states[min(225,len(states)-1)],duration).save(OUT/'overview.png')
    gameplay_frame(states[min(675,len(states)-1)],duration).save(OUT/'local_sensing.png')
    gif(OUT/'gameplay.mp4',OUT/'gameplay.gif',5,10)
    if args.gameplay_only:return
    scenes=encounter_states()
    frames=[0]*15+[i for i in range(1,37) for _ in range(3)]+[36]*15
    encode(OUT/'split_decisions.mp4',(decision_frame(scenes,i) for i in frames))
    decision_frame(scenes,9).save(OUT/'split_decisions.png')
    gif(OUT/'split_decisions.mp4',OUT/'split_decisions.gif',0,9.2)
    (DATA/'illustrative_decisions.json').write_text(json.dumps([{k:v for k,v in s.items() if k!='states'} for s in scenes],indent=2))
    report=json.loads((DATA/'benchmark.json').read_text());benchmark_figure(report)
    assets={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.is_file()}
    manifest=dict(assets=assets,gameplay_seed=trace['seed'],gameplay_seconds=duration,
        simulation_hz=30,gameplay_video_hz=15,gameplay_speed=1,split_demo_speed=1/6,
        trace_sha256=hashlib.sha256((DATA/'demo_trace.json.gz').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=report['models'])
    (DATA/'media_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
