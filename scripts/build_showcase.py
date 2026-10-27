"""Render full active matches and paired comparisons at two-times speed.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

OUT = ROOT/'docs/media/showcase'
DATA = ROOT/'outputs/showcase_v1'
CLASHES = ROOT/'outputs/showcase_clashes'
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
                if short.startswith('H'): short={'RULE-APEX':'HA','RULE-HUNTER':'HH','RULE-SURVIVOR':'HS'}.get(player['name'],'H'+player['id'][-1])
                text(d,(x,y-6),short,11,'white',True,'mm')
                text(d,(x,y+8),round(piece['mass']),10,'white',False,'mm')
    if observer is not None:
        x,y=point(observer['pos']);r=observer['vision_radius']*scale
        mask=Image.new('L',size,0);ImageDraw.Draw(mask).ellipse((x-r,y-r,x+r,y+r),fill=255)
        image=Image.composite(image,Image.new('RGB',size,'#E9EEF5'),mask)
        ImageDraw.Draw(image).ellipse((x-r,y-r,x+r,y+r),outline=COLORS[observer['group']],width=2)
    return image




def encode(path,frames,fps=30):
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


def short_name(state,pid):
    p=next(p for p in state['players'] if p['id']==pid)
    return p['name'].replace('PPO-','P').replace('DIFF-','D').replace('RULE-','H-')


def camera_target(state,events,focus=None):
    t=state['seconds']
    near=[e for e in events if e['kind']=='capture' and t-1<=e['seconds']<=t+3 and
          (focus is None or focus in (e['predator'],e['prey']))]
    if near:
        return np.array(min(near,key=lambda e:abs(e['seconds']-t))['pos'])
    best=None
    for p in state['players']:
        if focus is not None and p['id']!=focus:continue
        for q in state['players']:
            if p['id']==q['id']:continue
            for a in p['pieces']:
                for b in q['pieces']:
                    dist=math.dist(a['pos'],b['pos'])
                    gap=dist-a['radius']-b['radius']
                    if best is None or gap<best[0]:best=(gap,a['pos'],b['pos'],dist)
    if focus is not None and best[3]>5:
        return np.array(next(p for p in state['players'] if p['id']==focus)['pos'])
    return (np.array(best[1])+best[2])/2


def marked_world(state,events,size,center,scale):
    image=world(state,size,center=center,scale=scale);d=ImageDraw.Draw(image)
    for e in events:
        age=state['seconds']-e['seconds']
        if e['kind']!='capture' or not 0<=age<2:continue
        x=size[0]/2+(e['pos'][0]-center[0])*scale;y=size[1]/2-(e['pos'][1]-center[1])*scale
        if 0<x<size[0] and 0<y<size[1]:
            r=12+age*10
            d.ellipse((x-r,y-r,x+r,y+r),outline='#D14D57',width=3)
    return image


def recent_labels(state,events,focus=None,count=2):
    found=[]
    for e in reversed(events):
        if e['seconds']>state['seconds']:continue
        ids=(e['predator'],e['prey']) if e['kind']=='capture' else (e['player'],)
        if focus is not None and focus not in ids:continue
        if e['kind']=='capture':label=f"{short_name(state,e['predator'])} captured {short_name(state,e['prey'])}"
        elif e['kind']=='death':label=f"{short_name(state,e['player'])} full death / respawn"
        else:label=f"{short_name(state,e['player'])} executed a split"
        found.append(f"{e['seconds']:.1f}s · {label}")
        if len(found)==count:break
    return found or ['Awaiting first capture or split']


def gameplay_frame(state,events,center):
    image=Image.new('RGB',(1440,900),'#F3F6FA');d=ImageDraw.Draw(image)
    text(d,(30,20),'SIM2REAL / AGAR CONTROL',12,MUTED,True)
    text(d,(30,43),'PPO + diffusion in a shared arena',32,INK,True)
    text(d,(31,87),'Autonomous match · reactive opponents · 2× playback · actual captures marked in red',17,MUTED)
    card(d,(1190,22,1410,105));text(d,(1210,35),'SIMULATOR TIME / 2×',11,MUTED,True)
    text(d,(1210,57),f"{state['seconds']:05.1f}s / 120s",25,INK,True)
    card(d,(28,125,810,850));card(d,(834,125,1412,528));card(d,(834,546,1412,850))
    image.paste(marked_world(state,events,(692,692),(0,0),38),(73,143))
    overlay=Image.new('RGBA',(692,692),(0,0,0,0));od=ImageDraw.Draw(overlay)
    x,y=346+center[0]*38,346-center[1]*38
    od.rectangle((x-530/66*19,y-274/66*19,x+530/66*19,y+274/66*19),outline=INK,width=2)
    image.paste(overlay,(73,143),overlay)
    text(d,(854,144),'Clash camera',21,INK,True)
    text(d,(855,177),'Automatic spectator zoom on nearby encounters',15,MUTED)
    image.paste(marked_world(state,events,(530,274),center,66),(858,214))
    label=recent_labels(state,events,count=1)[0]
    text(d,(855,500),label,13,MUTED)
    text(d,(854,565),'Live agent state',21,INK,True)
    for x,label in [(1135,'MASS'),(1220,'CAPTURES'),(1302,'DEATHS'),(1384,'SPLITS')]:text(d,(x,600),label,10,MUTED,True,'ra')
    for i,p in enumerate(sorted(state['players'],key=lambda p:p['mass'],reverse=True)):
        y=630+i*28;d.ellipse((855,y+2,864,y+11),fill=COLORS[p['group']]);text(d,(876,y),p['name'],14,INK,True)
        for x,v in [(1135,f"{p['mass']:.1f}"),(1220,p['kills']),(1302,p['deaths']),(1384,p['splits'])]:text(d,(x,y),v,14,INK,False,'ra')
    x=35
    for label,color in [('Food','#A5B8A7'),('PPO ×2',COLORS['ppo']),('Diffusion ×2',COLORS['diffusion']),('Rules ×3',COLORS['heuristic']),('Virus hazard','#B97985')]:
        d.ellipse((x,866,x+9,875),fill=color);text(d,(x+16,862),label,12,MUTED);x+=140
    text(d,(1410,866),'30 Hz decisions · 2× playback · spectator view',13,MUTED,False,'ra')
    return image


def comparison_frame(states,events,centers):
    image=Image.new('RGB',(1440,900),'#F3F6FA');d=ImageDraw.Draw(image)
    text(d,(30,20),'PAIRED CURRENT-POLICY COMPARISON',12,MUTED,True)
    text(d,(30,43),'PPO vs diffusion',34,INK,True)
    text(d,(31,90),'Same starting state · six reactive heuristic opponents per run · 2× playback',17,MUTED)
    card(d,(1190,22,1410,105));text(d,(1210,35),'SIMULATOR TIME / 2×',11,MUTED,True)
    text(d,(1210,57),f"{states[0]['seconds']:05.1f}s / 120s",25,INK,True)
    for k,(state,ev,center,group) in enumerate(zip(states,events,centers,['ppo','diffusion'])):
        x=28+k*706;y=130;card(d,(x,y,x+678,852))
        text(d,(x+20,y+19),'PPO' if group=='ppo' else 'DIFFUSION',24,COLORS[group],True)
        text(d,(x+20,y+56),'One learned focus agent + six independently reacting opponents',14,MUTED)
        image.paste(marked_world(state,ev,(638,440),center,64),(x+20,y+90))
        image.paste(marked_world(state,ev,(148,148),(0,0),8),(x+20,y+550))
        p=next(p for p in state['players'] if p['id']=='player_0')
        for i,(label,value) in enumerate([('MASS',f"{p['mass']:.1f}"),('CAPTURES',p['kills']),('DEATHS',p['deaths']),('SPLITS',p['splits'])]):
            tx=x+194+i*111;text(d,(tx,y+550),label,11,MUTED,True);text(d,(tx,y+572),value,26,INK,True)
        for i,label in enumerate(recent_labels(state,ev,focus='player_0')):text(d,(x+194,y+628+i*25),label,12,MUTED)
        text(d,(x+194,y+690),'Deployed model actions + safety layer',12,MUTED)
    text(d,(30,868),'Separate rollouts from identical positions, masses and hazards; this is not a direct duel.',14,MUTED)
    text(d,(1410,868),'Blue: PPO · green: diffusion · amber: heuristics',13,MUTED,False,'ra')
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
    ax=axes[1,1];positions=np.arange(3);width=.33
    mean=[report['summary'][g]['inference_mean_ms'] for g in groups]
    p95=[report['summary'][g]['mean_per_match_inference_p95_ms'] for g in groups]
    ax.bar(positions-width/2,mean,width,color=colors,label='Mean')
    ax.bar(positions+width/2,p95,width,color=colors,alpha=.35,label='Mean per-match p95')
    ax.set(title='Policy inference on this Mac (CPU)',ylabel='Milliseconds per action',xticks=positions,xticklabels=labels)
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.supxlabel('12 independent matches · 2 PPO + 2 diffusion + 3 heuristics · shaded bands: match-bootstrap 95% CI\nTiming includes deployed safety checks under concurrent load; excludes rendering and is not browser FPS.',fontsize=10,color=MUTED)
    fig.savefig(OUT/'benchmarks.png',dpi=150,facecolor='white');plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gameplay-only',action='store_true',help='Rebuild the main video and GIF while retaining comparison and chart assets')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    recording=json.loads((CLASHES/'recordings.json').read_text());traces={}
    for item in recording['recordings']:
        path=CLASHES/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Recording hash mismatch')
        with gzip.open(path,'rt') as f:traces[item['kind']]=json.load(f)
    mixed=traces['mixed'];events=mixed['events'];frames=mixed['frames']
    first=next(e for e in events if e['kind']=='capture')
    # Two playback seconds at 2x correspond to four simulator seconds.
    start_index=max(0,round((first['seconds']-4)*mixed['capture_hz']))
    start_seconds=frames[start_index]['seconds']
    def mixed_frames():
        center=camera_target(frames[0],events)
        for index,state in enumerate(frames[:-1]):
            center=.82*center+.18*camera_target(state,events)
            if index>=start_index:
                yield gameplay_frame(state,events,center)
    encode(OUT/'gameplay.mp4',mixed_frames())
    # Keep the original simulator timestamps and every frame after the trim.
    poster=frames[min(len(frames)-1,round((first['seconds']+.3)*15))]
    gameplay_frame(poster,events,np.array(first['pos'])).save(OUT/'overview.png')
    later=[e for e in events if e['kind']=='capture'][-1]
    state=frames[min(len(frames)-1,round((later['seconds']+.3)*15))]
    gameplay_frame(state,events,np.array(later['pos'])).save(OUT/'local_sensing.png')
    left,right=traces['ppo'],traces['diffusion'];ev=[left['events'],right['events']]
    def paired_frames():
        centers=[camera_target(t['frames'][0],t['events'],focus='player_0') for t in (left,right)]
        for states in zip(left['frames'][:-1],right['frames'][:-1]):
            centers=[.82*c+.18*camera_target(s,e,focus='player_0') for c,s,e in zip(centers,states,ev)]
            yield comparison_frame(states,ev,centers)
    if not args.gameplay_only:
        encode(OUT/'policy_comparison.mp4',paired_frames())
    focus_event=next(e for e in left['events'] if e['kind']=='capture' and e['predator']=='player_0')
    index=min(len(left['frames'])-1,round((focus_event['seconds']+.3)*15))
    states=[left['frames'][index],right['frames'][index]]
    centers=[camera_target(s,e,focus='player_0') for s,e in zip(states,ev)]
    if not args.gameplay_only:
        comparison_frame(states,ev,centers).save(OUT/'policy_comparison.png')
    gif(OUT/'gameplay.mp4',OUT/'gameplay.gif',0,30)
    report=json.loads((DATA/'benchmark.json').read_text())
    if not args.gameplay_only:
        gif(OUT/'policy_comparison.mp4',OUT/'policy_comparison.gif',0,30)
        benchmark_figure(report)
    assets={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.is_file()}
    manifest=dict(schema_version=3,assets=assets,simulation_hz=30,source_recording_hz=15,
        video_hz=30,playback_speed=2,recording_simulation_seconds=120,preview_seconds=30,
        videos={'gameplay.mp4':dict(frames=len(frames)-1-start_index,
                    video_seconds=(len(frames)-1-start_index)/30,
                    simulation_start_seconds=start_seconds,simulation_end_seconds=frames[-1]['seconds'],
                    first_capture_simulation_seconds=first['seconds'],
                    first_capture_playback_seconds=(first['seconds']-start_seconds)/2),
                'policy_comparison.mp4':dict(frames=len(left['frames'])-1,
                    video_seconds=(len(left['frames'])-1)/30,
                    simulation_start_seconds=0,simulation_end_seconds=left['frames'][-1]['seconds'])},
        recordings_sha256=hashlib.sha256((CLASHES/'recordings.json').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=report['models'],
        poster_simulator_seconds=dict(overview=poster['seconds'],local_sensing=state['seconds'],comparison=states[0]['seconds']))
    (DATA/'media_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()





