"""Validate shipped benchmark arithmetic, provenance, media and doc links."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.benchmark_showcase import summarize, model_paths


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data=ROOT/'outputs/showcase_v1';media=ROOT/'docs/media/showcase'
    report=json.loads((data/'benchmark.json').read_text())
    manifest=json.loads((data/'media_manifest.json').read_text())
    _,models=model_paths()
    assert report['models']==manifest['models']==models['models']
    for source,digest in report['source_hashes'].items(): assert sha(ROOT/source)==digest,source
    assert sha(ROOT/'scripts/build_showcase.py')==manifest['renderer_sha256']
    assert sha(data/'demo_trace.json.gz')==report['trace_sha256']==manifest['trace_sha256']
    assert len(report['rows'])==len({r['seed'] for r in report['rows']})==12
    assert summarize(report['rows'],1800)==report['summary']
    for row in report['rows']:
        assert len(row['players'])==7 and len(row['timeline'])==60
        assert row['timeline'][-1]['seconds']==60
        for group,count in [('ppo',2),('diffusion',2),('heuristic',3)]:
            assert sum(p['group']==group for p in row['players'])==count
        assert all(np.isfinite(p['final_mass']) for p in row['players'])
    with gzip.open(data/'demo_trace.json.gz','rt') as f: trace=json.load(f)
    assert trace['seed']==report['rows'][0]['seed']==manifest['gameplay_seed']
    assert [f['step'] for f in trace['frames']]==list(range(0,1801,2))
    assert trace['frames'][-1]['seconds']==60
    before={p['id']:p['pos'] for p in trace['frames'][0]['players']}
    assert any(np.linalg.norm(np.array(p['pos'])-before[p['id']])>.1 for p in trace['frames'][225]['players'])
    assets={}
    for name,expected in manifest['assets'].items():
        path=media/name
        assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes'],name
        assert path.stat().st_size<10*1024*1024,name
        if path.suffix in ('.png','.gif'):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                item=dict(width=image.width,height=image.height,frames=getattr(image,'n_frames',1))
                if path.suffix=='.gif':
                    assert image.info['loop']==0 and image.n_frames>2
                    duration=0
                    for i in range(image.n_frames):
                        image.seek(i);duration+=image.info.get('duration',0)
                    item['seconds']=duration/1000
                    assert 8<item['seconds']<=10.5
                assets[name]=item
    for name,frames,seconds in [('gameplay.mp4',900,60.),('split_decisions.mp4',138,9.2)]:
        path=media/name
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(path)],text=True))['streams'][0]
        assert meta['codec_name']=='h264'
        assert (meta['width'],meta['height'])==(1440,900)
        assert int(meta['nb_frames'])==frames and abs(float(meta['duration'])-seconds)<.01
        assert meta['r_frame_rate']=='15/1'
        video=cv2.VideoCapture(str(path));samples=[]
        for frame in [0,frames//2,frames-1]:
            video.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=video.read()
            assert ok and image.shape==(900,1440,3)
            samples.append(hashlib.sha256(image.tobytes()).hexdigest())
        video.release();assert len(set(samples))==3
        assets[name]=meta
    scenes=json.loads((data/'illustrative_decisions.json').read_text())
    assert len(scenes)==4
    for scene in scenes:
        if scene['own']==90:
            assert scene['executed_splits']==1 and scene['capture_step']==1
        else: assert scene['executed_splits']==0
    output=data/'verification.json';checked=0
    for doc in [ROOT/'README.md',ROOT/'docs/media.md',ROOT/'docs/benchmarks.md',ROOT/'docs/checkpoint_history_2026-10-27.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
            if target.startswith(('https:','http:','#')):continue
            path=(doc.parent/target.split('#')[0]).resolve()
            assert path==output or path.exists(),f'Broken link in {doc.name}: {target}'
            checked+=1
    result=dict(passed=True,matches=12,agent_runs=84,simulation_steps=12*1800,
        policy_decisions=12*1800*7,trace_frames=901,local_links_checked=checked,
        model_source_and_data_hashes_verified=True,benchmark_arithmetic_verified=True,
        media=assets,benchmark_sha256=sha(data/'benchmark.json'),media_manifest_sha256=sha(data/'media_manifest.json'))
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
