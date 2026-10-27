"""Promote hash-verified policies only after both acceptance gates pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.arena_policies import load_package


def promote(ppo, diffusion, critic, ppo_report, diffusion_report, output):
    models = {'ppo': Path(ppo), 'diffusion': Path(diffusion), 'critic': Path(critic)}
    for algorithm, report_path in [('ppo', ppo_report), ('diffusion', diffusion_report)]:
        report = json.loads(Path(report_path).read_text())
        if report['algorithm'] != algorithm or not report['assisted']:
            raise ValueError('The report must evaluate the deployed assisted algorithm')
        summary = report['summary']
        if summary['matches'] < 12 or not summary['improved']:
            raise ValueError('Both policies must pass at least 12 acceptance matches')
        required = [algorithm] + (['critic'] if algorithm == 'diffusion' else [])
        evaluated_hashes = set(report['checkpoint_hashes'].values())
        for name in required:
            digest = hashlib.sha256(models[name].read_bytes()).hexdigest()
            if digest not in evaluated_hashes:
                raise ValueError(f'{name} differs from the evaluated checkpoint')
        for source, digest in report['source_hashes'].items():
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Evaluated source has changed: {source}')
    load_package(ppo, 'ppo'); load_package(diffusion, 'diffusion'); load_package(critic, 'return_critic')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 2, 'observation_version': 2, 'accepted_on': '2026-10-27',
        'models': {name: {'file': os.path.relpath(path.resolve(), output.parent.resolve()),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in models.items()},
        'evaluations': [os.path.relpath(Path(p).resolve(), output.parent.resolve())
                        for p in (ppo_report, diffusion_report)]}
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ppo', type=Path, default=ROOT/'outputs/arena_v2/ppo.pt')
    p.add_argument('--diffusion', type=Path, default=ROOT/'outputs/arena_v2_cosine/diffusion.pt')
    p.add_argument('--critic', type=Path, default=ROOT/'outputs/arena_v2_cosine/critic.pt')
    p.add_argument('--ppo-report', type=Path, default=ROOT/'outputs/arena_v2/ppo_acceptance.json')
    p.add_argument('--diffusion-report', type=Path, default=ROOT/'outputs/arena_v2_cosine/diffusion_acceptance.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/arena_v2/deployment.json')
    args = p.parse_args()
    promote(args.ppo, args.diffusion, args.critic, args.ppo_report, args.diffusion_report, args.output)
    print(f'Accepted deployment manifest: {args.output}')


if __name__ == '__main__':
    main()
