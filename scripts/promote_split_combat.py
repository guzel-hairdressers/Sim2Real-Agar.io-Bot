"""Promote split heads after physical combat and v2 retention checks.

Historical v2 reports remain historical: source changes invalidate them for
promotion. This gate verifies fresh reports against current code and weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.arena_policies import load_package
from src.evaluate_split_combat import summarize as summarize_combat
from src.evaluate_arena_v2 import summarize as summarize_arena


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(report, required):
    hashes = report.get('source_hashes', {})
    if not set(required).issubset(hashes):
        raise ValueError('Report omits required source hashes')
    for source, expected in hashes.items():
        if digest(ROOT/source) != expected:
            raise ValueError(f'Evaluated source has changed: {source}')


def promote(directory, combat_path, ppo_path, diffusion_path, output=None):
    directory = Path(directory)
    models = {'ppo': directory/'ppo.pt', 'diffusion': directory/'diffusion.pt',
              'critic': ROOT/'outputs/arena_v2_cosine/critic.pt'}
    combat = json.loads(Path(combat_path).read_text())
    common = ['envs/agar_env.py', 'src/arena_policies.py', 'src/arena_tactics.py',
              'src/train_champion_ppo.py', 'src/split_head.py']
    verify_sources(combat, common+['src/train_split_combat.py', 'src/evaluate_split_combat.py'])
    for name in ('ppo', 'diffusion'):
        if combat['checkpoint_hashes'].get(name+'.pt') != digest(models[name]):
            raise ValueError(f'{name} differs from combat checkpoint')
    calculated = summarize_combat(combat['rows'])
    for algorithm in ('ppo', 'diffusion'):
        subset = [r for r in combat['rows'] if r['algorithm'] == algorithm]
        seeds = {r['seed'] for r in subset}
        if len(seeds) != len(subset) or len(seeds) < 950:
            raise ValueError('Combat requires at least 950 distinct fresh cases per algorithm')
        training = load_package(models[algorithm], algorithm)['combat_training']
        training_start = training['seed']
        if any(training_start <= seed < training_start+training['cases'] for seed in seeds):
            raise ValueError('Combat cases overlap training/development data')
        if not calculated[algorithm]['passed']:
            raise ValueError(f'{algorithm} failed physical combat acceptance')
    for algorithm, path in [('ppo', ppo_path), ('diffusion', diffusion_path)]:
        report = json.loads(Path(path).read_text())
        verify_sources(report, common+['envs/arena_config.py', 'src/agar_diffusion_policy.py',
                                      'src/evaluate_arena_v2.py', 'src/heuristic_agent.py'])
        if report['algorithm'] != algorithm or not report['assisted'] or report.get('baseline_version') != 'arena_v2':
            raise ValueError('Retention must compare deployed policies against arena_v2')
        rows = report['rows']
        if len(rows) < 12 or len({r['seed'] for r in rows}) != len(rows) or report['config']['max_steps'] < 1800:
            raise ValueError('Retention requires 12 distinct full-length matches')
        evaluated = set(report['checkpoint_hashes'].values())
        for name in [algorithm]+(['critic'] if algorithm == 'diffusion' else []):
            if digest(models[name]) not in evaluated:
                raise ValueError(f'{name} differs from retention checkpoint')
        baseline = ROOT/('outputs/arena_v2/ppo.pt' if algorithm == 'ppo' else 'outputs/arena_v2_cosine/diffusion.pt')
        if digest(baseline) not in evaluated:
            raise ValueError('Retention baseline checkpoint is missing')
        summary = summarize_arena(rows)
        mass_ok = summary['new']['final_mass'] >= .9*summary['legacy']['final_mass']
        deaths_ok = sum(r['groups']['new']['deaths'] for r in rows) <= sum(r['groups']['legacy']['deaths'] for r in rows)+2
        if not mass_ok or not deaths_ok:
            raise ValueError(f'{algorithm} failed v2 retention tolerance')
    for name, algorithm in [('ppo', 'ppo'), ('diffusion', 'diffusion'), ('critic', 'return_critic')]:
        load_package(models[name], algorithm)
    output = Path(output) if output else directory/'deployment.json'
    reports = [Path(p) for p in (combat_path, ppo_path, diffusion_path)]
    manifest = {'schema_version': 2, 'observation_version': 2, 'accepted_on': '2026-10-27',
        'models': {name: {'file': os.path.relpath(path.resolve(), output.parent.resolve()), 'sha256': digest(path)}
                   for name, path in models.items()},
        'evaluations': [os.path.relpath(p.resolve(), output.parent.resolve()) for p in reports],
        'evaluation_sha256': {p.name: digest(p) for p in reports},
        'acceptance': 'paired projectile attacks plus 90% mass / +2 total deaths v2 retention tolerance'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=ROOT/'outputs/split_combat_v3')
    args = p.parse_args()
    promote(args.directory, args.directory/'combat_acceptance.json',
            args.directory/'ppo_retention.json', args.directory/'diffusion_retention.json')
    print(f'Accepted deployment: {args.directory}/deployment.json')


if __name__ == '__main__':
    main()
