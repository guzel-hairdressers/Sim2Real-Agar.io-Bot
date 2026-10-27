"""Promote outcome-validated policies with physical-only split constraints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.arena_policies import load_package
from src.evaluate_learned_risk import summarize as outcome_summary
from src.evaluate_risk_league import retention_summary


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(report,required):
    hashes=report.get('source_hashes',{})
    if not set(required).issubset(hashes):raise ValueError('Required source hashes missing')
    for path,expected in hashes.items():
        if digest(ROOT/path)!=expected:raise ValueError(f'Evaluated source changed: {path}')


def promote(directory,output=None):
    directory=Path(directory)
    protocol=json.loads((directory/'acceptance_protocol.json').read_text())
    common=['src/learned_risk.py','src/arena_policies.py','src/arena_tactics.py','envs/agar_env.py','src/heuristic_agent.py']
    models={a:directory/f'{a}.pt' for a in ('ppo','diffusion')}
    models['critic']=ROOT/'outputs/arena_v2_cosine/critic.pt'
    hashes={a:digest(p) for a,p in models.items()}
    for algorithm in ('ppo','diffusion'):
        package=load_package(models[algorithm],algorithm)
        if package.get('split_control')!='learned_risk_physical_only':raise ValueError('Risk veto mode is not physical-only')
        training=package['risk_training']
        if training['cases']<4000 or training['epochs']<240 or training['horizon']<60:raise ValueError('Longer training recipe is incomplete')
        if not training.get('sampling','').startswith('Uniform observed case prior'):raise ValueError('Uncalibrated training run cannot be promoted')
        sources(training,['src/learned_risk.py','src/train_learned_risk.py'])
        old=load_package(ROOT/f'outputs/split_combat_v3/{algorithm}.pt',algorithm)['state_dict']
        for key,value in old.items():
            if algorithm=='ppo' and key.startswith('split_head.'):continue
            if not torch.equal(value,package['state_dict'][key]):raise ValueError(f'Movement weights changed: {algorithm}/{key}')
    outcome_path=directory/'outcomes.json';outcome=json.loads(outcome_path.read_text())
    if outcome['seed']!=protocol['final_encounter_seed'] or outcome['cases_requested']!=protocol['final_encounter_cases']:
        raise ValueError('Outcome report does not use the declared final seeds')
    sources(outcome,common+['src/train_learned_risk.py','src/evaluate_learned_risk.py'])
    summary=outcome_summary(outcome['rows'])
    for algorithm in ('ppo','diffusion'):
        if outcome['checkpoint_hashes'].get(algorithm)!=hashes[algorithm]:raise ValueError('Outcome checkpoint hash mismatch')
        if not summary[algorithm]['passed']:raise ValueError(f'{algorithm} failed outcome acceptance')
        rows=[r for r in outcome['rows'] if r['algorithm']==algorithm]
        if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Duplicated outcome cases')
        if any(not outcome['seed']<=r['seed']<outcome['seed']+outcome['cases_requested'] for r in rows):raise ValueError('Outcome seed outside declared range')
        config=load_package(models[algorithm],algorithm)['risk_training']
        if any(config['seed']<=r['seed']<config['seed']+config['cases'] for r in rows):raise ValueError('Outcome seeds overlap training')
    reports=[outcome_path]
    for algorithm in ('ppo','diffusion'):
        path=directory/f'{algorithm}_retention.json';report=json.loads(path.read_text());reports.append(path)
        sources(report,common+['src/evaluate_risk_league.py','envs/arena_config.py'])
        if report['mode']!='retention' or report['algorithm']!=algorithm or report['config']['max_steps']<1800:raise ValueError('Invalid full-match retention protocol')
        if report['models'][algorithm]['sha256']!=hashes[algorithm]:raise ValueError('Retention checkpoint hash mismatch')
        if report['baseline_hashes'][algorithm]!=digest(ROOT/f'outputs/split_combat_v3/{algorithm}.pt'):raise ValueError('Retention baseline changed')
        if len({r['seed'] for r in report['rows']})!=len(report['rows']):raise ValueError('Duplicate retention seeds')
        expected={protocol['retention_seed_start']+37*i for i in range(protocol['retention_matches'])}
        if {r['seed'] for r in report['rows']}!=expected:raise ValueError('Retention seeds differ from declared final test')
        if not retention_summary(report['rows'])['passed']:raise ValueError(f'{algorithm} failed navigation retention')
    league_path=directory/'league.json';league=json.loads(league_path.read_text());reports.append(league_path)
    sources(league,common+['src/evaluate_risk_league.py','envs/arena_config.py'])
    if league['mode']!='league' or len(league['rows'])<12 or league['config']['max_steps']<1800:raise ValueError('Absolute current-model league is incomplete')
    if {r['seed'] for r in league['rows']}!={protocol['final_league_seed_start']+37*i for i in range(protocol['final_league_matches'])}:raise ValueError('League seeds differ from declared final test')
    for name in ('ppo','diffusion','critic'):
        if league['models'][name]['sha256']!=hashes[name]:raise ValueError('League checkpoint hash mismatch')
    if any(p['physical_violations'] for r in league['rows'] for p in r['players'] if p['group']!='heuristic'):raise ValueError('Physical split violation')
    output=Path(output) if output else directory/'deployment.json'
    manifest=dict(schema_version=2,observation_version=2,accepted_on='2026-10-27',
        split_control='learned_risk_physical_only',models={a:dict(file=os.path.relpath(p.resolve(),output.parent.resolve()),sha256=hashes[a]) for a,p in models.items()},
        evaluations=[os.path.relpath(p.resolve(),output.parent.resolve()) for p in reports],evaluation_sha256={p.name:digest(p) for p in reports})
    temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(output)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/learned_risk_v4');args=p.parse_args()
    promote(args.directory);print(f'Accepted: {args.directory}/deployment.json')
