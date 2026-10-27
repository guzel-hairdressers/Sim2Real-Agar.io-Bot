"""Warm-start neural movement from an assisted controller's demonstrations.

Assistance supplies training labels only. The exported policy has no runtime
steering overrides and keeps the learned physical-only split head unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.arena_policies import load_package
from src.arena_tactics import safety_filter
from src.benchmark_showcase import ROOT
from src.train_champion_ppo import HybridActorCritic


def labels(obs, teacher):
    targets = []
    weights = []
    for batch in torch.as_tensor(obs, dtype=torch.float32).split(2048):
        with torch.inference_mode():
            raw = teacher(batch)[0].numpy()
        for observation, continuous in zip(batch.numpy(), raw):
            action = safety_filter([*continuous, 0], observation)[:2]
            targets.append(action)
            weights.append(4. if np.max(np.abs(action - continuous)) > 1e-5 else 1.)
    return torch.tensor(np.array(targets)), torch.tensor(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT / 'outputs/balanced_v9/training/ppo_120832.pt')
    parser.add_argument('--teacher', type=Path, default=ROOT / 'outputs/split_combat_v3/ppo.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/arena_v2/demonstrations.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(6100001)
    package = load_package(args.base, 'ppo')
    net = HybridActorCritic(70, split_architecture='obs_mlp')
    net.load_state_dict(package['state_dict'])
    teacher = HybridActorCritic(70, split_architecture='obs_mlp')
    teacher.load_state_dict(load_package(args.teacher, 'ppo')['state_dict'])
    teacher.eval()
    data = np.load(args.data)
    x = torch.tensor(data['obs'], dtype=torch.float32)
    vx = torch.tensor(data['val_obs'], dtype=torch.float32)
    y, weights = labels(data['obs'], teacher)
    vy, vw = labels(data['val_obs'], teacher)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    parameters = list(net.actor_trunk.parameters()) + list(net.cont_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    best = float('inf')
    best_state = None
    history = []
    for epoch in range(args.epochs):
        net.train()
        for ids in torch.randperm(len(x)).split(512):
            mu = net(x[ids])[0]
            loss = ((mu - y[ids]).square().mean(1) * weights[ids]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            net.eval()
            with torch.inference_mode():
                prediction = torch.cat([net(b)[0] for b in vx.split(2048)])
                mse = float(((prediction - vy).square().mean(1) * vw).mean())
            row = dict(epoch=epoch + 1, weighted_heldout_mse=mse)
            history.append(row)
            print(json.dumps(row), flush=True)
            if mse < best:
                best = mse
                best_state = copy.deepcopy(net.state_dict())
    if best_state is None:
        raise ValueError('At least five epochs required')
    package['state_dict'] = best_state
    package['movement_control'] = 'learned'
    package['split_control'] = 'learned_full_control_physical_only'
    sources = ['src/distill_balanced_movement.py', 'src/arena_tactics.py',
               'src/train_champion_ppo.py', 'src/balanced_policies.py']
    config = dict(seed=6100001, teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                  base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  data_manifest=json.loads((args.data.parent / 'dataset_manifest.json').read_text()),
                  training_rows=len(x), heldout_rows=len(vx), epochs=args.epochs,
                  labels='v3 continuous neural means with original movement assistance; split label excluded',
                  split_head_frozen=True, runtime_movement_overrides=False,
                  source_hashes={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                  history=history, best_weighted_heldout_mse=best)
    package['movement_distillation'] = config
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output / 'ppo.pt')
    (args.output / 'distillation.json').write_text(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
