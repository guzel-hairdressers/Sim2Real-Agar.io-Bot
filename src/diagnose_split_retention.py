"""Cheap split-capability diagnostic on consumed v8 encounter observations.

This is development classification only, never a fresh outcome evaluation.
It detects a degenerate all-hold head before spending unseen test seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.balanced_policies import load_policy
from src.benchmark_showcase import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT / 'outputs/learned_risk_v8/outcomes.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [r for r in json.loads(args.reference.read_text())['rows'] if r['algorithm'] == 'ppo']
    observations = torch.tensor(np.array([r['obs'] for r in rows]), dtype=torch.float32)
    policy = load_policy('ppo', args.candidate)
    with torch.inference_mode():
        scores = policy.net(observations)[2].sigmoid().numpy()
    truth = np.array([r['label'] for r in rows], dtype=bool)
    proposed = (scores > .65) & np.array([r['physically_possible'] for r in rows], dtype=bool)
    outside = ~np.array([r['old_guard_allows'] for r in rows], dtype=bool)
    report = dict(purpose='Consumed v8 observations; development classification, not unseen outcomes',
                  checkpoint_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  reference_report_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  cases=len(rows), known_positive_opportunities=int(truth.sum()),
                  initial_proposals=int(proposed.sum()),
                  initial_precision=float((truth & proposed).sum() / max(1, proposed.sum())),
                  initial_recall=float((truth & proposed).sum() / max(1, truth.sum())),
                  useful_initial_splits_outside_old_guard=int((truth & proposed & outside).sum()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
