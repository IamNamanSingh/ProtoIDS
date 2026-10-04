#!/usr/bin/env python
"""
Noise probe: does averaging 3 seeds make the config comparison reliable?

The sweep is not bitwise reproducible on this GPU stack, so run-to-run AUROC
scatter (sd 0.068 at a FIXED seed) is the floor on what can be measured. Two
things follow, and this probe settles both before the full grid is spent:

  1. How large is the noise floor, really?
  2. If a config is averaged over S seeds, how much does the standard error of
     its mean shrink? If SE is still comparable to the between-config gaps we
     care about (0.01-0.05 AUROC), then 3 seeds is not enough and the full grid
     needs more - which is worth knowing now rather than after 84 minutes.

Runs one configuration across seeds on one cached protocol split. Determinism is
enforced but is best-effort, so the spread measured here IS the residual noise.

    PYTHONPATH=src python -m protoids.sweep \
        --processed experiments/processed/xiiotid_s1 \
        --stage k --limit 1 --out experiments/results/noise_probe \
        --device cuda --seed 42
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protoids.dataset import create_loaders_from_cache          # noqa: E402
from protoids.sweep import (GpuResidentLoader, enforce_determinism,  # noqa: E402
                            evaluate_config, score)

CONFIG = {
    'name': 'probe_k1_lam0.1_e64',
    'num_prototypes': 1,
    'embedding_dim': 64,
    'lambda_compact': 0.1,
    'dropout_rate': 0.2,
    'learning_rate': 1e-3,
    'epochs': 10,
    'threshold_percentile': 90.0,
}
WEIGHTS = {'known_accuracy': 0.4, 'unknown_f1': 0.4, 'auroc': 0.2}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', default='experiments/processed/xiiotid_s1')
    ap.add_argument('--seeds', type=int, nargs='*', default=[42, 7, 1234])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='experiments/results/noise_probe')
    args = ap.parse_args()

    tr, va, te, num_classes, input_dim = create_loaders_from_cache(
        args.processed, batch_size=256)
    with open(os.path.join(args.processed, 'split_meta.json')) as f:
        meta = json.load(f)

    device = torch.device(args.device)
    loaders = (GpuResidentLoader(tr, device, 42),
               GpuResidentLoader(va, device, 42),
               GpuResidentLoader(te, device, 42))
    cache = {'loaders': loaders, 'num_classes': num_classes,
             'input_dim': input_dim,
             'unknown_class_index': meta['unknown_class_index']}

    print(f"protocol : {meta['withheld_classes']}")
    print(f"config   : K={CONFIG['num_prototypes']} dim={CONFIG['embedding_dim']} "
          f"lambda={CONFIG['lambda_compact']} epochs={CONFIG['epochs']}")
    print(f"seeds    : {args.seeds}\n")

    rows: List[Dict] = []
    for seed in args.seeds:
        cfg = {**CONFIG, 'seed': seed}
        print(f"[seed {seed}]", flush=True)
        r = evaluate_config(cfg, args.processed, device, cache, None, verbose=False)
        r['objective'] = score(r, WEIGHTS)
        rows.append(r)
        v = r['validation']
        print(f"  known={v['known_accuracy']:.4f}  unkF1={v['unknown_f1']:.4f}  "
              f"AUROC={v['auroc']:.4f}  AUPR={v['aupr']:.4f}  "
              f"objective={r['objective']:.4f}  ({r['train_seconds']:.0f}s)\n", flush=True)

    def stats(key):
        vals = np.array([r['validation'][key] for r in rows], dtype=float)
        n = len(vals)
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        return vals, sd, se

    print(f"{'metric':<12}{'mean':>9}{'sd':>9}{'SE':>9}{'95% CI':>18}{'min':>9}{'max':>9}")
    print('-' * 75)
    summary = {}
    for metric in ('auroc', 'aupr', 'unknown_f1', 'known_accuracy'):
        vals, sd, se = stats(metric)
        half = 1.96 * se if len(vals) > 1 else 0.0
        summary[metric] = {'mean': float(vals.mean()), 'sd': sd, 'se': se,
                            'values': [float(v) for v in vals]}
        print(f"{metric:<12}{vals.mean():>9.4f}{sd:>9.4f}{se:>9.4f}"
              f"{f'+-{half:.4f}':>18}{vals.min():>9.4f}{vals.max():>9.4f}")

    obj = np.array([r['objective'] for r in rows], dtype=float)
    obj_sd = float(np.std(obj, ddof=1)) if len(obj) > 1 else 0.0
    obj_se = obj_sd / np.sqrt(len(obj)) if len(obj) > 1 else 0.0
    print(f"{'objective':<12}{obj.mean():>9.4f}{obj_sd:>9.4f}{obj_se:>9.4f}"
          f"{f'+-{1.96 * obj_se:.4f}':>18}{obj.min():>9.4f}{obj.max():>9.4f}")

    n = len(args.seeds)
    sd = summary['auroc']['sd']
    print(f"\nnoise floor (AUROC sd at a fixed protocol+config): {sd:.4f}")
    print(f"with {n} seeds, SE of the mean = {sd / np.sqrt(n):.4f}, "
          f"95% CI half-width = {1.96 * sd / np.sqrt(n):.4f}")

    gaps = [0.01, 0.02, 0.03, 0.05]
    print(f"\n{'target gap':>12}{'seeds needed':>14}{'verdict at n=' + str(n):>18}")
    print('-' * 48)
    for g in gaps:
        need = int(np.ceil((1.96 * sd / g) ** 2))
        se_now = sd / np.sqrt(n)
        verdict = "resolvable" if 1.96 * se_now < g else "NOT resolvable"
        print(f"{g:>12.2f}{need:>14}{verdict:>18}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'noise_probe.json')
    with open(path, 'w') as f:
        json.dump({'config': CONFIG, 'seeds': args.seeds,
                   'protocol': meta['withheld_classes'],
                   'summary': summary,
                   'objective': {'mean': float(obj.mean()), 'sd': obj_sd, 'se': obj_se},
                   'runs': [{'seed': r['config']['seed'],
                             'validation': r['validation'],
                             'objective': r['objective']} for r in rows]}, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
