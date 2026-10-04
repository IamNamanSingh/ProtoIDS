#!/usr/bin/env python
"""
Multi-seed, multi-protocol measurement with proper error bars.

The noise probe showed that a single run of a single config varies by
AUROC sd 0.060, so config gaps of 0.01-0.05 are unmeasurable and the original
18-config grid ranking was noise. This measures a small set of configs that
actually differ, over several seeds and all three open-set protocols, and
reports means with confidence intervals so the comparison is defensible.

Six configs, chosen because they span the dimensions that plausibly matter
rather than because they won a search:

    K in {1, 3}  x  dim in {32, 64}

    python -m protoids.multiseed --processed s1=... s2=... fam=... \
        --seeds 42 7 1234 5 99 2024 --out experiments/results/multiseed

The verdict for each config is based on the 95% CI of the paired per-protocol
difference against the incumbent (K=3, dim 32), because the two share seeds
and protocols and a paired test removes most of the shared variance.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protoids.dataset import create_loaders_from_cache            # noqa: E402
from protoids.sweep import (GpuResidentLoader, enforce_determinism,  # noqa: E402
                            evaluate_config, score)

WEIGHTS = {'known_accuracy': 0.4, 'unknown_f1': 0.4, 'auroc': 0.2}
METRICS = ('auroc', 'aupr', 'unknown_f1', 'known_accuracy')
GRID = [(k, e) for k in (1, 3) for e in (32, 64)]
INCUMBENT = (3, 32)


def ci95(sd: float, n: int) -> float:
    if n < 2:
        return float('nan')
    return 1.96 * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', action='append', required=True,
                    help='tag=path to a cached split dir; repeat per protocol')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 1234, 5, 99, 2024])
    ap.add_argument('--configs', type=int, nargs='*', default=None,
                    help='K values to test (dim 32 and 64 each); default 1 3')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='experiments/results/multiseed')
    ap.add_argument('--resume', action='store_true',
                    help='reuse finished (protocol,config,seed) cells')
    args = ap.parse_args()

    protocols: Dict[str, str] = {}
    for spec in args.processed:
        tag, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f"--processed needs tag=path, got {spec!r}")
        if not os.path.isfile(os.path.join(path, 'split_meta.json')):
            raise SystemExit(f"no split_meta.json in {path}")
        protocols[tag] = path
    if not protocols:
        raise SystemExit("no protocols given")

    ks = args.configs or [1, 3]
    grid = [(k, e) for k in ks for e in (32, 64)]
    seeds = args.seeds
    total = len(protocols) * len(grid) * len(seeds)
    print(f"protocols {len(protocols)} x configs {len(grid)} x seeds {len(seeds)} "
          f"= {total} runs")
    print(f"seeds: {seeds}")
    print(f"configs: {grid}\n")

    os.makedirs(args.out, exist_ok=True)
    cells_path = os.path.join(args.out, 'cells.json')
    cells: Dict[str, dict] = {}
    if args.resume and os.path.isfile(cells_path):
        with open(cells_path) as f:
            cells = json.load(f)
        print(f"resuming: {len(cells)} cells already done\n")

    device = torch.device(args.device)
    results: List[dict] = []

    for tag, path in protocols.items():
        tr, va, te, num_classes, input_dim = create_loaders_from_cache(
            path, batch_size=args.batch_size)
        with open(os.path.join(path, 'split_meta.json')) as f:
            meta = json.load(f)
        loaders = (GpuResidentLoader(tr, device, 42),
                   GpuResidentLoader(va, device, 42),
                   GpuResidentLoader(te, device, 42))
        cache = {'loaders': loaders, 'num_classes': num_classes,
                 'input_dim': input_dim,
                 'unknown_class_index': meta['unknown_class_index']}
        print(f"=== {tag}: withheld={meta['withheld_classes']} dim={input_dim} "
              f"known={len(meta['known_classes'])} ===")

        for (k, e), seed in itertools.product(grid, seeds):
            key = f"{tag}|k{k}_e{e}|s{seed}"
            if key in cells:
                results.append(cells[key])
                print(f"  {key:<26} cached")
                continue
            cfg = {'name': f"k{k}_e{e}", 'num_prototypes': k, 'embedding_dim': e,
                   'lambda_compact': 0.1, 'dropout_rate': 0.2, 'learning_rate': 1e-3,
                   'epochs': args.epochs, 'threshold_percentile': 90.0, 'seed': seed}
            t0 = time.time()
            r = evaluate_config(cfg, path, device, cache, None, verbose=False)
            r['objective'] = score(r, WEIGHTS)
            rec = {'protocol': tag, 'k': k, 'dim': e, 'seed': seed,
                   'objective': r['objective'],
                   **{m: r['validation'][m] for m in METRICS},
                   'seconds': time.time() - t0}
            cells[key] = rec
            results.append(rec)
            print(f"  k{k}_e{e} s{seed:<5} "
                  + "  ".join(f"{m}={rec[m]:.4f}" for m in METRICS)
                  + f"  ({rec['seconds']:.0f}s)", flush=True)
            with open(cells_path, 'w') as f:
                json.dump(cells, f, indent=2)

        del loaders, cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    summarise(results, grid, seeds, protocols, args.out)
    return 0


def summarise(results: List[dict], grid, seeds, protocols, out_dir):
    by: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for r in results:
        by[(r['k'], r['dim'])].append(r)

    n = len(seeds)
    print(f"\n{'config':<10}{'n':>4}" + "".join(f"{m:>22}" for m in METRICS))
    print(f"{'':<10}{'':>4}" + "".join(f"{'mean':>12}{'sd':>10}" for _ in METRICS))
    print("-" * 10 + "---" + "-" * (22 * len(METRICS)))
    table = {}
    for cfg in sorted(by):
        rs = by[cfg]
        line = f"k{cfg[0]}_e{cfg[1]:<6}"
        stats = {'n': len(rs)}
        for m in METRICS:
            vals = np.array([x[m] for x in rs], dtype=float)
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            stats[m] = {'mean': float(vals.mean()), 'sd': sd,
                        'ci95': ci95(sd, len(vals))}
            line += f"{vals.mean():>12.4f}{sd:>10.4f}"
        table[f"k{cfg[0]}_e{cfg[1]}"] = stats
        print(line)

    def paired(metric):
        base = {(r['protocol'], r['seed']): r[metric] for r in by[INCUMBENT]}
        out = {}
        for cfg in sorted(by):
            if cfg == INCUMBENT:
                continue
            d = [r[metric] - base[(r['protocol'], r['seed'])]
                 for r in by[cfg] if (r['protocol'], r['seed']) in base]
            if not d:
                continue
            d = np.array(d, dtype=float)
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            out[cfg] = {'mean_diff': float(d.mean()), 'sd_diff': sd,
                        'ci95_diff': ci95(sd, len(d)), 'n': len(d)}
        return out

    print(f"\npaired differences vs incumbent k{INCUMBENT[0]}_e{INCUMBENT[1]} "
          f"(same protocol+seed, so shared variance cancels)")
    for metric in ('auroc', 'aupr', 'unknown_f1'):
        pd = paired(metric)
        if not pd:
            continue
        print(f"\n  {metric}:")
        print(f"    {'config':<12}{'mean diff':>12}{'sd':>9}{'95% CI':>10}  verdict")
        for cfg, v in sorted(pd.items(), key=lambda kv: -kv[1]['mean_diff']):
            md, ci = v['mean_diff'], v['ci95_diff']
            if md - ci > 0:
                verdict = 'better'
            elif md + ci < 0:
                verdict = 'worse'
            else:
                verdict = 'indistinguishable'
            ci_txt = f'+-{ci:.4f}'
            print(f"    k{cfg[0]}_e{cfg[1]:<7}{md:>12.4f}{v['sd_diff']:>9.4f}"
                  f"{ci_txt:>10}  {verdict}")
        table.setdefault('paired', {})[metric] = {f"k{c[0]}_e{c[1]}": v for c, v in pd.items()}

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'seeds': seeds, 'n_seeds': n, 'protocols': list(protocols),
                   'incumbent': f"k{INCUMBENT[0]}_e{INCUMBENT[1]}",
                   'table': table, 'runs': results}, f, indent=2)
    print(f"\nwrote {os.path.join(out_dir, 'summary.json')}")


if __name__ == '__main__':
    raise SystemExit(main())
