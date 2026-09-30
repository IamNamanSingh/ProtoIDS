"""
Hyper-parameter sweep for ProtoIDS on a cached X-IIoTID split.

Why this exists: preprocessing (parsing 355 MB of CSV, fitting the scaler on
known-train rows, writing memmaps) depends only on the dataset and the withheld
class set. Re-doing it for every trial wastes almost all the compute, so the
split is built once with create_data_loaders(..., dataset_type='xiiotid') and
every trial afterwards reuses it through create_loaders_from_cache().

Selection discipline (this is the part that keeps the results publishable):
  * the UNKNOWN threshold is calibrated on KNOWN VALIDATION rows only
  * configurations are ranked on VALIDATION metrics only
  * the TEST split is scored once, for the single selected configuration
Tuning on the test split - or selecting on "accuracy" alone - would invalidate
the open-set claims, which are the point of the project.

    python -m protoids.sweep --processed experiments/processed/<dir> \
        --epochs 10 --out experiments/results/sweep_xiiotid
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protoids.dataset import create_loaders_from_cache            # noqa: E402
from protoids.protoids_model import ProtoIDS                      # noqa: E402
from protoids.training import train_protoids                      # noqa: E402


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _distances(model, loader, device):
    """Nearest-prototype cosine distance and the argmin class for every row."""
    model.eval()
    dists, preds, labels = [], [], []
    protos = model.prototype_layer.prototypes
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        emb = model.encoder(xb)[1] if isinstance(model.encoder(xb), tuple) \
            else model.encoder(xb)
        d = torch.cdist(emb, protos)
        min_d, arg = d.min(dim=1)
        dists.append(min_d.cpu().numpy())
        preds.append(arg.cpu().numpy())
        labels.append(yb.numpy())
    return (np.concatenate(dists), np.concatenate(preds), np.concatenate(labels))


def _open_set_metrics(dist, pred, y, unknown_idx, threshold):
    """Metrics for the KNOWN/UNKNOWN decision at a frozen threshold."""
    is_unknown = (y == unknown_idx)
    flagged = dist > threshold

    tp = int(np.sum(flagged & is_unknown))          # unknown correctly rejected
    fp = int(np.sum(flagged & ~is_unknown))         # known wrongly rejected
    fn = int(np.sum(~flagged & is_unknown))         # unknown accepted as known

    detect = tp / max(tp + fn, 1)                   # unknown recall
    far_unknown = fn / max(tp + fn, 1)              # project "True FAR"
    far_known = fp / max(fp + int(np.sum(~flagged & ~is_unknown)), 1)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    known = ~is_unknown
    kn_pred = pred[known]
    kn_true = y[known]
    acc = float(np.mean(kn_pred == kn_true)) if known.any() else 0.0

    return {
        'known_accuracy': acc,
        'unknown_detection_rate': detect,
        'true_far_unknown_accepted': far_unknown,
        'open_set_far_known_rejected': far_known,
        'unknown_f1': f1,
        'threshold': float(threshold),
        'n_known': int(known.sum()),
        'n_unknown': int(is_unknown.sum()),
    }


def calibrate_threshold(dist, y, unknown_idx, percentile: float = 90.0) -> float:
    """
    Project convention: the 90th percentile of the nearest-prototype distance
    over KNOWN validation rows. Unknown validation rows are never used.
    """
    known = y != unknown_idx
    if not known.any():
        raise ValueError("no known validation rows to calibrate on")
    return float(np.percentile(dist[known], percentile))


def evaluate_config(cfg: Dict, processed_dir: str, device: torch.device,
                    cache: Dict, output_dir: Optional[str] = None,
                    verbose: bool = True) -> Dict:
    """Train one configuration and score it on validation (never on test)."""
    tr, va, te, num_classes, input_dim = cache['loaders']
    unknown_idx = cache['unknown_class_index']

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    model = ProtoIDS(
        input_dim=input_dim, num_classes=num_classes,
        embedding_dim=cfg['embedding_dim'],
        num_prototypes_per_class=cfg['num_prototypes'],
        dropout_rate=cfg['dropout_rate'],
        unknown_class_index=unknown_idx,
    ).to(device)

    # KMeans-style prototype init on the training embeddings.
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for xb, yb in tr:
            out = model.encoder(xb.to(device))
            e = out[1] if isinstance(out, tuple) else out
            embs.append(e.cpu()); labs.append(yb)
    model.prototype_layer.initialize_prototypes(torch.cat(embs), torch.cat(labs))

    t0 = time.time()
    history = train_protoids(
        model=model, train_loader=tr, val_loader=va, device=device,
        num_epochs=cfg['epochs'], learning_rate=cfg['learning_rate'],
        lambda_compact=cfg['lambda_compact'], class_weights=None, verbose=False,
        checkpoint_dir=os.path.join(output_dir, cfg['name']) if output_dir else None,
    )
    train_secs = time.time() - t0

    d_val, p_val, y_val = _distances(model, va, device)
    thr = calibrate_threshold(d_val, y_val, unknown_idx, cfg['threshold_percentile'])
    val_metrics = _open_set_metrics(d_val, p_val, y_val, unknown_idx, thr)

    try:
        val_metrics['auroc'] = float(roc_auc_score(
            (y_val == unknown_idx).astype(int), -d_val))
        val_metrics['aupr'] = float(average_precision_score(
            (y_val == unknown_idx).astype(int), -d_val))
    except ValueError:
        val_metrics['auroc'] = float('nan')
        val_metrics['aupr'] = float('nan')

    # Rank-threshold-free separation, useful as a tie-breaker.
    val_metrics['mean_known_dist'] = float(d_val[y_val != unknown_idx].mean())
    val_metrics['mean_unknown_dist'] = float(d_val[y_val == unknown_idx].mean())
    val_metrics['final_train_loss'] = float(history[-1].get('train_loss', float('nan')))

    if verbose:
        print(f"  known_acc={val_metrics['known_accuracy']:.4f}  "
              f"unk_detect={val_metrics['unknown_detection_rate']:.4f}  "
              f"unk_F1={val_metrics['unknown_f1']:.4f}  "
              f"AUROC={val_metrics['auroc']:.4f}  AUPR={val_metrics['aupr']:.4f}  "
              f"({train_secs:.0f}s)", flush=True)

    return {'name': cfg['name'], 'config': cfg, 'validation': val_metrics,
            'train_seconds': train_secs, '_model': model, '_threshold': thr}


def score(m: Dict, weights: Dict[str, float]) -> float:
    """
    Composite objective, computed on VALIDATION only.

    Accuracy alone is the wrong target for an open-set detector, so the default
    weights keep known-class accuracy high while rewarding the ability to
    reject unseen attacks. Weights are explicit so the trade-off is a stated
    decision rather than an accident of the search.
    """
    v = m['validation']
    parts = {
        'known_accuracy': v['known_accuracy'],
        'unknown_f1': v['unknown_f1'],
        'auroc': v['auroc'] if not np.isnan(v['auroc']) else 0.0,
    }
    return float(sum(weights.get(k, 0.0) * val for k, val in parts.items()))


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def build_grid(stage: str, seed: int = 42) -> List[Dict]:
    """Coarse first, then refine around the winner. Keeps the search small."""
    base = dict(embedding_dim=32, num_prototypes=3, dropout_rate=0.2,
                learning_rate=1e-3, lambda_compact=0.1, epochs=10,
                threshold_percentile=90.0, seed=seed)
    grid: List[Dict] = []

    if stage == 'coarse':
        for k in (1, 3, 5):
            for lam in (0.0, 0.1, 0.5):
                for emb in (32, 64):
                    grid.append({**base, 'name': f"k{k}_lam{lam}_e{emb}",
                                 'num_prototypes': k, 'lambda_compact': lam,
                                 'embedding_dim': emb})
    elif stage == 'lambda':
        for lam in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            grid.append({**base, 'name': f"lam{lam}",
                         'lambda_compact': lam})
    elif stage == 'k':
        for k in (1, 2, 3, 4, 5, 8, 10):
            grid.append({**base, 'name': f"k{k}", 'num_prototypes': k})
    elif stage == 'threshold':
        for p in (80, 85, 90, 95, 97, 99):
            grid.append({**base, 'name': f"thr{p}", 'threshold_percentile': float(p)})
    elif stage == 'optim':
        for lr in (5e-4, 1e-3, 3e-3):
            for drop in (0.1, 0.2, 0.35):
                for ep in (20, 40):
                    grid.append({**base, 'name': f"lr{lr}_do{drop}_ep{ep}",
                                 'learning_rate': lr, 'dropout_rate': drop,
                                 'epochs': ep})
    else:
        raise ValueError(f"unknown stage: {stage}")

    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', required=True, help='cached processed dir')
    ap.add_argument('--stage', default='coarse',
                    choices=['coarse', 'lambda', 'k', 'threshold', 'optim'])
    ap.add_argument('--out', default=None, help='where to write results JSON')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=None, help='cap the number of trials')
    ap.add_argument('--w-known', type=float, default=0.4)
    ap.add_argument('--w-unknown', type=float, default=0.4)
    ap.add_argument('--w-auroc', type=float, default=0.2)
    ap.add_argument('--report-test', action='store_true',
                    help='score the selected config on the test split (once)')
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = {'known_accuracy': args.w_known, 'unknown_f1': args.w_unknown,
               'auroc': args.w_auroc}

    tr, va, te, num_classes, input_dim = create_loaders_from_cache(
        args.processed, batch_size=256)
    with open(os.path.join(args.processed, 'split_meta.json')) as f:
        meta = json.load(f)

    print(f"cached split : {args.processed}")
    print(f"  classes    : {num_classes} ({len(meta['known_classes'])} known"
          f" + {'1 unknown' if meta['open_set'] else 'closed-set'})")
    print(f"  withheld   : {meta['withheld_classes']}")
    print(f"  features   : {input_dim}")
    print(f"  rows       : {meta['counts']}")
    print(f"device       : {device}")
    print(f"objective    : {weights}\n")

    cache = {'loaders': (tr, va, te), 'unknown_class_index': meta['unknown_class_index']}
    grid = build_grid(args.stage, args.seed)
    if args.limit:
        grid = grid[:args.limit]

    results = []
    for i, cfg in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {cfg['name']}", flush=True)
        try:
            r = evaluate_config(cfg, args.processed, device, cache, args.out)
        except RuntimeError as exc:      # OOM on the GPU: skip, keep going
            print(f"  SKIPPED ({exc})", flush=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        r['objective'] = score(r, weights)
        results.append(r)
        print(f"  objective = {r['objective']:.4f}\n", flush=True)

    results.sort(key=lambda r: -r['objective'])
    print("\n=== ranked (validation only) ===")
    header = f"{'config':<22}{'objective':>10}{'known_acc':>11}{'unk_det':>9}{'unk_F1':>9}{'AUROC':>8}{'AUPR':>8}"
    print(header)
    print('-' * len(header))
    for r in results:
        v = r['validation']
        print(f"{r['name']:<22}{r['objective']:>10.4f}{v['known_accuracy']:>11.4f}"
              f"{v['unknown_detection_rate']:>9.4f}{v['unknown_f1']:>9.4f}"
              f"{v['auroc']:>8.4f}{v['aupr']:>8.4f}")

    if not results:
        raise SystemExit("every trial failed")

    best = results[0]
    print(f"\nbest: {best['name']}  (objective {best['objective']:.4f})")

    if args.report_test:
        print("\n=== TEST (frozen threshold from validation, scored once) ===")
        d_te, p_te, y_te = _distances(best['_model'], te, device)
        test_metrics = _open_set_metrics(d_te, p_te, y_te,
                                         cache['unknown_class_index'],
                                         best['_threshold'])
        try:
            is_unk = (y_te == cache['unknown_class_index']).astype(int)
            test_metrics['auroc'] = float(roc_auc_score(is_unk, -d_te))
            test_metrics['aupr'] = float(average_precision_score(is_unk, -d_te))
        except ValueError:
            test_metrics['auroc'] = test_metrics['aupr'] = float('nan')
        for k, v in test_metrics.items():
            print(f"  {k:<32} {v}")
        best['test'] = test_metrics

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f'sweep_{args.stage}.json')
        with open(path, 'w') as f:
            json.dump({
                'stage': args.stage,
                'objective_weights': weights,
                'processed': args.processed,
                'split_meta': meta,
                'best_config': best['config'],
                'best_objective': best['objective'],
                'results': [{k: v for k, v in r.items()
                             if not k.startswith('_')} for r in results],
            }, f, indent=2)
        print(f"\nwrote {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
