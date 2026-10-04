"""
Protocol-robust configuration selection for ProtoIDS on X-IIoTID.

Motivation
----------
A hyper-parameter sweep run inside a single open-set protocol selects a
configuration that is best *for that protocol*. Cross-validation then showed
the K=1/dim-64 config picked under protocol S1 is actually worse than K=3 under
protocol S2. So the search itself was protocol-dependent, and no configuration
had yet been shown to be robust.

This module answers the different question: which configuration is best
*across* protocols? It reads the per-protocol sweep results and ranks configs
by how consistently they do well, not by their score on any single protocol.

Ranking criteria, in order of preference:

  mean_rank    average rank across protocols - the main criterion. Rewards a
               config that is consistently near the top, and is immune to a
              single protocol being harder than the others.
  worst_rank   the worst rank a config achieves anywhere. Guards against
               configs that are excellent on two protocols and useless on the
               third.
  mean_aupr    mean AUPR, the most stable single metric across protocols.
  mean_auroc   mean AUROC. Reported, but not the selection criterion, because
               its spread is dominated by which classes are withheld.

Usage
-----
    python -m protoids.select_robust \
        --sweep experiments/processed/s1_k1 sweep_s1.json \
        --sweep experiments/processed/s2_k1 sweep_s2.json \
        --sweep experiments/processed/fam_k1 sweep_fam.json \
        --out experiments/results/robust_selection.json
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def load_sweeps(specs: List[str]) -> Dict[str, dict]:
    """
    Load one or more sweep JSONs, each produced by protoids.sweep.

    Each sweep's own split_meta tells us the withheld classes, so protocols
    are keyed by that rather than by whatever the caller named the file.
    """
    out = {}
    for spec in specs:
        path, _, label = spec.partition("=")
        if not os.path.isfile(path):
            raise SystemExit(f"sweep file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        meta = data.get("split_meta", {})
        withheld = tuple(sorted(meta.get("withheld_classes", [])))
        key = label or ("+".join(w[:12] for w in withheld) or "closed")
        data["_withheld"] = list(withheld)
        data["_label"] = key
        out[key] = data
    return out


def config_key(cfg: dict) -> Tuple:
    """The hyper-parameters that define a configuration (not the run name)."""
    return (cfg["num_prototypes"], cfg["embedding_dim"],
            round(float(cfg["lambda_compact"]), 6),
            round(float(cfg["learning_rate"]), 8),
            round(float(cfg["dropout_rate"]), 6),
            round(float(cfg["threshold_percentile"]), 3),
            int(cfg["epochs"]))


def short(key: Tuple) -> str:
    k, emb, lam, lr, drop, pct, ep = key
    return f"k{k}_e{emb}_lam{lam}_lr{lr}_do{drop}_p{pct}_ep{ep}"


def build_matrix(sweeps: Dict[str, dict]) -> Tuple[Dict[Tuple, Dict[str, dict]], List[str]]:
    """
    Assemble configs x protocols -> validation metrics.

    Only configurations present in EVERY protocol are kept, because a config
    missing from one protocol cannot be ranked across all of them.
    """
    by_cfg: Dict[Tuple, Dict[str, dict]] = defaultdict(dict)
    for proto, data in sweeps.items():
        for row in data["results"]:
            key = config_key(row["config"])
            by_cfg[key][proto] = row["validation"]
    complete = {k: v for k, v in by_cfg.items() if len(v) == len(sweeps)}
    dropped = len(by_cfg) - len(complete)
    return complete, sorted(sweeps)


def rank_configs(metrics: Dict[Tuple, Dict[str, dict]],
                 protocols: List[str]) -> List[dict]:
    """
    Rank by mean rank across protocols (lower is better), tie-broken by the
    worst rank and then by mean AUPR.
    """
    per_proto_order: Dict[str, List[Tuple]] = {}
    for p in protocols:
        rows = [(cfg, metrics[cfg][p]) for cfg in metrics]
        rows.sort(key=lambda kv: -kv[1].get("aupr", float("nan")))
        per_proto_order[p] = [cfg for cfg, _ in rows]

    out = []
    for cfg, per_p in metrics.items():
        ranks = [per_proto_order[p].index(cfg) + 1 for p in protocols]
        out.append({
            "config": short(cfg),
            "num_prototypes": cfg[0],
            "embedding_dim": cfg[1],
            "lambda_compact": cfg[2],
            "learning_rate": cfg[3],
            "dropout_rate": cfg[4],
            "threshold_percentile": cfg[5],
            "epochs": cfg[6],
            "ranks": ranks,
            "mean_rank": float(np.mean(ranks)),
            "worst_rank": int(max(ranks)),
            "mean_aupr": float(np.mean([per_p[p].get("aupr", float("nan"))
                                        for p in protocols])),
            "mean_auroc": float(np.mean([per_p[p].get("auroc", float("nan"))
                                         for p in protocols])),
            "mean_unknown_f1": float(np.mean([per_p[p].get("unknown_f1", float("nan"))
                                              for p in protocols])),
            "mean_known_acc": float(np.mean([per_p[p].get("known_accuracy", float("nan"))
                                             for p in protocols])),
            "per_protocol": {p: {k: per_p[p].get(k) for k in
                                 ("aupr", "auroc", "unknown_f1", "known_accuracy",
                                  "unknown_detection_rate")}
                             for p in protocols},
        })
    out.sort(key=lambda r: (r["mean_rank"], r["worst_rank"], -r["mean_aupr"]))
    for i, r in enumerate(out, 1):
        r["robust_rank"] = i
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", required=True,
                    help="path=label (label optional); repeat once per protocol")
    ap.add_argument("--out", default=None, help="write the ranking here")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    sweeps = load_sweeps(args.sweep)
    print(f"protocols: {len(sweeps)}")
    for label, data in sweeps.items():
        print(f"  {label:<44} withheld={data['_withheld']}")

    metrics, protocols = build_matrix(sweeps)
    if not metrics:
        raise SystemExit("no configuration appears in every protocol; "
                         "re-run the sweep with an identical grid for each")
    print(f"\nconfigurations present in all {len(protocols)} protocols: {len(metrics)}")

    ranked = rank_configs(metrics, protocols)

    print(f"\n{'#':>3} {'config':<34}{'meanR':>7}{'worstR':>7}{'AUPR':>8}{'AUROC':>8}{'unkF1':>8}{'known':>8}  ranks")
    print("-" * 100)
    for r in ranked[:args.top]:
        print(f"{r['robust_rank']:>3} {r['config']:<34}{r['mean_rank']:>7.2f}"
              f"{r['worst_rank']:>7}{r['mean_aupr']:>8.4f}{r['mean_auroc']:>8.4f}"
              f"{r['mean_unknown_f1']:>8.4f}{r['mean_known_acc']:>8.4f}  {r['ranks']}")

    if ranked:
        best = ranked[0]
        print(f"\nmost protocol-robust configuration: {best['config']}")
        print(f"  mean rank {best['mean_rank']:.2f} (worst {best['worst_rank']})")
        print(f"  mean AUPR {best['mean_aupr']:.4f}  mean AUROC {best['mean_auroc']:.4f}")
        for p in protocols:
            m = best["per_protocol"][p]
            print(f"    {p:<44} AUPR={m['aupr']:.4f} AUROC={m['auroc']:.4f} "
                  f"unkF1={m['unknown_f1']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"protocols": protocols, "ranking": ranked}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
