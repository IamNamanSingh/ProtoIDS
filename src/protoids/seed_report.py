"""
Aggregate the seed x protocol runs into a variance report.

The question this answers: the AUROC spread across open-set protocols is
~17 points. If seed-to-seed noise is a meaningful fraction of that, then the
protocol effect we reported is overstated; if it is small, the protocol
effect is real. The two sources of variation are compared directly, using the
same configurations throughout so they are not confounded.

    python -m protoids.seed_report --results experiments/results \
        --out experiments/results/seed_report.json

Variance decomposition, per configuration:
    protocol_sd   spread of the per-protocol mean, across protocols
    seed_sd       standard deviation across seeds, within a protocol
                  (pooled over protocols)
Both are computed on AUROC and AUPR, because the two behave very differently
here: AUROC is dominated by protocol choice, AUPR much less so.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

PAT = re.compile(r"^seed_(?P<proto>[a-z0-9]+)_(?P<tag>k\d+)_(?P<k>\d+)_e(?P<e>\d+)_s(?P<seed>\d+)$")


def collect(results_dir: str) -> List[dict]:
    rows = []
    for name in sorted(os.listdir(results_dir)):
        m = PAT.match(name)
        if not m:
            continue
        f = os.path.join(results_dir, name, "open_set_test_results.json")
        if not os.path.isfile(f):
            continue
        with open(f) as fh:
            t = json.load(fh)
        rows.append({
            "experiment": name,
            "protocol": m["proto"],
            "seed": int(m["seed"]),
            "num_prototypes": int(m["k"]),
            "embedding_dim": int(m["e"]),
            "auroc": t.get("auc_roc", t.get("auroc")),
            "aupr": t.get("auc_pr", t.get("aupr")),
            "unknown_f1": t.get("unknown_f1"),
            "known_accuracy": t.get("known_accuracy"),
            "threshold": t.get("threshold"),
        })
    return rows


def analyse(rows: List[dict]) -> dict:
    by_cfg: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[(r["num_prototypes"], r["embedding_dim"])].append(r)

    out = {"configurations": [], "protocols": sorted({r["protocol"] for r in rows}),
           "seeds": sorted({r["seed"] for r in rows}), "n_runs": len(rows)}

    for (k, e), rs in sorted(by_cfg.items()):
        cell = {"num_prototypes": k, "embedding_dim": e, "n_runs": len(rs),
                "by_protocol": {}}
        for metric in ("auroc", "aupr", "unknown_f1", "known_accuracy"):
            vals = [r[metric] for r in rs if r[metric] is not None]
            if not vals:
                continue
            cell[f"{metric}_mean"] = float(np.mean(vals))
            cell[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cell[f"{metric}_min"] = float(np.min(vals))
            cell[f"{metric}_max"] = float(np.max(vals))

        per_proto_mean = {}
        for p in out["protocols"]:
            v = [r["auroc"] for r in rs if r["protocol"] == p and r["auroc"] is not None]
            if v:
                per_proto_mean[p] = float(np.mean(v))
        cell["auroc_per_protocol_mean"] = per_proto_mean
        pm = list(per_proto_mean.values())
        if len(pm) > 1:
            cell["auroc_protocol_sd"] = float(np.std(pm, ddof=1))
            cell["auroc_protocol_range"] = float(max(pm) - min(pm))

        resid = []
        for p in out["protocols"]:
            v = [r["auroc"] for r in rs if r["protocol"] == p and r["auroc"] is not None]
            if len(v) > 1:
                m = float(np.mean(v))
                resid.extend(x - m for x in v)
        if len(resid) > 1:
            cell["auroc_seed_sd"] = float(np.std(resid, ddof=1))
        elif resid:
            cell["auroc_seed_sd"] = 0.0

        if "auroc_protocol_sd" in cell and cell.get("auroc_seed_sd") is not None:
            ratio = cell["auroc_seed_sd"] / cell["auroc_protocol_sd"] if cell["auroc_protocol_sd"] else float("inf")
            cell["seed_to_protocol_ratio"] = float(ratio)
        cell["by_protocol"] = {
            p: {"auroc": [r["auroc"] for r in rs if r["protocol"] == p],
                "aupr": [r["aupr"] for r in rs if r["protocol"] == p]}
            for p in out["protocols"]
        }
        out["configurations"].append(cell)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = collect(args.results)
    if not rows:
        raise SystemExit(f"no seed_* experiments found in {args.results}. "
                         f"Run scripts/seed_sensitivity.sh first.")
    rep = analyse(rows)

    print(f"runs: {rep['n_runs']}   protocols: {rep['protocols']}   seeds: {rep['seeds']}\n")
    print(f"{'cfg':<10}{'n':>4}{'AUROC mean':>12}{'AUROC sd':>10}{'proto sd':>10}"
          f"{'seed sd':>10}{'ratio':>8}{'AUPR mean':>11}")
    print("-" * 85)
    for c in rep["configurations"]:
        cfg = f"k{c['num_prototypes']}_e{c['embedding_dim']}"
        print(f"{cfg:<10}{c['n_runs']:>4}{c.get('auroc_mean',0):>12.4f}"
              f"{c.get('auroc_sd',0):>10.4f}{c.get('auroc_protocol_sd',0):>10.4f}"
              f"{c.get('auroc_seed_sd',0):>10.4f}"
              f"{c.get('seed_to_protocol_ratio',float('nan')):>8.2f}"
              f"{c.get('aupr_mean',0):>11.4f}")

    for c in rep["configurations"]:
        cfg = f"k{c['num_prototypes']}_e{c['embedding_dim']}"
        if "auroc_protocol_sd" not in c:
            continue
        ps = c.get("auroc_seed_sd", 0.0)
        pr = c["auroc_protocol_sd"]
        print(f"\n{cfg}: per-protocol AUROC means")
        for p, v in c["auroc_per_protocol_mean"].items():
            vals = c["by_protocol"][p]["auroc"]
            spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
            print(f"  {p:<6} {v:.4f}" + (f"   (seed spread {spread:.4f})" if len(vals) > 1 else ""))
        print(f"  protocol sd {pr:.4f}  vs  seed sd {ps:.4f}  ->  "
              f"{'protocol effect dominates' if pr > ps else 'seed noise is comparable'}")
    print("\nNote: the per-protocol test-set values above already vary by protocol "
          "design.\nSeed sd is the within-protocol component after removing each "
          "protocol mean.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"report": rep, "runs": rows}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
