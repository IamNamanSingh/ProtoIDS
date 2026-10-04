#!/usr/bin/env bash
# Phase 2: run the SAME hyper-parameter grid under every open-set protocol, so a
# configuration can be selected for robustness rather than for its score on the
# one protocol the search was originally fitted to.
#
# The grid is byte-identical across protocols on purpose: select_robust.py needs
# a shared config space to rank across protocols.
set -euo pipefail
cd /mnt/Storage/ProtoIDS
PY=.venv/bin/python
LOG=/tmp/opencode/xval
mkdir -p "$LOG"

for tag in s1 s2 fam; do
  proc="experiments/processed/xiiotid_${tag}"
  out="experiments/results/sweep_xiiotid_${tag}"
  [ -d "$proc" ] || { echo "missing $proc - run build_protocol_splits.sh"; continue; }
  done_marker="$out/sweep_coarse.json"
  if [ -f "$done_marker" ]; then
    echo "=== $tag: sweep already done ==="; continue
  fi
  echo "=== sweeping $tag ==="
  PYTHONPATH=src $PY -u -m protoids.sweep \
    --processed "$proc" \
    --stage coarse \
    --out "$out" \
    --device cuda --seed 42 \
    > "$LOG/sweep_${tag}.log" 2>&1 || { echo "  FAILED (see $LOG/sweep_${tag}.log)"; continue; }
  $PY - "$tag" <<'PYEOF'
import json, sys
tag = sys.argv[1]
d = json.load(open(f"experiments/results/sweep_xiiotid_{tag}/sweep_coarse.json"))
best = d["best_config"]
print(f"  {tag}: best on THIS protocol only -> k{best['num_prototypes']} "
      f"e{best['embedding_dim']} lam{best['lambda_compact']} (obj {d['best_objective']:.4f})")
PYEOF
done

echo
echo "=== protocol-robust selection ==="
PYTHONPATH=src $PY -u -m protoids.select_robust \
  --sweep "experiments/results/sweep_xiiotid_s1/sweep_coarse.json" \
  --sweep "experiments/results/sweep_xiiotid_s2/sweep_coarse.json" \
  --sweep "experiments/results/sweep_xiiotid_fam/sweep_coarse.json" \
  --out experiments/results/robust_selection.json
