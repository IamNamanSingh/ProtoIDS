#!/usr/bin/env bash
# Phase 3: seed sensitivity.
#
# Everything reported so far is a single run at seed 42. With AUROC spanning
# ~17 points across protocols, an unquantified seed variance would be
# indistinguishable from a protocol effect, so this measures it directly.
#
# Two things are varied:
#   1. seed alone (42 / 7 / 1234) on the same protocol - isolates run-to-run noise
#   2. protocol x seed - lets the seed variance be compared against the protocol
#      variance, which is the number that actually matters for the writeup
set -euo pipefail
cd /mnt/Storage/ProtoIDS
PY=.venv/bin/python
LOG=/tmp/opencode/xval
mkdir -p "$LOG"
SEEDS="${SEEDS:-42 7 1234}"

# Config: the project default (K=3, dim 32) and the swept best (K=1, dim 64).
# Both are measured, because the K=1 vs K=3 question is protocol-dependent and
# may also be seed-dependent.
for tag in s1 s2 fam; do
  proc="experiments/processed/xiiotid_${tag}"
  [ -d "$proc" ] || { echo "missing $proc"; continue; }
  [ -f "$proc/split_meta.json" ] || { echo "missing $proc/split_meta.json"; continue; }

  WITHHELD=$($PY -c "
import json
m=json.load(open('$proc/split_meta.json'))
print(' '.join(m['withheld_classes']))")

  for seed in $SEEDS; do
    for cfg in "k1:1:64" "k3:3:32"; do
      IFS=: read -r tagname k emb <<<"$cfg"
      exp="seed_${tag}_${tagname}_k${k}_e${emb}_s${seed}"
      if [ -f "experiments/results/$exp/open_set_test_results.json" ]; then
        continue
      fi
      echo "  $exp"
      $PY -u src/protoids/train_protoids.py \
        --dataset xiiotid --withhold_open_set \
        --withheld_classes $WITHHELD \
        --experiment_name "$exp" \
        --num_prototypes "$k" --embedding_dim "$emb" \
        --lambda_compact 0.1 --dropout_rate 0.2 --learning_rate 0.001 \
        --batch_size 256 --num_epochs 10 --device cuda --seed "$seed" \
        > "$LOG/${exp}.log" 2>&1 || echo "    FAILED (see $LOG/${exp}.log)"
    done
  done
done
echo "done. aggregating..."
