#!/usr/bin/env bash
# Cross-validate the X-IIoTID open-set result under two independent protocols.
#
# The published numbers came from ONE withheld-class set chosen by us, on a
# random split. X-IIoTID also drifts heavily over its 303-day span with attacks
# clustered early, so a random split is optimistic. These runs ask whether the
# headline result (K=1 beating K=3) survives a change of protocol.
#
#   Protocol S1  our original set   : Scanning_vulnerability, Dictionary, Reverse_shell
#   Protocol S2  disjoint second set: Exfiltration, BruteForce, Modbus_register_reading
#                 -> two unseen families + BruteForce, a SIBLING of the already
#                    withheld Dictionary, so this separates "unseen family" from
#                    "unseen type within a known family".
#   Protocol F   whole families     : Exfiltration, Lateral _movement, C&C
#                 -> 5 types across 3 attack categories the model never sees at all.
#
# Each protocol runs both the swept best (K=1, dim 64) and the documented
# default (K=3, dim 32) so the K claim is tested, not assumed.
set -euo pipefail
cd /mnt/Storage/ProtoIDS
export XIIOTID_DATA_DIR=datasets/xiiotid
PY=.venv/bin/python
LOG=/tmp/opencode/xval
mkdir -p "$LOG"

run() {  # name  withheld_args...
  local name="$1"; shift
  for cfg in "k1:1:64" "k3:3:32"; do
    IFS=: read -r tag k emb <<<"$cfg"
    local exp="xval_${name}_${tag}_k${k}_e${emb}"
    if [ -f "experiments/results/$exp/open_set_test_results.json" ]; then
      echo "  skip $exp (already done)"; continue
    fi
    echo "=== $exp ==="
    $PY -u src/protoids/train_protoids.py \
      --dataset xiiotid --withhold_open_set "$@" \
      --withheld_classes $WITHHELD \
      --experiment_name "$exp" \
      --num_prototypes "$k" --embedding_dim "$emb" \
      --lambda_compact 0.1 --dropout_rate 0.2 --learning_rate 0.001 \
      --batch_size 256 --num_epochs 10 --device cuda --seed 42 \
      > "$LOG/${exp}.log" 2>&1 || { echo "  FAILED, see $LOG/${exp}.log"; continue; }
    $PY - "$exp" <<'PYEOF'
import json, sys, os
exp = sys.argv[1]
r = f"experiments/results/{exp}"
v = json.load(open(f"{r}/open_set_results.json"))
t = json.load(open(f"{r}/open_set_test_results.json"))
print(f"  VAL  known={v['known_accuracy']:.4f} unkF1={v['unknown_f1']:.4f} "
      f"AUROC={v.get('auc_roc',0):.4f} AUPR={v.get('auc_pr',0):.4f} thr={v['threshold']:.4f}")
print(f"  TEST known={t['known_accuracy']:.4f} unkF1={t['unknown_f1']:.4f} "
      f"AUROC={t.get('auc_roc',0):.4f} AUPR={t.get('auc_pr',0):.4f} thr={t['threshold']:.4f}")
PYEOF
  done
}

run_family() {  # name family1 family2 ...
  local name="$1"; shift
  for cfg in "k1:1:64" "k3:3:32"; do
    IFS=: read -r tag k emb <<<"$cfg"
    local exp="xval_${name}_${tag}_k${k}_e${emb}"
    if [ -f "experiments/results/$exp/open_set_test_results.json" ]; then
      echo "  skip $exp (already done)"; continue
    fi
    echo "=== $exp ==="
    $PY -u src/protoids/train_protoids.py \
      --dataset xiiotid --withhold_open_set \
      --withheld_families "$@" \
      --experiment_name "$exp" \
      --num_prototypes "$k" --embedding_dim "$emb" \
      --lambda_compact 0.1 --dropout_rate 0.2 --learning_rate 0.001 \
      --batch_size 256 --num_epochs 10 --device cuda --seed 42 \
      > "$LOG/${exp}.log" 2>&1 || { echo "  FAILED, see $LOG/${exp}.log"; continue; }
    $PY - "$exp" <<'PYEOF'
import json, sys
exp = sys.argv[1]
r = f"experiments/results/{exp}"
v = json.load(open(f"{r}/open_set_results.json"))
t = json.load(open(f"{r}/open_set_test_results.json"))
print(f"  VAL  known={v['known_accuracy']:.4f} unkF1={v['unknown_f1']:.4f} "
      f"AUROC={v.get('auc_roc',0):.4f} AUPR={v.get('auc_pr',0):.4f}")
print(f"  TEST known={t['known_accuracy']:.4f} unkF1={t['unknown_f1']:.4f} "
      f"AUROC={t.get('auc_roc',0):.4f} AUPR={t.get('auc_pr',0):.4f}")
PYEOF
  done
}

echo "########## Protocol S2 - disjoint second withheld set ##########"
WITHHELD="Exfiltration BruteForce Modbus_register_reading"
echo "withheld: $WITHHELD"
run s2

echo
echo "########## Protocol F - three whole attack families ##########"
run_family fam Exfiltration "Lateral _movement" "C&C"

echo
echo "done. logs in $LOG"
