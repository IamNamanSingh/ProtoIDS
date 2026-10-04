#!/usr/bin/env bash
# Phase 1: build a cached processed split per open-set protocol.
#
# The sweep re-reads these memmaps instead of the 355 MB CSV, so the expensive
# parse+scaling is paid once per protocol rather than once per trial.
set -euo pipefail
cd /mnt/Storage/ProtoIDS
export XIIOTID_DATA_DIR=datasets/xiiotid
PY=.venv/bin/python

build() {  # tag  label  <train args...>
  local tag="$1" label="$2"; shift 2
  local proc="experiments/processed/xiiotid_${tag}"
  if [ -f "$proc/split_meta.json" ]; then
    echo "  $tag: cached split already present"; return
  fi
  echo "  $tag: building split ($label)"
  # train_protoids writes the split manifest; run 0 epochs so it preprocesses
  # and exits without spending GPU time on a throwaway model.
  $PY -u src/protoids/train_protoids.py \
    --dataset xiiotid --withhold_open_set "$@" \
    --experiment_name "prep_${tag}" \
    --num_prototypes 1 --embedding_dim 64 \
    --num_epochs 1 --batch_size 256 --device cuda --seed 42 \
    > "/tmp/opencode/xval/prep_${tag}.log" 2>&1 \
    || { echo "    FAILED (see /tmp/opencode/xval/prep_${tag}.log)"; return 1; }
  # move the cached split to the protocol-named dir so the sweep finds it
  mkdir -p "$(dirname "$proc")"
  [ -d "experiments/processed/prep_${tag}" ] && mv "experiments/processed/prep_${tag}" "$proc"
  echo "    -> $proc"
}

mkdir -p /tmp/opencode/xval
echo "=== building protocol splits ==="
build s1  "our original set"   --withheld_classes Scanning_vulnerability Dictionary Reverse_shell
build s2  "disjoint second set" --withheld_classes Exfiltration BruteForce Modbus_register_reading
build fam "three whole families" --withheld_families Exfiltration "Lateral _movement" "C&C"
echo
echo "=== cached splits ==="
for t in s1 s2 fam; do
  p="experiments/processed/xiiotid_${t}/split_meta.json"
  [ -f "$p" ] && $PY -c "
import json,sys
m=json.load(open('$p'))
print(f\"  $t  dim={m['input_dim']}  known={len(m['known_classes'])}  rows={m['counts']}  withheld={m['withheld_classes']}\")"
done
