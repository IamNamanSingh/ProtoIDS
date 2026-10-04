#!/usr/bin/env bash
# Start the multi-seed run as a DETACHED process on the Colab VM.
#
# Why detached: `colab exec` runs the job as a cell inside the VM's Jupyter
# kernel, so the job dies whenever the local WebSocket drops - switching WiFi,
# a VPN, a sleep/wake. Running it as a nohup'd VM-side process makes the job
# independent of the client: you can disconnect, change network, reconnect
# later. A short exec cell is used only to spawn it.
#
# Resumes from cells.json, so completed trials are never repeated.
set -euo pipefail
SESSION="${SESSION:-protoids-xiiotid}"
SEEDS="${SEEDS:-42 7 1234 5 99 2024}"
KS="${KS:-1 3}"
PROTOCOLS="${PROTOCOLS:-s1,s2,fam}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
REMOTE_LOG=/content/multiseed.log
export PATH="$HOME/.local/bin:$PATH"

colab sessions 2>/dev/null | grep -q "$SESSION" || {
  echo "no live session '$SESSION'"; exit 1; }

cat <<PY | colab exec -s "$SESSION" --timeout 180
import os, subprocess, sys
for p in ('/content/run_multiseed_remote.py',
          '/content/ProtoIDS/src/protoids/multiseed.py',
          '/content/ProtoIDS/experiments/processed/xiiotid_s1/split_meta.json'):
    assert os.path.exists(p), f'missing on VM: {p}'

env = dict(os.environ)
env.update(
    XIIOTID_DATA_DIR='/content/datasets/xiiotid',
    XIIOTID_SEEDS='$SEEDS',
    XIIOTID_KS='$KS',
    XIIOTID_PROTOCOLS='$PROTOCOLS',
    XIIOTID_EPOCHS='$EPOCHS',
    XIIOTID_BATCH='$BATCH',
    SESSION_TAG='$SESSION',
    PYTHONPATH='/content/ProtoIDS/src',
    PYTHONUNBUFFERED='1',
)
log = open('$REMOTE_LOG', 'w')
proc = subprocess.Popen(
    [sys.executable, '/content/run_multiseed_remote.py'],
    cwd='/content/ProtoIDS', env=env, stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True,
)
print('DETACHED_PID', proc.pid)
PY

echo
echo "job detached on the VM, log at $REMOTE_LOG"
echo "check progress:"
echo "  export PATH=\$HOME/.local/bin:\$PATH"
echo "  colab download -s $SESSION $REMOTE_LOG /tmp/remote.log && tail -20 /tmp/remote.log"
echo "or just count finished trials:"
echo "  colab download -s $SESSION /content/ProtoIDS/experiments/results/multiseed_${SESSION}/cells.json /tmp/c.json && python3 -c \"import json;print(len(json.load(open('/tmp/c.json'))),'done')\""
echo
echo "safe to disconnect / change network now."
