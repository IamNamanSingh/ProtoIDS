#!/usr/bin/env bash
# Provision a Colab T4 session with the X-IIoTID data already in place.
#
# Split out from the run scripts so a session can be created once and reused
# for many experiments. The previous attempt failed because the VM died
# mid-upload and the CLI blocked on a dead connection, so this verifies the
# transfer by checksum and refuses to continue on a dead session.
#
#   scripts/colab_session_up.sh              # default: T4, upload the zip
#   GPU=L4 scripts/colab_session_up.sh
#   DATA_MODE=kaggle scripts/colab_session_up.sh
#   SESSION=protoids-xiiotid DATA_MODE=none scripts/colab_session_up.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-T4}"
SESSION="${SESSION:-protoids-xiiotid}"
DATA_MODE="${DATA_MODE:-auto}"
PART_MB="${PART_MB:-32}"
REMOTE=/content
QUICK_TIMEOUT=180
TIMEOUT="${TIMEOUT:-3600}"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

command -v colab >/dev/null || { echo "colab CLI missing: uv tool install google-colab-cli"; exit 1; }

alive() {
  colab sessions 2>/dev/null | grep -q "$SESSION"
}

echo "== 1. session =="
if alive; then
  echo "   reusing existing session"
else
  colab new -s "$SESSION" --gpu "$GPU"
fi
colab status -s "$SESSION" || true
alive || { echo "session is not usable, aborting"; exit 1; }

echo
echo "== 2. dependencies =="
colab install -s "$SESSION" torch numpy pandas scikit-learn joblib pyarrow matplotlib kagglehub

echo
echo "== 3. data =="
CSV_LOCAL=""
[[ -d datasets/xiiotid ]] && CSV_LOCAL="$(find datasets/xiiotid -maxdepth 1 -iname '*.csv' -print -quit)"
ZIP_LOCAL="$HOME/Downloads/archive.zip"

ready() {
  echo "import glob; p=glob.glob('/content/datasets/xiiotid/*.csv'); print('CSV_READY' if p else 'NO_CSV')" \
    | colab exec -s "$SESSION" --timeout "$QUICK_TIMEOUT" 2>/dev/null | grep -q CSV_READY
}

try_exec() {
  local secs="$1"; shift
  alive || return 1
  cat | colab exec -s "$SESSION" --timeout "$secs" "$@"
}

if ready; then
  echo "   CSV already on the VM"
else
  case "$DATA_MODE" in
    none)
      echo "   DATA_MODE=none and no CSV on the VM"
      ;;
    kaggle)
      [[ -f "$HOME/.kaggle/kaggle.json" ]] || { echo "no ~/.kaggle/kaggle.json"; exit 1; }
      colab upload -s "$SESSION" "$HOME/.kaggle/kaggle.json" "$REMOTE/kaggle.json"
      try_exec "$QUICK_TIMEOUT" <<'PY'
import os, shutil
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy('/content/kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
PY
      try_exec "$TIMEOUT" <<'PY'
import sys; sys.path.insert(0, '/content/ProtoIDS/src')
from data.download_xiiotid import ensure_xiiotid
print(ensure_xiiotid(dest_dir='/content/datasets/xiiotid', source='auto'))
PY
      ;;
    *)
      # The zip is ~114 MB versus 355 MB for the raw CSV, and the contents API
      # rejects large uploads, so transfer the zip in parts and verify it.
      SRC=""
      if [[ -n "$ZIP_LOCAL" && -f "$ZIP_LOCAL" ]]; then SRC="$ZIP_LOCAL"; fi
      if [[ -z "$SRC" && -n "$CSV_LOCAL" ]]; then SRC="$CSV_LOCAL"; fi
      [[ -n "$SRC" ]] || { echo "no local archive or CSV to send; use DATA_MODE=kaggle"; exit 1; }

      parts=/tmp/opencode/colab_parts
      echo "   sending $(basename "$SRC") ($(du -m "$SRC" | cut -f1) MB) in ${PART_MB} MB parts"
      "$PY" "$REPO_ROOT/scripts/remote/split_file.py" split "$SRC" "$parts" --part-mb "$PART_MB" >/dev/null
      try_exec "$QUICK_TIMEOUT" <<<"import os; os.makedirs('/content/upload', exist_ok=True)" >/dev/null
      total=$(ls -1 "$parts" | grep -c '\.part[0-9]' || true)
      n=0
      for p in "$parts"/*.part*; do
        n=$((n + 1))
        alive || { echo "   session died during upload (part $n/$total)"; exit 1; }
        printf '   part %s/%s %s\n' "$n" "$total" "$(basename "$p")"
        colab upload -s "$SESSION" "$p" "/content/upload/$(basename "$p")" >/dev/null \
          || { echo "   upload of part $n failed; aborting"; exit 1; }
      done
      colab upload -s "$SESSION" "$parts/manifest.json" "/content/upload/manifest.json" >/dev/null
      colab upload -s "$SESSION" "$REPO_ROOT/scripts/remote/split_file.py" \
        "/content/upload/split_file.py" >/dev/null
      echo "   reassembling and verifying on the VM"
      try_exec "$TIMEOUT" <<'PY'
import os, sys, zipfile
D='/content/upload'
sys.path.insert(0, D)
from split_file import join
out = join(D, os.path.join(D, 'archive.zip'))
os.makedirs('/content/datasets/xiiotid', exist_ok=True)
if zipfile.is_zipfile(out):
    with zipfile.ZipFile(out) as z:
        z.extractall('/content/datasets/xiiotid')
    print('extracted', os.listdir('/content/datasets/xiiotid'))
else:
    import shutil
    shutil.move(out, '/content/datasets/xiiotid/')
    print('moved csv', os.listdir('/content/datasets/xiiotid'))
PY
      ;;
  esac
fi

echo
echo "== 4. verify on the VM =="
if ! try_exec "$QUICK_TIMEOUT" <<'PY'
import glob, os, sys
p = sorted(glob.glob('/content/datasets/xiiotid/*.csv'))
if not p:
    sys.exit('NO CSV ON VM')
f = p[0]
print('path :', f)
print('size : %.1f MB' % (os.path.getsize(f) / 1e6))
import pandas as pd
h = pd.read_csv(f, nrows=0)
print('cols :', len(h.columns))
import torch
print('torch:', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
then
  echo "verification FAILED - the session does not have usable data"
  exit 1
fi

echo
echo "session '$SESSION' ready on $GPU."
echo "  url      : \$(colab url -s $SESSION)"
echo "  run with : scripts/colab_t4_run.sh"
echo "  teardown : colab stop -s $SESSION"
