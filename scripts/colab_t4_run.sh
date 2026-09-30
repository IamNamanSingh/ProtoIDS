#!/usr/bin/env bash
# Run the X-IIoTID ProtoIDS experiment on a REMOTE Colab GPU (T4 / L4 / A100)
# using the official `google-colab-cli`.
#
#   https://pypi.org/project/google-colab-cli/   (github.com/googlecolab)
#
# One-off setup:
#   uv tool install google-colab-cli
#   export PATH="$HOME/.local/bin:$PATH"
#   colab usage            # triggers the OAuth flow; paste the auth code
#
# Usage:
#   scripts/colab_t4_run.sh                       # T4, 10 epochs
#   GPU=L4 EPOCHS=20 scripts/colab_t4_run.sh      # different GPU / budget
#   DATA_MODE=upload scripts/colab_t4_run.sh       # push the 355 MB CSV
#   DATA_MODE=none   WITHNESS=... scripts/colab_t4_run.sh   # reuse an existing VM
#
# DATA_MODE:
#   auto   (default) reuse a CSV already on the VM, else upload it
#   upload always upload datasets/xiiotid/*.csv
#   kaggle let the VM pull it from Kaggle (needs ~/.kaggle/kaggle.json there)
#   none   do not transfer data at all
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-T4}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
K="${K:-3}"
LAMBDA="${LAMBDA:-0.1}"
SEED="${SEED:-42}"
DATA_MODE="${DATA_MODE:-auto}"
SESSION="${SESSION:-protoids-xiiotid}"
REMOTE_DIR=/content
LOCAL_TARBALL=/tmp/opencode/protoids_repo.tar.gz
REMOTE_ENTRY="$REPO_ROOT/scripts/remote/run_xiiotid_remote.py"
# `colab exec` kills the kernel after 30s by default, which is far too short for
# preprocessing + training. Raise it well past the expected runtime.
TIMEOUT="${TIMEOUT:-14400}"
QUICK_TIMEOUT=120
# Size of each transfer part. The Colab contents API rejects large uploads, so
# 32 MB keeps every part comfortably under the limit.
PART_MB="${PART_MB:-32}"
WITHHELD="${WITHHELD:-}"

# Any python works for the local split/join helper.
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

command -v colab >/dev/null || { echo "colab CLI not found: uv tool install google-colab-cli"; exit 1; }

cleanup() { colab stop -s "$SESSION" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== 1/6 packaging the repo =="
# Ship the working tree (tracked + new files), but never the data or artifacts:
# no datasets, memmaps, venv or .git. Using --cached --others --exclude-standard
# keeps .gitignore authoritative, so ignoring datasets/ and *.dat is enough.
git ls-files -z --cached --others --exclude-standard \
  | tar --null -czf "$LOCAL_TARBALL" --files-from=-
ls -lh "$LOCAL_TARBALL"
echo "   files in tarball: $(tar -tzf "$LOCAL_TARBALL" | wc -l)"

echo "== 2/6 provisioning a $GPU runtime =="
colab new -s "$SESSION" --gpu "$GPU"
colab status -s "$SESSION"

echo "== 3/6 installing dependencies on the VM =="
colab install -s "$SESSION" torch numpy pandas scikit-learn joblib pyarrow matplotlib kagglehub

echo "== 4/6 uploading code =="
colab upload -s "$SESSION" "$LOCAL_TARBALL" "$REMOTE_DIR/protoids_repo.tar.gz"
colab upload -s "$SESSION" "$REMOTE_ENTRY" "$REMOTE_DIR/run_xiiotid_remote.py"

LOCAL_CSV=""
if [[ -d datasets/xiiotid ]]; then
  LOCAL_CSV="$(find datasets/xiiotid -maxdepth 1 -iname '*.csv' -print -quit)"
fi

# `colab upload` goes through Jupyter's /api/contents, which 400s on a
# few-hundred-MB file. Push the dataset in parts, then reassemble on the VM and
# verify the SHA-256 so a truncated part cannot pass as a complete file.
upload_csv() {
  local csv="$1"
  local name; name="$(basename "$csv")"
  local parts=/tmp/opencode/xiiotid_parts

  echo "   splitting $(basename "$csv") ($(du -m "$csv" | cut -f1) MB) into ${PART_MB} MB parts"
  "$PY" "$REPO_ROOT/scripts/remote/split_file.py" split "$csv" "$parts" \
    --part-mb "$PART_MB" >/dev/null

  colab exec -s "$SESSION" --timeout "$QUICK_TIMEOUT" \
    -c "print(__import__('os').makedirs('/content/datasets/xiiotid', exist_ok=True))" >/dev/null 2>&1 || true

  local total; total="$(ls -1 "$parts" | grep -c '\.part[0-9]' || true)"
  local n=0
  for part in "$parts"/*.part*; do
    n=$((n + 1))
    printf '   part %s/%s  %s\n' "$n" "$total" "$(basename "$part")"
    colab upload -s "$SESSION" "$part" "/content/datasets/xiiotid/$(basename "$part")" >/dev/null
  done

  colab upload -s "$SESSION" "$REPO_ROOT/scripts/remote/split_file.py" \
    "/content/datasets/xiiotid/split_file.py" >/dev/null
  echo "   reassembling and checksumming on the VM"
  colab exec -s "$SESSION" --timeout "$TIMEOUT" -c "
import os, sys
D = '/content/datasets/xiiotid'
sys.path.insert(0, D)
from split_file import join
join(D, os.path.join(D, '${name}'))
for f in os.listdir(D):
    if f.endswith(tuple('0123456789')) and '.part' in f or f in ('manifest.json', 'split_file.py'):
        os.remove(os.path.join(D, f))
print('ready:', os.path.join(D, '${name}'))
"
}

case "$DATA_MODE" in
  none)   echo "   skipping data transfer" ;;
  upload)
    [[ -n "$LOCAL_CSV" ]] || { echo "no CSV in datasets/xiiotid"; exit 1; }
    upload_csv "$LOCAL_CSV"
    ;;
  kaggle)
    if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
      colab upload -s "$SESSION" "$HOME/.kaggle/kaggle.json" "$REMOTE_DIR/kaggle.json"
      colab exec -s "$SESSION" --timeout "$QUICK_TIMEOUT" -c "
import os, json, shutil
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy('/content/kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
print('kaggle credentials installed on the VM')
" >/dev/null
    else
      echo "   WARNING: no ~/.kaggle/kaggle.json; the VM cannot pull the dataset"
    fi
    ;;
  auto)
    if [[ -n "$LOCAL_CSV" ]]; then
      upload_csv "$LOCAL_CSV"
    elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
      echo "   no local CSV; uploading Kaggle credentials so the VM can fetch it"
      colab upload -s "$SESSION" "$HOME/.kaggle/kaggle.json" "$REMOTE_DIR/kaggle.json"
      colab exec -s "$SESSION" --timeout "$QUICK_TIMEOUT" -c "
import os, shutil
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy('/content/kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
" >/dev/null
    else
      echo "   WARNING: no local CSV and no Kaggle credentials; the run will stop early"
    fi
    ;;
  *) echo "unknown DATA_MODE=$DATA_MODE"; exit 1 ;;
esac

echo "== 5/6 running the experiment on the $GPU =="
colab exec -s "$SESSION" -f "$REMOTE_ENTRY" \
  --env "XIIOTID_EPOCHS=$EPOCHS" \
  --env "XIIOTID_BATCH=$BATCH" \
  --env "XIIOTID_K=$K" \
  --env "XIIOTID_LAMBDA=$LAMBDA" \
  ${WITHHELD:+--env "XIIOTID_WITHHELD=$WITHHELD"} \
  || echo "the remote run exited non-zero; the log above has the traceback"

echo "== 6/6 fetching artifacts =="
STAMP="$(date +%Y%m%d-%H%M%S)"
LOCAL_OUT="artifacts/colab-${GPU}-${STAMP}"
mkdir -p "$LOCAL_OUT"
for f in summary.json; do
  colab download -s "$SESSION" "/content/ProtoIDS/$f" "$LOCAL_OUT/$f" 2>/dev/null || true
done
colab exec -s "$SESSION" -c "
import os, glob
for p in glob.glob('/content/ProtoIDS/experiments/results/*/*.json'):
    print(os.path.relpath(p, '/content/ProtoIDS'))
" 2>/dev/null | tail -20 || true

echo
echo "done. local copy of the summary: $LOCAL_OUT"
echo "full artifacts are on the VM at /content/ProtoIDS/experiments/results/"
echo "re-attach with:  colab url -s $SESSION      (session is stopped on exit)"
