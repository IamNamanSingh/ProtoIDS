#!/usr/bin/env bash
# Launch the multi-seed measurement on the live Colab T4 session.
#
# Reuses the session provisioned by scripts/colab_session_up.sh, so no VM is
# created and the dataset is already on disk. Writes the remote log to
# /tmp/opencode/multiseed_remote.log for `watch` to follow.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"

SESSION="${SESSION:-protoids-xiiotid}"
LOG=/tmp/opencode/multiseed_remote.log
ENTRY="$REPO_ROOT/scripts/remote/run_multiseed_remote.py"
REMOTE_ENTRY=/content/run_multiseed_remote.py
SEEDS="${SEEDS:-42 7 1234 5 99 2024}"
KS="${KS:-1 3}"
PROTOCOLS="${PROTOCOLS:-s1,s2,fam}"
EPOCHS="${EPOCHS:-10}"
TIMEOUT="${TIMEOUT:-21600}"

command -v colab >/dev/null || { echo "colab CLI missing"; exit 1; }
colab sessions 2>/dev/null | grep -q "$SESSION" || {
  echo "no live session '$SESSION' - run scripts/colab_session_up.sh first"; exit 1; }

echo "== uploading the repo =="
TARBALL=/tmp/opencode/protoids_repo.tar.gz
git ls-files -z --cached --others --exclude-standard \
  | tar --null -czf "$TARBALL" --files-from=-
echo "  $(du -h "$TARBALL" | cut -f1), $(tar -tzf "$TARBALL" | wc -l) files"
colab upload -s "$SESSION" "$TARBALL" /content/protoids_repo.tar.gz >/dev/null
colab upload -s "$SESSION" "$ENTRY" "$REMOTE_ENTRY" >/dev/null

echo "== unpacking on the VM =="
echo "
import tarfile, pathlib
d = pathlib.Path('/content/ProtoIDS')
d.mkdir(parents=True, exist_ok=True)
with tarfile.open('/content/protoids_repo.tar.gz') as t:
    t.extractall(d, filter='data')
print('src ok:', (d / 'src/protoids/dataset.py').exists())
" | colab exec -s "$SESSION" --timeout 600

echo "== launching (seeds: $SEEDS | K: $KS | protocols: $PROTOCOLS) =="
nohup bash -c "
  echo \"
import os
os.environ['XIIOTID_DATA_DIR'] = '/content/datasets/xiiotid'
os.environ['XIIOTID_SEEDS'] = '$SEEDS'
os.environ['XIIOTID_KS'] = '$KS'
os.environ['XIIOTID_PROTOCOLS'] = '$PROTOCOLS'
os.environ['XIIOTID_EPOCHS'] = '$EPOCHS'
os.environ['SESSION_TAG'] = '$SESSION'
import runpy, sys
sys.argv = ['run_multiseed_remote.py']
runpy.run_path('$REMOTE_ENTRY', run_name='__main__')
\" | colab exec -s '$SESSION' --timeout $TIMEOUT
" > "$LOG" 2>&1 &
disown

echo
echo "remote job started. watch it with:"
echo "  watch -n 15 'tail -n 12 $LOG'"
echo "  watch -n 15 'colab status -s $SESSION; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'"
