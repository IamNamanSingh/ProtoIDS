#!/usr/bin/env bash
# Launch a Jupyter server that the Colab frontend can attach to as a LOCAL RUNTIME.
#
# Reference: https://research.google.com/colaboratory/local-runtimes.html
# The two flags that matter are --NotebookApp.allow_origin (lets the Colab
# frontend open the websocket) and --NotebookApp.allow_credentials.
#
# Usage:
#   scripts/colab_local_runtime.sh            # 127.0.0.1:8888
#   PORT=9000 scripts/colab_local_runtime.sh  # custom port
#   HOST=0.0.0.0 scripts/colab_local_runtime.sh   # for SSH port-forwarding from another box
#
# Then in Colab: Connect -> "Connect to local runtime..." -> paste the printed URL.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The Colab frontend expects a "content" directory to exist inside the notebook
# root; without it, file upload / composer features break.
mkdir -p content

# Pick an interpreter: $PYTHON, else the repo venv, else whatever is on PATH.
if [[ -n "${PYTHON:-}" ]]; then
  JUPYTER="$PYTHON -m jupyter notebook"
elif [[ -x "$REPO_ROOT/.venv/bin/jupyter" ]]; then
  JUPYTER="$REPO_ROOT/.venv/bin/jupyter notebook"
else
  JUPYTER="jupyter notebook"
fi

PORT="${PORT:-8888}"
HOST="${HOST:-127.0.0.1}"

cat <<EOF
Starting Jupyter for the Colab local runtime.

  root     : $REPO_ROOT
  bind     : $HOST:$PORT
  interpreter: $($REPO_ROOT/.venv/bin/python -V 2>/dev/null || python3 -V)

Next:
  1. Copy the http://.../?token=... URL printed below.
  2. In Colab: Connect -> "Connect to local runtime..." -> paste that URL.
  3. Open colab/ProtoIDS_XIIoTID.ipynb.

EOF

# shellcheck disable=SC2086
exec $JUPYTER \
  --notebook-dir="$REPO_ROOT" \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.allow_credentials=True \
  --NotebookApp.port_retries=0 \
  --ip="$HOST" \
  --port="$PORT" \
  --no-browser \
  --ServerApp.root_dir="$REPO_ROOT"
