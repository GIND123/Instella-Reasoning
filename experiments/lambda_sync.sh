#!/usr/bin/env bash
# Push the local working tree to a Lambda Labs box.
#
# The repo is private, so cloning on the box would need a token. Syncing the working
# tree instead reproduces the property the Modal image gives us via add_local_dir(
# copy=True): what executes remotely is byte-identical to what is on this laptop, so
# local code cannot drift from what produced the results.
#
# The exclude list mirrors modal_fullscale.py:118 — run directories come from HF, not
# from here, and model weights are downloaded on the box.
#
#   ./experiments/lambda_sync.sh 192.222.x.x

set -euo pipefail

HOST="${1:?usage: lambda_sync.sh <instance-ip> [user]}"
USER="${2:-ubuntu}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="/home/$USER/instella-reasoning"

echo "==> syncing $SRC -> $USER@$HOST:$DEST"
rsync -az --delete --stats \
    --exclude '.git' \
    --exclude 'experiments/runs' \
    --exclude 'models' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude '.DS_Store' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude 'data/raw' \
    --exclude 'outputs' \
    "$SRC/" "$USER@$HOST:$DEST/"

echo "==> done. next:"
echo "  ssh $USER@$HOST 'bash $DEST/experiments/lambda_bootstrap.sh'"
