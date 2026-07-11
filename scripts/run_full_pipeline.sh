#!/usr/bin/env bash
# One-shot: build the whole Reasoning Reliability Atlas from a config.
#
# Usage:
#   bash scripts/run_full_pipeline.sh [CONFIG]
#
# Default CONFIG (configs/pipeline/full.yaml) runs entirely on CPU with the bundled
# example data and no model — the reproducible smoke path. Point the config's `data`
# at your downloaded JSONL and set `generation.model: amd/Instella-3B` for the real
# run on a GPU.
set -euo pipefail

CONFIG="${1:-configs/pipeline/full.yaml}"

echo "== Reasoning Reliability Atlas =="
echo "config: $CONFIG"
instella-reasoning run-all --config "$CONFIG"
