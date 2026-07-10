#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/training/amd_base_upstream.yaml}"

if [[ ! -d "external/Instella/.git" ]]; then
  mkdir -p external
  git clone https://github.com/AMD-AGI/Instella external/Instella
fi

COMMAND="$(instella-reasoning train-command --config "$CONFIG_PATH")"
echo "$COMMAND"
exec bash -lc "$COMMAND"
