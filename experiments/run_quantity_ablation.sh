#!/usr/bin/env bash
# G1 - the quantity ablation across the Instella-3B ladder.
#
# One item set, N checkpoints, identical decoding. Every generation setting except the
# model is held fixed here rather than read per-checkpoint, for the same reason as
# run_phase1.sh: a differing token budget changes the truncation rate and a differing
# batch size changes greedy tie-breaking, and either would enter the contrast as though
# it were a model difference. n_shot and the chat template still follow the checkpoint's
# own interface, because a base checkpoint loops without exemplars.
#
# The comparison that carries the result is *within parent, across condition*, so the
# checkpoint axis is secondary here — but all conditions for one checkpoint must come
# from one engine, and the engine is stamped per row so a mixed pool is detectable.
#
# Resumable: an existing non-empty output for a (checkpoint) is skipped, so a Spot
# preemption costs at most the checkpoint in flight.
#
#   bash experiments/run_quantity_ablation.sh
#   TAGS="instruct" bash experiments/run_quantity_ablation.sh
#   PUSH=0 bash experiments/run_quantity_ablation.sh          # skip the HF sync

set -euo pipefail

REPO="${REPO:-$HOME/instella-reasoning}"
RUN="${RUN:-$REPO/experiments/runs/qty-ablation-v1}"
TAGS="${TAGS:-stage2 sft instruct}"
ENGINE="${ENGINE:-vllm}"
MAX_NEW="${MAX_NEW:-2048}"
TEMP="${TEMP:-0.0}"
GPU_UTIL="${GPU_UTIL:-0.90}"
HF_REPO="${HF_REPO:-GOVINDFROM/Instella-Reasoning}"
PUSH="${PUSH:-1}"

VLLM_PY="${VLLM_PY:-$HOME/.venvs/vllm/bin/python}"
HF_PY="${HF_PY:-$HOME/.venvs/instella/bin/python}"
BENCH="$RUN/base/qty_ablation.jsonl"

[[ -s "$BENCH" ]] || { echo "missing item set: $BENCH" >&2; echo "run build_quantity_ablation.py first" >&2; exit 2; }
[[ -x "$VLLM_PY" ]] || { echo "no vllm venv at $VLLM_PY; run experiments/lambda_vllm_bootstrap.sh" >&2; exit 2; }

mkdir -p "$RUN/generations" "$RUN/logs" "$RUN/analysis"

_field() { PYTHONPATH="$REPO/src" "$HF_PY" - "$1" "$2" <<'PY'
import sys
from instella_reasoning.checkpoints import resolve
print(getattr(resolve(sys.argv[1]), sys.argv[2]))
PY
}

echo "== quantity ablation: $(wc -l < "$BENCH") rows, tags [$TAGS], engine $ENGINE"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

for tag in $TAGS; do
  model="$(_field "$tag" load_path)"
  n_shot="$(_field "$tag" n_shot)"
  is_base="$(_field "$tag" is_base)"
  chat_flag=""; [[ "$is_base" == "True" ]] && chat_flag="--no-chat-template"

  gen="$RUN/generations/qty__${tag}.jsonl"
  log="$RUN/logs/qty__${tag}.log"

  if [[ -s "$gen" ]]; then
    echo "  == $tag: skip, $(wc -l < "$gen") rows already present"
    continue
  fi

  echo "  == $tag (model=$model shot=$n_shot base=$is_base) =="
  VLLM_USE_V1=0 VLLM_LOGGING_LEVEL=WARNING HF_HOME="${HF_HOME:-$HOME/hf}" \
  HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  PYTHONPATH="$REPO/src:$HOME/arch" \
  "$VLLM_PY" "$REPO/experiments/vllm_generate.py" \
    --benchmark "$BENCH" --model "$model" --output "$gen" \
    --max-new-tokens "$MAX_NEW" --temperature "$TEMP" --n-samples 1 \
    --n-shot "$n_shot" --gpu-memory-utilization "$GPU_UTIL" $chat_flag \
    2>&1 | tee "$log"

  # Analyse and sync after every checkpoint, not at the end: on Spot the box can vanish
  # with 30 seconds' notice and an unsynced checkpoint is a wasted GPU-hour.
  PYTHONPATH="$REPO/src" "$HF_PY" "$REPO/experiments/analyze_quantity_ablation.py" \
    --run "$RUN" || true
  if [[ "$PUSH" == "1" ]]; then
    PYTHONPATH="$REPO/src" "$HF_PY" "$REPO/experiments/hf_sync.py" push \
      --repo "$HF_REPO" --path "experiments/runs/qty-ablation-v1" \
      --message "qty ablation: $tag" || true
  fi
done

echo "== quantity ablation complete"
PYTHONPATH="$REPO/src" "$HF_PY" "$REPO/experiments/analyze_quantity_ablation.py" --run "$RUN"
