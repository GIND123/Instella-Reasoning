#!/usr/bin/env bash
# Phase 1 — the premise-deletion probe across the Instella trajectory.
#
# One item set, four checkpoints, identical decoding. The measurement is whether a model
# reproduces the parent's answer to a question the prompt no longer determines, and the
# comparison that carries it is *within item, across checkpoint* — so every generation
# setting except the model itself is held fixed here rather than read per-checkpoint from
# the registry. A differing token budget changes the truncation rate; a differing batch
# size changes greedy tie-breaking through padding. Either would enter the contrast as
# though it were a model difference.
#
# n_shot and the chat template DO vary, because they follow the checkpoint's own
# interface (base checkpoints loop without few-shot exemplars). That is a property of the
# model being measured, not a knob of the measurement.
#
# ENGINE picks the backend, and it must be constant across every checkpoint in a contrast:
#   vllm         — continuous batching; retires each sequence at its own stop point
#   transformers — static padded batches at BATCH; the reference implementation
# The engine is stamped into each row's metadata so a mixed pool is detectable later.
#
#   bash experiments/run_phase1.sh                        # train arm, greedy, vllm
#   ARM=test bash experiments/run_phase1.sh               # clean-arm control
#   TEMP=0.7 NSAMPLES=5 bash experiments/run_phase1.sh    # a rate needs k draws
#   ENGINE=transformers TAGS=stage1 bash experiments/run_phase1.sh

set -euo pipefail

REPO="${REPO:-$HOME/instella-reasoning}"
RUN="${RUN:-$REPO/experiments/runs/ckpt-axis-v1}"
ARM="${ARM:-train}"                    # train | test
TAGS="${TAGS:-stage1 stage2 sft instruct}"
ENGINE="${ENGINE:-vllm}"               # vllm | transformers
MAX_NEW="${MAX_NEW:-2048}"
BATCH="${BATCH:-8}"                    # transformers engine only
TEMP="${TEMP:-0.0}"
NSAMPLES="${NSAMPLES:-1}"              # draws per item; >1 requires TEMP>0
LIMIT="${LIMIT:-0}"                    # 0 = all rows
GPU_UTIL="${GPU_UTIL:-0.90}"           # vllm engine only
MIN_TERM="${MIN_TERM:-0.0}"            # 0 = record, never abort; the rate IS the result

HF_PY="$HOME/.venvs/instella/bin/python"
VLLM_PY="$HOME/.venvs/vllm/bin/python"

case "$ARM" in
  train) BENCH="$RUN/base/gsm8k_train_deletion.jsonl" ;;
  test)  BENCH="$RUN/base/gsm8k_test_deletion.jsonl" ;;
  *) echo "ARM must be train|test" >&2; exit 2 ;;
esac
[[ -s "$BENCH" ]] || { echo "missing benchmark: $BENCH" >&2; exit 2; }

case "$ENGINE" in
  vllm) [[ -x "$VLLM_PY" ]] || { echo "no vllm venv; run experiments/lambda_vllm_bootstrap.sh" >&2; exit 2; } ;;
  transformers) : ;;
  *) echo "ENGINE must be vllm|transformers" >&2; exit 2 ;;
esac

mkdir -p "$RUN/generations" "$RUN/logs"

_field() { PYTHONPATH="$REPO/src" "$HF_PY" - "$1" "$2" <<'PY'
import sys
from instella_reasoning.checkpoints import resolve
print(getattr(resolve(sys.argv[1]), sys.argv[2]))
PY
}

echo "== phase1 arm=$ARM engine=$ENGINE rows=$(wc -l < "$BENCH") tags=[$TAGS]"
echo "   T=$TEMP k=$NSAMPLES tokens=$MAX_NEW"

for tag in $TAGS; do
  model="$(_field "$tag" load_path)"
  n_shot="$(_field "$tag" n_shot)"
  is_base="$(_field "$tag" is_base)"
  chat_flag=""; [[ "$is_base" == "True" ]] && chat_flag="--no-chat-template"

  stem="phase1__${tag}__${ARM}__T${TEMP}__k${NSAMPLES}__${ENGINE}"
  gen="$RUN/generations/${stem}.jsonl"
  log="$RUN/logs/${stem}.log"

  echo "  == $tag  (model=$model shot=$n_shot base=$is_base) =="
  if [[ "$ENGINE" == "vllm" ]]; then
    limit_flag=""; [[ "$LIMIT" != "0" ]] && limit_flag="--limit $LIMIT"
    # shellcheck disable=SC2086
    VLLM_USE_V1=0 VLLM_LOGGING_LEVEL=WARNING HF_HOME="$HOME/hf" HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO/src:$HOME/arch" \
    "$VLLM_PY" "$REPO/experiments/vllm_generate.py" \
      --benchmark "$BENCH" --model "$model" --output "$gen" \
      --max-new-tokens "$MAX_NEW" --temperature "$TEMP" --n-samples "$NSAMPLES" \
      --n-shot "$n_shot" --gpu-memory-utilization "$GPU_UTIL" $chat_flag $limit_flag \
      2>&1 | tee "$log"
  else
    if [[ -s "$gen" ]]; then echo "     skip ($gen exists)"; continue; fi
    limit_flag=""; [[ "$LIMIT" != "0" ]] && limit_flag="--limit $LIMIT"
    # shellcheck disable=SC2086
    instella-reasoning generate \
      --benchmark "$BENCH" --model "$model" --output "$gen" \
      --max-new-tokens "$MAX_NEW" --batch-size "$BATCH" --temperature "$TEMP" \
      --dtype bf16 --n-shot "$n_shot" $chat_flag $limit_flag \
      --min-termination-rate "$MIN_TERM" 2>&1 | tee "$log"
  fi
done

echo "== phase1 arm=$ARM engine=$ENGINE complete"
