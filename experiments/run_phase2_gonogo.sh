#!/usr/bin/env bash
# Phase 2 go/no-go: the extremes first, then the middle doses only if the pair separates.
#
# 0x runs BEFORE 64x, for a reason that costs nothing: 0x is the same fixed token budget
# made entirely of filler, so its probe recall should come back at Instella-3B's already
# measured rate (2.83% on the train arm). That is a free end-to-end validation of the whole
# harness against a known quantity. If 0x lands far from it, continue-pretraining itself
# damaged the model, and we learn that before spending anything on 64x.
#
# A smoke arm runs first at a tiny budget. This is new code on the critical path of a
# causal claim, and a crash on step 3 of the real 0x arm costs eight minutes of H100 to
# discover something two minutes would have.
#
# The stopping rule is NOT applied here. This script produces the numbers; the decision to
# run the middle doses is a human one, and it is only meaningful if the positive controls
# inside each arm passed. A flat 64x-vs-0x with failed controls measures the harness.

set -euo pipefail

REPO="${REPO:-$HOME/instella-reasoning}"
RUN="${RUN:-$REPO/experiments/runs/ckpt-axis-v1}"
MIX="$RUN/phase2/mixture"
OUT="$RUN/phase2"
TOTAL_TOKENS="${TOTAL_TOKENS:-8388608}"
DOSES="${DOSES:-0 64}"      # go/no-go pair by default; middle doses once it passes
SKIP_SMOKE="${SKIP_SMOKE:-0}"
HF_PY="$HOME/.venvs/instella/bin/python"
VLLM_PY="$HOME/.venvs/vllm/bin/python"

export HF_HOME="$HOME/hf" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/src"

mkdir -p "$OUT" "$RUN/logs"
cd "$REPO"

if [[ "$SKIP_SMOKE" == "0" ]]; then
echo "===== SMOKE (dose 1, tiny budget, no save) ====="
"$HF_PY" experiments/inject_pretrain.py --dose 1 --mixture-dir "$MIX" \
  --out-dir "$OUT/smoke" --total-tokens 262144 --micro-batch 2 --grad-accum 4 \
  2>&1 | tee "$RUN/logs/phase2_smoke.log"
fi

for dose in $DOSES; do
  echo "===== ARM dose=${dose}x  (budget $TOTAL_TOKENS tokens, held constant) ====="
  "$HF_PY" experiments/inject_pretrain.py --dose "$dose" --mixture-dir "$MIX" \
    --out-dir "$OUT" --total-tokens "$TOTAL_TOKENS" --save \
    2>&1 | tee "$RUN/logs/phase2_dose${dose}.log"
done

# Probe both arms with the identical Phase 1 instrument: same deletion items, same engine,
# same decoding, and crucially the same prompting interface. These checkpoints derive from
# amd/Instella-3B, a BASE checkpoint, which Phase 1 probed at n_shot=4 with no chat
# template. Probing them at n_shot=0 instead would make the 0x arm incomparable to the
# 2.83% it exists to validate against, and nothing in the output would show it.
# Scoring splits injected from held-out by parent id afterwards.
for dose in $DOSES; do
  ckpt="$OUT/dose${dose}"
  [[ -d "$ckpt" ]] || { echo "no checkpoint at $ckpt, skipping probe"; continue; }
  echo "===== PROBE dose=${dose}x ====="
  VLLM_USE_V1=0 VLLM_LOGGING_LEVEL=WARNING PYTHONPATH="$REPO/src:$HOME/arch" \
  "$VLLM_PY" experiments/vllm_generate.py \
    --benchmark "$RUN/base/gsm8k_test_deletion.jsonl" \
    --model "$ckpt" \
    --output "$RUN/generations/phase2__dose${dose}__test__T0.0__k1__vllm.jsonl" \
    --max-new-tokens 2048 --temperature 0.0 --n-shot 4 --no-chat-template \
    2>&1 | tee "$RUN/logs/phase2_probe_dose${dose}.log"
done

echo "===== phase2 go/no-go complete ====="
