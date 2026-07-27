#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FULL-SCALE MEMORISATION SUITE — one run, ~13-15 GPU hours.
#
# WHAT THIS MEASURES
#   Does the model do better on problems it PROVABLY memorised?
#
#   GSM8K *train* items are verbatim in Instella's stage-2 data; GSM8K *test* items are
#   not. Membership is VERIFIED per item by exact 13-gram containment against the corpus,
#   not inferred from an embedding proxy. Cross that with numeric perturbation and the
#   difference-in-differences
#
#       DiD = (seen - unseen | original) - (seen - unseen | perturbed)
#
#   is the share of the seen-item advantage that does not survive perturbation: the
#   memorisation component. Because the same perturbation distribution is applied to both
#   arms, the integer-magnitude confound (arXiv:2605.28700) cancels in the double difference.
#
#   The model axis is the full Instella-3B trajectory. Stage1 -> Stage2 is the step that
#   introduces GSM8K-derived data, so it is a controlled data intervention rather than an
#   observational comparison.
#
# RESUMABLE BY DESIGN
#   Every stage writes a marker file; every (model, block) writes a score file. On start
#   the run PULLS from HF and skips anything already finished, so a reclaimed Colab VM
#   just continues. Generations are uploaded too — a scoring bug found later can then be
#   fixed WITHOUT re-spending GPU hours (this is not hypothetical; it cost a whole run).
#
#   Existing HF results are protected: hf_sync.py refuses to write the reliability-B* runs.
#
# USAGE (Colab)
#   bash experiments/run_fullscale_suite.sh
#   N_PER_ARM=60 SUITE_TIER=1 bash experiments/run_fullscale_suite.sh    # short rehearsal
#
# RUN PREFLIGHT FIRST. It is the gate that protects the budget:
#   python scripts/preflight_fullscale.py
# ---------------------------------------------------------------------------
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- budget knobs -----------------------------------------------------------
N_PER_ARM="${N_PER_ARM:-250}"          # verified items per arm (seen / unseen)
SURFACE_K="${SURFACE_K:-2}"            # answer-preserving variants per item
NUMERIC_K="${NUMERIC_K:-2}"            # answer-changing numeric variants per item
GSMSYM_PER_TEMPLATE="${GSMSYM_PER_TEMPLATE:-3}"   # official GSM-Symbolic instances/template
RESAMPLE_ITEMS="${RESAMPLE_ITEMS:-100}"           # decoding-noise control items
RESAMPLE_N="${RESAMPLE_N:-5}"                     # samples per item at T>0
RESAMPLE_TEMP="${RESAMPLE_TEMP:-0.7}"
# The decoding-noise control is a *null* for the consistency comparison, not a headline
# measurement, so it only needs the checkpoints whose consistency is actually being
# interpreted. Running it on all four costs ~1.2 GPU-hours and answers nothing extra.
RESAMPLE_MODELS="${RESAMPLE_MODELS:-instruct math}"
BATCH="${BATCH:-8}"
SUITE_TIER="${SUITE_TIER:-1}"          # 1 = core 4 checkpoints, 2 = all 6
CORPUS_LIMIT="${CORPUS_LIMIT:-0}"      # 0 = full Instella-GSM8K-synthetic
MIN_TERMINATION="${MIN_TERMINATION:-0.85}"
SEED="${SEED:-6198}"

# ICC on this project's pilot data is ~0.48, so the 5th variant of an item is worth ~7%
# of a fresh item. Cluster size stays small (1 original + 2 surface + 2 numeric) and the
# budget goes into MORE ITEMS. Raising SURFACE_K/NUMERIC_K is almost always the wrong trade.

OUT="${OUT:-experiments/runs/fullscale-S${N_PER_ARM}}"
HF_SYNC="${HF_SYNC:-1}"
HF_RESULTS_REPO="${HF_RESULTS_REPO:-GOVINDFROM/Instella-Reasoning}"

mkdir -p "$OUT"/{base,arms,variants,generations,scores,atlas,contamination,analysis,figures,review,markers}

_hf_pull() { [[ "$HF_SYNC" == "1" ]] || return 0
  python experiments/hf_sync.py pull --repo "$HF_RESULTS_REPO" --path "$OUT" || true; }
_hf_push() { [[ "$HF_SYNC" == "1" ]] || return 0
  python experiments/hf_sync.py push --repo "$HF_RESULTS_REPO" --path "$OUT" --message "${1:-checkpoint}" || true; }
_done()  { touch "$OUT/markers/$1"; }
_todo()  { [[ ! -f "$OUT/markers/$1" ]]; }

echo "=============================================================="
echo " Full-scale memorisation suite"
echo " run dir : $OUT"
echo " arms    : ${N_PER_ARM} seen + ${N_PER_ARM} unseen (verified)"
echo " cluster : 1 original + ${SURFACE_K} surface + ${NUMERIC_K} numeric"
echo " tier    : ${SUITE_TIER}"
echo "=============================================================="

echo "== Sync: restore prior progress (HF_SYNC=$HF_SYNC) =="
_hf_pull

# Resolve the checkpoint list from the registry (single source of truth for token
# budgets, few-shot policy, and assembly paths).
mapfile -t MODEL_TAGS < <(python - "$SUITE_TIER" <<'PY'
import sys
from instella_reasoning.checkpoints import tier
for c in tier(int(sys.argv[1])):
    print(c.tag)
PY
)
echo "Checkpoints this run: ${MODEL_TAGS[*]}"

_ckpt_field() {  # $1 = tag, $2 = field
  python - "$1" "$2" <<'PY'
import sys
from instella_reasoning.checkpoints import resolve
c = resolve(sys.argv[1])
print(getattr(c, sys.argv[2]))
PY
}

# --------------------------------------------------------------------------
# STAGE 1 — corpus + contamination index (CPU only, no GPU time)
# --------------------------------------------------------------------------
CORPUS="$OUT/contamination/synthetic_corpus.jsonl"
if _todo stage1_corpus; then
  echo "== Stage 1: load Instella-GSM8K-synthetic (the seen-arm ground truth) =="
  climit=""; [[ "$CORPUS_LIMIT" != "0" ]] && climit="--limit $CORPUS_LIMIT"
  [[ -s "$CORPUS" ]] || instella-reasoning load-corpus \
    --hf-path amd/Instella-GSM8K-synthetic --source instella-gsm8k-synthetic \
    --split train --output "$CORPUS" $climit
  _done stage1_corpus
  _hf_push "stage1: corpus"
fi

# --------------------------------------------------------------------------
# STAGE 2 — verified seen/unseen arms
# --------------------------------------------------------------------------
if _todo stage2_splits; then
  echo "== Stage 2: verified seen/unseen arms =="
  [[ -s "$OUT/base/gsm8k_train.jsonl" ]] || instella-reasoning load-benchmark \
    --benchmark gsm8k_train --output "$OUT/base/gsm8k_train.jsonl" --limit $((N_PER_ARM * 4))
  [[ -s "$OUT/base/gsm8k_test.jsonl" ]] || instella-reasoning load-benchmark \
    --benchmark gsm8k --output "$OUT/base/gsm8k_test.jsonl" --limit $((N_PER_ARM * 4))

  instella-reasoning build-splits \
    --train "$OUT/base/gsm8k_train.jsonl" --test "$OUT/base/gsm8k_test.jsonl" \
    --corpus "$CORPUS" --output "$OUT/arms/gsm8k_arms.jsonl" \
    --containment-output "$OUT/analysis/containment.json" \
    --n-per-arm "$N_PER_ARM" --seed "$SEED"
  _done stage2_splits
  _hf_push "stage2: verified arms"
fi

# --------------------------------------------------------------------------
# STAGE 3 — variant clusters + magnitude gate + hand-review gate
# --------------------------------------------------------------------------
VARIANTS="$OUT/variants/arms_variants.jsonl"
if _todo stage3_variants; then
  echo "== Stage 3: variant clusters =="
  instella-reasoning make-variants \
    --benchmark "$OUT/arms/gsm8k_arms.jsonl" --output "$OUT/variants/_raw.jsonl" \
    --numeric-k "$NUMERIC_K" --max-variants "$SURFACE_K" --seed "$SEED"
  instella-reasoning validate-variants \
    --benchmark "$OUT/variants/_raw.jsonl" --output "$OUT/variants/validation.json"

  # Magnitude gate: refuse to spend GPU hours on a perturbation that just made the
  # arithmetic bigger. The previous generator produced a median answer ratio of 2.61x.
  python - "$OUT/variants/_raw.jsonl" <<'PY'
import json, sys
from instella_reasoning.gsm_symbolic import magnitude_report
from instella_reasoning.records import read_benchmark
report = magnitude_report(read_benchmark(sys.argv[1]))
print("magnitude audit:", json.dumps(report.to_dict()))
if not report.in_band:
    raise SystemExit(
        "ABORT: numeric variants are not magnitude-neutral. The measured effect would be "
        "a bigger-arithmetic effect, not a reasoning effect (arXiv:2605.28700)."
    )
PY

  # Hand-verification harness. GSM-Symbolic hand-annotated all 100 of its templates; the
  # auto-derived ones get the same treatment. REVIEW_CSV, when present, gates the run.
  instella-reasoning export-templates \
    --variants "$OUT/variants/_raw.jsonl" --output "$OUT/review/templates_to_review.csv"
  REVIEW_CSV="${REVIEW_CSV:-$OUT/review/templates_reviewed.csv}"
  if [[ -s "$REVIEW_CSV" ]]; then
    instella-reasoning apply-template-review \
      --variants "$OUT/variants/_raw.jsonl" --review "$REVIEW_CSV" \
      --output "$VARIANTS" ${REQUIRE_REVIEW:+--require-complete}
  else
    echo "  NOTE: no reviewed CSV at $REVIEW_CSV — using auto-derived templates as-is."
    echo "        Review $OUT/review/templates_to_review.csv and re-run to apply verdicts."
    cp "$OUT/variants/_raw.jsonl" "$VARIANTS"
  fi
  _done stage3_variants
  _hf_push "stage3: variants"
fi

# --------------------------------------------------------------------------
# STAGE 4 — official GSM-Symbolic + decoding-noise control
# --------------------------------------------------------------------------
GSMSYM="$OUT/variants/gsm_symbolic_official.jsonl"
RESAMPLE="$OUT/variants/resample_control.jsonl"
if _todo stage4_aux; then
  echo "== Stage 4: official GSM-Symbolic (hand-written templates) + noise control =="
  for cfg in gsm_symbolic gsm_symbolic_p1; do
    f="$OUT/variants/${cfg}.jsonl"
    [[ -s "$f" ]] || instella-reasoning load-benchmark --benchmark "$cfg" \
      --output "$f" --max-per-parent "$GSMSYM_PER_TEMPLATE"
  done
  cat "$OUT/variants/gsm_symbolic.jsonl" "$OUT/variants/gsm_symbolic_p1.jsonl" > "$GSMSYM"

  instella-reasoning make-resample-suite \
    --benchmark "$OUT/arms/gsm8k_arms.jsonl" --output "$RESAMPLE" \
    --n-samples "$RESAMPLE_N" --limit "$RESAMPLE_ITEMS"
  _done stage4_aux
  _hf_push "stage4: gsm-symbolic + noise control"
fi

# --------------------------------------------------------------------------
# STAGE 5 — generation + scoring (the GPU stage)
# --------------------------------------------------------------------------
# Blocks: name | benchmark file | temperature. The score file is the completion marker.
run_block() {   # $1 tag  $2 block  $3 benchmark  $4 temperature
  local tag="$1" block="$2" bench="$3" temp="$4"
  local sco="$OUT/scores/${tag}__${block}.jsonl"
  local gen="$OUT/generations/${tag}__${block}.jsonl"
  [[ -s "$bench" ]] || { echo "  skip $block (no input)"; return 0; }
  if [[ -s "$sco" ]]; then echo "  == skip $tag/$block (already scored) =="; return 0; fi

  local model max_new n_shot is_base chat_flag
  model="$(_ckpt_field "$tag" load_path)"
  max_new="$(_ckpt_field "$tag" max_new_tokens)"
  n_shot="$(_ckpt_field "$tag" n_shot)"
  is_base="$(_ckpt_field "$tag" is_base)"
  chat_flag=""; [[ "$is_base" == "True" ]] && chat_flag="--no-chat-template"

  echo "  == generate $tag/$block  (model=$model tokens=$max_new shot=$n_shot T=$temp) =="
  instella-reasoning generate --benchmark "$bench" --model "$model" --output "$gen" \
    --batch-size "$BATCH" --max-new-tokens "$max_new" --n-shot "$n_shot" \
    --temperature "$temp" $chat_flag \
    --min-termination-rate "$MIN_TERMINATION" --fail-degenerate 1.0
  instella-reasoning score-generations --benchmark "$bench" --generations "$gen" --output "$sco"
  _hf_push "scores ${tag}/${block}"
}

for tag in "${MODEL_TAGS[@]}"; do
  echo "== Checkpoint: $tag =="
  # Assemble a local HF-loadable copy when the Hub repo ships vLLM-only remote code.
  assembly="$(_ckpt_field "$tag" local_assembly)"
  if [[ "$assembly" != "None" && ! -d "$assembly" ]]; then
    echo "  assembling $assembly"
    python experiments/fetch_instella_math_hf.py --out "$assembly" \
      || echo "  (assembly failed; skipping $tag)"
  fi

  run_block "$tag" arms   "$VARIANTS" 0.0
  run_block "$tag" gsmsym "$GSMSYM"   0.0
  if [[ " $RESAMPLE_MODELS " == *" $tag "* ]]; then
    run_block "$tag" resample "$RESAMPLE" "$RESAMPLE_TEMP"
  else
    echo "  == skip $tag/resample (not in RESAMPLE_MODELS) =="
  fi

  merged="$OUT/scores/${tag}__ALL.jsonl"
  : > "$merged"
  for block in arms gsmsym resample; do
    f="$OUT/scores/${tag}__${block}.jsonl"; [[ -s "$f" ]] && cat "$f" >> "$merged"
  done
  _hf_push "merged ${tag}"
done

# --------------------------------------------------------------------------
# STAGE 6 — measurement-validity audit (before any number is believed)
# --------------------------------------------------------------------------
echo "== Stage 6: termination audit =="
instella-reasoning check-termination --generations "$OUT"/generations/*.jsonl \
  --output "$OUT/analysis/termination.json" --min-rate "$MIN_TERMINATION" \
  || echo "  WARNING: at least one model was truncated — see analysis/termination.json"

# --------------------------------------------------------------------------
# STAGE 7 — headline analysis
# --------------------------------------------------------------------------
echo "== Stage 7: memorisation DiD + cluster-robust regression =="
instella-reasoning memorization --scores "$OUT"/scores/*__ALL.jsonl \
  --output "$OUT/analysis/memorization.json"

echo "== Stage 7b: atlas per checkpoint =="
CONTAM="$OUT/contamination/gsm8k_contam.jsonl"
[[ -s "$CONTAM" ]] || echo '' > "$CONTAM"
for tag in "${MODEL_TAGS[@]}"; do
  merged="$OUT/scores/${tag}__ALL.jsonl"
  [[ -s "$merged" ]] || continue
  instella-reasoning atlas --scores "$merged" --contamination "$CONTAM" \
    --output "$OUT/atlas/${tag}_atlas.json" --markdown "$OUT/atlas/${tag}_atlas.md" \
    || echo "  (atlas skipped for $tag)"
done

# --------------------------------------------------------------------------
# STAGE 8 — figures
# --------------------------------------------------------------------------
echo "== Stage 8: publication figures =="
instella-reasoning figures --scores "$OUT"/scores/*__ALL.jsonl \
  --analysis "$OUT/analysis/memorization.json" --output-dir "$OUT/figures" \
  --containment "$OUT/analysis/containment.json" \
  --termination "$OUT/analysis/termination.json" --n-items "$N_PER_ARM" \
  || echo "  (figures need the viz extra: pip install -e '.[viz]')"

_hf_push "final: analysis + atlas + figures"
echo
echo "Done. Artifacts in $OUT/"
echo "  analysis/memorization.json   <- the headline DiD"
echo "  analysis/termination.json    <- measurement validity"
echo "  analysis/containment.json    <- treatment-assignment evidence"
echo "  figures/                     <- f1..f8"
[[ "$HF_SYNC" == "1" ]] && echo "Saved to https://huggingface.co/datasets/$HF_RESULTS_REPO (path: $OUT)"
