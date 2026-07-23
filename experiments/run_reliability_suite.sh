#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 10-GPU-HOUR RELIABILITY SUITE — the run that actually measures reasoning.
#
# The base-benchmark suite (run_gpu_suite.sh) generates on ORIGINAL items only, so
# every cluster is a singleton and consistency is trivially 1.0 — the atlas can only
# report accuracy (it now labels those cells ACCURACY-ONLY). This suite closes that
# gap: it generates each model over the *variant clusters* (answer-preserving surface
# variants + GSM-Symbolic-style numeric variants), so consistency is genuinely probed
# and Reliability = Accuracy x Consistency becomes a real memorisation-vs-reasoning
# signal (the study's core contribution, A2/A3).
#
# It also builds a higher-coverage contamination index from the full
# Instella-GSM8K-synthetic training set, so A1 is no longer limited to the earlier
# 5,000-document smoke scan. Whether this produces contaminated test items is an
# empirical result, not an assumption.
#
# BUDGET: targets 10 T4-GPU-hours using the configurable estimate below. Actual time
# depends strongly on generated sequence lengths and hardware. All CPU stages are free
# of GPU time. Tune BASE_ITEMS / MODELS / BENCHMARKS to trade coverage for hours.
#
# Usage (Colab T4):   bash experiments/run_reliability_suite.sh
#   Override defaults with env vars, e.g.:
#     BASE_ITEMS=100 BENCHMARKS="gsm8k" bash experiments/run_reliability_suite.sh
#
# RESUMABLE: a per-(model,benchmark) score file is the completion marker; re-running
# skips finished pieces, so a reclaimed Colab VM just continues on the next cell run.
# ---------------------------------------------------------------------------
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- budget knobs -----------------------------------------------------------
# BASE_ITEMS originals per benchmark; each fans out to ~1 original + up to 4 answer-
# preserving surface variants + NUMERIC_K numeric variants (~10x). GSM8K is the
# headline; add math/logiqa2 only if you have budget to spare.
BASE_ITEMS="${BASE_ITEMS:-120}"
NUMERIC_K="${NUMERIC_K:-5}"
BENCHMARKS_STR="${BENCHMARKS:-gsm8k}"          # e.g. "gsm8k math" for two skills
read -r -a BENCHMARKS <<< "$BENCHMARKS_STR"
BATCH="${BATCH:-8}"
MAX_NEW="${MAX_NEW:-512}"
# Used only for the printed budget estimate. Override after timing the sanity pass.
SECONDS_PER_ITEM="${SECONDS_PER_ITEM:-6}"
# Contamination corpus: full synthetic set is ~small; cap for a first pass if needed.
CORPUS_LIMIT="${CORPUS_LIMIT:-0}"              # 0 = full Instella-GSM8K-synthetic

# --- Hugging Face result sync (periodic save + cross-machine resume) --------
# On start we PULL the run dir from HF so a teammate's / an earlier session's finished
# work is restored, then the per-benchmark score files act as completion markers and the
# suite skips them — no GPU hours are re-spent. After each (model,benchmark) and at the
# end we PUSH, so a reclaimed Colab VM never loses progress. Set HF_SYNC=0 to disable.
HF_SYNC="${HF_SYNC:-1}"
HF_RESULTS_REPO="${HF_RESULTS_REPO:-GOVINDFROM/Instella-Reasoning}"
HF_INCLUDE_GENERATIONS="${HF_INCLUDE_GENERATIONS:-0}"   # 1 = also upload bulky raw generations
_hf_gen_flag=""; [[ "$HF_INCLUDE_GENERATIONS" == "1" ]] && _hf_gen_flag="--include-generations"

_hf_pull() {
  [[ "$HF_SYNC" == "1" ]] || return 0
  python experiments/hf_sync.py pull --repo "$HF_RESULTS_REPO" --path "$OUT" || true
}
_hf_push() {  # $1 = commit message
  [[ "$HF_SYNC" == "1" ]] || return 0
  python experiments/hf_sync.py push --repo "$HF_RESULTS_REPO" --path "$OUT" \
    --message "${1:-checkpoint}" $_hf_gen_flag || true
}

# Models: scale + post-training axis. Instruct is the headline (also run in bf16 for
# the reportable accuracy number). Set SUITE_MODELS to a subset to split sessions.
_resolve_model() {
  case "$1" in
    olmo) echo amd/AMD-OLMo-1B ;; base) echo amd/Instella-3B ;;
    instruct) echo amd/Instella-3B-Instruct ;; math) echo models/Instella-3B-Math-hf ;;
    *) echo "$1" ;;
  esac
}
if [[ -n "${SUITE_MODELS:-}" ]]; then
  MODELS=(); for m in $SUITE_MODELS; do MODELS+=("$(_resolve_model "$m")"); done
else
  MODELS=(amd/AMD-OLMo-1B amd/Instella-3B amd/Instella-3B-Instruct models/Instella-3B-Math-hf)
fi

OUT="${OUT:-experiments/runs/reliability-B${BASE_ITEMS}-K${NUMERIC_K}}"
mkdir -p "$OUT"/{base,variants,generations,scores,atlas,contamination}
echo "Run dir: $OUT | base=$BASE_ITEMS numeric_k=$NUMERIC_K benchmarks=${BENCHMARKS[*]}"
echo "Models this session: ${MODELS[*]}"

# Restore any prior/teammate progress before doing anything, so resume can skip it.
echo "== Sync: pull prior results from $HF_RESULTS_REPO (HF_SYNC=$HF_SYNC) =="
_hf_pull

# --- rough GPU-hour estimate (printed, not enforced) ------------------------
python - "$BASE_ITEMS" "$NUMERIC_K" "${#MODELS[@]}" "${#BENCHMARKS[@]}" "$SECONDS_PER_ITEM" <<'PY'
import sys
base, k, n_models, n_bench = map(int, sys.argv[1:5])
seconds_per_item = float(sys.argv[5])
per_item = 1 + 4 + k                      # original + surface + numeric variants
gens = base * per_item * n_models * n_bench
hrs = gens * seconds_per_item / 3600
print(f"~{gens} generations -> ~{hrs:.1f} GPU-hours at the assumed "
      f"{seconds_per_item:g}s/item. Time the sanity pass and override SECONDS_PER_ITEM.")
if hrs > 9:
    print("WARNING: estimate exceeds ~9h — lower BASE_ITEMS or trim MODELS/BENCHMARKS.")
PY

# --- Stage 0: assemble Math checkpoint if it's in this session --------------
if [[ " ${MODELS[*]} " == *"Instella-3B-Math-hf"* ]]; then
  echo "== Stage 0a: assemble HF-loadable Instella-3B-Math =="
  python experiments/fetch_instella_math_hf.py
fi

# Build an immutable, run-local slice even when data/processed already contains a larger
# cached benchmark. This keeps BASE_ITEMS truthful and prevents silent budget overruns.
_prepare_base() {
  local b="$1"
  local source="data/processed/$b.jsonl"
  local target="$OUT/base/$b.jsonl"
  [[ -s "$source" ]] || instella-reasoning load-benchmark \
    --benchmark "$b" --output "$source" --limit "$BASE_ITEMS"
  if [[ ! -s "$target" ]]; then
    python - "$source" "$target" "$BASE_ITEMS" <<'PY'
import sys
from pathlib import Path

source, target, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(lines[:limit]) + "\n", encoding="utf-8")
if len(lines) < limit:
    print(f"WARNING: requested {limit} base items but {source} contains only {len(lines)}")
PY
  fi
}

# --- Stage 1: higher-coverage contamination index ---------------------------
# CPU/embedding only — no GPU time. The scan determines whether contaminated test items
# exist; it does not assume that full-corpus coverage guarantees positive hits.
mkdir -p data/processed
_prepare_base gsm8k
GSM_BASE="$OUT/base/gsm8k.jsonl"
CONTAM="$OUT/contamination/gsm8k_contam.jsonl"
if [[ ! -s "$CONTAM" ]]; then
  echo "== Stage 1: contamination index vs full Instella-GSM8K-synthetic =="
  corpus="$OUT/contamination/synthetic_corpus.jsonl"
  climit=""; [[ "$CORPUS_LIMIT" != "0" ]] && climit="--limit $CORPUS_LIMIT"
  [[ -s "$corpus" ]] || instella-reasoning load-corpus \
    --hf-path amd/Instella-GSM8K-synthetic --source instella-gsm8k-synthetic \
    --split train --output "$corpus" $climit
  # bruteforce = exact (faiss can segfault on some numpy builds); MiniLM is mandatory
  # for a real scan (the hashing fallback is semantic-blind — see WORKLOG §6).
  instella-reasoning scan-contamination-embedding \
    --benchmark "$GSM_BASE" --corpus "$corpus" --output "$CONTAM" \
    --embedder-backend sentence-transformers --index-backend bruteforce --top-k 20
fi

# --- Stage 2: base data + variant clusters (CPU) ----------------------------
echo "== Stage 2: benchmark data + variant clusters =="
mkdir -p data/processed
for b in "${BENCHMARKS[@]}"; do
  _prepare_base "$b"
  base="$OUT/base/$b.jsonl"
  var="$OUT/variants/${b}_variants.jsonl"
  if [[ ! -s "$var" ]]; then
    instella-reasoning make-variants --benchmark "$base" --output "$var" --numeric-k "$NUMERIC_K"
    # Audit: answer-preservation rate + well-formedness (reviewer concern M2).
    instella-reasoning validate-variants --benchmark "$var" \
      --output "$OUT/variants/${b}_validation.json"
  fi
done
# Checkpoint the CPU-side prep (contamination index + variant clusters) before any GPU work.
_hf_push "contamination + variants prepared"

# --- Stage 3: generate + score over the VARIANT clusters --------------------
# NOTE: both generate AND score use the *variant* file, so parent_id / variant_type /
# answer_changing survive into the scores -> the atlas can cluster and probe consistency.
for model in "${MODELS[@]}"; do
  mtag="${model##*/}"
  merged="$OUT/scores/${mtag}__ALL.jsonl"
  for b in "${BENCHMARKS[@]}"; do
    var="$OUT/variants/${b}_variants.jsonl"
    [[ -s "$var" ]] || { echo "skip $b (no variants)"; continue; }
    gen="$OUT/generations/${mtag}__${b}.jsonl"
    sco="$OUT/scores/${mtag}__${b}.jsonl"
    if [[ -s "$sco" ]]; then echo "== skip $mtag / $b (already scored) =="; continue; fi
    # Headline accuracy fidelity: primary model on the primary benchmark in bf16;
    # everything else 4-bit to stay in budget (precision is recorded per row).
    prec="--load-in-4bit"
    [[ "$mtag" == "Instella-3B-Instruct" && "$b" == "${BENCHMARKS[0]}" ]] && prec=""
    echo "== generate $mtag / $b (${prec:-bf16}) =="
    instella-reasoning generate --benchmark "$var" --model "$model" \
      --output "$gen" $prec --batch-size "$BATCH" --max-new-tokens "$MAX_NEW" \
      --fail-degenerate 0.25
    instella-reasoning score-generations --benchmark "$var" \
      --generations "$gen" --output "$sco"
    # Periodic save: checkpoint each finished (model,benchmark) so a VM reclaim loses nothing.
    _hf_push "scores ${mtag}/${b}"
  done
  : > "$merged"
  for b in "${BENCHMARKS[@]}"; do
    sco="$OUT/scores/${mtag}__${b}.jsonl"; [[ -s "$sco" ]] && cat "$sco" >> "$merged"
  done
  # Atlas now has real clusters -> consistency is probed -> GENUINE/FRAGILE are earned.
  instella-reasoning atlas --scores "$merged" --contamination "$CONTAM" \
    --output "$OUT/atlas/${mtag}_atlas.json" --markdown "$OUT/atlas/${mtag}_atlas.md" \
    || echo "  (atlas skipped)"
  _hf_push "atlas ${mtag}"
done

# --- Stage 4: headline analyses (CPU) ---------------------------------------
INSTRUCT="$OUT/scores/Instella-3B-Instruct__ALL.jsonl"
if [[ -s "$INSTRUCT" ]]; then
  echo "== A1: difficulty-adjusted, FDR-corrected contamination gap =="
  instella-reasoning accuracy-gap --scores "$INSTRUCT" --contamination "$CONTAM" \
    --benchmark "$OUT/base/${BENCHMARKS[0]}.jsonl" \
    --output "$OUT/accuracy_gap.json" \
    --stratified-output "$OUT/accuracy_gap_stratified.json" || echo "  (gap skipped)"
  echo "== report + figures =="
  instella-reasoning report --scores "$INSTRUCT" --contamination "$CONTAM" \
    --output "$OUT/report.md" || echo "  (report skipped)"
  instella-reasoning plots --scores "$INSTRUCT" --contamination "$CONTAM" \
    --output-dir "$OUT/figures" || echo "  (figures need the viz extra)"
fi

echo "== A5 scale + A6 RL emergence (now on real reliability) =="
instella-reasoning emergence \
  --small-scores "$OUT/scores/AMD-OLMo-1B__ALL.jsonl" --small-name OLMo-1B \
  --large-scores "$OUT/scores/Instella-3B__ALL.jsonl" --large-name Instella-3B \
  --contamination "$CONTAM" --output "$OUT/emergence_scale.json" \
  || echo "  (needs both score files)"
instella-reasoning emergence \
  --small-scores "$OUT/scores/Instella-3B-Instruct__ALL.jsonl" \
  --large-scores "$OUT/scores/Instella-3B-Math-hf__ALL.jsonl" \
  --small-name Instella-3B-Instruct --large-name Instella-3B-Math \
  --pre-rl-scores "$OUT/scores/Instella-3B-Instruct__ALL.jsonl" \
  --post-rl-scores "$OUT/scores/Instella-3B-Math-hf__ALL.jsonl" \
  --contamination "$CONTAM" --output "$OUT/emergence_rl.json" \
  || echo "  (needs both score files)"

# Final save: whole run dir (atlas, gap, report, figures, emergence) to HF.
_hf_push "final: atlas + gap + report + emergence"

echo "Done. Reliability artifacts in $OUT/  (atlas verdicts are now consistency-probed)."
[[ "$HF_SYNC" == "1" ]] && echo "Saved to https://huggingface.co/datasets/$HF_RESULTS_REPO (path: $OUT)."
