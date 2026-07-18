#!/usr/bin/env bash
# Turnkey CPU experiment suite: every experiment stage that runs without a GPU.
# Mirrors the reference run in experiments/runs/2026-07-17_cpu-batch/ (same seed, same
# limits, same backends) so a re-run elsewhere (e.g. Colab) can be diffed against it with
# experiments/compare_runs.py. See experiments/README.md for the plan this serves.
#
# Stages:
#   0. download benchmarks + contamination corpus  (skipped for files already present)
#   1. A2/A3  make-variants + validate-variants    (all answer-bearing benchmarks)
#   2. A1     scan-contamination-embedding         (gsm8k, math vs GSM8K-synthetic corpus)
#   3. A4     attribute-embedding                  (gsm8k, math vs GSM8K-synthetic corpus)
#
# Usage:  bash experiments/run_cpu_suite.sh [BENCH_LIMIT] [CORPUS_LIMIT] [RUN_DIR]
#   defaults: 200 5000 experiments/runs/<date>_cpu-suite
#   Use the same limits as the reference run (200 5000) for a comparable re-run.
#
# NOTE on faiss: some local stacks (Python 3.14 / numpy 2.5) segfault inside faiss, and the
# scan then fails silently with an empty output. We pin --index-backend bruteforce (exact
# search, same results as faiss-flat) for portability and comparability.
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false

BENCH_LIMIT="${1:-200}"
CORPUS_LIMIT="${2:-5000}"
RUN="${3:-experiments/runs/$(date +%F)_cpu-suite}"
SEED=6198
EMB=(--embedder-backend sentence-transformers --index-backend bruteforce)

BENCHMARKS=(gsm8k math logiqa2 arc_challenge bbh reclor)   # answer-scored
CONTAM_BENCHMARKS=(gsm8k math)                              # scanned vs the math corpus
CORPUS="data/processed/instella_gsm8k_synth.jsonl"

mkdir -p "$RUN"/{variants,contamination,attribution} data/processed
blim=""; [[ "$BENCH_LIMIT" != "0" ]] && blim="--limit $BENCH_LIMIT"
clim=""; [[ "$CORPUS_LIMIT" != "0" ]] && clim="--limit $CORPUS_LIMIT"

echo "== Stage 0: data =="
for b in "${BENCHMARKS[@]}" humaneval; do
  if [[ ! -s "data/processed/$b.jsonl" ]]; then
    instella-reasoning load-benchmark --benchmark "$b" --output "data/processed/$b.jsonl" $blim
  fi
done
if [[ ! -s "$CORPUS" ]]; then
  instella-reasoning load-corpus --hf-path amd/Instella-GSM8K-synthetic \
    --source instella-gsm8k-synthetic --output "$CORPUS" $clim
fi

echo "== Stage 1: variants + M2 validation (A2/A3) =="
for b in "${BENCHMARKS[@]}"; do
  nk=0; { [[ "$b" == "gsm8k" ]] || [[ "$b" == "math" ]]; } && nk=3
  instella-reasoning make-variants --benchmark "data/processed/$b.jsonl" \
    --output "$RUN/variants/${b}_variants.jsonl" --numeric-k "$nk" --seed "$SEED"
  instella-reasoning validate-variants --benchmark "$RUN/variants/${b}_variants.jsonl" \
    --output "$RUN/variants/${b}_validation.json"
done

echo "== Stage 2: contamination scan (A1) =="
for b in "${CONTAM_BENCHMARKS[@]}"; do
  instella-reasoning scan-contamination-embedding \
    --benchmark "data/processed/$b.jsonl" --corpus "$CORPUS" \
    --output "$RUN/contamination/${b}_contam.jsonl" "${EMB[@]}"
done

echo "== Stage 3: embedding attribution (A4) =="
for b in "${CONTAM_BENCHMARKS[@]}"; do
  instella-reasoning attribute-embedding \
    --benchmark "data/processed/$b.jsonl" --corpus "$CORPUS" \
    --output "$RUN/attribution/${b}_attrib.jsonl" "${EMB[@]}"
done

echo ""
echo "Done. Artifacts in $RUN/"
echo "Compare against the reference run with:"
echo "  python experiments/compare_runs.py experiments/runs/2026-07-17_cpu-batch $RUN"
