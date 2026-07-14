#!/usr/bin/env bash
# Download reasoning benchmarks and Instella training-corpus shards from HuggingFace
# into the project's JSONL contract. Requires the `hf` extra:  pip install -e ".[hf]"
#
# Usage:
#   bash scripts/download_data.sh [BENCH_LIMIT] [CORPUS_LIMIT]
#
# The limits keep a first pass small (a Colab T4 smoke run). Drop them (pass 0) for a
# full download. See docs/DATA_DOWNLOAD.md for the complete AMD data procedure and
# licensing notes.
set -euo pipefail

BENCH_LIMIT="${1:-200}"
CORPUS_LIMIT="${2:-5000}"
PROC="data/processed"
mkdir -p "$PROC"

# Work around HF Xet backend 401s (cas-server.xethub.hf.co) on some datasets by
# falling back to plain HTTPS LFS downloads. Set HF_TOKEN in the environment for
# higher rate limits / gated datasets: `export HF_TOKEN=hf_xxx`.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

bench_arg=""; [[ "$BENCH_LIMIT" != "0" ]] && bench_arg="--limit $BENCH_LIMIT"
corpus_arg=""; [[ "$CORPUS_LIMIT" != "0" ]] && corpus_arg="--limit $CORPUS_LIMIT"

echo "== Reasoning benchmarks =="
instella-reasoning load-benchmark --benchmark gsm8k          --output "$PROC/gsm8k.jsonl"          $bench_arg
instella-reasoning load-benchmark --benchmark arc_challenge  --output "$PROC/arc_challenge.jsonl"  $bench_arg
instella-reasoning load-benchmark --benchmark math           --output "$PROC/math.jsonl"           $bench_arg || \
  echo "  (MATH config may need --hf-name; skipping is non-fatal)"

echo "== Instella training corpus (contamination ground truth) =="
# Smoking gun for GSM8K: index the AMD synthetic set fully where possible.
instella-reasoning load-corpus \
  --hf-path amd/Instella-GSM8K-synthetic \
  --source instella-gsm8k-synthetic \
  --output "$PROC/instella_gsm8k_synth.jsonl" $corpus_arg || \
  echo "  (Instella-GSM8K-synthetic download failed; set HF_TOKEN and retry. Non-fatal.)"

# Instella-Math SFT source (paraphrase contamination pathway).
instella-reasoning load-corpus \
  --hf-path nvidia/OpenMathInstruct-2 \
  --source openmathinstruct-2 \
  --output "$PROC/openmathinstruct2.jsonl" $corpus_arg || \
  echo "  (OpenMathInstruct-2 is large; consider a smaller --limit)"

echo "== Done. Processed files in $PROC/ =="
ls -la "$PROC"
