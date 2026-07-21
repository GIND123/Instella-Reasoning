#!/usr/bin/env bash
# Turnkey GPU suite: the parts of the experiment plan that need a T4 (model generation).
# Runs generate -> score -> atlas for every (model x benchmark), then emergence across
# models. CPU-side stages (contamination, variants, attribution) are done in
# experiments/runs/ and reused here. See experiments/GPU_SUITE.md.
#
# Usage (Colab T4):   bash experiments/run_gpu_suite.sh [LIMIT] [RUN_DIR]
#   LIMIT caps items per benchmark for a fast pass (default 200; use 0 for full).
#
# RESUMABLE: a per-(model,benchmark) score file is the completion marker. Re-running
# skips any (model,benchmark) already scored, so an interrupted run (Colab reclaim, VM
# recycle) just continues where it left off — re-run the same cell and it fast-forwards.
# The run dir is fixed per LIMIT (not date-stamped) so resuming across days finds it.
set -euo pipefail
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # stream generation logs so a live run never looks "stopped"

LIMIT="${1:-200}"
lim=""; [[ "$LIMIT" != "0" ]] && lim="--limit $LIMIT"

OUT="${2:-experiments/runs/gpu-suite-L${LIMIT}}"
mkdir -p "$OUT"/{generations,scores,atlas}
echo "Run dir: $OUT (resumable — already-scored model/benchmarks are skipped)"

# Answer-scored benchmarks (humaneval is contamination-only -> excluded from generate/score).
BENCHMARKS=(gsm8k math logiqa2 arc_challenge bbh reclor)
# Models in scale order; 4-bit fits a T4. OLMo-1B is the attribution-validation baseline.
# Instella-3B-Math is used via a local HF-loadable copy: its HF repo ships vLLM-only
# remote code, so fetch_instella_math_hf.py swaps in the Instruct repo's modeling file
# (checkpoints are tensor-name-identical; see that script's docstring).
MODELS=(amd/AMD-OLMo-1B amd/Instella-3B amd/Instella-3B-Instruct models/Instella-3B-Math-hf)

echo "== Stage 0a: assemble HF-loadable Instella-3B-Math =="
python experiments/fetch_instella_math_hf.py

echo "== Stage 0: benchmark data (download any missing) =="
mkdir -p data/processed
for b in "${BENCHMARKS[@]}"; do
  [[ -s "data/processed/$b.jsonl" ]] || \
    instella-reasoning load-benchmark --benchmark "$b" --output "data/processed/$b.jsonl" $lim
done

CONTAM="experiments/runs/2026-07-17_cpu-batch/contamination/gsm8k_contam.jsonl"

for model in "${MODELS[@]}"; do
  mtag="${model##*/}"
  merged="$OUT/scores/${mtag}__ALL.jsonl"
  for b in "${BENCHMARKS[@]}"; do
    bench="data/processed/$b.jsonl"
    [[ -f "$bench" ]] || { echo "skip $b (no data — run scripts/download_data.sh)"; continue; }
    gen="$OUT/generations/${mtag}__${b}.jsonl"
    sco="$OUT/scores/${mtag}__${b}.jsonl"
    if [[ -s "$sco" ]]; then
      echo "== skip $mtag / $b (already scored) =="   # resume marker
      continue
    fi
    echo "== generate $mtag / $b =="
    instella-reasoning generate --benchmark "$bench" --model "$model" \
      --output "$gen" --load-in-4bit --batch-size 8 --max-new-tokens 512 $lim
    instella-reasoning score-generations --benchmark "$bench" \
      --generations "$gen" --output "$sco"
  done
  # Rebuild the merged score file from whatever per-benchmark files exist (resume-safe).
  : > "$merged"
  for b in "${BENCHMARKS[@]}"; do
    sco="$OUT/scores/${mtag}__${b}.jsonl"
    [[ -s "$sco" ]] && cat "$sco" >> "$merged"
  done
  # Per-model Atlas from that model's merged scores + the CPU-side contamination hits.
  instella-reasoning atlas --scores "$merged" --contamination "$CONTAM" \
    --output "$OUT/atlas/${mtag}_atlas.json" --markdown "$OUT/atlas/${mtag}_atlas.md" \
    || echo "  (atlas skipped)"
done

# A5 scale emergence (OLMo-1B vs Instella-3B) and A6 RL effect (3B-Instruct vs 3B-Math).
echo "== A5 scale emergence =="
instella-reasoning emergence \
  --small-scores "$OUT/scores/AMD-OLMo-1B__ALL.jsonl" --small-name OLMo-1B \
  --large-scores "$OUT/scores/Instella-3B__ALL.jsonl" --large-name Instella-3B \
  --output "$OUT/emergence_scale.json" || echo "  (needs both score files)"
echo "== A6 RL effect =="
instella-reasoning emergence \
  --pre-rl-scores  "$OUT/scores/Instella-3B-Instruct__ALL.jsonl" \
  --post-rl-scores "$OUT/scores/Instella-3B-Math-hf__ALL.jsonl" \
  --output "$OUT/emergence_rl.json" || echo "  (needs both score files)"

echo "Done. Artifacts in $OUT/"
