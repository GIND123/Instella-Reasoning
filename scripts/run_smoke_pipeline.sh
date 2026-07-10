#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

instella-reasoning scan-contamination \
  --benchmark examples/mini_benchmark.jsonl \
  --corpus examples/mini_corpus.jsonl \
  --output outputs/mini_contamination.jsonl

instella-reasoning score-generations \
  --benchmark examples/mini_benchmark.jsonl \
  --generations examples/mini_generations.jsonl \
  --output outputs/mini_scores.jsonl

instella-reasoning attribute \
  --contamination outputs/mini_contamination.jsonl \
  --scores outputs/mini_scores.jsonl \
  --output outputs/mini_attribution.jsonl

instella-reasoning report \
  --scores outputs/mini_scores.jsonl \
  --contamination outputs/mini_contamination.jsonl \
  --output outputs/mini_report.md
