# Run log — 2026-07-17 CPU batch

Every CPU-runnable experiment on real (smoke-scale) data. Model generation excluded (GPU).
Env: Python 3.14 venv, real MiniLM embeddings, `--index-backend bruteforce` (faiss segfaults
locally). Seed 6198. Benchmarks capped at 200 items; corpus at 5,000 docs.

## Data prepared (Tier 0)
7 benchmarks pulled to `data/processed/`: gsm8k, math, logiqa2, arc_challenge, bbh, reclor
(200 each) + humaneval (164). Corpus: Instella-GSM8K-synthetic (5,000 docs).
Code fixes that unblocked this: corpus `messages` schema, logiqa2 nested-JSON schema, new
reclor + humaneval loaders (see git diff).

## A2/A3 — variant build + M2 validation (`variants/`)
All answer-preserving surface variants validated at **rate 1.000** (no answer drift):

| Benchmark | preserving | rate | numeric (answer-changing) | degenerate-text flags |
|-----------|-----------:|-----:|--------------------------:|----------------------:|
| gsm8k | 530 | 1.000 | 130 | 163 |
| math | 324 | 1.000 | 0 | 90 |
| logiqa2 | 461 | 1.000 | 0 | 17 |
| arc_challenge | 417 | 1.000 | 0 | 0 |
| bbh | 200 | 1.000 | 0 | 0 |
| reclor | 457 | 1.000 | 0 | 0 |

Interpretation: M2 cleared — a consistency drop downstream reflects the model, not broken
variants. Caveat: gsm8k/math carry many "degenerate-text" flags (163/90) — inspect before
the reliability run; some generated variants may read awkwardly even while answer-preserving.

## A1 — contamination scan (`contamination/`)
Real MiniLM, default C/PC/N thresholds, benchmark vs Instella-GSM8K-synthetic (5k):
- **gsm8k:** 6 hit-rows written; 0 contaminated / 2 partial / 198 not-detected. Top-1 cosine
  median 0.61, max 0.78; 13-gram overlap 0.0 across the board.
- **math:** 0 hits (expected — MATH problems are not in the GSM8K-synthetic corpus).

Interpretation: at this coverage (one source, 5k docs), GSM8K **test** items have no
near-duplicates — a legitimate lower bound. GSM8K item 0 is the *test*-split "Janet's ducks";
the corpus derives from GSM8K *train*, so verbatim overlap is not expected here.

## A4 — embedding attribution (`attribution/`)
`attribute-embedding`, gsm8k + math vs synthetic corpus: 200 profiles each. Per-item fields:
gini, concentration, near_duplicate_share, top1_cosine, source_shares, top_source. Feeds the
concentrated-vs-diverse verdict (aggregated per sub-skill at atlas time).

## Blocked here
- **calibrate-contamination / validate-reliability** — need a small hand-labeled set
  (paraphrase truth; genuine/fragile clusters). Not fabricated.
- **Everything generation-dependent** (A1 gap, A2/A3 reliability, A5/A6 scale/RL) — GPU, see
  `../../GPU_SUITE.md`.
- **clutrr, ttt_bench** — no accessible HF dataset.
