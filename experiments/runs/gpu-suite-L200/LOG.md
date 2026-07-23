# GPU-suite results — L200 (200 items/benchmark, 4-bit, bf16, Colab T4)

Generation + scoring for **4 models × 6 benchmarks × 200 items = 24 complete runs**
(verified 200/200 each). Raw `generations/` are gitignored (bulky, re-derivable); `scores/`
and `atlas/` are committed.

## Accuracy (exact-match, % correct)

| Model | gsm8k | math | logiqa2 | arc | bbh | reclor |
|---|---:|---:|---:|---:|---:|---:|
| AMD-OLMo-1B (base 1B) | 1.0 | 2.0 | 14.5 | 18.0 | 7.0 | 22.0 |
| Instella-3B (base) | 52.0 | 9.0 | 27.5 | 32.5 | 4.0 | 31.5 |
| Instella-3B-Instruct | **75.0** | **19.5** | **48.5** | **76.5** | 6.0 | **53.5** |
| Instella-3B-Math | 30.0 | 9.0 | 33.0 | 57.5 | **38.5** | 32.5 |

## What the numbers say (and don't)

- **Scale (A5, OLMo-1B → Instella-3B):** arithmetic is `emergent` (0.01 → 0.52 reliability);
  other skills `amplified`/`scale_resistant`. See `emergence.json`.
- **Instruction tuning:** Instella-3B-Instruct is the strongest general model — big jumps on
  gsm8k (75%), arc (76.5%), reclor (53.5%).
- **RL / math post-training (A6, Instruct → Math):** `emergence.json` shows **regression on
  most skills** (arithmetic −0.45, logic/reading −0.15 to −0.21) but a large **gain on
  multi_step/bbh (+0.325)**. Net: not math-domain-specific. **Caveat:** the Math checkpoint
  was loaded via the assembled HF copy and showed higher degeneracy; part of the gsm8k drop
  may be answer-format/extraction, not pure capability loss — needs a spot-check before this
  is stated as a finding.

## IMPORTANT LIMITATION — reliability is not yet measured (now encoded in the atlas)

The atlas reports **consistency = 1.000 for every cell**, so Reliability collapses to plain
Accuracy. That is because this run generated on the **base benchmarks only** — the
semantics-preserving **variant clusters were not generated**, so each "cluster" is a single
item and consistency is trivially perfect. The memorization-vs-reasoning signal (the study's
core contribution, A2/A3) is therefore **not yet tested**.

As of the atlas-honesty fix, every cell here is labelled **`ACCURACY-ONLY (consistency
untested)`** instead of GENUINE/FRAGILE — the atlas will no longer emit a reasoning verdict
from a base-only run (see `analysis/atlas.py::MIN_VARIANTS_FOR_CONSISTENCY`). Likewise
`emergence.json` now carries `consistency_probed: false`. To actually measure reliability,
run **`experiments/run_reliability_suite.sh`** (configured to target a 10-hour GPU budget):
it generates over the `make-variants` clusters and re-atlases, at which point the verdicts
become earned. Time its sanity pass before relying on the estimate.

## Contamination cross-reference is thin

Atlas cells are almost all `clean` because the only contamination scan available is the
smoke-scale gsm8k-vs-synthetic scan (0 contaminated / 2 partial / 198 clean). A real A1
accuracy-gap needs a fuller contamination index.

## Provenance
bf16 + 4-bit NF4 on a T4; transformers 4.57.x (pinned <5); seed 6198. Per-item precision is
recorded in each score row's `metadata`.
