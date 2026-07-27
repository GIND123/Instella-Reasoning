# Experiment status — done vs. needed (handoff)

Two runs exist. One is **wide but shallow** (accuracy across 6 benchmarks), the other is
**narrow but deep** (reliability = accuracy × consistency, gsm8k only). Scoping up = making
the deep run cover the other benchmarks.

## ✅ Done

### 1. GPU suite — breadth (accuracy only) · `runs/gpu-suite-L200/`
4 models × **6 benchmarks** × 200 items. Accuracy %:

| Model | gsm8k | math | logiqa2 | arc | bbh | reclor |
|---|---:|---:|---:|---:|---:|---:|
| AMD-OLMo-1B | 1.0 | 2.0 | 14.5 | 18.0 | 7.0 | 22.0 |
| Instella-3B | 52.0 | 9.0 | 27.5 | 32.5 | 4.0 | 31.5 |
| Instella-3B-Instruct | 75.0 | 19.5 | 48.5 | 76.5 | 6.0 | 53.5 |
| Instella-3B-Math | 30.0 | 9.0 | 33.0 | 57.5 | 38.5 | 32.5 |

- No consistency (base benchmarks only → every cluster singleton → consistency trivially 1.0).
- **Stored:** git (`main`) + Google Drive (`instella-gpu-suite-L200`). **Not on HF.**

### 2. Reliability suite — depth (accuracy × consistency) · `runs/reliability-B120-K5-clean/`
4 models × **gsm8k only** × 120 items × variant clusters. Consistency genuinely probed.

| Model | Accuracy | Consistency | Reliability |
|---|---:|---:|---:|
| AMD-OLMo-1B | 0.01 | 0.51 | 0.003 |
| Instella-3B | 0.50 | 0.69 | 0.43 |
| Instella-3B-Instruct | 0.66 | 0.79 | 0.59 |
| Instella-3B-Math | 0.23 | 0.49 | 0.13 |

- A5 scale (1B→3B): reliability 0.004 → 0.41 (`amplified`).
- A6 RL (Instruct→Math): reliability 0.59 → 0.13 (`regression`).
- Run fully in **bf16** (not the script's mixed bf16/4-bit — 4-bit is broken on the Blackwell GPU).
- **Stored:** HF `GOVINDFROM/Instella-Reasoning` → `reliability-B120-K5-clean`, git branch `results/reliability-suite-B120`.

## ❌ Needed to scope up (in priority order)

1. **Reliability on the other 5 benchmarks** — math, logiqa2, arc_challenge, bbh, reclor currently
   have accuracy but **no consistency**. Run the reliability suite with
   `BENCHMARKS="gsm8k math logiqa2 arc_challenge bbh reclor"` so every accuracy cell above gets a
   reliability number → a full Reliability Atlas. *Biggest gap.*
2. **More items** — 120 → larger per benchmark for statistical power.
3. **Validated contamination/attribution** — current 451 gsm8k hits are retrieval-similarity
   *candidates*, not gradient-verified; A1 gap is triage-only until validated.
4. **Math-checkpoint spot-check** — Instella-3B-Math had 26% degenerate output (assembled HF
   checkpoint); confirm how much of its low score is extraction noise vs. real capability loss.

## Notes for whoever runs the scale-up
- Colab notebook: `notebooks/reliability_suite_colab.ipynb` (clones `main`, bf16 patch, fresh `OUT` dir).
- On non-T4 GPUs, keep the bf16 patch (`prec=""`) — bitsandbytes 4-bit produces degenerate output on Blackwell.
- Use a fresh `OUT=` dir each scaled run so stale prep isn't restored from HF (that silently caps item count).
- Runtime on a Blackwell GPU: gsm8k-only 120-item run ≈ 40 min; expect ~6× that for all 6 benchmarks.
