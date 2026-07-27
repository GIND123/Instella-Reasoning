# Full-scale build plan — one 15-hour run, workshop-grade

Companion to [AUDIT_2026-07-26.md](AUDIT_2026-07-26.md). That doc says what is broken;
this one says what to build. Target: a single uninterrupted run with no wasted GPU hours.

> **Status: built and verified.** Components A–F are implemented, tested (163 passing,
> lint clean), and exercised end-to-end through the real CLI. Component G (attribution /
> TracIn) is deferred — see §2. Operating manual:
> [`experiments/FULLSCALE.md`](../experiments/FULLSCALE.md).
>
> Two numbers in §1 were measured during the build and differ from the estimates below:
> the numeric-template yield rose from 22.5% to **55%** after the role-disambiguation fix
> (better than the ~45% projected), and the realised answer-magnitude ratio is **1.00**
> (the target band is 0.80–1.25).

---

## 0. The finding that should reshape the paper

The whole Instella-3B post-training chain is public. I enumerated the Hub:

```
amd/Instella-3B-Stage1      <- pretraining stage 1 only (4.07T tokens, OLMoE-mix)
amd/Instella-3B             <- + Stage 2 (Dolmino, Tulu-3, Instella-GSM8K-synthetic)
amd/Instella-3B-SFT         <- + supervised fine-tuning
amd/Instella-3B-Instruct    <- + DPO
amd/Instella-3B-Math-SFT    <- + math SFT
amd/Instella-3B-Math        <- + RL
```

**Six checkpoints, one architecture, one tokenizer, documented data at every step.**
The current run uses three of them plus an unrelated 1B model whose only comparison axis
[METHODOLOGY.md §6](METHODOLOGY.md) already admits is confounded.

This matters because **Stage 2 is where AMD explicitly added GSM8K-targeted data**
([arXiv:2511.10628](https://arxiv.org/abs/2511.10628): Stage 2 "further enhance capabilities
on MMLU, BBH, and GSM8K", including `Instella-GSM8K-synthetic`). So:

- **Stage1 is the clean control** — the only checkpoint that has *not* seen GSM8K-targeted data.
- **Stage1 → Stage2 is a controlled data intervention**, not an observational comparison.

Combined with the seen/unseen split from the audit (GSM8K **train** is verbatim in Stage-2
data; **test** is not), the design becomes a 2×2×2 with a *causal* arm:

|  | Stage1 (no GSM8K data) | Stage2 (GSM8K data added) |
|---|---|---|
| **GSM8K train** (in Stage-2 corpus) | baseline | ← memorisation lands here |
| **GSM8K test** (not in corpus) | baseline | ← generalisation lands here |

`(Stage2 − Stage1)` on train items, minus `(Stage2 − Stage1)` on test items, is a
difference-in-differences estimate of **how much of Stage 2's GSM8K gain is memorisation.**
No closed model can produce this number. That is the paper.

Then re-measure the same quantity after perturbation, and you separate *memorised answers*
from *memorised procedures* — which is the actual "reasoning or remembering" question.

**Recommendation: reframe around the checkpoint trajectory.** Keep your abstract's framing
and metric; swap the model axis from "1B vs 3B vs 3B-Math" (confounded, and the 1B is
degenerate) to "the six-checkpoint Instella-3B developmental trajectory" (controlled).

---

## 1. Sample sizes, from measured data

I computed the intra-cluster correlation of correctness from your committed run:

```
Instella-3B            ICC=0.507   Instella-3B-Instruct   ICC=0.477
AMD-OLMo-1B            ICC=0.227   Instella-3B-Math       ICC=0.166
median ICC = 0.48
```

**ICC 0.48 is the single most important planning number.** Variants of the same item are
highly correlated, so extra variants buy far less than extra items:

```
marginal effective-n contribution of variant #k
  #1 +1.000   #2 +0.354   #3 +0.181   #4 +0.110   #5 +0.074   #6 +0.053
```

The 5th variant of an item is worth 7% of a fresh item. **Cap at 4–5 items per cluster and
spend everything else on more base items.** Your current run used ~4.6 — right at the
efficient frontier, so the fix is more *items*, not more variants.

Power at α=.05, two-sided, 80%, using the measured DEFF:

| contrast | design | detectable effect at n=250 items/arm | verdict |
|---|---|---|---|
| perturbation drop (within-item, McNemar) | paired | ~8 pp | **over-powered**, 250 is generous |
| seen vs unseen (between-arm) | 5 obs/item, DEFF 2.9 | ~13 pp | adequate |
| DiD (the headline) | contrast of contrasts | ~20 pp | **adequate only for a large effect** |

Measured discordance in your data is already extreme (ψ = 0.84–0.95), so the *existence* of
a perturbation drop needs only ~70–190 pairs. The budget is driven entirely by the
between-arm and DiD contrasts.

**Be clear-eyed about the DiD.** At 250 items/arm you can detect a ~20 pp memorisation
effect. If the true effect is smaller, you get a **bounded null** — "no evidence that
verified memorisation explains Instella's GSM8K accuracy; the 95% CI excludes effects
larger than X pp." On a fully open model with a *verified* treatment assignment, that is a
publishable workshop result. A noisy positive is not. Design for the honest interval, not
for a hoped-for star.

**Settled sample size: 250 GSM8K-train + 250 GSM8K-test base items, 5 items per cluster
(1 original + 2 surface + 2 numeric).**

---

## 2. What I will build

### Component A — Data layer

| # | Build | Detail | Status of dependency |
|---|---|---|---|
| A1 | `gsm8k_train` loader | Adds the "seen" arm. Registry entry + split override | `openai/gsm8k` parquet ✅ |
| A2 | **Exact-match treatment verification** | For every train item, assert verbatim 13-gram presence in the corpus; for every test item, assert absence. Emits `seen_verified.json`. **This is what makes the treatment assignment ground truth instead of a cosine proxy.** | — |
| A3 | Difficulty-matched sampler | Pairs train/test items on `difficulty.estimate_difficulty` (reasoning-step count) so the arms are balanced before generation | existing module |
| A4 | `gsm_symbolic_official` loader | `apple/GSM-Symbolic` — `main/test.jsonl`, `p1/test.jsonl`, `p2/test.jsonl`, plain JSONL, no script | ✅ verified reachable |
| A5 | Corpus loaders extended | `nvidia/OpenMathInstruct-2` (55 parquet ✅), `allenai/tulu-3-sft-mixture` (6 parquet ✅), `allenai/dolmino-mix-1124` (7,355 files — sampled, coverage reported) | ✅ |
| A6 | **Drop CLUTRR and dm_math** | Both are script-only on the Hub (`CLUTRR/v1` → `v1.py`, `deepmind/math_dataset` → `math_dataset.py`); `datasets` ≥3 removed script support. Not worth pinning an old `datasets` inside a 15-hour run | ⛔ blocked, confirmed |

### Component B — Variant generator (fixes the wrong labels)

| # | Build | Fixes |
|---|---|---|
| B1 | Reject templates where a leaf's token count in the calc chain ≠ its count in the question | The `2/2` "half" conflation — 15% wrong labels |
| B2 | Decimal-atomic tokenisation (`\d+(?:\.\d+)?`) | `10*1.2` → leaves `{10,1,2}`, which rewrote *"1.2 times"* → *"5.2 times"* |
| B3 | **Magnitude-neutral resampling** + a hard run assertion that median answer-magnitude ratio ∈ [0.8, 1.25] | The 2.61× inflation — the exact confound [arXiv:2605.28700](https://arxiv.org/abs/2605.28700) names |
| B4 | Positional (span-based) substitution instead of value-based | Recovers much of the 40.8% "leaf not exactly once in question" rejection → raises yield from 22.5% toward ~45% |
| B5 | Semantic-correctness tests: ≥20 hand-checked variants with asserted gold answers | The gap that let all of the above through 123 passing tests |
| B6 | Hand-verification harness | Exports surviving templates to CSV, re-imports your verdicts, hard-fails generation if any template is unreviewed. GSM-Symbolic hand-annotated all 100; reviewers expect it |
| B7 | Stronger distractor variants | Current `irrelevant_context` recycles ~3 null clauses ("A friend watched the whole time"). Keep them as the *unambiguous* control — that side-steps the [2026 GSM-NoOp ambiguity critique](https://www.lesswrong.com/posts/Ze4C99Dasj74YKCFh/revisiting-gsm-symbolic-do-2026-frontier-models-still-fail) — and add `p1`/`p2` from official GSM-Symbolic as the strong condition |

### Component C — Generation (fixes the truncation artifact)

| # | Build | Fixes |
|---|---|---|
| C1 | Per-model `max_new_tokens`; default 1024, **1536 for Math / Math-SFT** | Math emitted `####` in 2% of outputs; 145/556 contained the gold answer but scored wrong |
| C2 | **Termination-rate gate**: abort the cell if <85% of outputs terminate naturally | Turns silent truncation into a loud failure. Non-negotiable for a single-shot run |
| C3 | 5-shot prompting for base checkpoints (Stage1, Stage2, OLMo-1B) | 408/556 OLMo outputs were degenerate loops; zero-shot CoT does not work on 1B/3B *base* models |
| C4 | Temperature-sampling mode (T=0.7, n=5) | The decoding-noise control — proves the perturbation drop exceeds run-to-run variance |
| C5 | Format-compliance logging per model | `####`-emission ranged 2%→54%; right now the models are being compared on instruction-following |

### Component D — Metric reconciliation (your proposal vs the code)

Your abstract defines consistency as *"the fraction of variants whose correctness label
matches that of the original problem."* The code defines it as *modal-answer share over
answer-preserving variants* ([METHODOLOGY.md §3](METHODOLOGY.md)).

These are different metrics and they disagree. The code's version is stronger — it detects
a model that is consistently *wrong in the same way* (a memorisation signature) whereas
correctness-label-agreement scores that identically to consistent correctness.

| # | Build |
|---|---|
| D1 | Keep the code's modal-answer-share as primary; report your label-agreement version as a secondary column; document the difference in the paper's metric section. Reviewers reward this |

### Component E — Analysis (what the 2026 critique demands)

| # | Build |
|---|---|
| E1 | **GLMM**: `correct ~ seen * perturbed * checkpoint + log(answer_magnitude) + (1\|item)`, logistic, via `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`. Replaces the pooled z-test as the headline test — [arXiv:2605.28700](https://arxiv.org/abs/2605.28700) makes per-question random effects table stakes |
| E2 | DiD estimator with a parent-cluster bootstrap CI |
| E3 | Magnitude-matched sensitivity: re-run the primary contrast on variants whose gold answer is within ±20% of the original |
| E4 | Checkpoint-trajectory figure: accuracy / consistency / reliability across all six checkpoints, seen vs unseen |
| E5 | Keep BH-FDR across all reported contrasts |
| E6 | Fix `compute_accuracy_gap` to report a `partial`-vs-clean contrast too, and to emit an explicit `insufficient_treatment_group` status rather than `gap: 0.0, p: 1.0` — the current output looks like a measured null but is an empty analysis |

### Component F — Orchestration and the anti-waste gate

| # | Build |
|---|---|
| F1 | `experiments/run_fullscale_suite.sh` — the single run, per-cell HF checkpointing |
| F2 | **`scripts/preflight_fullscale.py`** — the gate that protects your 15 hours. Runs on CPU + a 10-minute GPU smoke and hard-fails on: unreviewed templates, magnitude ratio out of band, termination rate <85%, any model that can't load, HF write failure, missing seen/unseen verification, stale `OUT` dir collision. **Nothing launches until this is green.** |
| F3 | `HF_INCLUDE_GENERATIONS=1` by default | 
| F4 | New notebook `notebooks/fullscale_colab.ipynb` in the shape of the existing one, with the bf16 + degenerate-gate patches folded into the script instead of `sed` |

### Component G — Attribution (Phase 4 of your proposal), Tier 2

| # | Build |
|---|---|
| G1 | Embedding attribution over the expanded corpus — already built, just needs A5 |
| G2 | TracIn checkpoint-proximity on AMD-OLMo-1B, 100 FAISS-prefiltered items, as the validation of embedding attribution | 

Keep G honest and small. Your abstract already frames it as a validation step, and
[METHODOLOGY.md §5](METHODOLOGY.md) correctly calls embedding attribution associational.
If time is short, G is the first thing to cut — it is the only component whose absence
does not weaken the headline claim.

---

## 3. Run budget

Throughput basis: your run did 2,224 generations × 4 models at 512 tokens in ~1.5 h on the
RTX PRO 6000 (96 GB) → ≈1,500 gen/h. At 1024 tokens, assume **≈1,000 gen/h.**

| Block | Content | Items | Models | Gens | Hours |
|---|---|---:|---:|---:|---:|
| **B1** Core | 250 train + 250 test, × (1 orig + 2 surface + 2 numeric) | 2,500 | 4¹ | 10,000 | 10.0 |
| **B2** Official GSM-Symbolic | `main` + `p1`, 100 each | 200 | 4 | 800 | 0.8 |
| **B3** Decoding-noise control | 100 items × 5 samples @ T=0.7 | 500 | 2² | 1,000 | 1.0 |
| **B4** Buffer | re-runs, Math at 1536, failed cells | — | — | 1,700 | 1.7 |
| | | | | **13,500** | **13.5** |

¹ Tier-1 checkpoints: `Stage1`, `Instella-3B`, `Instella-3B-Instruct`, `Instella-3B-Math`.
² `Instella-3B-Instruct`, `Instella-3B-Math`.

**Tier 2, only if B4 buffer is untouched (~1.5 h):** add `Instella-3B-SFT` and
`Instella-3B-Math-SFT` on a 100-item subset. That separates SFT from DPO and SFT from RL,
which upgrades the A6 story from "post-training regressed" to "the *RL step specifically*
did X". Worth it if you have the hours.

**Dropped, deliberately:** AMD-OLMo-1B from the headline (degenerate output, confounded
axis — keep it only for G2 TracIn, where full precision on 1B is the actual reason it is in
your proposal); CLUTRR and dm_math (unreachable); MATH/LogiQA2/ARC/BBH from this run. A
4-page workshop paper that nails GSM8K across six checkpoints beats a thin sweep of four
benchmarks. The multi-benchmark atlas is the conference version.

---

## 4. Checkpoint accessibility — verified working, two gotchas

I authenticated with the token in `.env` (`GOVINDFROM`, fine-grained write) and inspected
the repo: `GOVINDFROM/Instella-Reasoning`, private dataset, 88 files, three runs
(`reliability-B120-K5-clean` is the valid one; `B120-K5` and `B15-K5` share identical
15-item variant files because `B120-K5` resumed from stale prep).

Resume works as designed: `pull` at start restores prior work, per-`(model, benchmark)`
score files are completion markers, `push` after each cell. A reclaimed Colab VM loses
nothing.

Two things that will cost you hours if not fixed first:

1. **`generations/` are never uploaded** — `_DEFAULT_IGNORE` in
   [hf_sync.py:36](../experiments/hf_sync.py#L36). Only `scores/` survive, so nothing can be
   re-scored offline; every scoring fix needs full re-generation. This is exactly why the
   truncation bug cost a whole run. **Set `HF_INCLUDE_GENERATIONS=1`** (F3). Estimated
   ~200 MB for 13.5k generations — trivial next to the 1.45 GB corpus already up there.
2. **Score files are completion markers**, so a changed variant set in an existing `OUT`
   dir is silently skipped. Use a **fresh `OUT` dir** (`fullscale-S250`), and F2 will
   hard-fail if it detects a collision.

Also cached and free: `contamination/synthetic_corpus.jsonl` (1.45 GB) is already on HF, so
Stage 1 will not re-download or re-embed.

---

## 5. Sequencing against 2026-08-29

| Window | Work | Gate |
|---|---|---|
| Jul 27 – Jul 30 | Components A, B, C, D | `pytest` green incl. B5 semantic tests |
| Jul 31 – Aug 1 | B6 hand-verification of surviving templates (expect 80–130 at the improved yield; budget ~4 h of your time) | every template reviewed |
| Aug 2 | Component F, preflight, 10-min GPU smoke | **F2 preflight green** |
| Aug 3 – Aug 4 | **The run** (~13.5 h, checkpointed) | termination rate ≥85% per cell |
| Aug 5 – Aug 9 | Component E analysis, figures | — |
| Aug 10 – Aug 27 | Write; ~1.5 h GPU held in reserve for reviewer-anticipating ablations | — |

The only serial human dependency is B6. Start it the moment B1–B4 land.

---

## 6. What I need from you before building

1. **Confirm the reframe** — six-checkpoint trajectory + seen/unseen, GSM8K only. This is a
   real change from your abstract's four-benchmark scope, and it is the change that makes
   the claim defensible. If you want the four benchmarks kept, say so and I will re-cost it
   (roughly: it doubles the run and halves the power on every contrast).
2. **Confirm the workshop and its deadline.** I have been planning against 2026-08-29, which
   is the general NeurIPS 2026 workshop submission date; individual workshops differ, and a
   two-week swing changes the sequencing above.
3. **Whether to keep Component G** (attribution + TracIn). It is in your abstract, it is the
   thread that makes the "unified" claim in your title literally true, and it is also the
   first thing that should be cut if the schedule slips.

Say go and I will start with A, B, and C — those are the blocking fixes, they are all
CPU-side, and nothing else can be validated until they land.
