# Full-scale memorisation suite — what it runs and why

The operating manual for `experiments/run_fullscale_suite.sh`. Design rationale is in
[`docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md); the defects it exists to fix are in
[`docs/AUDIT_2026-07-26.md`](../docs/AUDIT_2026-07-26.md).

## The question

> Does the model do better on problems it **provably memorised**?

GSM8K **train** items are verbatim in Instella's stage-2 training data (via
`amd/Instella-GSM8K-synthetic`); GSM8K **test** items are not. Membership is *verified per
item* by exact 13-gram containment, not inferred from an embedding proxy. Cross that with
numeric perturbation:

|              | original | numerically perturbed |
|--------------|----------|-----------------------|
| **seen**     | A        | B                     |
| **unseen**   | C        | D                     |

`DiD = (A − C) − (B − D)` is the part of the seen-item advantage that does not survive
perturbation — the memorisation component. Applying the same perturbation distribution to
both arms means the integer-magnitude confound of
[arXiv:2605.28700](https://arxiv.org/abs/2605.28700) cancels in the double difference.

**A tight interval around zero is a result**, not a failed experiment: it bounds how much
of the model's benchmark accuracy memorisation can explain, on a model whose training data
is actually inspectable.

## The model axis

The whole Instella-3B chain is public, so each step is a controlled data intervention:

```
Stage1 ──stage-2 data──▶ Instella-3B ──SFT──▶ SFT ──DPO──▶ Instruct ──math SFT──▶ Math-SFT ──RL──▶ Math
   ▲                          ▲
control: no GSM8K data    treated: GSM8K-targeted data enters here
```

`SUITE_TIER=1` runs the reliable headline trajectory: `stage1, stage2, instruct`.
`SUITE_TIER=2` additionally runs `sft`, `math_sft`, and `math`; it is diagnostic because
the assembled Math endpoint reached only 75% semantic completion at 3072 tokens in a live
smoke test and must not silently enter the headline result.

`AMD-OLMo-1B` is deliberately **off** the headline axis: its comparison to Instella-3B is
confounded (different data, different token budget) and its zero-shot output was degenerate
on 408/556 items in the pilot. It stays in the registry at tier 3 for gradient attribution,
where full precision on a 1B model is the actual reason to want it.

## Stages

| # | Stage | Cost | Marker |
|---|---|---|---|
| 1 | Load `Instella-GSM8K-synthetic` (the seen-arm ground truth) | CPU | `markers/stage1_corpus` |
| 2 | Verified seen/unseen arms + difficulty matching | CPU, ~5 min | `markers/stage2_splits` |
| 3 | Variant clusters, **magnitude gate**, hand-review harness | CPU | `markers/stage3_variants` |
| 4 | Official `apple/GSM-Symbolic` + decoding-noise control | CPU | `markers/stage4_aux` |
| 5 | Generate + score, per checkpoint × block | **GPU** | per-block score file |
| 6 | Termination audit | CPU | — |
| 7 | DiD + cluster-robust regression + atlas | CPU | — |
| 8 | Figures F1–F8 | CPU | — |

Two stages can **abort the run on purpose**:

* **Stage 3 magnitude gate** — if the median perturbed/original answer ratio leaves
  `[0.80, 1.25]`, the suite stops. The previous generator sat at 2.61×, which would have
  measured a bigger-arithmetic effect and called it reasoning fragility.
* **Stage 5 termination gate** — if fewer than `MIN_TERMINATION` (default 85%) of a
  checkpoint's completions terminate naturally, that cell fails. At 512 tokens the Math
  checkpoint emitted its answer marker on 2% of items and 26% of its responses contained
  the gold answer yet scored wrong; a better parser does not fix that, only more tokens do.

## Budget knobs

```bash
N_PER_ARM=250            # verified items per arm  <- the power-driving number
SURFACE_K=2              # answer-preserving variants per item
NUMERIC_K=2              # answer-changing numeric variants per item
GSMSYM_PER_TEMPLATE=3    # official GSM-Symbolic instances per template
RESAMPLE_ITEMS=100  RESAMPLE_N=5  RESAMPLE_TEMP=0.7   # decoding-noise control
RESAMPLE_MODELS="instruct math"   # Math is used only in diagnostic Tier 2
SUITE_TIER=1             # 1 = reliable core 3, 2 = diagnostic all 6
BATCH=8  MIN_TERMINATION=0.85  SEED=6198
```

At the defaults, on the ~1,000 generations/hour observed for this hardware at 1024 tokens:

| block | generations | hours |
|---|---:|---:|
| core arms (2 × 250 items × 5 variants × 3 checkpoints) | 7,500 | 7.5 |
| official GSM-Symbolic (200 templates × 4 × 3 checkpoints) | 2,400 | 2.4 |
| decoding-noise control (100 × 6 × Instruct) | 600 | 0.6 |
| **total** | **10,500** | **10.5** |

This leaves room in a long Colab runtime for retries and syncing. Tier 2 is not included in
that budget because the long-CoT Math checkpoints require separate output-validity work.

**Do not raise `SURFACE_K`/`NUMERIC_K` to buy power.** Measured intra-cluster correlation
on the pilot data is ≈0.48, so the marginal effective-n of the k-th variant is

```
#1 +1.00   #2 +0.35   #3 +0.18   #4 +0.11   #5 +0.07
```

The 5th variant of an item is worth 7% of a fresh item. Budget goes into `N_PER_ARM`.
Figure F8 plots this and the resulting minimum detectable effect.

## Resume

Every stage writes a marker; every (checkpoint, block) writes a score file. On start the
run pulls from HF and skips what is already finished. A reclaimed Colab VM loses nothing —
re-run the same cell.

**`generations/` are uploaded** (`HF_INCLUDE_GENERATIONS=1` by default). They used to be
excluded as "re-derivable"; they are not re-derivable without the GPU, and when a scoring
bug surfaced after a completed run the raw text was gone and the whole run had to be
repeated. A couple of hundred MB is cheap insurance against ~14 GPU-hours.

**Existing results are protected.** `hf_sync.py` refuses to write the `reliability-B*`
paths, so the pilot runs referenced by the audit stay intact.

## Running it

```bash
python scripts/preflight_fullscale.py --smoke      # the gate — must print SAFE TO LAUNCH
bash experiments/run_fullscale_suite.sh
```

Colab: [`notebooks/fullscale_colab.ipynb`](../notebooks/fullscale_colab.ipynb).

A short rehearsal through the identical code path, into its own directory:

```bash
OUT=experiments/runs/rehearsal-S40 N_PER_ARM=40 GSMSYM_PER_TEMPLATE=2 RESAMPLE_ITEMS=20 \
  bash experiments/run_fullscale_suite.sh
```

## Outputs

| path | what it answers |
|---|---|
| `analysis/memorization.json` | the headline DiD, cluster-robust regression, GLMM |
| `analysis/containment.json` | is the seen/unseen assignment real? (F4) |
| `analysis/termination.json` | were the models scored on complete outputs? (F6) |
| `figures/f1..f8` | the publication figure set |
| `review/` | hand-verification CSVs for the auto-derived templates |
| `generations/` | raw model text — re-score without re-running the GPU |

## Hand verification

The auto-derived numeric templates are precision-first but not infallible; GSM-Symbolic
hand-annotated all 100 of its own. Stage 3 exports
`review/templates_to_review.csv`; mark each row `ok`/`bad`, save as
`review/templates_reviewed.csv`, re-run. One `bad` row condemns the whole template — a
wrong instantiation means the substitution rule is unsound, not just that instance. Set
`REQUIRE_REVIEW=1` to make an unreviewed template a hard failure for the final run.
