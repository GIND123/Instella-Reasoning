<h1 align="center">Reasoning or Remembering — a premise-deletion probe across the Instella-3B training trajectory</h1>

<p align="center">
  A premise-deletion probe run across four Instella-3B checkpoints against a
  verified-clean control arm, calibrated by controlled injection at five exposure levels,
  with cluster-robust inference over parent problems.
</p>

<p align="center">
  <a href="#abstract"><b>Abstract</b></a> ·
  <a href="#results"><b>Results</b></a> ·
  <a href="#figures"><b>Figures</b></a> ·
  <a href="https://huggingface.co/datasets/GOVINDFROM/Instella-Reasoning">Artifacts</a> ·
  <a href="experiments/runs/ckpt-axis-v1/analysis/RESULTS.md">Results (active study)</a> ·
  <a href="docs/FIGURES.md">Provenance</a> ·
  <a href="docs/METHODOLOGY.md">Methodology</a> ·
  <a href="docs/AUDIT_2026-07-26.md">Audit</a> ·
  <a href="#command-reference">Commands</a> ·
  <a href="docs/COMPUTE.md">Compute</a>
</p>

---

> ## ⚠ Scope — the active study is the **checkpoint axis**, not the item axis
>
> **The item-level design described in the rest of this README is superseded.** It compared
> high- against low-containment GSM8K *train* items to estimate a memorisation component.
> That estimand does not exist: `allenai/tulu-3-sft-mixture` is in Instella's Stage-2
> mixture and carries ~97% of GSM8K train at containment ≥0.999, so **both arms of the
> within-train contrast are exposed** and no difference between them can be attributed to
> membership. See [`docs/CORPUS_CORRECTION.md`](docs/CORPUS_CORRECTION.md).
>
> The active study replaces the item axis with the **checkpoint axis**: the same
> premise-deletion items run across `Stage1 → Instella-3B → SFT → Instruct`, against a
> GSM8K-**test** control arm verified clean, calibrated by controlled injection at five
> exposure levels.
>
> | | |
> |---|---|
> | Design & pipeline | [`docs/CHECKPOINT_AXIS_STUDY.md`](docs/CHECKPOINT_AXIS_STUDY.md) |
> | Results | [`experiments/runs/ckpt-axis-v1/analysis/RESULTS.md`](experiments/runs/ckpt-axis-v1/analysis/RESULTS.md) |
> | Every number's provenance | [`docs/FIGURES.md`](docs/FIGURES.md) |
> | Injection design | [`docs/PHASE2_DESIGN.md`](docs/PHASE2_DESIGN.md) |
> | Figures | `paper/figures/F{1,2,3}_*.pdf` |
> | Current state / handoff | [`STATE.md`](STATE.md) |
>
> **Headline: the accuracy drop under perturbation is a failure of inference, not recall.**
> Deletion-recall rises across the Stage-2 data intervention (+1.70 pp, McNemar p=4.6e-4)
> — but it rises just as much on items provably absent from the corpus (+1.95 pp).
> Difference-in-differences **−0.26 pp, 95% CI [−1.66, +1.14]**, cluster-bootstrapped over
> parent problems. Two further nulls agree: recall is non-monotonic in Stage-2 containment
> within the train arm, and the contrast survives sampling at T=0.7 and T=1.0.
>
> The null is **bounded, not bare**. Controlled injection shows the probe detects recall
> only at 64× verbatim repetition (verbatim reproduction 0.946, p=0.035); at 16× the model
> reproduces 49% of an injected document verbatim and the probe detects nothing.
>
> The earlier `reliability-B*` runs under `experiments/runs/` are **superseded pilots**;
> [`docs/AUDIT_2026-07-26.md`](docs/AUDIT_2026-07-26.md) documents why four of their five
> headline findings are artifacts. Those numbers must not be cited.

## Abstract

Benchmark contamination is routinely treated as grounds for discounting a reported score: if
evaluation items appear in training data, accuracy is presumed to reflect recall rather than
reasoning. The inference is rarely tested, because membership is normally inferred from an
embedding or perplexity proxy rather than established.

The present study establishes membership exactly and tests the inference directly. Instella-3B is
released at successive training stages, and its stage-2 corpus (`amd/Instella-GSM8K-synthetic`) is
derived from GSM8K **train** and not from GSM8K **test**. Membership of each benchmark item is
therefore verified by exact 13-gram containment against that corpus rather than estimated, and
items whose containment falls between the thresholds are excluded from both arms rather than
forced into one. Crossing verified membership with numeric perturbation yields a
difference-in-differences estimator that isolates the memorisation component and cancels the
integer-magnitude confound identified in arXiv:2605.28700. Intervals are obtained from a bootstrap
that resamples parent items, matching the measured intra-cluster correlation of approximately 0.48.

Across four checkpoints spanning the full training pipeline — 12,468 scored generations, 250
verified-seen and 250 verified-unseen items per checkpoint — the memorisation
difference-in-differences is flat and every interval spans zero: +0.014 at `stage1`, +0.008 at
`stage2`, +0.010 after supervised fine-tuning, and +0.040 after DPO. The estimate does not move at
the release boundary where GSM8K-derived data enters training, although accuracy on seen items
rises by 58 points across that same boundary.

A second measurement excludes the most obvious rebuttal. Because the corpus derives from GSM8K
train, every GSM8K test item is a known negative and the detector's false-positive rate is
measurable rather than arguable. The standard any-n-gram rule flags 3 of 1,000 known negatives, a
false-positive rate of 0.3 percent. Detection is therefore accurate, and the absence of a
memorisation effect cannot be attributed to a noisy treatment label.

---

## Study status

The experimental programme is complete. All generation blocks across all four checkpoints have
been generated, scored, analysed and plotted, and all artifacts are mirrored to Hugging Face. No
computation is pending.

| | |
|---|---|
| Rows generated and scored | **12,468 / 12,468 (100%)** |
| Generation blocks complete | **10 / 10 (100%)** |
| Checkpoints measured | **4** — `stage1`, `stage2`, `sft`, `instruct` |
| Quality gates passed | **9 / 10** — `stage1/arms` reaches 84.9% against an 85% gate |
| Rows eligible for optional rework | **304 / 12,468 (2.4%)** |
| Figures | **17** — F1–F8 headline, S1–S4 supplementary, J1–J5 judge, in [`docs/figures/`](docs/figures/) |
| Test suite | 173 passed, 2 skipped |
| GPU time consumed by the corrected run | ~4.9 h (A100-40GB) |
| GPU time required to complete outstanding work | **0 h** |

The authoritative run directory is `experiments/runs/fullscale-S250-v2`. The earlier
`experiments/runs/fullscale-S250` remains complete and reproducible but its arms were constructed
from a corrupted containment measurement, described under [Defects identified and
corrected](#defects-identified-and-corrected). Figures and numbers from the earlier run must not be
quoted as headline results.

---

## Research question and design rationale

### The question that cannot be asked, and the one that can

The literal contamination question — whether GSM8K **test** items appear in Instella's training
data — is unanswerable against the available corpus. `amd/Instella-GSM8K-synthetic` is derived from
GSM8K train, so a scan of test items returns no matches at any threshold and the contaminated group
is empty by construction.

The answerable substitute exploits a property of the benchmark. GSM8K ships train and test splits
written by the same annotators to the same specification at comparable difficulty. Instella's
training data contains material derived from train and not from test. Two arms follow:

- **seen** — GSM8K train items verified as present in the training corpus
- **unseen** — GSM8K test items verified as absent from it

Verification is exact rather than semantic. Each item is decomposed into every run of 13
consecutive normalised word tokens, and containment is the fraction of those n-grams occurring
anywhere in the 1.45 GB corpus. A copied item scores near 1.0 and an unrelated item scores 0.0.
Items falling between the thresholds of 0.10 and 0.80 are excluded from both arms, on the principle
that an indefensible treatment label is more damaging than a smaller sample. The 13-gram window
follows Brown et al. (2020) and matches the window used elsewhere in the repository, so the two
stages agree.

### Why perturbation is required

A raw accuracy advantage on seen items establishes nothing, since train items may simply be easier.
Each item is therefore also rewritten with different numeric values, preserving structure and
reasoning depth while changing the arithmetic. Memorised answers lose their advantage once the
numbers change; reasoned answers retain it.

### The estimator

The memorisation component is the double difference

```
DiD = (seen − unseen | original) − (seen − unseen | perturbed)
```

A positive DiD indicates that the seen advantage evaporates under perturbation, which is the
signature of memorisation. A DiD near zero indicates that whatever advantage exists survives new
numbers, which is not. Differencing twice removes any factor affecting both arms equally, including
the possibility that perturbed problems involve systematically larger arithmetic.

### Checkpoints

Instella-3B is published at successive training stages, which converts an observational comparison
into an approximation of a controlled intervention:

| tag | model | GSM8K-derived data | role |
|---|---|---|---|
| `stage1` | `amd/Instella-3B-Stage1` | **no** | **control** — the only checkpoint never exposed to GSM8K |
| `stage2` | `amd/Instella-3B` | yes | **treated** — the stage-2 mix explicitly targets GSM8K |
| `sft` | `amd/Instella-3B-SFT` | yes | isolates supervised fine-tuning from DPO |
| `instruct` | `amd/Instella-3B-Instruct` | yes | post-DPO; the headline general model |

Under the memorisation hypothesis the DiD should be approximately zero at `stage1` and positive at
`stage2`. Including `sft` prevents the two post-training stages from being confounded, so that any
change can be attributed to one of them rather than to post-training in aggregate.

### Item clusters and blocks

Each of the 500 arm items expands into a cluster of five rows: one original, two answer-preserving
rewrites (rephrasing, distractor insertion) and two answer-changing numeric perturbations, giving
2,017 rows per checkpoint. Answer-preserving and answer-changing variants are kept in separate
terms, the former driving a consistency measure and the latter the difference-in-differences.

| block | rows | checkpoints | purpose |
|---|---|---|---|
| `arms` | 2,017 | all four | the headline difference-in-differences |
| `gsmsym` | 800 | all four | Apple's hand-written GSM-Symbolic templates, as external validity |
| `resample` | 600 | `stage2`, `instruct` | identical prompts sampled five times at T = 0.7, as the decoding-noise null |

The `resample` block carries more weight than its size suggests. At temperature 0 there is no
sampling variance, so an observed inconsistency across variants has no null against which to be
compared and the consistency statistic is uninterpretable without it. It is run for the treated base
checkpoint and the headline model, the two whose consistency is actually interpreted; running it on
`stage1` would consume approximately 1.5 GPU-hours and answer nothing further.

### Inference

Confidence intervals derive from a bootstrap resampling **parent items** rather than individual
rows. The five rows of one problem are not five independent observations; the measured
intra-cluster correlation is approximately 0.48. Resampling rows would contract intervals by
roughly the design effect of 2.4 and yield confidently incorrect error bars.

---

## Experimental record

| | v1 — `fullscale-S250` | v2 — `fullscale-S250-v2` |
|---|---|---|
| Arms | 194 seen / 194 unseen | **250 seen / 250 unseen** |
| Treatment labels | 43% of the seen arm mislabelled | all verified |
| Difficulty matching | performed against incorrect bins | correct, 84/83/83 |
| `containment.json` | values up to 7.78 | corrected |
| Checkpoints | 3 | **4** |
| Status | complete, superseded | **complete, authoritative** |

The first run executed across Colab (`stage1`) and Modal (`stage2`, `instruct`) and completed in
full. A subsequent audit of its containment file exposed the identifier collision described below,
which invalidated the treatment assignment. The second run was staged into a fresh directory
reusing 4,119 of the 6,051 existing generations — possible because variants are seeded per
`(item, type)` and retained items therefore keep byte-identical text — and regenerated only the
1,932 genuinely new rows. Two further checkpoints and the `stage2` decoding-noise control were
added subsequently.

---

## Defects identified and corrected

Several defects were material to the reported numbers, and two would have propagated into the
published result had they not been found. They are recorded in full because the corrections are
part of the evidence that the final numbers are trustworthy.

**1. Identifier collision between GSM8K train and test.** Both splits use the scheme
`gsm8k_NNNNN`, so loading 1,000 items from each produces pairwise collision across all 1,000.
`verify_containment` keys its gram index by `item.id` while the owner map retains both twins, so
the numerator accumulated matches against n-grams absent from the denominator. Containment, a
fraction bounded above by 1.0, reached **7.78**. Re-measurement with `train::` and `test::`
namespaced identifiers established that:

- **83 of 194 seen-arm items in the first run (43%) were not verifiably seen**, with true
  containment between 0.39 and 0.79, below the 0.80 threshold the design requires;
- the eligible pools are in fact 303 seen and 998 unseen, supporting substantially larger arms.

**2. Difficulty bins inherited the same collision.** `assign_difficulty_bins` is likewise keyed by
`item.id`, so each train item inherited the difficulty of its test twin. The first run's arms were
consequently never difficulty-matched despite the balance report reporting success. The second run
bins over namespaced identifiers.

**3. Answer extraction drifts across checkpoints.** `extract_numeric` falls back through three
tiers: the `#### N` marker, then an "answer is …" phrase, then the final number in the completion.
Marker rates differ substantially — 84.9% at `stage1`, 98.3% at `stage2`, 35.4% at `sft`, 33.6% at
`instruct`, the last two reflecting conversational post-training style. Tier 3 carries different
meaning per checkpoint, recovering the correct answer 70.4% of the time at `sft` but approximately
1% at `stage1`, where it fires on looping output. The effect on the estimator was measured rather
than assumed: the second difference of the tier-1 rate is at most 0.058 and cancels in the double
difference.

**4. Blocking remote invocation propagated local cancellation.** `modal run --detach` preserves the
application, but `Function.remote()` blocks the local client and cancellation of that call
propagates into the container. A two-hour run terminated at `instruct/resample` as a result.
Launching through `Function.spawn()` removes the long-lived local process entirely.

**5. Remote synchronisation silently reverted local edits.** The suite begins with
`snapshot_download(local_dir=".")`, which synchronises local files downward to match the remote. An
in-place edit of a run directory is therefore undone before generation begins; a complete corrected
rebuild was overwritten in exactly this way and the subsequent run reported every block as already
scored. Corrections are now staged into a new run directory, which additionally preserves the
earlier run intact.

**6. Unpinned library version.** `pyproject` specifies `>=4.44,<5`, but a range permits different
checkpoints to run under different minor versions, which enters the estimator as though it were a
model difference. The version is now pinned exactly to `transformers==4.56.0`, matching the version
under which `stage1` was originally generated.

**7. The decoding-noise control was counted as difference-in-differences evidence.** Stage 7 is
invoked with `--scores <run>/scores/*__ALL.jsonl`, merging arms, gsmsym and resample.
`make_resample_suite` emits, for each item, the cluster's original **in addition to** N copies;
that original reuses the arms block's `benchmark_id` while being generated at T = 0.7, and the
copies inherit the parent's `arm` label without being answer-changing. All 600 rows consequently
entered the `seen|original` cell:

```
stage2   seen|original  n=1345      (745 arms + 600 resample)
instruct seen|original  n=1345
stage1   seen|original  n= 745      (no resample block)
sft      seen|original  n= 745
```

Three faults occurred simultaneously. Temperature-0.7 samples entered a temperature-0 contrast; the
resampled items were weighted once per copy; and, because the control is run only for the
checkpoints whose consistency is interpreted, the estimator shifted for those checkpoints alone.
The third is the most serious, since it renders a measurement difference indistinguishable from a
checkpoint difference in precisely the `stage1` to `stage2` comparison on which the study depends.
The correction excludes control variants at the shared entry points and computes the estimator from
the arms block alone; F3 required an additional de-duplication because it reads records directly;
F7 is unaffected, the resample block being legitimately its null. Recovery tests now inject a
deliberately unbalanced resample block and assert that the estimate, the cell counts and the
cluster counts remain unchanged.

**8. A suppressed non-zero exit preserved a stale audit.** The finalisation step ignored return
codes, so a malformed `check-termination` invocation failed without notice and left a
`termination.json` describing seven files alongside an analysis refreshed to ten. Return codes are
now checked, with `check-termination` explicitly permitted to exit non-zero because in that command
a non-zero status reports a finding — a block below the gate — rather than an error.

## Results

Every number below is from `experiments/runs/fullscale-S250-v2`, 250 verified-seen and 250
verified-unseen items per checkpoint, cluster-robust bootstrap over parent items.

### The headline: the memorisation DiD is flat across the entire training pipeline

| checkpoint | training stage | seen orig | unseen orig | seen pert | unseen pert | adv. orig | adv. pert | **DiD** | CI95 |
|---|---|---|---|---|---|---|---|---|---|
| `stage1` | never saw GSM8K | 0.0752 | 0.0853 | 0.0658 | 0.0897 | −0.0101 | −0.0238 | **+0.0137** | [−0.037, +0.064] |
| `stage2` | **GSM8K data enters** | 0.6550 | 0.4547 | 0.5679 | 0.3759 | +0.2004 | +0.1920 | **+0.0083** | [−0.103, +0.117] |
| `sft` | + supervised fine-tuning | 0.7852 | 0.7267 | 0.6420 | 0.5931 | +0.0586 | +0.0489 | **+0.0097** | [−0.098, +0.118] |
| `instruct` | + DPO | 0.8121 | 0.5602 | 0.6502 | 0.4379 | +0.2519 | +0.2123 | **+0.0396** | [−0.071, +0.149] |

All four checkpoints are measured on identical cells — 745 seen-original, 739 unseen-original,
243 seen-perturbed, 290 unseen-perturbed, 250 clusters per arm — so nothing in this table is a
sample-composition artifact.

**Every interval spans zero.** The seen-arm advantage is large for `stage2` and `instruct`
(+0.20 and +0.25 on originals) but it **does not shrink under perturbation** (+0.19 and +0.21).
That is the whole result: the advantage is not the kind that disappears when the numbers change,
which is what memorisation would look like.

### What moves along the trajectory, and what does not

| transition | introduces GSM8K data | Δ accuracy seen | Δ accuracy unseen | **Δ DiD** |
|---|---|---|---|---|
| `stage1` → `stage2` | **yes** | +0.5799 | +0.3694 | **−0.0054** |
| `stage2` → `sft` | no | +0.1302 | +0.2720 | +0.0014 |
| `sft` → `instruct` | no | +0.0268 | −0.1664 | +0.0299 |

This is the sharpest statement the design supports. The `stage1`→`stage2` step is the release
boundary where GSM8K-derived data enters training — the closest thing to a controlled data
intervention a public model offers. Across it, accuracy on seen items rises by **58 points**, and
the memorisation DiD moves by **−0.005**. The capability arrives; the memorisation signature does
not. Neither post-training stage changes that.

### Robustness

**Magnitude-matched re-estimate.** Perturbed items whose gold answer shifted in magnitude are
dropped, then the DiD is recomputed — because a perturbation effect could otherwise be a
bigger-arithmetic effect (arXiv:2605.28700):

| checkpoint | matched DiD | CI95 |
|---|---|---|
| `stage1` | −0.0042 | [−0.092, +0.081] |
| `stage2` | +0.0191 | [−0.198, +0.216] |
| `sft` | −0.0576 | [−0.272, +0.139] |
| `instruct` | +0.0042 | [−0.212, +0.214] |

Still no effect, with intervals widened by the smaller matched sample.

**Three independent arm constructions.** The conclusion does not depend on how the arms were
built — including the construction that was later found to be 43% mislabelled:

| checkpoint | v1 full (194, mislabelled) | v1 verified-only (111) | **v2 (250, all verified)** |
|---|---|---|---|
| `stage1` | +0.0128 | +0.0066 | **+0.0137** |
| `stage2` | −0.0592 | −0.0069 | **+0.0083** |
| `instruct` | −0.0735 | −0.0192 | **+0.0396** |

Point estimates wander between −0.074 and +0.040 with no stable sign, which is the signature of a
true effect near zero rather than of an effect being missed.

**Extraction-tier leakage.** `extract_numeric` falls back in three tiers (the `#### N` marker →
an "answer is …" phrase → the last number in the text), and the marker rate differs enormously
across checkpoints — 84.9% for `stage1`, 98.3% for `stage2`, 35.4% for `sft`, 33.6% for
`instruct`. What matters is not that difference but whether it is *correlated with the treatment*,
which is the second difference of the tier-1 rate:

| checkpoint | accuracy DiD | tier-1 DiD | verdict |
|---|---|---|---|
| `stage1` | +0.0137 | −0.0127 | cancels |
| `stage2` | +0.0083 | −0.0136 | cancels |
| `sft` | +0.0097 | **−0.0583** | marginally above the ±0.05 flag |
| `instruct` | +0.0396 | +0.0143 | cancels |

`sft` is the one checkpoint where extraction composition differs across cells by more than the
threshold. Its practical impact is limited — tier-3 recovers the correct answer 70.4% of the time
for `sft`, close to its overall accuracy, so a shift in tier composition moves little. It is
recorded here rather than smoothed over.

### Supporting analyses

**Contamination-detector false-positive rate.** The corpus derives from GSM8K *train*, so every
*test* item is a **known negative** — it cannot be contaminated. That makes the standard
detector's error rate measurable rather than arguable:

| rule | train flagged | test flagged | false-positive rate |
|---|---|---|---|
| any 13-gram match (Brown et al., 2020) | 866 (86.6%) | 3 (0.3%) | **0.3%** |
| containment ≥ 0.50 | 584 (58.4%) | 1 (0.1%) | 0.1% |
| containment ≥ 0.80 (this study) | 303 (30.3%) | 0 (0.0%) | **0.0%** |

Test-item containment: median 0.000, p99 0.000, max 0.538. **The detector works**, which removes
"you found nothing because your detector is noisy" as an explanation for the null. This analysis
was run expecting the opposite result — that shared GSM8K templates would produce many false
positives — and the finding it produced is the more useful one.

Note also that **86.6% of train items have *some* overlap but only 30.3% reach ≥0.80.** The
synthetic corpus mostly *derives from* GSM8K train rather than copying it, which is exactly why a
binary contaminated/clean label is the wrong instrument and a continuous measure with an explicit
ambiguous band is the right one.

**Measurement validity.** A model cut off at the token cap scores as a weak model when it is
really a truncated measurement. All 12,468 generations audited:

| file | n | reached semantic stop | `####` marker | median chars | gate |
|---|---|---|---|---|---|
| `stage1__arms` | 2,017 | **84.9%** | 84.9% | 196 | **below 85%** |
| `stage1__gsmsym` | 800 | 86.1% | 86.1% | 215 | pass |
| `stage2__arms` | 2,017 | 98.4% | 98.3% | 277 | pass |
| `stage2__gsmsym` | 800 | 96.9% | 96.4% | 320 | pass |
| `stage2__resample` | 600 | 99.8% | 97.0% | 302 | pass |
| `sft__arms` | 2,017 | 99.5% | 35.4% | 505 | pass |
| `sft__gsmsym` | 800 | 99.4% | 31.0% | 651 | pass |
| `instruct__arms` | 2,017 | 99.8% | 33.6% | 700 | pass |
| `instruct__gsmsym` | 800 | 99.9% | 33.4% | 852 | pass |
| `instruct__resample` | 600 | 100.0% | 26.0% | 693 | pass |

`stage1/arms` is the single block below the gate. The cause is diagnosed, not assumed: v1 repaired
261 truncated rows at 2,048 tokens and v2's 644 new rows were generated at 1,024, so the block
mixes budgets (reused rows terminate at 85.6%, new rows at 83.5%). The truncation is **balanced
across the DiD cells** — its second difference is −0.013 — so it attenuates every cell alike
rather than biasing the contrast, and `stage1` is the control arm whose DiD is ~0 under every
construction.

---

## Figures

All eight are regenerated from the final data and live in [`docs/figures/`](docs/figures/).
Design rules are enforced in code rather than left to taste: one y-axis per panel (a dual-scale
chart lets the author choose the visual conclusion), fixed categorical colours so a checkpoint
keeps its hue when a filter changes the series count, CVD-safe palettes (worst adjacent separation
dE 9.4 under simulated protanopia/deuteranopia/tritanopia, above the 8 floor), identity carried by
direct labels so it is never colour-alone, and error bars that are the same parent-item bootstrap
used in the analysis.

### F1 — Trajectory

*Where along the training pipeline does accuracy appear, and does it appear on seen items only?*

![F1 trajectory](docs/figures/f1_trajectory.png)

Accuracy climbs steeply from `stage1` to `stage2` on **both** arms (+58 points seen, +37 points
unseen). If stage-2 data bought memorisation, the seen line would lift away from the unseen line.
It does not — both rise together.

### F2 — Difference-in-differences forest

*How large is the memorisation component, with intervals?* **This is the headline figure.**

![F2 DiD forest](docs/figures/f2_did_forest.png)

Four checkpoints, four intervals, all crossing zero.

### F3 — Perturbation slopes

*Does accuracy survive numeric perturbation, per checkpoint?*

![F3 perturbation slopes](docs/figures/f3_perturbation_slopes.png)

A slope chart rather than grouped bars, because the quantity of interest is the *change* within a
model and a slope encodes change as slope instead of as a difference between bar heights the eye
has to subtract. Every checkpoint drops under perturbation — that is a real robustness finding —
but seen and unseen drop *together*, which is why the DiD is flat.

### F4 — Containment evidence

*Is the seen/unseen treatment assignment actually verified?*

![F4 containment](docs/figures/f4_containment.png)

The two arms separate almost completely: train items concentrate near 1.0, test items at 0.0
(median 0.000, p99 0.000). **This figure was unplottable in v1** — it drew containment values above
1.0 on an axis that measures a fraction, which was the visual symptom of the id-collision bug.

### F5 — Magnitude control

*Is the perturbation effect just bigger arithmetic?*

![F5 magnitude control](docs/figures/f5_magnitude_control.png)

Median answer-magnitude ratio 1.057, inside the [0.8, 1.25] band. The perturbation changes the
answer without making the arithmetic systematically harder, so the observed drop is not a
magnitude effect.

### F6 — Measurement validity

*Are the models being scored on complete outputs?*

![F6 measurement validity](docs/figures/f6_measurement_validity.png)

Termination and marker rates per file. `stage1`'s lower bar is the documented mixed-token-budget
issue above, not a formatting break.

### F7 — Consistency decomposition

*Is the inconsistency real perturbation sensitivity, or just decoding noise?*

![F7 consistency decomposition](docs/figures/f7_consistency_decomposition.png)

The decoding-noise bar is the null: identical prompts resampled at T=0.7. Without it, an observed
inconsistency across variants has nothing to be compared against. Run for `stage2` (the treated
base checkpoint) and `instruct` (the headline model) — the two whose consistency is actually
interpreted.

### F8 — Design power

*Is the sample size adequate, given the measured ICC?*

![F8 design power](docs/figures/f8_design_power.png)

The honest limitation. At ICC ≈ 0.48 and 250 items per arm, the design resolves effects of roughly
12–15pp. It does **not** resolve the 5pp end of the range claimed in the contamination literature.

### Supplementary figures

Four further panels present evidence otherwise available only as tables. Each addresses an
objection raised specifically against a null result: whether the treatment label is trustworthy
(S1, S2), whether scoring is comparable across checkpoints (S3), and whether the absence of an
effect is visible in the raw cells rather than only in a derived scalar (S4).

#### S1 — Containment distribution by split

![S1 containment distribution](docs/figures/s1_containment_distribution.png)

The two splits separate almost completely on a logarithmic count axis. GSM8K train concentrates
near containment 1.0; GSM8K test concentrates at 0.0, with median 0.000 and 99th percentile 0.000.
The dashed lines mark the 0.80 and 0.10 thresholds, and the sparse region between them is the
ambiguous band excluded from both arms.

#### S2 — Detector false-positive rate against known negatives

![S2 detector false positive rate](docs/figures/s2_detector_false_positive.png)

Because the corpus derives from GSM8K train, every GSM8K test item is a known negative and the
false-positive rate is measurable rather than argued. The orange series stays at or below 0.3
percent across the entire threshold sweep while the blue series falls from 86.6 to 30.3 percent,
establishing that exact detection separates the splits without flagging items that cannot be
contaminated.

#### S3 — Extraction-tier composition

![S3 extraction tiers](docs/figures/s3_extraction_tiers.png)

Marker emission differs sharply between the base checkpoints and the post-trained ones, which
raises the question of whether the four checkpoints are scored comparably. The composition is shown
directly; the corresponding second differences, reported under
[Robustness](#robustness), establish that the difference is not correlated with the treatment and
therefore cancels.

#### S4 — Cell accuracies underlying each estimate

![S4 DiD cells](docs/figures/s4_did_cells.png)

The difference-in-differences is a contrast of contrasts, and a single scalar conceals the
quantities producing it. Displaying all four cells per checkpoint permits the null to be verified
against the raw accuracies: the seen bars exceed the unseen bars at `stage2` and `instruct`, and
the gap persists rather than closing when the numbers change.

### Judge figures

#### J1 — Judge balanced accuracy by verified membership

![J1 judge balanced accuracy](docs/figures/j1_judge_balanced_accuracy.png)

Both panels carry an explicit chance line, because the headline observation is how close the
bars sit to it rather than how far apart they sit from each other. The right panel is the
reference-based control: the membership gap persists there at comparable magnitude, which is
what rules memorisation out as its cause.

#### J2 — Abstention and raw agreement, reported separately

![J2 judge abstention](docs/figures/j2_judge_abstention.png)

A judge declining to commit is not a judge disagreeing. Collapsing the two would inflate
apparent reliability, so abstention is plotted as its own series. Rates stay between 0.0 and
1.6 percent, so the balanced-accuracy figures are not an artifact of selective abstention.

#### J3 — Judge reliability per cell

![J3 judge cells](docs/figures/j3_judge_cells.png)

Answer-preserving rewrites leave the gold answer intact, so movement between original and
perturbed within an arm is judge fragility rather than task difficulty. This is the panel that
addresses paraphrase robustness directly.

---

---

## Artifacts that manufacture memorisation effects

Three separate analyses each produced an apparently significant memorisation effect that
dissolved under a control. All three are recorded with the discarded numbers intact, because
each is a trap any study of this design can fall into and none is signposted in the
literature the design draws on.

### A1 — Cohen's kappa is not comparable across arms with unequal base rates

Judge reliability was first measured as chance-corrected agreement with exact-match ground
truth. The seen-minus-unseen gap was large and excluded zero at every checkpoint:

| target | mode | κ seen | κ unseen | Δκ | CI95 |
|---|---|---|---|---|---|
| `instruct` | reference-free | +0.005 | +0.173 | −0.168 | [−0.228, −0.113] |
| `stage2` | reference-free | +0.011 | +0.303 | −0.292 | [−0.356, −0.228] |

Read naively, judge reliability collapses on memorised items. It is the **kappa paradox**.
Kappa depends on the marginals as well as on the agreement, and the arms have very different
ground-truth base rates — `instruct` is correct on 77% of seen rows against 53% of unseen.
An arm near 0.5 has far more kappa headroom than a skewed one, so the skew alone produces the
gap. Replaced by **balanced accuracy**, the unweighted mean of sensitivity and specificity,
which conditions on the true class and is therefore base-rate independent.

### A2 — A control block silently entered the estimator

Stage 7 is invoked over `scores/*__ALL.jsonl`, which merges arms, gsmsym and resample.
`make_resample_suite` emits, per item, the cluster's original **in addition to** N copies; that
original reuses the arms block's `benchmark_id` while being generated at T = 0.7, and the
copies inherit the parent's `arm` label without being answer-changing. All 600 rows landed in
the `seen|original` cell, for the two checkpoints that run the control and not the two that
do not:

```
stage2   seen|original  n=1345      (745 arms + 600 resample)
instruct seen|original  n=1345
stage1   seen|original  n= 745      (no resample block)
sft      seen|original  n= 745
```

Temperature-0.7 samples entered a temperature-0 contrast, the resampled items were weighted
once per copy, and the estimator moved **for two checkpoints only** — making a measurement
difference indistinguishable from a checkpoint difference in exactly the `stage1` to `stage2`
comparison the study rests on.

### A3 — Pooled consistency confounds membership with accuracy

Numeric perturbation changes arithmetic while preserving problem structure, so the
difference-in-differences detects recall of an *answer* but is blind to recall of a *solution
procedure*: a memorised template still executes correctly on new numbers and yields exactly
the flat estimate observed. Answer-preserving variants attack that gap from the other side,
since rephrasing and distractor insertion leave the gold answer and the required procedure
intact while changing surface form.

Pooled, the probe fired, and fired in the right place — absent at the control checkpoint,
present at every checkpoint after GSM8K enters training:

| checkpoint | C seen | C unseen | Δ | CI95 |
|---|---|---|---|---|
| `stage1` (never saw GSM8K) | 0.6833 | 0.6587 | +0.0247 | [−0.016, +0.067] |
| `stage2` (**GSM8K enters**) | 0.8893 | 0.8473 | +0.0420 | [+0.004, +0.080] |
| `sft` | 0.9233 | 0.8860 | +0.0373 | [+0.004, +0.071] |
| `instruct` | 0.9160 | 0.8780 | +0.0380 | [+0.003, +0.072] |

Stratifying on whether the cluster's original was answered correctly removes it entirely:

| checkpoint | stratum | n seen / unseen | Δ | CI95 |
|---|---|---|---|---|
| `stage2` | original correct | 176 / 116 | +0.0063 | [−0.028, +0.044] |
| `stage2` | original wrong | 74 / 134 | −0.0172 | [−0.084, +0.052] |
| `instruct` | original correct | 210 / 134 | +0.0145 | [−0.012, +0.043] |
| `instruct` | original wrong | 40 / 116 | **−0.1651** | [−0.244, −0.086] |

Accuracy and self-consistency are mechanically linked, and the seen arm holds far more correct
clusters (176 of 250 against 116 at `stage2`). Pooling let that composition masquerade as a
membership effect — **Simpson's paradox**. Inside both strata every interval spans zero, and
`instruct`'s incorrect stratum runs the opposite way.

### What survives

Two independent instruments — numeric perturbation and paraphrase consistency — agree that no
memorisation effect is detectable once the confound in each is held constant. The null is
therefore better supported than a single-instrument result would be, and the three controls
above are the reason it can be believed.

---

## Judge reliability under verified membership

An LLM judge grading a benchmark it may itself have memorised is an instrument whose
calibration depends on the thing it is measuring. Because the corpus is public, membership of
each graded item in the **judge's own** training data is verifiable, which no study using a
closed-weight judge can establish. Instella grades Instella for exactly that reason.

Two conditions. **Reference-free** withholds the gold answer, so the judge must evaluate the
reasoning, and is the only setting where memorisation could plausibly help. **Reference-based**
supplies it and serves as the control: an effect appearing there cannot be memorisation of the
solution, since the solution is given.

16,126 verdicts, four target checkpoints, two conditions, judge held at temperature 0
(same-verdict rates fall from above 95% to roughly 70% between T = 0 and T = 1, so a sampled
judge would confound the contrast with its own noise):

| target | mode | base rate seen / unseen | BA seen | BA unseen | Δ | CI95 |
|---|---|---|---|---|---|---|
| `stage1` | reference-free | 0.073 / 0.087 | 0.5296 | 0.6224 | −0.0928 | [−0.120, −0.066] |
| `stage2` | reference-free | 0.635 / 0.436 | 0.5042 | 0.6661 | −0.1619 | [−0.195, −0.130] |
| `sft` | reference-free | 0.751 / 0.690 | 0.5081 | 0.5118 | −0.0037 | [−0.017, +0.009] |
| `instruct` | reference-free | 0.773 / 0.530 | 0.5016 | 0.5825 | −0.0809 | [−0.109, −0.054] |
| `stage2` | reference-based | 0.635 / 0.436 | 0.5181 | 0.6400 | −0.1219 | [−0.155, −0.089] |
| `instruct` | reference-based | 0.773 / 0.526 | 0.5137 | 0.5678 | −0.0541 | [−0.083, −0.026] |

Three judges of increasing capability were used, because a single weak judge cannot
distinguish an instrument defect from its own noise floor.

| judge | mode | BA seen | BA unseen | Δ | CI95 |
|---|---|---|---|---|---|
| Instella 3B | reference-based | 0.5137 | 0.5678 | −0.0541 | [−0.083, −0.026] |
| Qwen2.5 7B | reference-based | 0.9370 | 0.9728 | −0.0359 | [−0.068, −0.008] |
| Qwen2.5 14B | reference-based | 0.9459 | 0.9689 | −0.0230 | [−0.057, +0.006] |
| Instella 3B | reference-free | 0.5016 | 0.5825 | −0.0809 | [−0.109, −0.054] |
| Qwen2.5 7B | reference-free | 0.5998 | 0.7998 | **−0.2000** | [−0.248, −0.149] |
| Qwen2.5 14B | reference-free | 0.6362 | 0.8202 | **−0.1840** | [−0.231, −0.133] |

*Grading `instruct` outputs. Equivalent tables for the other three targets are in
`analysis/judge_reliability.json`.*

**A competent judge grades verifiably memorised problems substantially less accurately, and
only when no reference answer is supplied.** Three observations establish it.

First, capability is real, not assumed. Given the reference answer, the 7B and 14B judges reach
balanced accuracy between 0.90 and 0.98, against 0.51 for the 3B judge. Conclusions about the
instrument are therefore drawn from instruments that work.

Second, the control behaves as a control should. As capability rises, the reference-based
membership gap converges toward zero — −0.122 to −0.002 to +0.008 across the ladder when
grading `stage2`. Supplying the correct answer for the specific variant removes the effect.

Third, withholding it does not, and scale does not rescue it. The reference-free gap on
`instruct` runs −0.081, −0.200, −0.184 across the ladder: it grows once the judge is competent
enough to exhibit it and then holds. A small-model artifact would shrink.

The mechanism this implies is substitution of recall for verification. Asked to grade a problem
present in its own training data without being told the answer, the judge appears to compare
the candidate solution against the answer it remembers rather than checking the reasoning
presented. Numerically perturbed variants have different correct answers, so a remembered
answer marks correct solutions wrong. Supplying the answer for the variant at hand removes the
reliance on recall, and with it the effect.

Two qualifications. The 3B judge sits within 0.04 of chance on the seen arm in every
configuration, so its numbers describe a noise floor rather than a judge. And the effect is
concentrated on `stage2` and `instruct` outputs; `stage1` outputs are wrong almost everywhere
(7% accuracy) and therefore trivially gradeable, while `sft` shows no gap in either condition,
which the present data cannot explain.

A design testing only the reference-free condition would have reported this as a property of
the judge. A design testing only a 3B judge would have found a gap in both conditions and
concluded the opposite. The control and the ladder are jointly load-bearing.

#### J4 — The dissociation

![J4 judge dissociation](docs/figures/j4_judge_dissociation.png)

Reference-based converges to zero as the judge scales; reference-free does not.

#### J5 — Judge competence up the ladder

![J5 judge ladder](docs/figures/j5_judge_ladder.png)

Included so the dissociation is read against judges demonstrably above chance. The dashed line
is chance for a base-rate-independent binary metric.

## Analysis artifacts

Under `experiments/runs/fullscale-S250-v2/analysis/` on Hugging Face:

| file | contents |
|---|---|
| `memorization.json` | Headline DiD, per-cell counts and accuracies, cluster-robust intervals, magnitude-matched re-estimate, cluster-robust logistic regression, trajectory and transition deltas |
| `memorization_purified.json` | DiD restricted to verified-verdict items. Identical to the headline in v2, which confirms every arm item carries a verified label |
| `containment_verified.json` | Corrected 13-gram containment for all 2,000 candidates, keyed `train::`/`test::` |
| `containment.json` | Same evidence in the shape F4 consumes |
| `detector_falsepositive.json` | Threshold sweep and false-positive rate against known negatives |
| `extraction_tiers.json` | Which extraction tier produced each answer, per checkpoint and per DiD cell |
| `termination.json` | Semantic-stop audit across all 10 generation files |
| `judge_reliability.json` | Judge balanced accuracy and abstention by verified membership, both conditions; the discarded kappa figures retained alongside |
| `consistency_by_arm.json` | Paraphrase consistency by arm, pooled and stratified on original-item correctness |

`atlas/` holds a per-checkpoint JSON + Markdown summary for all four checkpoints. `generations/`
holds raw model text — kept deliberately, because a scoring bug found after the fact once
destroyed a whole run when only scores had been saved.

## What is left

**Nothing is required.** All four checkpoints are generated, scored, analysed and plotted. The
items below are improvements, listed with honest costs.

### 1. `stage1` truncated-row repair — ~1.6 h GPU, fits a 3 h budget

`stage1/arms` reached 84.9% semantic-stop against an 85% gate. The cause is fully diagnosed and
is *not* model collapse: v1 repaired 261 truncated rows at 2,048 tokens, and v2's 644 new rows
never got that treatment.

```
reused (v1, some at 2048 tokens)   1373   85.6%
new    (1024 tokens)                644   83.5%
                                          ---- block average 84.9%
```

Critically, the truncation is **balanced across the DiD cells** — its second difference is
−0.013, well inside ±0.05 — so it attenuates every cell alike rather than biasing the contrast.
`stage1` is also the control arm, whose DiD is ~0 in all three constructions. Repairing it makes
the block internally consistent and clears the gate; it will not change the conclusion.

```bash
modal run experiments/modal_fullscale.py::repair_truncated --write
```

Then delete `scores/stage1__arms.jsonl` and re-run the suite to re-score and refresh the figures.

### 2. Scale-up for real statistical power — ~15–20 h GPU, does NOT fit a 3 h budget

The current arms cap at 250 because only 30.3% of train candidates clear the 0.80 threshold and
the suite loads just 1,000 of GSM8K train's 7,473 items. Loading all of them would yield ~2,200
verified-seen items, with arms then capped by the test split at ~1,300 per arm.

That is √(1300/250) ≈ 2.3× narrower intervals — **CI ≈ ±0.048**, which genuinely excludes 5pp
effects and would let you write the strong version of the claim. The cost is ~19,500 generations,
and `stage1` runs at 9.3 s/item, so budget 15–20 h. **This is the single highest-value remaining
experiment, and the only work separating a statement of no evidence from a statement that the
effect sizes reported elsewhere are excluded.**

### 3. Known limitations to state in the paper, not fix

- **One model family.** Every result is Instella-3B. Nothing here generalises to other models.
- **One benchmark.** GSM8K only, in a study about benchmark construct validity.
- **`stage1`→`stage2` is not a clean intervention.** Stage 2 adds Dolmino *and* Tulu-3 *and*
  GSM8K-synthetic simultaneously; the change cannot be attributed to GSM8K data alone.
- **seen/unseen is train-vs-test.** `splits.py` is explicit that the literal contamination
  question is unanswerable here. A reviewer may argue this measures a train/test generalisation
  gap. Worth pre-empting directly.

---

## Running on Modal

Colab sessions were reclaimed mid-run with no signal and no way to reattach. The suite now runs
as a detached Modal app driven from a local terminal — `experiments/modal_fullscale.py`. The
local working tree is baked into the image, so the private repo needs no token and the code that
runs cannot drift from the code on disk.

```bash
modal secret create instella-hf HF_TOKEN=hf_...          # once
modal run experiments/modal_fullscale.py::probe          # GPU health + throughput, ~3 min
modal run experiments/modal_fullscale.py::status         # inventory, no GPU
modal run --detach experiments/modal_fullscale.py --out experiments/runs/fullscale-S250-v2
modal app list                                           # find the app id
modal app logs <app-id>                                  # reattach to logs
```

Analysis entry points, all CPU-only and safe to run any time:

| function | what it does |
|---|---|
| `status` | Per-block progress against exact input row counts |
| `containment_repair` | Re-measures containment with namespaced ids; audits the built arms |
| `purified_did` | DiD restricted to verified-verdict items |
| `extraction_audit` | Extraction-tier provenance, including per DiD cell |
| `termination_by_cell` | Termination split by DiD cell and by row origin |
| `detector_falsepositive` | Detector error rate against known negatives |
| `stage_v2` | Stages a corrected run directory, reusing prior generations |
| `repair_truncated` | GPU — regenerates only truncated rows at a larger budget |

Durability has three independent layers: per-batch `fsync` with id-keyed resume, a Modal Volume
committed every 5 minutes, and an HF push after each scored block.

> **The Colab cells below are superseded** by the Modal workflow above. They are retained because
> they document the earlier runs, and because the preflight and quality-gate logic they describe
> still governs the suite.

### Cell 1 - clone or update `main`

Add Colab secrets named `github` (GitHub read token; optional if the repo is public) and
`hf` (Hugging Face token with write access). This cell keeps credentials out of the Git
remote URL and handles an existing checkout without relying on branch tracking.

```python
import base64
import os
import subprocess
from pathlib import Path

from google.colab import userdata

REPO = Path("/content/Instella-Reasoning")
REPO_URL = "https://github.com/GIND123/Instella-Reasoning.git"

def colab_secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

github_token = colab_secret("github")
git_env = os.environ.copy()
if github_token:
    git_env.update({
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + base64.b64encode(
            f"x-access-token:{github_token}".encode()
        ).decode(),
    })

if not (REPO / ".git").is_dir():
    subprocess.run(
        ["git", "clone", REPO_URL, str(REPO)],
        check=True,
        env=git_env,
    )

subprocess.run(["git", "remote", "set-url", "origin", REPO_URL], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "main"], cwd=REPO, check=True)
subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO, check=True, env=git_env)
subprocess.run(["git", "merge", "--ff-only", "origin/main"], cwd=REPO, check=True)
subprocess.run(
    ["git", "merge-base", "--is-ancestor", "bd5074b", "HEAD"],
    cwd=REPO,
    check=True,
)

commit = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=REPO,
    text=True,
).strip()
print("Repository ready:", REPO)
print("Current commit:", commit)
print("GitHub:", "https://github.com/GIND123/Instella-Reasoning")
```

### Cell 2 - install, restore, test, and preflight

This cell restores the HF run, runs the repository tests, proves HF write access, checks
the GPU/models/datasets, and performs the Tier-1 generation smoke. Continue only after
`SAFE TO LAUNCH`.

```python
import os
import subprocess
import sys
from pathlib import Path

from google.colab import userdata

REPO = Path("/content/Instella-Reasoning")
RUN_DIR = "experiments/runs/fullscale-S250"
HF_REPO = "GOVINDFROM/Instella-Reasoning"
os.chdir(REPO)

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = userdata.get("hf")
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("Missing write-enabled Colab secret named 'hf'.")

os.environ.update({
    "OUT": RUN_DIR,
    "HF_RESULTS_REPO": HF_REPO,
    "HF_HUB_DISABLE_XET": "1",
    "HF_INCLUDE_GENERATIONS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "SUITE_TIER": "1",
})

print("Installing runtime dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-e",
    ".[dev,hf,retrieval,viz,stats]",
], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "statsmodels",
], check=True)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Select Runtime > Change runtime type > GPU.")
print("GPU:", torch.cuda.get_device_name(0))

print("\nRestoring the current run from Hugging Face...")
subprocess.run([
    sys.executable, "experiments/hf_sync.py", "pull",
    "--repo", HF_REPO,
    "--path", RUN_DIR,
], check=True)

print("\nRunning repository tests...")
subprocess.run([sys.executable, "-m", "pytest"], check=True)

print("\nRunning Tier-1 GPU/HF preflight...")
preflight_json = f"{RUN_DIR}/analysis/preflight.json"
subprocess.run([
    sys.executable, "scripts/preflight_fullscale.py",
    "--smoke",
    "--out", RUN_DIR,
    "--repo", HF_REPO,
    "--tier", "1",
    "--json", preflight_json,
], check=True)

print("\nSaving successful preflight...")
subprocess.run([
    sys.executable, "experiments/hf_sync.py", "push",
    "--repo", HF_REPO,
    "--path", RUN_DIR,
    "--message", "successful Tier-1 preflight",
], check=True)

print("\nSAFE TO LAUNCH CELL 3.")
```

### Cell 3 - monitored, resumable full launch

This is the cell to rerun after a Colab interruption. It pulls HF first, launches the
remaining suite, reports live file/GPU progress every 60 seconds, and attempts a recovery
push on every exit. A score file is the completion marker; partial row counts alone are not.

```python
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import userdata

REPO = Path("/content/Instella-Reasoning")
RUN_DIR = Path("experiments/runs/fullscale-S250")
HF_REPO = "GOVINDFROM/Instella-Reasoning"
os.chdir(REPO)

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = userdata.get("hf")
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("Missing write-enabled Colab secret named 'hf'.")

run_env = os.environ.copy()
run_env.update({
    "OUT": str(RUN_DIR),
    "HF_RESULTS_REPO": HF_REPO,
    "HF_HUB_DISABLE_XET": "1",
    "HF_INCLUDE_GENERATIONS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "SUITE_TIER": "1",
    "N_PER_ARM": "250",
    "BATCH": "8",
})

print("GPU check...")
import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Select Runtime > Change runtime type > GPU.")
print("GPU:", torch.cuda.get_device_name(0))

print("\nRestoring completed and partial work from Hugging Face...")
subprocess.run([
    sys.executable, "experiments/hf_sync.py", "pull",
    "--repo", HF_REPO,
    "--path", str(RUN_DIR),
], check=True, env=run_env)

def count_jsonl(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (FileNotFoundError, OSError):
        return 0

def print_progress():
    generation_files = [
        Path(path) for path in glob.glob(str(RUN_DIR / "generations/*.jsonl"))
    ]
    score_files = [
        Path(path) for path in glob.glob(str(RUN_DIR / "scores/*.jsonl"))
        if not path.endswith("__ALL.jsonl")
    ]
    newest = max(generation_files, key=lambda path: path.stat().st_mtime, default=None)
    gpu = subprocess.run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False).stdout.strip()
    current = (
        f"{newest.name}: {count_jsonl(newest)} rows"
        if newest is not None else "no generation file yet"
    )
    print(
        f"[progress] {current} | completed score blocks={len(score_files)}"
        + (f" | GPU {gpu}" if gpu else ""),
        flush=True,
    )

print("\nLaunching or resuming the full Tier-1 analysis...")
process = subprocess.Popen(
    ["bash", "experiments/run_fullscale_suite.sh"],
    env=run_env,
)

exit_code = None
try:
    while process.poll() is None:
        time.sleep(60)
        print_progress()
    exit_code = process.returncode
except KeyboardInterrupt:
    print("\nInterrupt received; stopping the suite cleanly...")
    process.send_signal(2)
    try:
        exit_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        exit_code = process.wait()
finally:
    print("\nCheckpointing all current local progress to Hugging Face...")
    subprocess.run([
        sys.executable, "experiments/hf_sync.py", "push",
        "--repo", HF_REPO,
        "--path", str(RUN_DIR),
        "--message", "Cell 3 exit checkpoint",
    ], check=False, env=run_env)

if exit_code != 0:
    raise RuntimeError(
        f"Suite stopped with exit code {exit_code}. "
        "Local work was checkpointed; rerun Cell 3 to resume after diagnosing "
        "any termination-gate message above."
    )

print("\nFULL TIER-1 ANALYSIS COMPLETE.")
print("Results:", RUN_DIR / "analysis/memorization.json")
print("Figures:", RUN_DIR / "figures")
print("HF:", "https://huggingface.co/datasets/GOVINDFROM/Instella-Reasoning")
```

Expected first messages for the current checkpoint:

```text
== skip stage1/arms (already scored) ==
== skip stage1/gsmsym (already scored) ==
== generate stage2/arms ==
```

HF normally changes after a scored block finishes, not while a block is generating. The
one-minute Cell 3 monitor is the live signal during long blocks. Full design, budget,
artifacts, recovery behavior, and validity gates are documented in
[`experiments/FULLSCALE.md`](experiments/FULLSCALE.md).

When a language model answers a reasoning problem correctly, **accuracy alone cannot
tell you whether it reasoned or remembered.** Instella is one of the very few
competitive model families that is *fully open* — weights, code, data recipe, **and
the training data itself**. That makes the question empirically testable: for every
benchmark problem the actual training corpus can be searched for near-duplicates, then
measure whether correct answers survive semantically-equivalent rewrites, then trace
which training documents drove them.

This repository is the **runnable research pipeline** for that study. It is built to
be **cloned and run end-to-end on a Colab T4 or a CPU-only laptop** — heavy pieces
(real embeddings, FAISS, model generation, gradient attribution) are optional extras
that degrade gracefully to dependency-free fallbacks so the whole thing runs, and its
tests pass, anywhere.

## The four-stage method

```
              ┌──────────────┐   ┌───────────────┐   ┌───────────────┐   ┌──────────────┐
   benchmarks │ 1. FILTER    │   │ 2. DIAGNOSE   │   │ 3. ATTRIBUTE  │   │ 4. PRESCRIBE │
   + Instella │ contamination│──▶│ reliability = │──▶│ which training│──▶│ data curation│
   train data │  C / PC / N  │   │ acc × consist.│   │ docs behind it│   │ guidelines   │
              └──────────────┘   └───────────────┘   └───────────────┘   └──────────────┘
                    │                    │                    │                   │
              contamination.jsonl   atlas.json          attribution.jsonl     report.md
                                  (Reliability Atlas)   (concentrated/diverse)  + figures
```

1. **Filter** — for every benchmark item, embed + 13-gram search Instella's training
   data and label it **Contaminated / Partially-contaminated / Clean**.
2. **Diagnose** — expand each item into a consistency cluster (numeric, entity,
   reorder, distractor, rephrase variants) and score **Reliability = Accuracy ×
   Consistency**. High accuracy with low consistency is the *memorization signature*.
3. **Attribute** — characterize each item's nearest training neighbours
   (**concentrated** near-duplicates ⇒ memorization; **diverse** ⇒ generalization),
   then refine the interesting ones with TracIn-CP / Concept Influence.
4. **Prescribe** — assemble the **Reasoning Reliability Atlas**: a per-sub-skill map of
   which reasoning is genuine, fragile, or absent, and what training data drives each.

## Run instructions

Pick **one** of the three blocks below and copy-paste it whole. Each is self-contained
and starts from a fresh clone. If you just want to see it work, use **A**.

### Cloning a private repo (Colab / CI)

If this repository is **private**, a plain `git clone https://github.com/...` fails in a
non-interactive shell (Colab, CI) with `fatal: could not read Username for
'https://github.com'` — git is trying to prompt for credentials it can't get.
Authenticate with a token instead.

**Colab** (store a token in *Secrets* as `github`, with repo `Contents: read`):

```python
from google.colab import userdata
import subprocess
token = userdata.get("github")
# subprocess (not !git) keeps the token out of the printed cell output.
subprocess.run(
    ["git", "clone",
     f"https://x-access-token:{token}@github.com/GIND123/Instella-Reasoning.git"],
    check=True,
)
%cd Instella-Reasoning
# Scrub the token from the saved remote URL (you can still pull read-only after this):
subprocess.run(
    ["git", "remote", "set-url", "origin",
     "https://github.com/GIND123/Instella-Reasoning.git"],
    check=True,
)
```

**Shell / CI** (token in `$GITHUB_TOKEN`):

```bash
git clone https://x-access-token:${GITHUB_TOKEN}@github.com/GIND123/Instella-Reasoning.git
```

The `git clone https://github.com/...` lines shown in blocks A–C below work as-is for a
**public** repo; swap in the token form above if yours is private.

### A. Run it now — CPU only, no GPU, no downloads (~1 min)

Runs the **entire pipeline** on bundled example data. Works on Linux/Mac/Colab.

```bash
git clone https://github.com/GIND123/Instella-Reasoning
cd Instella-Reasoning
pip install -e ".[dev]"

instella-reasoning run-all --config configs/pipeline/full.yaml
```

That's it. Open the results:

```bash
cat outputs/atlas_run/report.md          # the reliability report
ls  outputs/atlas_run/                    # atlas.json, contamination.jsonl, scores.jsonl, ...
```

> Want the Atlas **figures** (PNG) too? Add the viz extra: `pip install -e ".[dev,viz]"`
> and re-run — figures land in `outputs/atlas_run/figures/`.

<details>
<summary>Windows PowerShell version of block A</summary>

```powershell
git clone https://github.com/GIND123/Instella-Reasoning
cd Instella-Reasoning
pip install -e ".[dev]"

instella-reasoning run-all --config configs/pipeline/full.yaml
type outputs\atlas_run\report.md
```
</details>

### B. Run it on Google Colab (T4 GPU, real Instella model)

First set **Runtime → Change runtime type → T4 GPU**, then paste this into **one Colab
cell**. It preflights the GPU, installs the GPU extras, authenticates to HuggingFace,
downloads a small real slice, verifies output on a tiny sample, then runs the batch.

```python
# Public repo: this line works as-is. PRIVATE repo: replace it with the token-based
# clone from "Cloning a private repo" above (a plain clone fails non-interactively).
!git clone https://github.com/GIND123/Instella-Reasoning
%cd Instella-Reasoning
# The hf extra pins transformers<5 on purpose: Instella ships custom remote modeling code
# for the 4.4x API, and transformers 5.x silently breaks it (degenerate output). Do not
# upgrade transformers past 5 for this model.
!pip install -q -e ".[hf,retrieval,viz,stats]" && pip install -q bitsandbytes

# 0) Preflight: is a real GPU attached? (a CPU-only runtime makes generation unusable)
!python scripts/preflight.py

# 1) HuggingFace auth + Xet workaround (add an HF token in Colab Secrets as HF_TOKEN)
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 2) Smoke-test the install on bundled data, then download a small real slice
!instella-reasoning run-all --config configs/pipeline/full.yaml
!bash scripts/download_data.sh 200 5000

# 3) VERIFY on 4 items first — the completion must be coherent GSM8K reasoning.
#    (Chat templating for -Instruct models is automatic.) --fail-degenerate 0.25 makes the
#    cell exit non-zero (and print a loud banner) if the output loops/empties out, so a
#    broken run stops HERE instead of silently poisoning the full batch and the Atlas.
!instella-reasoning generate \
    --benchmark data/processed/gsm8k.jsonl \
    --model amd/Instella-3B-Instruct \
    --output outputs/gsm8k_smoke.jsonl \
    --limit 4 --max-new-tokens 256 --batch-size 2 --fail-degenerate 0.25
!head -c 800 outputs/gsm8k_smoke.jsonl

# 4) Full run. Drop --load-in-4bit for the accurate bf16 number (see note); keep it
#    only for a fast, lower-fidelity pass.
!instella-reasoning generate \
    --benchmark data/processed/gsm8k.jsonl \
    --model amd/Instella-3B-Instruct \
    --output outputs/gsm8k_generations.jsonl \
    --batch-size 8

# 5) Quality gate — refuse to proceed if >10% of completions are degenerate
#    (a formatting bug should never silently flow into the Atlas).
!instella-reasoning check-generations \
    --generations outputs/gsm8k_generations.jsonl \
    --output outputs/gsm8k_flagged.jsonl --fail-threshold 0.1

!instella-reasoning score-generations \
    --benchmark data/processed/gsm8k.jsonl \
    --generations outputs/gsm8k_generations.jsonl \
    --output outputs/gsm8k_scores.jsonl
```

> **Three things that decide whether you get real answers vs. gibberish:**
> 1. **A GPU must actually be attached.** If `preflight.py` reports `cpu_only_build` /
>    `cuda_available=False`, the 3B model falls back to CPU and generation is unusably
>    slow — fix the runtime type before going further.
> 2. **`transformers` must be < 5.** Instella ships custom remote modeling code for the
>    4.4x attention-mask/`cache_position` API; transformers 5.x removed it and the forward
>    pass degenerates into looping text. The `hf` extra pins `transformers>=4.44,<5` for
>    you — don't `pip install -U transformers` past it. On a T4 you can also add
>    `--dtype fp16` for speed (Turing has no native bf16), and `--revision <sha>` to pin
>    the checkpoint.
> 3. **Instruct models need their chat template**, which the pipeline applies
>    automatically. For **accuracy numbers**, prefer **bf16** (omit `--load-in-4bit`);
>    4-bit NF4 is a fast, lower-fidelity mode for smoke checks, not headline metrics.

> Prefer the notebook UI? Open
> [`notebooks/instella_reasoning_atlas.ipynb`](notebooks/instella_reasoning_atlas.ipynb)
> in Colab and Run all — it does the same steps in separate cells.

### C. Full study — all extras, all data

```bash
git clone https://github.com/GIND123/Instella-Reasoning
cd Instella-Reasoning
pip install -e ".[all]"                    # hf + retrieval + viz + stats + train

bash scripts/download_data.sh 0 0          # 0 0 = no limit → full benchmarks + corpus shards
# For a reportable run use the rigorous config (numeric variants, difficulty-adjusted gap,
# real MiniLM embeddings, bf16); edit its data/model paths first.
instella-reasoning run-all --config configs/pipeline/rigorous.yaml
```

See [`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md) for the complete AMD data procedure
and licensing before a full download.

### D. The reliability run — the headline result (≤20 GPU-h, auto-saving, resumable)

Blocks A–C measure **accuracy**. The paper's actual claim (reasoning vs memorising) needs
**consistency**, which requires generating each model over *variant clusters*, not base
items. [`experiments/run_reliability_suite.sh`](experiments/run_reliability_suite.sh) does
exactly that — contamination index → variant clusters → generate-over-variants → score →
atlas (consistency-probed) → difficulty-adjusted gap → emergence → report + figures — and
**periodically saves every artifact to Hugging Face so any teammate can resume where the
last session stopped, never re-spending GPU hours.**

#### GPU-hour budget (single T4)

Time is dominated by generation. `gens = base_items × (1 + 4 surface + numeric_k) × models
× benchmarks`. The runner conservatively defaults to **6 s/item** on a T4 (blend of
OLMo-1B, 4-bit 3B, and full-precision 3B); replace this assumption with the sanity run's
measured rate. The suite prints its estimate on startup and warns past ~9 h.

| Config (all 4 models) | generations | ~time @6s | fits 20 h? |
|---|---:|---:|:--:|
| **GSM8K only, 120 items** (default) | 4,800 | **~8 h** | ✅ |
| GSM8K only, 200 items | 8,000 | ~13.3 h | ✅ |
| **GSM8K + MATH, 120 items** (recommended) | 9,600 | **~16 h** | ✅ tight |
| GSM8K + MATH + LogiQA2, 120 items | 14,400 | ~24 h | ❌ trim/split |
| Instruct only, GSM8K, 120 items (smoke) | 1,200 | ~2 h | ✅ |

> **Calibrate before you commit the budget.** Run a `BASE_ITEMS=15` sanity pass (~15 min),
> read the reported wall-clock, then set `SECONDS_PER_ITEM` to your measured rate so the
> printed estimate is accurate for your hardware. Split the four models across sessions with
> `SUITE_MODELS` (e.g. `SUITE_MODELS="instruct"`) — they all write to the same run dir and
> the HF sync stitches the pieces together.

#### One Colab cell (Runtime → T4 GPU first)

Assumes your **GitHub token** is in Colab Secrets as **`github`** (only needed if the repo is
private) and your **Hugging Face token** as **`hf`** (needs *write* access, since results are
pushed to your dataset repo).

```python
import os, subprocess
from google.colab import userdata

# 1) Clone. Public repo: the plain clone works. Private repo: uncomment the token form.
!git clone https://github.com/GIND123/Instella-Reasoning
# tok = userdata.get("github")
# subprocess.run(["git","clone",
#   f"https://x-access-token:{tok}@github.com/GIND123/Instella-Reasoning.git"], check=True)
%cd Instella-Reasoning

# 2) Install (hf extra brings huggingface_hub used by the result sync) + 4-bit loader.
!pip install -q -e ".[hf,retrieval,viz,stats]" && pip install -q "bitsandbytes>=0.46.1"

# 3) Tokens from Colab Secrets. HF_TOKEN drives both model downloads AND the result sync.
os.environ["HF_TOKEN"] = userdata.get("hf")          # HF secret named "hf" (write access)
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 4) Run. Defaults = GSM8K, 120 items, 4 models (~8 GPU-h). Results stream to HF as they
#    finish; a reclaimed VM just re-runs this cell and continues from the last checkpoint.
!BENCHMARKS="gsm8k math" bash experiments/run_reliability_suite.sh
```

#### How the auto-save + resume works (for the whole team)

- **On start** the suite *pulls* the run directory from the private
  `GOVINDFROM/Instella-Reasoning` dataset on Hugging Face. Each finished
  `(model, benchmark)` has a score file that acts as a
  **completion marker**, so generation skips whatever is already done.
- **As it runs** it *pushes* after every `(model, benchmark)`, after each atlas, and at the
  end — so a Colab timeout, VM recycle, or a teammate picking it up on another machine
  **loses at most the one benchmark in flight**, never the whole run.
- **Any teammate** runs the *same cell* (with their own `hf` secret) and it fast-forwards to
  the first unfinished piece. No coordination needed; no GPU hours wasted re-generating.

Knobs (env vars): `HF_RESULTS_REPO` (default `GOVINDFROM/Instella-Reasoning`),
`HF_SYNC=0` to disable syncing, `HF_INCLUDE_GENERATIONS=1` to also upload the bulky raw
generations (scores alone are enough to resume), `BASE_ITEMS`, `NUMERIC_K`, `BENCHMARKS`,
`SUITE_MODELS`, `SECONDS_PER_ITEM`. Full mechanics: [`experiments/GPU_SUITE.md`](experiments/GPU_SUITE.md).

### Research-grade run (headline numbers)

For results you intend to report, the pipeline adds methodology controls that keep
quantization noise, formatting bugs, and difficulty confounds out of the Atlas. Start with
the three quick rules below; then use the [one-command rigorous run](#the-one-command-rigorous-run)
for the full set (GSM-Symbolic numeric variants, difficulty-adjusted gap, threshold
calibration, metric validation). The methodology audit behind each control is in
[`docs/REVIEW.md`](docs/REVIEW.md).

1. **bf16 is the headline, 4-bit is a fast pass.** Omit `--load-in-4bit` for the number
   you report; NF4 measurably shifts accuracy. Every generation records its `precision`,
   and `report.md` prints a **Generation provenance** block that warns when any number came
   from 4-bit. Run both to quantify the quantization gap:
   ```bash
   instella-reasoning generate --benchmark data/processed/gsm8k.jsonl \
       --model amd/Instella-3B-Instruct --output outputs/gsm8k_bf16.jsonl --batch-size 8
   instella-reasoning generate --benchmark data/processed/gsm8k.jsonl \
       --model amd/Instella-3B-Instruct --output outputs/gsm8k_4bit.jsonl \
       --load-in-4bit --batch-size 8
   ```
2. **Gate on degenerate output.** `check-generations` flags looping/empty completions
   (the missing-chat-template failure mode) via distinct-token ratio, repeated-trigram, and
   longest-run signals. `run-all` runs this gate automatically and surfaces it in the report;
   run it standalone in CI with `--fail-threshold`:
   ```bash
   instella-reasoning check-generations --generations outputs/gsm8k_bf16.jsonl \
       --output outputs/flagged.jsonl --fail-threshold 0.05   # exits non-zero if > 5% degenerate
   ```
3. **Use the full MATH benchmark.** `load-benchmark --benchmark math` (no `--hf-name`) now
   loads and interleaves **all seven MATH subjects** for a balanced set; add `--hf-name geometry`
   only to restrict to one subject.
   ```bash
   instella-reasoning load-benchmark --benchmark math --output data/processed/math.jsonl   # all 7 subjects
   ```

#### The one-command rigorous run

`configs/pipeline/rigorous.yaml` turns all of the above on at once — numeric variants,
consistency clusters, real MiniLM embeddings, bf16 generation, and the difficulty-adjusted
gap (written automatically as `accuracy_gap_stratified.json` whenever scores exist):

```bash
instella-reasoning run-all --config configs/pipeline/rigorous.yaml
```

Or drive the controls stage-by-stage:

```bash
# 1. Consistency clusters WITH GSM-Symbolic-style numeric variants, then audit them.
instella-reasoning make-variants --benchmark data/processed/gsm8k.jsonl \
    --output data/processed/gsm8k_variants.jsonl --numeric-k 5
instella-reasoning validate-variants --benchmark data/processed/gsm8k_variants.jsonl

# 2. Difficulty-adjusted, cluster-robust, FDR-corrected contamination gap.
instella-reasoning accuracy-gap --scores outputs/scores.jsonl \
    --contamination outputs/contamination.jsonl --output outputs/accuracy_gap.json \
    --benchmark data/processed/gsm8k.jsonl --stratified-output outputs/accuracy_gap_stratified.json

# 3. Calibrate contamination thresholds against a hand-labeled set (PR/F1).
instella-reasoning calibrate-contamination --contamination outputs/contamination.jsonl \
    --ground-truth data/labels/contamination_truth.jsonl --output outputs/calibration.json
```

Why each exists — the methodology audit and how the code answers it — is in
[`docs/REVIEW.md`](docs/REVIEW.md).

### Run a single stage

`run-all` chains everything, but every stage is also a standalone command you can copy
and run on its own — see the [command reference](#command-reference). Example: just the
contamination scan on the bundled example data:

```bash
instella-reasoning scan-contamination-embedding \
  --benchmark examples/mini_benchmark.jsonl \
  --corpus    examples/mini_corpus.jsonl \
  --output    outputs/contamination.jsonl \
  --embedder-backend hashing --index-backend bruteforce
```

## Install profiles

| Extra | Install | Enables |
|---|---|---|
| base | `pip install -e .` | Core JSONL pipeline, contamination, metrics, stats, atlas — CPU, no third-party ML deps |
| `dev` | `.[dev]` | pytest + ruff |
| `hf` | `.[hf]` | HuggingFace `datasets`/`transformers` — benchmark/corpus loading, model generation |
| `retrieval` | `.[retrieval]` | `sentence-transformers` + `faiss` + `numpy` — real MiniLM/GTE embeddings & FAISS |
| `viz` | `.[viz]` | `matplotlib` — the Atlas figure set |
| `stats` | `.[stats]` | `scipy` — exact p-values (dependency-free fallback otherwise) |
| `train` | `.[train]` | `torch`/`peft`/`trl`/`wandb` — gradient attribution + upstream training launch |
| `all` | `.[all]` | everything |

## Command reference

Every command reads and writes JSONL/JSON so long jobs can be sharded, resumed, and merged.

| Command | Stage | Purpose |
|---|---|---|
| `run-all` | orchestrator | **One config → the whole Atlas** (contamination → score → gap → attribution → atlas → report → figures). |
| `load-benchmark` | data | Convert a HuggingFace reasoning benchmark (gsm8k, math, arc_challenge, logiqa2, bbh) to the JSONL contract. |
| `load-corpus` | data | Stream a HuggingFace corpus/dataset into `CorpusDocument` JSONL (streaming when `--limit` set). |
| `make-variants` | 2 | Expand a benchmark into consistency clusters; `--numeric-k` adds GSM-Symbolic-style answer-changing numeric variants. |
| `validate-variants` | 2 | Audit a variant suite: answer-preservation rate + text well-formedness. |
| `scan-contamination` | 1 | Lightweight lexical contamination scan. |
| `scan-contamination-embedding` | 1 | Embedding + 13-gram scan → C / PC / N labels. |
| `generate` | 2 | Generate completions with Transformers (chat template, 4-bit/`--dtype`/`--revision`, batching, CoT, `--limit`, `--fail-degenerate`). Auto-runs the degeneracy gate. |
| `score-generations` | 2 | Benchmark-aware answer extraction + exact-match scoring. |
| `check-generations` | 2 | Quality gate: flag degenerate (looping/empty) completions; `--fail-threshold` for CI. |
| `accuracy-gap` | 2 | Contaminated-vs-clean accuracy, two-proportion z-test + BH-FDR; `--stratified-output` adds a difficulty-adjusted, cluster-robust gap. |
| `calibrate-contamination` | 1 | Sweep the cosine threshold against a labeled set → precision/recall/F1. |
| `validate-reliability` | 3 | Check the Reliability metric ranks genuine > fragile clusters on a labeled set. |
| `attribute` | 3 | Retrieval-correctness attribution proxy (triage queue). |
| `attribute-embedding` | 3 | Tier-1 attribution: neighbour concentration (Gini), source shares, concentrated/diverse verdict. |
| `atlas` | 3 | Build the Reasoning Reliability Atlas (JSON + Markdown). |
| `emergence` | 5 | Scale transitions + RL-effect analysis across model score files. |
| `plots` | 6 | Render the Atlas figure set (needs `viz`). |
| `report` | 6 | Markdown reliability report. |
| `train-command` | — | Render an upstream Instella `torchrun` command from a YAML launch config. |

## Package map

```
src/instella_reasoning/
├── records.py               # JSONL dataclasses + IO (the exchange contract)
├── text.py                  # normalization, n-grams, overlap metrics
├── datasets/loaders.py      # HF benchmark/corpus → JSONL (skill-tagged; MATH all-subjects)
├── prompting.py             # CoT templates + benchmark-aware answer extractors
├── answer_equivalence.py    # math-aware scoring: string -> numeric -> SymPy symbolic  [Phase 2]
├── provenance.py            # run manifest: git commit + package versions + config   [repro]
├── evaluation.py            # model generation (chat template/4-bit/dtype/CoT) + scoring
├── quality.py               # degeneracy gate: flag looping/empty completions   [Phase 2]
├── difficulty.py            # model-independent difficulty (steps/level/length) [confound control]
├── gsm_symbolic.py          # GSM-Symbolic-lite: validated numeric variants     [Phase 3]
├── validation.py            # contamination PR calibration + reliability construct-validity
├── embedding.py             # MiniLM/GTE embedder + dependency-free hashing fallback
├── faiss_index.py           # FAISS (flat/ivfpq/hnsw) + brute-force fallback, save/load
├── contamination.py         # lexical + embedding/13-gram scanners → C/PC/N   [Phase 1]
├── perturbations.py         # answer-preserving variant generators + validation  [Phase 3]
├── metrics.py               # Reliability = accuracy × consistency (type-aware)
├── attribution.py           # retrieval-correctness proxy (triage)             [Phase 3]
├── attribution_embedding.py # Tier-1 neighbour characterization (Gini/sources) [Phase 4]
├── attribution_gradient.py  # TracIn-CP + Concept Influence (torch-gated)      [Phase 4]
├── emergence.py             # transitions, Schaeffer test, RL effect, CoT divergence [Phase 5]
├── reporting.py             # summary + atlas + gap + attribution report
├── pipeline.py              # end-to-end orchestrator (run-all)
├── training.py              # upstream Instella torchrun command builder
├── cli.py                   # the `instella-reasoning` CLI
└── analysis/
    ├── accuracy_gap.py      # contaminated-vs-clean gap: z-test + FDR + difficulty-adjusted [Phase 2]
    ├── atlas.py             # the Reasoning Reliability Atlas builder          [Phase 3/6]
    ├── stats.py             # bootstrap CI, Mann-Whitney, effect sizes, BH-FDR [Phase 6]
    └── plots.py             # publication figures (validated palette)          [Phase 6]
```

## The Reasoning Reliability Atlas

The central deliverable. Each `(sub-skill × contamination-level)` cell reports
accuracy, consistency, reliability, and a verdict:

| Accuracy | Consistency | Reliability | Verdict |
|---|---|---|---|
| High | High | High | **GENUINE** — understands the problem structure |
| High | Low | Low | **FRAGILE** — recognises the original, fails variants (memorization) |
| Low | High | Low | **GAP (consistent)** — lacks the skill, but consistently |
| Low | Low | Very low | **GAP** — no grasp of the problem type |

The headline analysis cross-references this with contamination: if contaminated items
show high accuracy but low consistency while clean items show the reverse, benchmark
contamination is inflating accuracy without building reasoning.

## Data & models

Instella's full training data is public, which is what makes ground-truth
contamination search possible. The exact HuggingFace dataset IDs, the priority order
for indexing, streaming/sampling strategy, sizes, and licensing (ResearchRAIL) are in
**[`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md)**. Model variants (OLMo-1B →
Instella-3B → 3B-Instruct → 3B-Math) and their VRAM footprints are there too.

## Compute

A single **T4 (16 GB) + CPU** covers the whole study; the analysis layer runs
dependency-free on a laptop. Full budget, per-model VRAM, FAISS index sizing, and the
attribution fallback ladder are in **[`docs/COMPUTE.md`](docs/COMPUTE.md)**.

## Portability contract

- **Core pipeline** (loaders, embedding, FAISS, contamination, perturbations, metrics,
  stats, atlas, report) runs on **CPU-only Windows/Linux/Mac** at smoke/sample scale
  with **no third-party ML dependency**.
- **Heavy pieces** (model generation, real embeddings, TracIn) are gated behind extras
  and fail with a clear install message rather than an import error.
- **All paths, configs, and CLI commands work identically** on Colab and Windows.

## Testing

```bash
pytest              # unit + analysis tests (gsm_symbolic, difficulty, stratified gap,
                    #   quality gate, validation, perturbations, atlas, stats, emergence, plots)
ruff check .        # lint
make smoke          # example pipeline end-to-end
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint + tests + the
smoke pipeline on every push. The model-generation path is covered by hermetic tests
(chat-template formatting, quality gate) that need no GPU or model download.

## Upstream Instella training

The scaffold does not replace AMD's trainer; it wraps the research workflow around it
and can render launch commands. See [`docs/GITHUB_TRAINING.md`](docs/GITHUB_TRAINING.md).

```bash
git clone https://github.com/AMD-AGI/Instella external/Instella
instella-reasoning train-command --config configs/training/amd_base_upstream.yaml
```

## Documentation

| Doc | Contents |
|---|---|
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Phased roadmap: modules, data, tests, analyses, plots |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Method + metric definitions |
| [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md) | JSONL schemas |
| [`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md) | AMD data download procedure + licensing |
| [`docs/COMPUTE.md`](docs/COMPUTE.md) | Compute budget + hardware |
| [`docs/GITHUB_TRAINING.md`](docs/GITHUB_TRAINING.md) | Self-hosted runner training |
| [`docs/proposal/`](docs/proposal/) | Full research proposal + literature compendium |

## Source basis

Grounded in AMD's Instella repository (<https://github.com/AMD-AGI/Instella>), the
`amd/Instella-3B` model card, and the proposal/compendium in `docs/proposal/`.
Instella is a 3.11B-parameter decoder-only model (36 layers, 32 heads, hidden 2560),
4096-token context, OLMo tokenizer (~50K vocab), trained on ROCm/MI300X with
FlashAttention-2, torch.compile, bf16, and FSDP over ~4.15T tokens across two
pretraining stages plus SFT and DPO.

## License & citation

MIT for this scaffold. Instella checkpoints and the AMD GSM8K-synthetic dataset are
released for research under AMD's ResearchRAIL terms — confirm each card before
training, redistribution, or publication.

```bibtex
@article{liu2025instella,
  title={Instella: Fully Open Language Models with Stellar Performance},
  author={Liu, Jiang and Wu, Jialian and Yu, Xiaodong and Su, Yusheng and Mishra, Prakamya and Ramesh, Gowtham and Ranjan, Sudhanshu and Manem, Chaitanya and Sun, Ximeng and Wang, Ze and Brahma, Pratik Prabhanjan and Liu, Zicheng and Barsoum, Emad},
  journal={arXiv preprint arXiv:2511.10628},
  year={2025}
}
```
