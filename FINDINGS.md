# Findings of record

Every number here has been produced from generation files and independently re-derived at
least once. Where a claim was withdrawn, the reason is given rather than the claim deleted,
because the same idea will otherwise be re-proposed by whoever reads this next.

Two manuscripts are live:

| paper | venue | deadline | length | source |
|---|---|---|---|---|
| MATH-AI | 6th Workshop on Mathematical Reasoning and AI, NeurIPS 2026 | **25 Sep 2026** | 4 pages body | `paper/mathai/neurips_2026.tex` |
| JUDGe | Can We Trust the Judge?, NeurIPS 2026 | **29 Aug 2026** | 6 pages body | `paper/judge/main.tex` |

Both are double-blind, non-archival, and have **no rebuttal phase**, so every objection has
to be pre-empted in the text. JUDGe additionally uses reciprocal reviewing: one author must
register to review.

Inference throughout is a cluster bootstrap over **parent problems**, 4000 draws, seed 6198.
The measured intraclass correlation is ~0.48, so resampling rows rather than parents would
contract intervals by a design effect near 2.4. An asterisk below means the interval
excludes zero.

---

## 1. MATH-AI: what holds

### 1.1 Exposure does not raise answer recall (the headline)

Premise-deletion probe, 1591 train / 1588 test rows over 891 / 888 parents, four checkpoints.

| checkpoint | train | test |
|---|--:|--:|
| Stage 1 | 1.13% | 1.64% |
| Instella 3B | 2.83% | 3.59% |
| SFT | 3.08% | 3.34% |
| Instruct | 3.58% | 3.15% |

Difference in differences at the boundary where the stage-two corpus enters:
**−0.26 pp [−1.66, +1.14]**. Later boundaries +0.25 (SFT) and +0.94 (Instruct), both
crossing zero.

Survives: restriction to 1–2 digit answers (−0.16), restriction to terminated generations
(−0.26), extension down the trajectory, a more sensitive outcome measure (§3.1 below), and
a within-arm containment gradient using no train/test contrast at all — bins run
**3.7 / 1.5 / 0.9 / 4.3 / 4.1 percent** from least to most contained, non-monotonic.

Design resolution: **1.89 pp at 80% power** over 891 parent clusters. Every observed gap is
below that floor.

### 1.2 Memorisation returns the procedure, not the answer

Controlled injection, fixed 8.4M-token budget, five doses, positive controls in every arm.
Change relative to a never-injected control:

| dose | verbatim | answer recall | procedure reproduction |
|---|--:|---|---|
| 1× | 15.3% | +0.06 [−3.06, +3.20] | +0.48 [−2.47, +3.29] |
| 4× | 18.0% | +0.09 [−2.42, +2.70] | +0.32 [−2.33, +3.01] |
| 16× | 49.1% | −0.22 [−3.66, +3.07] | +2.10 [−1.27, +5.33] |
| 64× | 94.6% | +1.52 [−1.93, +5.08] | **+4.76 [+1.30, +8.23]*** |

Document repetition in the released corpus tops out at 4, well left of where the effect
appears.

### 1.3 Models compose

Appending one dependent step to problems Instruct already solves: **88.0% remain correct**
(368/418), with **zero** reversions to the pre-extension answer.

### 1.4 Abstention under a licensing prompt (new, 2026-08-23)

Identical 3179 deletion items, prompt differing only in naming `#### unanswerable`:

| checkpoint | abstains | 95% CI | under answer-forcing prompt |
|---|--:|---|--:|
| Stage 1 | 7.5% | [6.1, 9.0] | ~0.1% |
| Instella 3B | 11.1% | [9.5, 12.8] | ~0.0% |
| SFT | 42.4% | [39.8, 45.1] | ~0.4% |
| Instruct | **53.2%** | [50.6, 55.9] | ~2.1% |

Validated: all 1709 detected abstentions at Instruct are explicit refusals ending in the
licensed token; **none** came from the "no extractable number" fallback; none truncated.
Train vs test indistinguishable at every checkpoint (53.2 vs 54.3 at Instruct). The
strongest checkpoint still answers **44%** of provably unanswerable problems.

### 1.5 Post-training fixes surface robustness

Paired, gold verified identical on every pair: rewording −6.6 pp (Instella 3B) → −0.8 n.s.
(Instruct); inserted off-topic sentence −3.6 → −0.6 n.s.

---

## 2. MATH-AI: what was withdrawn, and why

### 2.1 "An inserted number costs 6.2 points" — WITHDRAWN

The old comparison was `irrelevant_context` (5 words, off-topic, no number) against
`distractor_quantity` (15 words, domain-related, number). Three things varied at once.

Crossed design, all four insertions one 20-word frame with two words swapped, 1764 parents,
complete 2×2 on every one. **Main effect of the number, pre-registered as primary:**

| checkpoint | effect | 95% CI |
|---|--:|---|
| Instella 3B | −0.59 | [−1.62, +0.45] |
| SFT | −0.79 | [−1.93, +0.31] |
| Instruct | −0.42 | [−1.53, +0.68] |

Null everywhere. Topicality null too. What survives is a cost of insertion itself, ~2 pp at
stage two and Instruct — **SFT shows no insertion cost at all** and is the exception. The
6.2 figure was carried by length and topicality.

### 2.2 Irrelevance-marker ablation — holds at ONE checkpoint only

Stripping "and has no bearing on this question", holding the sentence at 20 words:

| checkpoint | marked | unmarked | marker effect |
|---|--:|--:|---|
| Instella 3B | −2.21 | −5.67 | **−3.46 [−4.90, −2.01]*** |
| SFT | +0.09 | +0.11 | +0.03 [−1.33, +1.36] |
| Instruct | −2.21 | −2.78 | −0.57 [−1.93, +0.79] |

One of three is a single result, not a pattern. Lives in an appendix with that caveat
stated. It connects to Shi et al. (ICML 2023), who report that instructing a model to
ignore irrelevant information mitigates the loss — but it does **not** explain the residual
2 pp at Instruct, which neither the number, topicality, nor the marker accounts for.

### 2.3 GSM-Symbolic clause ladder (−34 to −39 pp) — WITHDRAWN

P1/P2 instances are **independently generated, structurally harder problems**, not the same
items with clauses appended. Template heads match across rungs; instances do not. So the
drop partly measures difficulty. Consistent with Ivanova (2024).

### 2.4 Premise reordering (−12.8 pp) — WITHDRAWN

The generator permutes premises without checking cross-sentence dependency, so many
variants are broken rather than reordered ("*This week she sent 50 less than double what
she sent last week*" placed before the sentence introducing last week). Filtering to items
with no pronoun, temporal connective or back-reference in later premises leaves 24% of the
set, and the effect becomes **+2.2 [−6.5, +10.9]** on 46 problems at Instruct. Manual
inspection still finds violations surviving the filter. **Do not rebuild this.**

### 2.5 Arithmetic step validity across the trajectory — NOT USABLE

Tempting and it looks striking, so it will be re-proposed unless recorded. Verifying every
`a op b = c` the model writes, on 1331 GSM8K test items per checkpoint:

| checkpoint | accuracy | steps/gen | step validity |
|---|--:|--:|--:|
| Stage 1 | 0.017 | 1.19 | 0.659 |
| Instella 3B | 0.512 | 2.23 | 0.838 |
| SFT | 0.672 | 1.50 | 0.819 |
| Instruct | 0.716 | 1.13 | 0.811 |

Reads as "post-training raises accuracy 20 points without raising the validity of the
reasoning it shows". **It does not survive its coverage check.** Generations containing no
parseable step: 30.7% at stage two, 39.5% at SFT, 46.4% at Instruct, 92.6% at Stage 1. The
answer format also shifts across post-training, from `####` at 53% of stage-two generations
to `\boxed{}` at 68-69% after. So validity is computed on a shrinking, non-randomly selected
subset under a changing output format: different populations, not a trend.

Would need a step extractor robust to prose and to both answer formats, plus a coverage
matched subsample, before it could be claimed. Not attempted.

### 2.6 "Stage 2 memorises deleted premise values (+4.15 pp)" — WITHDRAWN

Looked like item-specific memorisation at exactly the right boundary. Killed by three
checks: the Stage2→SFT placebo fires at −4.10 in the opposite direction; it is not an
emission-count artifact (+3.97 after conditioning); and decisively, **within the train arm
alone it does not scale with containment** (high ≥0.50 minus low <0.10 = −3.54
[−11.83, +3.97]). It is a generalisation gap, not exposure.

---

## 3. JUDGe: what holds

### 3.1 The dissociation, five judges, three families

Balanced accuracy seen minus unseen, grading instruct outputs, 2017 solutions each:

| judge | reference given | reference withheld | ratio |
|---|---|---|--:|
| Instella 3B Instruct | −0.054* | −0.081* | 1.5 |
| Llama 3.1 8B Instruct | −0.074* | **−0.243*** | 3.3 |
| Qwen2.5 7B Instruct | −0.036* | −0.200* | 5.6 |
| Qwen2.5 14B Instruct | −0.023 | −0.184* | 8.0 |
| Qwen2.5 32B Instruct | −0.062* | −0.191* | 3.1 |

**Argue from the within-judge ratio, not from a trend across the ladder.** With five judges
the control gaps are not monotone in capability, so the earlier "the control converges as
judges improve" claim does not hold. The ratio compares one judge against itself on
identical items and identical solutions, so it needs no assumption about the ladder.

Absolute balanced accuracy (given seen / given unseen / withheld seen / withheld unseen):
Instella 3B 0.514 / 0.568 / 0.502 / 0.583 · Llama 8B 0.860 / 0.934 / 0.511 / 0.754 ·
Qwen 7B 0.937 / 0.973 / 0.600 / 0.800 · Qwen 14B 0.946 / 0.969 / 0.636 / 0.820 ·
Qwen 32B 0.888 / 0.950 / 0.648 / 0.839. (Corrected 2026-08-24: the first two cells previously read 0.951 / 0.968, which are the **stage2** target's control cells, not instruct's. Table 1 of the manuscript was unaffected, its 32B control gap of $-0.062$ being the instruct figure throughout.)

### 3.2 The effect is specificity, not sensitivity (new, 2026-08-23)

| judge | sensitivity gap | specificity gap |
|---|---|---|
| Llama 3.1 8B | +0.011 [−0.005, +0.034] | **−0.496 [−0.573, −0.415]*** |
| Qwen2.5 32B | −0.004 [−0.024, +0.016] | **−0.378 [−0.472, −0.276]*** |

Llama rejects **52.3%** of wrong solutions to unseen problems and **2.7%** of wrong
solutions to seen ones. The judges become *credulous* on familiar problems, not confused.

This **falsified the paper's original mechanism**, which claimed the judge compares against
a remembered answer — that predicts correct solutions being marked wrong, i.e. a
sensitivity loss. A direct test on answer-changing variants (where a remembered answer
would be the wrong answer) also found no concentration of false negatives on the seen arm.

### 3.3 The error-subtlety confound is controlled

The generator is better on seen problems, so its errors there may be subtler. Both
conditions grade the **same solutions**, so the control separates it:

| judge | specificity gap, reference given | reference withheld |
|---|---|---|
| Llama 3.1 8B | −0.151 [−0.236, −0.068]* | −0.496 [−0.573, −0.415]* |
| Qwen2.5 32B | −0.124 [−0.207, −0.042]* | −0.378 [−0.472, −0.276]* |

About a third of the gap is error subtlety. Withholding the reference triples it, and that
excess cannot be a property of the solutions.

### 3.4 Prompt robustness

Reworded prompt, same information: Llama −0.254 [−0.292, −0.212] vs −0.243 originally;
Qwen 32B −0.218 [−0.262, −0.170] vs −0.191. Control gaps stay small (−0.051, −0.086).

### 3.5 Ceiling in solution quality

Grading the weaker stage-two generator, Qwen2.5 32B gives **−0.035 [−0.075, +0.006]**
(spans zero) where Llama 8B gives −0.192. Qwen 32B's reference-free specificity there is
0.854 — the errors are blunt enough to catch without a reference, so there is little left
to lose. Predicted by the specificity account. Measured on one generator pair; report as an
observation, not a law.

### 3.6 The kappa paradox

Cohen's kappa manufactures a seen-minus-unseen gap of **−0.292 [−0.356, −0.228]** on
verdicts where balanced accuracy gives **+0.054**, purely from a base-rate difference of
77% vs 53%. Feinstein & Cicchetti (1990). Reported as a cautionary result.

---

## 3.7 Premise verification, with three controls (new, 2026-08-25)

Under a prompt naming `#### unanswerable`, abstention on provably underdetermined items
against the **answerable control** (untouched parents, where declining is an error):

| checkpoint | underdetermined | answerable | discrimination | 95% CI |
|---|--:|--:|--:|---|
| Stage 1 | 0.043 | 0.025 | +1.84 | [+0.97, +2.70] |
| Instella 3B | 0.110 | 0.008 | +10.10 | [+8.94, +11.30] |
| SFT | 0.416 | 0.035 | +38.08 | [+36.06, +40.02] |
| Instruct | 0.538 | 0.011 | **+52.63** | [+50.77, +54.57] |

**Graded by chain length.** Slope per extra calculator step, pruned vs answerable control:

| checkpoint | pruned | control | difference |
|---|--:|--:|---|
| Instella 3B | −2.55 | −0.21 (spans 0) | −2.34 [−3.12, −1.57] |
| SFT | −4.06 | −0.42 (spans 0) | −3.64 [−5.08, −2.15] |
| Instruct | −5.22 | −0.11 (spans 0) | −5.11 [−6.36, −3.82] |

**Perturbation control** (all answerable, instruct): clean 0.0113, off-topic distractor with
unused number 0.0170, in-domain distractor 0.0125, against 0.538 for genuine deletion. Rules
out "the text reads oddly" as the trigger.

**Truncation** never exceeds 0.7% at any chain length; restricting to finished generations
reproduces every rate exactly.

`depth_delta` is constant at −1 across all probe items, so gradation by amount removed is not
available. Chain length is the usable axis.

---

## 3.8 Sentence-splitting edge case, measured and immaterial (2026-08-25)

The probe splits premises on `(?<=[.!?])\s+`, which also splits after an honorific. Where
the fragment after the honorific is the one deleted, the stem is malformed
(`Mr. After three years...`). Strictly affected: **22 of 3088** GSM8K deletion items (0.71%)
and **5 of 1091** MATH items (0.46%).

Excluding them shifts the discrimination headline by at most **0.05 pp**:

| checkpoint | all items | mangled excluded | shift |
|---|--:|--:|--:|
| Instella 3B | +10.10 | +10.12 | +0.02 |
| SFT | +38.08 | +38.03 | −0.05 |
| Instruct | +52.63 | +52.61 | −0.02 |

No rerun performed; the GPU cost is not justified by a 0.05 pp shift. The MATH builder is
guarded going forward. Reportable as a robustness check rather than a defect.

---

## 3.9 Second domain and cross-family replication (2026-08-25)

**Balanced accuracy of the decline decision** (0.500 = chance):

| checkpoint | GSM8K | MATH |
|---|--:|--:|
| Stage 1 | 0.509 | 0.516 |
| Instella 3B | 0.550 | 0.557 |
| SFT | 0.690 | 0.611 |
| Instruct | **0.763** | **0.675** |

MATH probe built independently: 818 parents, 1091 variants, all 7 subjects, necessity check
passes 1091/1091. Same shape, lower level — expected, since harder problems the model cannot
solve inflate the answerable-decline rate and shrink the measured gap (conservative bias).

**Depth decay replicates on MATH**: −7.11 pp per extra sentence [−9.87, −4.67], control slope
spans zero. (GSM8K: −5.11 per calculator step [−6.36, −3.82].)

**Across families**, same GSM8K items, instruct checkpoints:

| model | declines pruned | declines answerable | discrimination | bal acc |
|---|--:|--:|--:|--:|
| Qwen2.5-3B Instruct | 0.632 | 0.043 | +58.92 | **0.795** |
| Instella-3B Instruct | 0.538 | **0.011** | +52.63 | 0.763 |
| Qwen2.5-1.5B Instruct | 0.609 | 0.090 | +51.88 | 0.759 |
| OLMo-2-1B Instruct | 0.439 | 0.292 | +14.68 | 0.573 |

**Instella is second, not first.** Qwen2.5-3B beats it 0.795 vs 0.763. Instella does have the
lowest false-decline rate of the four (1.1%). Do not claim Instella is best — it is
competitive, and far ahead of OLMo-2-1B.

OLMo-2 shows why balanced accuracy is needed: discrimination +14.68 clears zero, but it
declines 29.2% of answerable problems, so its decisions sit near chance.

---

## 3.10 JUDGe: the error-subtlety objection, closed (2026-08-26)

An external reviewer called this reject-level: the generator is more accurate on seen items
(0.772 vs 0.526), so its errors there may be **subtler**, and subtle errors are harder to
catch without a reference. That alone could produce the whole result with no membership role.

**Four controls, all in `paper/judge/extra_controls.py` → `extra_controls.json`:**

**1. Raw counts** (Llama 3.1 8B, seen arm, 225 wrong solutions):

| condition | rejected | specificity |
|---|--:|--:|
| reference withheld | **6** / 225 | 0.027 |
| reference given | **162** / 225 | 0.720 |

Same 225 solutions. A judge catching 72% with a reference is not failing on subtlety.

**2. Interaction** (withheld gap − given gap, identical solutions):

| judge | specificity | sensitivity |
|---|---|---|
| Llama 3.1 8B | **−0.345 [−0.445, −0.243]*** | +0.007 (spans 0) |
| Qwen2.5 32B | **−0.254 [−0.349, −0.160]*** | −0.004 (spans 0) |

**NOT sufficient alone** — removes main effects of solution properties but *not* an
interaction between them and condition. If subtle errors are disproportionately harder to
catch without a reference, this still moves. Hence control 3.

**3. Difficulty stratification** (parent calculator chain length), reference withheld:

| judge | per stratum | weighted | unstratified |
|---|---|--:|--:|
| Llama 3.1 8B | 2st -0.637 · 3st -0.524 · 4st -0.429 · 5st -0.242 | **-0.4880** | -0.5001 |
| Qwen2.5 32B | 2st -0.483 · 3st -0.568 · 4st -0.192 · 5st -0.080 | **-0.3811** | -0.4158 |

Gap holds inside **every** stratum and moves <0.04 when strata are weighted equally.
**Difficulty is not the explanation.** This is the control that actually closes the objection.

**4. Placebo split**: halve the unseen arm at random, relabel one half "seen", 2000 draws →
mean specificity gap **+0.001**, intervals centred on zero. The pipeline does not manufacture
gaps from arm labels.

**Limitation recorded:** chain length *proxies* difficulty. A severity measure (how far a
wrong answer lands from gold) is **not recoverable** — most judged items are variants whose
own gold answers were never retained. Only 500 of 2017 have gold in any committed file.

**Also changed:** title dropped Instella (it carries the *smallest* gap in Table 1, so naming
it as the site of degradation pointed at the one judge not showing the effect); cites
Li et al. preference leakage (arXiv 2502.01534); abstract membership claim scoped to
"judges built on that corpus".

**Process note:** commit 82e13f0 claimed these edits landed. Only the abstract qualifier did —
the others matched text since rewritten and were silently lost. Reapplied and verified by
string check in 47933b8. **Verify edits landed, do not trust a script's success message.**

---

## 4. Jiang Liu's four comments — status

| | comment | status |
|---|---|---|
| J1 | Scope "unexposed" to the specific released dataset | **Closed.** "Known negative *with respect to that corpus*"; Stage 1 described as "precedes the corpus". OLMoE-mix scan gives a measured bound: 0 of 7473 GSM8K train and 0 of 1319 test at containment ≥0.3 in 400,000 rows. |
| J2 | Synthetic-data procedure description correct | **Closed.** Citable as personal communication. |
| J3 | Use `train_119K`, not the released pool | **Closed, renumbered 2026-08-23.** Matched rescan of both corpora through one code path on identical items (n=891 train parents): 12.57% at ≥0.80 against the consumed subset vs 23.34% against the full pool. Earlier 7.56%/29.75% pair used two different extraction paths and is withdrawn; see `docs/EXPOSURE_TWO_CORPORA.md`. |
| J4 | Post-training gets GSM8K via OpenMathInstruct-2 | **Closed 2026-08-23.** The SFT and Instruct boundaries are now flagged as not clean membership contrasts, biased *toward* the effect, and reported for completeness rather than as estimates. |

**Containment definition:** use **max over individual documents**, not union over the
corpus. `docs/CORPUS_CORRECTION.md` §1 uses the union rule (15.12%) and is superseded.
`outputs/corpus_scan/train119k_summary.json` uses per-document but a different text
extraction (local JSONL `text` field, 722 parents) and must not be compared against a
Hub-streamed scan; it is retained for provenance only. The figures the paper uses come from
the matched rescan in `docs/EXPOSURE_TWO_CORPORA.md`. Calibration holds under the
per-document rule against the **full 1,367,882-row pool**, the largest corpus either model
saw: GSM8K test 0/888 at ≥0.5, 1/888 at ≥0.3, max 0.487.

**Instella-MoE (released 2026-07-24):** GSM8K-synthetic enters only at long-context
extension phase 2 and uses the **full `train` split** (329M tokens), per
`docs/data_preparation.md` in AMD-AGI/Instella-MoE. So Midtrain→Base is a second data-entry
boundary at ~11.5× the corpus. Containment against that pool is now measured (see above and
`docs/EXPOSURE_TWO_CORPORA.md`): 18.52% of train parents verbatim vs 9.65% under
`train_119K`, so the MoE boundary carries roughly twice the verbatim exposure at the same
items. The deletion probe on Midtrain→Base is **not run and will not be for this
submission**. The go/no-go loaded the model successfully and it reasons correctly on GSM8K,
so the architecture is not the obstacle; throughput is. Measured on A100-80GB through plain
transformers at batch 12: **30.2 tokens/sec aggregate**, giving ~15.0 h for one checkpoint
over 3,179 items and ~29.9 h for the Midtrain/Base pair, against a remaining budget of
~9.5 h. All 48 probe generations also ran the full 512 tokens without emitting a stop
token, so that estimate is a floor rather than a worst case. vLLM is not an escape: the
config declares `model_type: deepseek_v3` but ships custom Gated-MLA/FarSkip classes via
`trust_remote_code`, so vLLM's native DeepSeek-V3 path does not apply. Risk: AMD's inference stack is ROCm-only; config declares `model_type: deepseek_v3`
with custom Gated-MLA/FarSkip classes, so generation runs through plain transformers rather
than vLLM.

---

## 5. Infrastructure

Modal workspace `dasashreeya`, secret `instella-hf` (HF token, write verified).

| app function | what it does | cost |
|---|---|---|
| `modal_boost.py::g1` | quantity ablation, 8933 rows × 3 ckpts | ~4 min/ckpt |
| `modal_boost.py::g1b` | marker ablation, 3528 rows × 3 ckpts | ~4.5 min/ckpt |
| `modal_boost.py::g3` | abstention licensing, 3179 rows × 4 ckpts | ~5 min/ckpt |
| `modal_boost.py::g2` | judge extension, 2017 rows × 2 modes | 2–13 min/cell |
| `modal_boost.py::g2b` | judge prompt robustness | 3–9 min/cell |
| `modal_boost.py::g2_verify` | rebuilds arms and checks ids before any GPU | CPU |

**Arms reconstruction:** only the 500 parents are published
(`fullscale-S250-v2/arms/gsm8k_arms.jsonl`). Variants rebuild deterministically at seed
6198 via `make_variant_suite(numeric_variants=2)` + `make_structural_variants(premise_removal_k=3)`.
Verified **2017 of 2017** benchmark ids matched for both stage2 and instruct, zero
unmatched. Always run `g2_verify` before spending GPU.

Analysis scripts: `analyze_quantity_ablation.py`, `analyze_marker.py`, `analyze_abstention.py`
— all with estimands fixed in the docstring **before** the run.

Volumes: `instella-boost-runs`, `instella-runs` (teammate's G1). Everything mirrors to HF
`GOVINDFROM/Instella-Reasoning` under `experiments/runs/{qty-ablation-v1, abstain-v1,
judge-ext-v1}`.

Local toolchain note: repo requires Python ≥3.10 (`dataclass(slots=True)`); the venv at
`/tmp/hfenv` is 3.9 and cannot import `instella_reasoning`. LaTeX builds with Tectonic.

---

## 6. Open items, in value order

1. **Injection replication on a second base model** (OLMo-2-1B, Qwen2.5-1.5B). §1.2 is
   MATH-AI's key positive finding and rests on one model. ~5 GPU-h.
2. **Instella-MoE Midtrain→Base** — kills "one family, one scale". Gate on a 2-hour
   go/no-go; AMD's stack is ROCm-only. ~3 GPU-h.
3. **The SFT anomaly.** SFT shows no insertion cost, no marker effect, and in JUDGe no
   membership gap in either condition, despite output quality comparable to Instruct.
   Unexplained in both papers.
4. Full-corpus tulu-3 route attribution — the current scan is a 300k-row prefix reaching
   only 6 subsets.
5. Anonymise both papers (repo and HF names appear in appendices) and register JUDGe's
   reciprocal review.

---

## 7. Working rules

- **Do not report a percentage acceptance estimate as fact.** Wide variance, three
  reviewers, no rebuttal.
- **Fix estimands before the run.** Every analysis script here does, in its docstring.
- **Anything not in this file has not been stress-tested** — the burden is on it.
- Three headline candidates died under their own controls (§2.3, §2.4, §2.5) and one
  mechanism was falsified (§3.2). That record is a feature of the work and belongs in the
  papers, not hidden.
