# Reasoning Reliability Atlas — design, rationale, and run book

Run tag: `atlas-v1` · App: `llm-reasoning` · Artifacts: `GOVINDFROM/Instella-Reasoning` (dataset repo)

---

## 0. Final results

### 0.1 Memorisation, measured without a train/test confound

Both arms are GSM8K **train** items, matched on reasoning-step count, differing only in how
heavily the stage-2 augmentation reproduced them (containment at or above 0.80 versus at or
below 0.10). No test item appears in the design, so the "this is a generalisation gap"
objection does not apply. Recall is measured on items whose required premise was deleted,
so reproducing the parent's answer cannot come from the prompt.

Items whose parent answer still appears in the pruned prompt are excluded: emitting it there is
copying rather than recall, and the leak correlates with containment (6.9 percent of
high-containment items against 3.1 percent of low), so leaving those in inflates the estimate by
about the size of the effect. 23 rows are dropped, leaving balanced 217/217 arms.

| checkpoint | GSM8K in training | recall high | recall low | gap | 95% CI |
|---|---|--:|--:|--:|---|
| AMD-OLMo-1B | no | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| Instella-3B-Stage1 | no | 0.009 | 0.005 | +0.005 | [-0.010, +0.022] |
| Instella-3B | yes | 0.065 | 0.046 | +0.018 | [-0.027, +0.063] |
| Instella-3B-SFT | yes | 0.028 | 0.028 | 0.000 | [-0.030, +0.030] |
| Instella-3B-Instruct | yes | 0.060 | 0.023 | +0.037 | [-0.002, +0.079] |
| **Instella-3B-Math** | yes | 0.037 | 0.009 | **+0.028** | **[+0.002, +0.057]** |

Only the reinforcement-learned checkpoint excludes zero, and marginally; the instruction-tuned one
sits just inside it. **The negative controls are exact:** the never-exposed 1B model recalls the
parent answer on precisely zero non-leaked items, which validates the probe rather than merely
assuming it.

**Detectable-effect bound.** At 217 rows per arm, a low-arm base rate of 0.03, and the measured
intraclass correlation of 0.48 over 2.8 removals per parent (design effect 1.87, effective n about
125 per arm), the design resolves gaps of **0.094** at 80 percent power. Observed gaps of 0.018 to
0.037 sit well below that. The defensible statement is therefore that memorisation effects above
roughly 0.09 are excluded, not that the effect is zero.

**Two containment-correlated confounds, both now controlled.** They are worth reporting because
anyone building a premise-deletion probe will meet them:

- *Answer leakage.* The necessity check confirms the deleted premise value is gone; it says nothing
  about the answer, which can survive elsewhere in the text. Now rejected at generation time.
- *Answer guessability.* High-containment items have far smaller answers (mean 865 against
  1,721,923; 30 against 16 percent single-digit), so a guess lands on them more often. Restricting
  to one- and two-digit answers leaves no checkpoint excluding zero.

### 0.2 Fragility, on externally authored perturbations

From the published GSM-Symbolic release, so the perturbations are not ours and cannot have been
tuned. Template head versus instances isolates the numeric effect; p1 and p2 add one and two
clauses.

| checkpoint | numeric | +1 clause | +2 clauses |
|---|---|--:|--:|
| Instella-3B | -0.133 [-0.217, -0.043] | -0.173 | **-0.357** |
| Instella-3B-SFT | -0.187 [-0.273, -0.103] | -0.163 | **-0.413** |
| Instella-3B-Instruct | -0.167 [-0.257, -0.083] | -0.170 | **-0.430** |

**Paired within template.** The three rungs share template numbering, so each can be compared
against its own template instead of the pooled mean, holding item difficulty exactly fixed. This
is the control that dissolved an earlier consistency probe; the fragility result survives it, on
the 50 templates present in all three rungs:

| checkpoint | main | p1 | p2 | p1 - main | p2 - main |
|---|--:|--:|--:|---|---|
| Instella-3B | 0.440 | 0.233 | 0.100 | [-0.320, -0.087] | [-0.447, -0.233] |
| Instella-3B-SFT | 0.593 | 0.407 | 0.207 | [-0.293, -0.080] | [-0.500, -0.267] |
| Instella-3B-Instruct | 0.593 | 0.427 | 0.200 | [-0.267, -0.060] | [-0.527, -0.260] |

Every interval excludes zero. Fragility is large and robust to exact difficulty matching, while
the memorisation component is bounded near zero on the same items under the same scorer. That
contrast, not either half alone, is the result.

### 0.3 Reliability atlas across checkpoints and reasoning subskills

Reliability is accuracy multiplied by cross-variant consistency over answer-preserving variants.
Approximately 61,000 generations across nine benchmarks. `arithmetic_symbolic` carries no
answer-preserving variants by design (the release ships its own cluster structure), so consistency
and reliability are withheld rather than imputed.

| checkpoint | arithmetic | mathematical | commonsense science | logical deduction | logical reading | multi step |
|---|--:|--:|--:|--:|--:|--:|
| AMD-OLMo-1B | 0.008 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 |
| Instella-3B-Stage1 | 0.010 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |
| Instella-3B | 0.415 | 0.088 | 0.075 | 0.037 | 0.049 | 0.166 |
| Instella-3B-SFT | 0.546 | 0.262 | 0.274 | 0.275 | 0.273 | 0.462 |
| Instella-3B-Instruct | 0.564 | 0.235 | 0.201 | 0.220 | 0.236 | 0.495 |

Benchmarks: GSM8K, MATH, ARC-Challenge, BBH, LogiQA, ReClor, and GSM-Symbolic main/p1/p2.
MATH was retained throughout and appears as the `mathematical` subskill (758 rows per checkpoint);
its numbers became usable only after the answer-extraction fix in 0.4.

Post-training raises arithmetic reliability from 0.415 to 0.564 while competition mathematics
stays at 0.235 and logical deduction at 0.220, so the gains are narrower than a headline average
implies.

### 0.4 Two measurement-validity findings

1. **Long chain-of-thought truncation.** A 512-token budget applied to a checkpoint post-trained
   with 16K-token rollouts produced a phantom 46-point regression in an earlier analysis.
   Corrected: 0.227 to 85.5 against a published 92.48.
2. **Nested-brace answer extraction.** The common pattern `\boxed\{([^{}]+)\}` cannot match a
   nested group, so `\boxed{\frac{5}{4}}` fails on both the gold and the prediction side. 54 of
   200 MATH gold answers were being compared as entire solution texts. Fixing it moved MATH from
   0.212 to 0.257 and BBH from 0.435 to 0.632.

Both are reusable results: any harness evaluating reasoning-specialised checkpoints at default
token budgets, or scoring LaTeX answers with that regex, is affected.

### 0.5 What did not survive

Recorded so the discarded numbers stay inspectable.

- **`depth_extension` as a memorisation probe.** It appends to the problem while leaving the
  original verbatim, so a recalled answer stays usable. It costs only 3.8 points where numeric
  perturbation costs 18.7, and the externally authored ladder shows added clauses are in fact
  *more* damaging, not less. The earlier "models break on values, not composition" reading was an
  artifact of this probe.
- **The k=1 recall estimate.** At one removal per item the gap was +0.049; at three removals per
  item, on the same design, it fell to +0.027 and one checkpoint reversed sign. The larger sample
  is the estimate. Reporting the k=1 figure after seeing both would be selection on outcome.

---

## 1. The question, and why the previous design could not answer it

The project asks whether benchmark accuracy reflects reasoning or recall of training data.
The earlier design answered this with a difference-in-differences estimator: the seen-minus-unseen
accuracy gap on original items, minus that same gap on **numerically perturbed** items. A positive
value would mean the seen advantage evaporates once the numbers change — the signature of a
memorised answer.

That estimator came back flat at every checkpoint, with every interval spanning zero. The natural
reading was "no memorisation." **That reading is not supported**, for a reason visible in the
model family's own training description.

The second-stage synthetic mathematics corpus was built by taking GSM8K training problems,
abstracting their numerical values into Python function parameters, and then re-instantiating
those parameters with new values to produce fresh question–answer pairs. Numeric perturbation is
therefore *the same operation that generated the training data*. A numerically perturbed
evaluation item lies inside the training distribution by construction.

Consequence: a flat estimate is the predicted outcome **whether the model reasons or memorises**.
The instrument cannot move, so the null is uninformative rather than evidence of absence.

## 2. The axis that can answer it

The augmentation varied **leaf values** while holding the **solution program** fixed. The
discriminating axis is therefore the program, not the magnitudes:

| Perturbation | What changes | Relative to augmentation support | Informative? |
|---|---|---|---|
| surface rewrites (rephrasing, entity substitution, premise reordering, irrelevant context) | wording | inside | robustness only |
| `numeric_perturbation` / `gsm_symbolic` | leaf values | **inside** | no |
| `distractor_quantity` | adds an unused quantity the program must ignore | **outside** | yes |
| `depth_extension` | adds one dependent step; program depth +1 | **outside** | yes |

**Hypothesis under test.** The model memorised a *solution program*, not an answer. This predicts:

1. flat difference-in-differences under numeric perturbation — **already observed**;
2. accuracy collapse under `depth_extension` — **this run tests it**.

If both hold, the result is direct evidence for procedure-level memorisation measured against a
verified training corpus. If accuracy holds under `depth_extension` too, the model is genuinely
composing, which is an equally reportable result.

## 3. Implementation

### 3.1 Structural perturbations (`src/instella_reasoning/structural.py`)

- `depth_extension` — appends a clause referring to the parent result anaphorically ("Finally,
  what is 3 times that amount?") and recomputes gold in closed form. Anaphora avoids entity
  parsing, so the generator applies to any item with an integer gold answer. Marked
  `metadata["answer_changing"] = True` so `metrics.is_answer_changing` excludes it from
  consistency, which otherwise would compare correctness labels across different gold answers.
- `distractor_quantity` — inserts a sentence carrying one unused number, chosen to avoid
  colliding with any integer already in the question, placed before the final interrogative.
  Gold is preserved by construction.

Both are pure functions of `(item, rng)` seeded per `(item id, kind)`, so variant text is stable
regardless of call order or batch composition. Items whose gold answer is not an exact integer are
skipped rather than approximated.

### 3.2 Generation (`experiments/modal_reasoning.py`)

Served with vLLM. Three defects in the previous harness are fixed here:

| Axis | Checkpoint was trained for | Old harness | Now |
|---|---|---|---|
| output budget | 8K–16K tokens (long chain-of-thought) | 512, later 3072 | 16384 for `M3-math` |
| answer format | `\boxed{}` | forced `####` | per-model `answer_style` |
| extraction | `\boxed{}` | `####` first | style-dependent precedence, with tier recorded |

The reasoning-specialised checkpoint is a long chain-of-thought model whose post-training raised
context from 4K to 32K and ran policy optimisation at 8K–16K rollout lengths. Generating it under
a 512–3072 token cap truncates it mid-derivation. The observable signature was recorded in the old
code: 26% of its responses contained the gold answer yet scored wrong, and completion never
exceeded 75%. Published accuracy for that checkpoint is 92.48 on GSM8K; the old harness measured
0.227. **The prior "reliability regression after math post-training" was therefore a harness
artifact, not a finding.**

vLLM does not ship this architecture. The reasoning-specialised repository does ship a
vLLM-native model class, and all 3B checkpoints share one weight-compatible architecture, so a
single out-of-tree registration serves the whole ladder. The class is downloaded under a distinct
module name because the transformers-style file shipped by the other repositories has the same
basename, and the dynamic-module loader would otherwise resolve whichever landed first.
Registration uses the string form so the class survives import inside a worker process.

Pin rationale: the registered class takes a `VllmConfig` in `__init__` (an API that postdates
vLLM 0.6.3) and imports the V0 sampler (removed once V1 became mandatory). `vllm==0.8.5.post1`
with `VLLM_USE_V1=0` is inside the window that satisfies both.

### 3.3 Metric

Reliability = accuracy x consistency, per checkpoint and reasoning subskill.

Consistency is computed over **answer-preserving** variants only. Answer-changing variants
(numeric and `depth_extension`) enter accuracy and the perturbation contrast instead, because
agreement of correctness labels across variants with different gold answers is not meaningful.

Multiple-choice benchmarks admit no exact recomputable gold answer, so they carry surface variants
only. The atlas reports `n/a` for their structural columns rather than imputing a value.

## 4. Reference numbers (validation gate)

Published, from the model family's paper (Tables 4 and 6) and the reasoning-specialised model card.
GSM8K few-shot settings: 8-shot for base checkpoints, 5-shot MMLU, 3-shot BBH, 0-shot Minerva MATH.

| Alias | Checkpoint | GSM8K | MATH | ARC-C | BBH | MMLU |
|---|---|--:|--:|--:|--:|--:|
| `M3-stage1` | Stage 1 | 10.8 | — | 53.9 | 34.3 | 54.7 |
| `M3-base` | Stage 2 | 59.8 | — | 52.8 | 39.7 | 58.3 |
| `M3-sft` | + SFT | 71.7 | 40.5 | — | 46.0 | 58.8 |
| `M3-instruct` | + DPO | 73.9 | 42.5 | — | 46.8 | 58.9 |
| `M3-math` | + GRPO | **92.48** | 86.49 (MATH500) | — | — | — |

The gate runs zero-shot with each checkpoint's native answer style, so base checkpoints landing a
few points below published is expected. A checkpoint landing *tens* of points low means the harness
is wrong. **No downstream number is trustworthy until this gate passes.**

## 5. Run book

```bash
# 0. one-time: Modal secret llm-reasoning-hf holding HF_TOKEN
modal run experiments/modal_reasoning.py::smoke    --aliases "M3-instruct,M3-math"
modal run experiments/modal_reasoning.py::validate --aliases "M3-base,M3-instruct,M3-math" --n 200

# 1. build variant clusters
modal run experiments/modal_reasoning.py::build --limit 200

# 2. generate (per checkpoint; resumable, skips existing outputs)
modal run experiments/modal_reasoning.py::generate --alias M3-instruct
modal run experiments/modal_reasoning.py::generate --alias M3-base
modal run experiments/modal_reasoning.py::generate --alias M1
modal run experiments/modal_reasoning.py::generate --alias M3-math --benchmarks "gsm8k,math"

# 3. analyse, plot, publish
modal run experiments/modal_reasoning.py::atlas
modal run experiments/modal_reasoning.py::push
```

On Windows, prefix with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` — the CLI prints non-ASCII status
glyphs that crash under the default console codepage when stdout is a pipe.

**Do not edit anything under `src/` while a Modal build is running.** The directory is copied into
the image at build time and a mid-copy write aborts the build.

## 6. Figures

`src/instella_reasoning/analysis/atlas_figures.py` renders each figure to **both `.pdf` (vector,
for the paper) and `.png`**. Titles are omitted because the paper carries the description in
`\caption`.

| Figure | Content |
|---|---|
| `a1_perturbation_slope_<skill>` | accuracy across original → distractor → numeric → depth, one line per checkpoint, with the augmentation-support boundary marked. **The discriminating figure.** |
| `a2_reliability_heatmap` | reliability per checkpoint x subskill |
| `a3_accuracy_consistency_<skill>` | reliability decomposed into its two factors |

Palette is the validated four-slot categorical order (blue, orange, aqua, yellow): worst adjacent
CVD separation dE 9.1, worst normal-vision separation dE 22.9, both clear of their floors. Two
slots sit below 3:1 contrast on a light surface, so every mark also carries a direct value label
and identity is never conveyed by colour alone.

## 7. Known limitations to state in the paper

1. **Train-versus-test, not contaminated-versus-clean.** The verified-membership contrast compares
   GSM8K train items (in the corpus) against test items (not in it). A reader may argue this
   measures a train/test generalisation gap. The structural contrast is *within-item* and does not
   inherit this confound, which is one reason it carries the argument.
2. **Membership is verified for one family only.** Any judge or comparison model outside it has an
   unknown corpus.
3. **`depth_extension` composes rather than restructures.** It appends a step to the end of the
   program; it does not reorder or re-wire the existing dependency graph. A model could in
   principle fail it through arithmetic carry-through rather than through program recall. Reporting
   `distractor_quantity` alongside it partially separates these, since that variant adds no
   arithmetic at all.
4. **Anaphoric phrasing is a fixed template.** Sensitivity of the effect to the wording of the
   appended clause is untested.
