<h1 align="center">Instella Reasoning Atlas</h1>

<p align="center">
  <b>Reasoning or Remembering?</b><br>
  Diagnosing whether AMD Instella solves reasoning tasks by <i>generalizing</i>,
  by <i>memorizing</i> training examples, or by relying on <i>fragile shortcuts</i>.
</p>

<p align="center">
  <a href="https://github.com/GIND123/Instella-Reasoning"><b>GitHub</b></a> ·
  <a href="experiments/FULLSCALE.md"><b>Full-scale suite</b></a> ·
  <a href="notebooks/fullscale_colab.ipynb">Colab</a> ·
  <a href="https://huggingface.co/datasets/GOVINDFROM/Instella-Reasoning">HF results</a> ·
  <a href="docs/AUDIT_2026-07-26.md">Audit</a> ·
  <a href="docs/BUILD_PLAN.md">Design</a> ·
  <a href="#command-reference">Commands</a> ·
  <a href="docs/COMPUTE.md">Compute</a> ·
  <a href="docs/proposal/">Proposal</a>
</p>

---

> **The current experiment is the seen/unseen memorisation suite** —
> [`experiments/FULLSCALE.md`](experiments/FULLSCALE.md). It asks whether the model does
> better on problems it *provably memorised*: GSM8K **train** items are verbatim in
> Instella's stage-2 training data, GSM8K **test** items are not, and membership is
> verified per item by exact 13-gram containment rather than inferred from an embedding
> proxy. Crossed with numeric perturbation across the validated Instella-3B checkpoint
> trajectory, the difference-in-differences isolates the memorisation component and cancels
> the magnitude confound.
>
> ```bash
> python scripts/preflight_fullscale.py --smoke   # the gate — must print SAFE TO LAUNCH
> bash experiments/run_fullscale_suite.sh
> ```
>
> The earlier `reliability-B*` runs under `experiments/runs/` are **superseded pilots**;
> [`docs/AUDIT_2026-07-26.md`](docs/AUDIT_2026-07-26.md) explains why four of their five
> headline findings are artifacts. Do not cite those numbers.

## Study status — read this first

**The experiment is finished.** All seven generation blocks across all three checkpoints are
generated, scored, analysed, plotted, and mirrored to Hugging Face. Nothing is mid-flight and
no GPU job is pending. What remains is one optional quality repair and one optional
scale-up — both described under [What is left](#what-is-left), with costs.

| | |
|---|---|
| Rows generated and scored | **9,051 / 9,051 (100%)** |
| Generation blocks complete | **7 / 7 (100%)** |
| Quality gates passed | **6 / 7** — `stage1/arms` sits at 84.9% against an 85% gate |
| Rows needing optional rework | **304 / 9,051 (3.4%)** |
| Analysis, atlas, and F1–F8 figures | Produced and pushed |
| GPU time already spent on the corrected run | ~2.3 h (A100-40GB) |
| GPU time required to finish everything outstanding | **~1.6 h** — fits a 3 h budget |

The authoritative run is `experiments/runs/fullscale-S250-v2`. The earlier
`experiments/runs/fullscale-S250` is kept, complete, and reproducible, but its arms were
built on a corrupted measurement — see [Bugs found and fixed](#bugs-found-and-fixed). Do not
quote v1 numbers as the headline.

---

## What the study asks, for a reader with no context

Large language models are scored on public benchmarks. If a benchmark's questions were in a
model's training data, the score may reflect **memorisation** rather than **reasoning** — the
model recites an answer it has seen instead of working it out. This is called *contamination*,
and reviewers routinely demand that results be discounted because of it.

**The obvious version of the question is unanswerable here.** You would want to ask "are GSM8K
*test* items in Instella's training data?" But the corpus we can inspect,
`amd/Instella-GSM8K-synthetic`, is *derived from GSM8K train*. Scanning test items against it
returns zero matches no matter how thresholds are set, so the contaminated group is empty and
there is nothing to compare.

**The answerable version.** GSM8K ships a *train* split and a *test* split, written by the same
annotators to the same spec, at the same difficulty. Instella's training data contains material
derived from **train** and not from **test**. So:

- **"seen" arm** = GSM8K *train* items that are provably in the training corpus
- **"unseen" arm** = GSM8K *test* items that are provably absent from it

"Provably" is the important word. Membership is not guessed from a similarity score. For each
item we compute **13-gram containment**: chop the question into every run of 13 consecutive
words, then measure what fraction of those runs appear anywhere in the 1.45 GB training corpus.
An item copied into training scores near 1.0; an unrelated item scores 0.0. Items landing in
between are **excluded from both arms** rather than forced into one, because a treatment label
you cannot defend is worse than a smaller sample.

**Why perturbation is needed.** Suppose seen items score higher. That alone proves nothing —
maybe train items are simply easier. So every item is also rewritten with **different numbers**
(same structure, same reasoning steps, new arithmetic). A model that *memorised* an answer
loses its advantage the moment the numbers change. A model that *reasons* keeps it.

**The measurement: difference-in-differences (DiD).** Take the seen-minus-unseen accuracy gap on
original items, then subtract the same gap on perturbed items:

```
DiD = (seen − unseen | original) − (seen − unseen | perturbed)
```

- **DiD > 0** → the seen advantage evaporates under perturbation → memorisation
- **DiD ≈ 0** → whatever advantage exists survives new numbers → not memorisation

Subtracting twice cancels anything affecting both arms equally — including the known confound
that perturbed problems may just involve bigger arithmetic.

---

## The experimental design

**Checkpoints.** Instella-3B is released at successive training stages, which turns an
observational comparison into something closer to a controlled intervention:

| tag | model | GSM8K-derived data? | role |
|---|---|---|---|
| `stage1` | `amd/Instella-3B-Stage1` | **No** | **Control** — the only checkpoint that never saw GSM8K |
| `stage2` | `amd/Instella-3B` | **Yes** | **Treated** — stage-2 mix explicitly targets GSM8K |
| `instruct` | `amd/Instella-3B-Instruct` | Yes | + DPO; the headline general model |

If memorisation drives GSM8K scores, `stage1` should show DiD ≈ 0 and `stage2` should not.

**Item clusters.** Each of the 500 arm items expands to 5 rows — 1 original, 2
answer-*preserving* rewrites (rephrasing, distractor text), and 2 answer-*changing* numeric
perturbations. 500 × 5 ≈ 2,017 rows per checkpoint. Preserving and changing variants are kept
in separate terms: changing ones drive the DiD, preserving ones drive a consistency measure.

**Three blocks per checkpoint:**

| block | rows | what it is for |
|---|---|---|
| `arms` | 2,017 | The headline DiD |
| `gsmsym` | 800 | Apple's hand-written GSM-Symbolic templates — external validity |
| `resample` | 600 | Same item generated 5× at temperature 0.7 — the **decoding-noise null** |

The `resample` block matters more than its size suggests. At temperature 0 there is no sampling
variance, so an observed inconsistency across variants has nothing to be compared against. This
block supplies that baseline.

**Statistics.** Confidence intervals come from a bootstrap that resamples **parent items**, not
individual rows. The five rows of one problem are not five independent observations — measured
ICC on this data is ~0.48. Resampling rows instead would shrink intervals by roughly the design
effect (~2.4×) and produce confidently wrong error bars.

---

## What was actually run

| | v1 — `fullscale-S250` | v2 — `fullscale-S250-v2` |
|---|---|---|
| Arms | 194 seen / 194 unseen | **250 seen / 250 unseen** |
| Treatment labels | 43% of seen arm mislabelled | All verified |
| Difficulty matching | Against wrong bins | Correct — 84/83/83 |
| `containment.json` | Values up to 7.78 (impossible) | Corrected |
| Status | Complete, superseded | **Complete, authoritative** |

Chronology: v1 ran across Colab (stage1) and Modal (stage2, instruct) and completed fully. An
audit of its containment file then exposed the id-collision bug below, which invalidated the
treatment assignment. v2 was staged into a fresh directory reusing 4,119 of v1's 6,051
generations — possible because variants are seeded per `(item, type)`, so retained items keep
byte-identical text — and regenerated only the 1,932 genuinely new rows.

---

## Bugs found and fixed

These are documented in detail because several are the kind a reviewer will ask about, and
because finding one that invalidated 43% of our own treatment arm is part of the record.

**1. GSM8K train/test id collision — the serious one.** Both splits use the id scheme
`gsm8k_NNNNN`, so loading 1,000 of each makes every id collide pairwise (1,000/1,000 collisions
confirmed). `verify_containment` keys its gram index by `item.id`, so the test twin overwrote
the train twin's gram set while the owner map retained both. The numerator then accumulated
matches against grams absent from the denominator, and containment — a *fraction*, mathematically
bounded by 1.0 — reached **7.78**. Re-measuring with `train::`/`test::` namespaced ids showed:

- **83 of 194 v1 seen-arm items (43%) were never verifiably seen** — true containment 0.39–0.79,
  below the 0.80 threshold the design requires.
- Eligible pools are actually 303 seen and 998 unseen, supporting far larger arms than v1 used.

**2. Difficulty bins hit the same collision.** `assign_difficulty_bins` is also keyed by
`item.id`, so every train item inherited its *test twin's* difficulty score. The v1 arms were
therefore never actually difficulty-matched, despite the balance report saying they were. v2
bins over namespaced ids.

**3. Answer-extraction drifts across checkpoints.** `extract_numeric` falls back in three tiers:
the `#### N` marker, then an "answer is …" phrase, then the last number anywhere in the text.
Marker rates differ enormously — `stage1` 86%, `stage2` 98%, `instruct` **34%** (the DPO model
answers conversationally). Tier 3 means completely different things per checkpoint: it recovers
the right answer 72.5% of the time for `instruct`, but only ~1% for `stage1`, where it fires on
looping output. **This does not contaminate the DiD**: the tier-1 rate's *second difference* is
≤0.032 everywhere, so the artifact cancels in the double difference. Verified rather than assumed.

**4. `.remote()` cancellation killed a 2-hour run.** `modal run --detach` keeps the *app* alive,
but `Function.remote()` blocks the local client, and cancelling that local call propagates into
the container. A run died at `instruct/resample`. Fixed by launching with `.spawn()`.

**5. `_hf_pull` silently reverts local edits.** The suite starts with
`snapshot_download(local_dir=".")`, which syncs local files *down* to match the remote. An
in-place edit of a run directory is undone before generation starts — an entire corrected rebuild
was wiped this way and reported "already scored". Fixed by staging corrections into a *new* run
directory, which also leaves v1 intact.

**6. Floating `transformers` version.** `pyproject` pins `>=4.44,<5`, but a range puts different
checkpoints on different minor versions, which enters the DiD as if it were a model difference.
Now pinned exactly to `transformers==4.56.0`, matching what stage1 was originally generated under.

---

## Results

**Headline DiD, cluster-robust bootstrap over parent items.** All three arm constructions:

| checkpoint | v1 full (194, 43% mislabelled) | v1 verified-only (111) | **v2 (250, all verified)** |
|---|---|---|---|
| `stage1` | +0.0128 [−0.047, +0.073] | +0.0066 [−0.061, +0.072] | **+0.0137 [−0.037, +0.064]** |
| `stage2` | −0.0592 [−0.191, +0.067] | −0.0069 [−0.161, +0.153] | **+0.0083 [−0.103, +0.117]** |
| `instruct` | −0.0735 [−0.198, +0.047] | −0.0192 [−0.175, +0.134] | **+0.0396 [−0.071, +0.149]** |

**Every interval spans zero.** No checkpoint shows a memorisation advantage that survives
numeric perturbation — including `stage2`, whose training data explicitly targets GSM8K.

**What this supports.** Point estimates wander between −0.074 and +0.040 with no stable sign
across three independent constructions. That is the signature of a true effect near zero, not of
an effect being missed. Paired with the detector result below — membership is being identified
correctly — the claim is: *detection is accurate, membership is verified, and the accuracy
advantage still is not there.*

**What this does NOT support.** The upper confidence bounds are +0.064 (`stage1`), +0.117
(`stage2`), +0.149 (`instruct`). Contamination effects claimed in the literature are typically
5–15pp, so this design **excludes the top of that range but not the middle**. Do not write "we
rule out contamination effects of the size others report" — the data does not carry it. The
honest claim is "no evidence of a memorisation advantage; effects above ~12–15pp are excluded."

### Supporting analyses

**Contamination-detector false-positive rate.** The corpus derives from GSM8K *train*, so every
*test* item is a **known negative** — it cannot be contaminated. That makes the standard
detector's error rate measurable rather than arguable:

| rule | train flagged | test flagged | false-positive rate |
|---|---|---|---|
| any 13-gram match (Brown et al., 2020) | 866 (86.6%) | 3 (0.3%) | **0.3%** |
| containment ≥ 0.50 | 584 (58.4%) | 1 (0.1%) | 0.1% |
| containment ≥ 0.80 (this study) | 303 (30.3%) | 0 (0.0%) | **0.0%** |

Test-item containment: median 0.000, p99 0.000, max 0.538. **The detector works.** This was run
expecting the opposite — that shared GSM8K templates would produce many false positives — and it
did not. The finding is more useful than the one predicted: it removes "your detector is noisy"
as an explanation for the null.

Note also that **86.6% of train items have *some* overlap but only 30.3% reach ≥0.80**. The
synthetic corpus mostly *derives from* GSM8K train rather than copying it, which is exactly why a
binary contaminated/clean label is the wrong instrument and a continuous measure with an explicit
ambiguous band is the right one.

**Measurement validity (termination).** A model cut off at the token limit scores as a weak model
when it is really a truncated measurement:

| checkpoint | reached a semantic stop | `####` marker | median chars | flagged degenerate |
|---|---|---|---|---|
| `stage1` | **84.9%** | 84.9% | 196 | 19% |
| `stage2` | 98.4% | 98.3% | 277 | 3% |
| `instruct` | 99.8% | 33.6% | 700 | 3% |

`stage1`'s 19% is genuine model weakness, not a formatting break — a broken chat template or a
`transformers` 5.x mismatch shows ~10% marker rate, not 85%. Those items score as incorrect,
which is conservative. `instruct`'s low marker rate is DPO conversational style, already shown
above not to affect the DiD.

---

## Every figure explained

Rendered to `experiments/runs/fullscale-S250-v2/figures/`. Each answers one question a reviewer
will actually ask.

| figure | question it answers |
|---|---|
| **F1** `f1_trajectory` | Where along the training pipeline does accuracy appear, and does it appear on seen items only? |
| **F2** `f2_did_forest` | How large is the memorisation component, with intervals? *This is the headline figure.* |
| **F3** `f3_perturbation_slopes` | Does accuracy survive numeric perturbation, per checkpoint? |
| **F4** `f4_containment` | Is the seen/unseen treatment assignment actually verified? *Was unplottable in v1 — it drew containment above 1.0 on a "fraction" axis. Fixed in v2.* |
| **F5** `f5_magnitude_control` | Is the perturbation effect just bigger arithmetic? |
| **F6** `f6_measurement_validity` | Are the models being scored on complete outputs? |
| **F7** `f7_consistency_decomposition` | Is the inconsistency real perturbation sensitivity, or just decoding noise? |
| **F8** `f8_design_power` | Is the sample size adequate, given the measured ICC? |

Design rules are enforced in code, not left to taste: one y-axis per panel (a dual-scale chart
lets the author pick the visual conclusion), fixed categorical colours so a checkpoint keeps its
hue when a filter changes the series count, colours validated CVD-safe (worst adjacent separation
dE 9.4 under simulated protanopia/deuteranopia/tritanopia, above the 8 floor), identity also
carried by direct labels so it is never colour-alone, and error bars that are the same parent-item
bootstrap used in the analysis.

## Every analysis file explained

Under `experiments/runs/fullscale-S250-v2/analysis/`:

| file | contents |
|---|---|
| `memorization.json` | The headline DiD, cells, and cluster-robust intervals |
| `memorization_purified.json` | DiD restricted to verified-verdict items. In v2 identical to the headline, which confirms every arm item carries a verified label |
| `containment_verified.json` | Corrected 13-gram containment for all 2,000 candidates, keyed `train::`/`test::` |
| `containment.json` | Same evidence in the shape F4 consumes |
| `detector_falsepositive.json` | Threshold sweep and false-positive rate against known negatives |
| `extraction_tiers.json` | Which extraction tier produced each answer, per checkpoint and per DiD cell |
| `termination.json` | Semantic-stop audit across every generation file |

`atlas/` holds a per-checkpoint JSON + Markdown summary. `generations/` holds raw model text —
kept deliberately, because a scoring bug found after the fact once destroyed a whole run when
only scores had been saved.

---

## What is left

**Nothing is required.** Both remaining items are improvements, listed with honest costs.

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
experiment, and it is the only thing standing between "no evidence of an effect" and "we exclude
the effects others report."**

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
benchmark problem we can search the actual training corpus for near-duplicates, then
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
