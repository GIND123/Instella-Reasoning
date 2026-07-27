<h1 align="center">Instella Reasoning Atlas</h1>

<p align="center">
  <b>Reasoning or Remembering?</b><br>
  Diagnosing whether AMD Instella solves reasoning tasks by <i>generalizing</i>,
  by <i>memorizing</i> training examples, or by relying on <i>fragile shortcuts</i>.
</p>

<p align="center">
  <a href="experiments/FULLSCALE.md"><b>Full-scale suite</b></a> ·
  <a href="notebooks/fullscale_colab.ipynb">Colab</a> ·
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

## Current validated full-scale workflow

**This is the authoritative handoff for agents and Colab operators.** Use
[`notebooks/fullscale_colab.ipynb`](notebooks/fullscale_colab.ipynb) and
[`experiments/run_fullscale_suite.sh`](experiments/run_fullscale_suite.sh) for the headline
run. The generic Atlas demo and superseded reliability examples later in this README are not
the current experiment.

- **Headline scope:** `SUITE_TIER=1` runs `stage1`, `stage2`, and `instruct`. This preserves
  the controlled Stage-1 → Stage-2 intervention where GSM8K-derived data enters, plus the
  aligned Instruct endpoint. The costed run is about 10.5 GPU-hours on the measured hardware.
- **Math is diagnostic-only:** `math` moved to `SUITE_TIER=2` after live smoke tests reached
  only 75% semantic completion even at 3072 tokens. Do not lower or bypass the 85%
  completion gate to force it into headline results.
- **Completion semantics:** a generation is complete when it emits EOS, reaches the `####`
  final-answer marker, or crosses into the next few-shot exemplar. This prevents valid base
  checkpoint answers from being mislabeled as token-cap truncations.
- **Colab secrets:** add `github` (GitHub repo read access) and `hf` (Hugging Face write
  access). The notebook authenticates Git with an in-memory HTTP header; do not put either
  token in a remote URL or printed command.
- **Hugging Face destination:** results live in the private dataset repo
  `GOVINDFROM/Instella-Reasoning`, under
  `experiments/runs/fullscale-S250/`. The successful preflight report is
  `analysis/preflight.json` inside that run path.
- **Cross-runtime resume:** the suite pulls this run directory before starting and pushes
  after every completed preparation stage and `(checkpoint, block)`, including raw
  generations by default. Re-run the same launch cell in a new Colab environment to resume.
  Work in the currently active block normally remains local until that block finishes;
  interrupted raw generations can also be pushed manually and resumed item-by-item.

### Current Hugging Face snapshot (2026-07-27)

The private dataset repo `GOVINDFROM/Instella-Reasoning` currently holds the recoverable
state below under `experiments/runs/fullscale-S250/`. This is a progress checkpoint, not a
finished experiment.

| Area | Current contents |
|---|---|
| `markers/` | Successful preflight write probe and all four preparation markers: `stage1_corpus`, `stage2_splits`, `stage3_variants`, `stage4_aux` |
| `analysis/` | `preflight.json` and the verified-arm containment report |
| `contamination/` | The approximately 1.45 GB `synthetic_corpus.jsonl` used for exact containment |
| `base/` | Normalized GSM8K train and test inputs |
| `arms/` | The verified seen/unseen GSM8K arms |
| `variants/` | Raw and validated arm variants, official GSM-Symbolic inputs, and the decoding-noise resample control |
| `review/` | Exported template-review CSV |
| `generations/` | `stage1__arms.jsonl`: 1,553/1,553 rows, approximately 1.72 MB |
| `scores/`, `atlas/`, `figures/` | No completed result artifacts yet |

The `stage1/arms` generation reached every benchmark item, but only **83.2%** of outputs
reached EOS or the final-answer marker, below the required **85%** termination gate.
Consequently it is deliberately **unscored** and must not be treated as a result. Of its
1,553 rows, 1,292 are valid and approximately 261 truncated rows need a targeted retry with
a larger token budget. Preserve the original file, retain rows whose
`metadata.finished` value is true, regenerate only the unfinished IDs at 2,048 tokens, then
run `score-generations` and push the resulting `scores/stage1__arms.jsonl`.

At this checkpoint all CPU/data preparation is complete, the first raw-generation block is
complete but not yet valid, and none of the seven Tier-1 GPU blocks has a score completion
marker. A fresh runtime must pull from HF before doing any repair or launch work. After the
repaired `stage1__arms.jsonl` passes the gate and is scored, rerunning the suite will skip
that block and continue with `stage1/gsmsym`.

Required gate and launch:

```bash
export OUT=experiments/runs/fullscale-S250
export SUITE_TIER=1
export HF_RESULTS_REPO=GOVINDFROM/Instella-Reasoning

python scripts/preflight_fullscale.py --smoke --tier 1 \
  --out "$OUT" --repo "$HF_RESULTS_REPO" \
  --json "$OUT/analysis/preflight.json"
# Continue only after: SAFE TO LAUNCH

python experiments/hf_sync.py push \
  --repo "$HF_RESULTS_REPO" --path "$OUT" \
  --message "successful Tier-1 preflight"

bash experiments/run_fullscale_suite.sh
```

On a fresh Pro runtime, clone or pull `main`, install
`.[hf,retrieval,viz,stats]`, expose the same `hf` secret as `HF_TOKEN`, and run the same
launch command with the same `OUT`. The initial HF pull restores completed work. Full design,
budget, artifacts, and failure gates are documented in
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
