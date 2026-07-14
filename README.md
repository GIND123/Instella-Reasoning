<h1 align="center">Instella Reasoning Atlas</h1>

<p align="center">
  <b>Reasoning or Remembering?</b><br>
  Diagnosing whether AMD Instella solves reasoning tasks by <i>generalizing</i>,
  by <i>memorizing</i> training examples, or by relying on <i>fragile shortcuts</i>.
</p>

<p align="center">
  <a href="#run-instructions">Run instructions</a> ·
  <a href="notebooks/instella_reasoning_atlas.ipynb">Colab notebook</a> ·
  <a href="#command-reference">Commands</a> ·
  <a href="docs/COMPUTE.md">Compute</a> ·
  <a href="docs/DATA_DOWNLOAD.md">Data download</a> ·
  <a href="docs/proposal/">Proposal</a>
</p>

---

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
!git clone https://github.com/GIND123/Instella-Reasoning
%cd Instella-Reasoning
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
#    (Chat templating for -Instruct models is automatic; see the note below.)
!instella-reasoning generate \
    --benchmark data/processed/gsm8k.jsonl \
    --model amd/Instella-3B-Instruct \
    --output outputs/gsm8k_smoke.jsonl \
    --limit 4 --max-new-tokens 256 --batch-size 2
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

> **Two things that decide whether you get real answers vs. gibberish:**
> 1. **A GPU must actually be attached.** If `preflight.py` reports `cpu_only_build` /
>    `cuda_available=False`, the 3B model falls back to CPU and generation is unusably
>    slow — fix the runtime type before going further.
> 2. **Instruct models need their chat template**, which the pipeline now applies
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
| `generate` | 2 | Generate completions with Transformers (chat template, 4-bit/bf16, batching, CoT prompts, `--limit`). |
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
├── evaluation.py            # model generation (chat template/4-bit/bf16/CoT) + scoring
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
