<h1 align="center">Instella Reasoning Atlas</h1>

<p align="center">
  <b>Reasoning or Remembering?</b><br>
  Diagnosing whether AMD Instella solves reasoning tasks by <i>generalizing</i>,
  by <i>memorizing</i> training examples, or by relying on <i>fragile shortcuts</i>.
</p>

<p align="center">
  <a href="#quick-start-3-tiers">Quick start</a> ·
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
   train data │  C / PC / N  │   │ acc × consist.│   │ docs caused it│   │ guidelines   │
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

## Quick start (3 tiers)

### Tier 1 — CPU smoke, no GPU, no downloads (~1 minute)

Runs **the entire pipeline** on bundled example data with a dependency-free embedder.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

instella-reasoning run-all --config configs/pipeline/full.yaml   # contamination→score→gap→attribution→atlas→report
pytest                                                            # the full test suite
```

Artifacts land in `outputs/atlas_run/` (`report.md`, `atlas.json`, `contamination.jsonl`, …).

### Tier 2 — Colab T4 (real embeddings + Instella)

Open [`notebooks/instella_reasoning_atlas.ipynb`](notebooks/instella_reasoning_atlas.ipynb)
in Colab (**Runtime → T4 GPU**) and run all cells. It clones, installs, runs the CPU
smoke path, then downloads real data and generates with Instella-3B in 4-bit.

### Tier 3 — full study

```bash
pip install -e ".[all]"                 # hf + retrieval + viz + stats + train
bash scripts/download_data.sh 200 5000  # benchmarks + Instella corpus shards (see docs/DATA_DOWNLOAD.md)
# edit configs/pipeline/full.yaml: set generation.model, expand_variants: true, data paths
instella-reasoning run-all --config configs/pipeline/full.yaml
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
| `make-variants` | 2 | Expand a benchmark into semantics-preserving consistency clusters. |
| `scan-contamination` | 1 | Lightweight lexical contamination scan. |
| `scan-contamination-embedding` | 1 | Embedding + 13-gram scan → C / PC / N labels. |
| `generate` | 2 | Generate completions with Transformers (4-bit, batching, CoT prompts). |
| `score-generations` | 2 | Benchmark-aware answer extraction + exact-match scoring. |
| `accuracy-gap` | 2 | Contaminated-vs-clean accuracy with a two-proportion z-test. |
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
├── datasets/loaders.py      # HF benchmark/corpus → JSONL (skill-tagged)
├── prompting.py             # CoT templates + benchmark-aware answer extractors
├── evaluation.py            # model generation (4-bit/batch/CoT) + exact-match scoring
├── embedding.py             # MiniLM/GTE embedder + dependency-free hashing fallback
├── faiss_index.py           # FAISS (flat/ivfpq/hnsw) + brute-force fallback, save/load
├── contamination.py         # lexical + embedding/13-gram scanners → C/PC/N   [Phase 1]
├── perturbations.py         # 5 semantics-preserving variant generators        [Phase 3]
├── metrics.py               # Reliability = accuracy × consistency
├── attribution.py           # retrieval-correctness proxy (triage)             [Phase 3]
├── attribution_embedding.py # Tier-1 neighbour characterization (Gini/sources) [Phase 4]
├── attribution_gradient.py  # TracIn-CP + Concept Influence (torch-gated)      [Phase 4]
├── emergence.py             # transitions, Schaeffer test, RL effect, CoT divergence [Phase 5]
├── reporting.py             # summary + atlas + gap + attribution report
├── pipeline.py              # end-to-end orchestrator (run-all)
├── training.py              # upstream Instella torchrun command builder
├── cli.py                   # the `instella-reasoning` CLI
└── analysis/
    ├── accuracy_gap.py      # contaminated-vs-clean gap + z-test               [Phase 2]
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
pytest              # unit + analysis tests (perturbations, atlas, stats, attribution, emergence, plots)
ruff check .        # lint
make smoke          # example pipeline end-to-end
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint + tests + the
smoke pipeline on every push.

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
