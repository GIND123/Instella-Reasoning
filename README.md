# Instella Reasoning Atlas

Diagnosing whether AMD Instella solves reasoning tasks by generalizing, by
matching memorized training examples, or by relying on fragile shortcuts.

This repository is a research scaffold for the "Reasoning or Remembering?"
project described in [`docs/proposal/`](docs/proposal/). It is designed to be
cloned, extended, and run on either a small local machine for smoke tests or on
a self-hosted GitHub runner / AMD ROCm cluster for full Instella-scale runs.

The code here does not replace AMD's official Instella trainer. It wraps the
research workflow around it: contamination search, robustness evaluation,
retrieval-style attribution, report generation, and launch templates for
upstream Instella training.

The current base build is intentionally portable. Core JSONL processing, lexical
contamination checks, hashing-based embedding search, scoring, reporting, and
tests run on CPU-only Windows or Linux. Hugging Face loading, sentence
embeddings, FAISS, model generation, and training launch helpers are available
through optional extras.

## Source Basis

This scaffold is grounded in:

- AMD's Instella GitHub repository: <https://github.com/AMD-AGI/Instella>
- Hugging Face model card content for `amd/Instella-3B` and related checkpoints
- The local research proposal and literature compendium in `docs/proposal/`

Key Instella facts used by the scaffold:

- 3.11B parameter decoder-only language model family.
- 36 decoder layers, 32 attention heads, hidden size 2560, MLP size 13824.
- 4096 token context length and OLMo tokenizer with about 50K vocabulary.
- Stage 1 pretraining: about 4.065T tokens.
- Stage 2 pretraining: about 57.575B additional tokens.
- SFT: about 8.902B tokens for 3 epochs.
- DPO: about 760M preference tokens.
- Training stack: ROCm, MI300X, FlashAttention-2, torch compile, bf16, and FSDP.

## Repository Layout

```text
.
|-- configs/
|   |-- pipeline/                 # Research pipeline configs
|   `-- training/                 # Upstream Instella launch configs
|-- data/
|   |-- raw/                      # Local raw benchmark/corpus files
|   |-- processed/                # Normalized JSONL artifacts
|   `-- indices/                  # Retrieval indexes and manifests
|-- docs/
|   |-- GITHUB_TRAINING.md        # Self-hosted GitHub runner guide
|   |-- IMPLEMENTATION_PLAN.md    # Phased research execution roadmap
|   |-- METHODOLOGY.md            # Research method and metrics
|   |-- DATA_MANIFEST.md          # Data layout and artifact conventions
|   `-- proposal/                 # User-provided proposal/reference material
|-- examples/                     # Tiny smoke-test benchmark and corpus
|-- scripts/                      # Shell entrypoints for local/cluster runs
|-- src/instella_reasoning/       # Python package and CLI
|   |-- analysis/                 # Accuracy-gap and follow-on analyses
|   `-- datasets/                 # Hugging Face benchmark/corpus loaders
|-- tests/                        # Unit tests for the research scaffold
`-- outputs/                      # Generated reports, scans, predictions
```

## Quick Start

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Useful optional install profiles:

```bash
pip install -e ".[hf]"          # Hugging Face datasets and model utilities
pip install -e ".[retrieval]"   # sentence-transformers, FAISS, scikit-learn
pip install -e ".[train]"       # torch/PEFT/TRL/W&B training dependencies
pip install -e ".[all]"         # everything above plus dev tools
```

Run the smoke test pipeline:

```bash
instella-reasoning scan-contamination \
  --benchmark examples/mini_benchmark.jsonl \
  --corpus examples/mini_corpus.jsonl \
  --output outputs/mini_contamination.jsonl

instella-reasoning score-generations \
  --benchmark examples/mini_benchmark.jsonl \
  --generations examples/mini_generations.jsonl \
  --output outputs/mini_scores.jsonl

instella-reasoning report \
  --scores outputs/mini_scores.jsonl \
  --contamination outputs/mini_contamination.jsonl \
  --output outputs/mini_report.md
```

Or use the wrapper script:

```bash
bash scripts/run_smoke_pipeline.sh
```

Run the test suite:

```bash
pytest
```

## Base Build Capabilities

The CLI currently exposes the full baseline path needed to start the study:

| Command | Purpose |
|---|---|
| `load-benchmark` | Convert supported Hugging Face reasoning benchmarks to the project JSONL contract. |
| `load-corpus` | Stream a Hugging Face corpus or dataset into `CorpusDocument` JSONL rows. |
| `scan-contamination` | Run the lightweight lexical contamination scanner over local JSONL files. |
| `scan-contamination-embedding` | Run embedding retrieval plus 13-gram overlap and label items as contaminated, partial, or clean. |
| `generate` | Generate model completions with Transformers, optional 4-bit loading, batching, and CoT prompts. |
| `score-generations` | Extract final answers with benchmark-aware parsers and compute exact-match correctness. |
| `accuracy-gap` | Compare contaminated, partial, and clean accuracy with a two-proportion z-test. |
| `attribute` | Build retrieval-correctness attribution proxy rows from contamination and scoring outputs. |
| `report` | Write a Markdown reliability report from scores and contamination hits. |
| `train-command` | Render upstream Instella `torchrun` commands from YAML launch configs. |

## Phase 1: Embedding Contamination Search

The embedding scan implements the proposal's C/PC/N classification (cosine + 13-gram
overlap). It runs anywhere: install the `retrieval` extra for real MiniLM/GTE
embeddings + FAISS, or fall back to the dependency-free hashing embedder and
brute-force index for smoke tests and CI.

```bash
# Full setup (Colab T4 or a machine with the retrieval extra):
pip install -e ".[hf,retrieval]"

# Download a benchmark and an Instella training shard from HuggingFace:
instella-reasoning load-benchmark --benchmark gsm8k --output data/processed/gsm8k.jsonl --limit 200
instella-reasoning load-corpus \
  --hf-path amd/Instella-GSM8K-synthetic \
  --output data/processed/gsm8k_synth.jsonl --limit 5000

# Embedding + 13-gram contamination scan (MiniLM + FAISS when installed):
instella-reasoning scan-contamination-embedding \
  --benchmark data/processed/gsm8k.jsonl \
  --corpus data/processed/gsm8k_synth.jsonl \
  --output outputs/gsm8k_contamination.jsonl \
  --top-k 20

# Dependency-free fallback (no GPU, no downloads):
instella-reasoning scan-contamination-embedding \
  --benchmark examples/mini_benchmark.jsonl \
  --corpus examples/mini_corpus.jsonl \
  --output outputs/mini_contamination_embedding.jsonl \
  --embedder-backend hashing --index-backend bruteforce
```

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the full phased
roadmap (data sources, tests, analyses, and required plots).

## Research Pipeline

1. Build a benchmark item file in JSONL with `load-benchmark` or a custom exporter.
2. Build or stream a corpus manifest for Instella training data with `load-corpus`.
3. Search each benchmark item against the corpus for exact, near-duplicate,
   paraphrase-candidate, contaminated, partial, and clean matches.
4. Generate Instella answers for original and perturbed benchmark variants.
5. Score exact answer accuracy and grouped answer consistency.
6. Compute contaminated-vs-clean accuracy gaps for each benchmark/model slice.
7. Combine contamination hits and evaluation scores into a Markdown report.
8. Optionally run retrieval-style attribution over the top corpus hits.

The core exchange format is JSONL so long jobs can be restarted, sharded, and
merged without a database.

Example baseline workflow:

```bash
instella-reasoning load-benchmark \
  --benchmark gsm8k \
  --output data/processed/gsm8k.jsonl \
  --limit 200

instella-reasoning load-corpus \
  --hf-path amd/Instella-GSM8K-synthetic \
  --output data/processed/gsm8k_synth.jsonl \
  --limit 5000

instella-reasoning scan-contamination-embedding \
  --benchmark data/processed/gsm8k.jsonl \
  --corpus data/processed/gsm8k_synth.jsonl \
  --output outputs/gsm8k_contamination.jsonl \
  --top-k 20

instella-reasoning generate \
  --benchmark data/processed/gsm8k.jsonl \
  --model amd/Instella-3B \
  --output outputs/gsm8k_generations.jsonl \
  --load-in-4bit

instella-reasoning score-generations \
  --benchmark data/processed/gsm8k.jsonl \
  --generations outputs/gsm8k_generations.jsonl \
  --output outputs/gsm8k_scores.jsonl

instella-reasoning accuracy-gap \
  --scores outputs/gsm8k_scores.jsonl \
  --contamination outputs/gsm8k_contamination.jsonl \
  --output outputs/gsm8k_accuracy_gap.json
```

## GitHub Training

GitHub-hosted runners are not suitable for real Instella training. Use a
self-hosted GitHub runner attached to a ROCm machine or cluster node, then
trigger `.github/workflows/amd-base-self-hosted.yml` manually.

See [`docs/GITHUB_TRAINING.md`](docs/GITHUB_TRAINING.md) for runner labels,
required secrets, cache layout, and multi-node notes.

## Upstream Instella Training

Clone the official trainer separately:

```bash
git clone https://github.com/AMD-AGI/Instella external/Instella
```

Then generate a torchrun command from the template config:

```bash
instella-reasoning train-command \
  --config configs/training/amd_base_upstream.yaml
```

The command builder supports single-node and multi-node `torchrun`, checkpoint
resume paths, and extra upstream config overrides.

## License and Use

This scaffold is for research engineering around Instella. Instella model
checkpoints and the AMD GSM8K synthetic dataset are released for academic and
research purposes under their ResearchRAIL terms. Confirm the license on each
model and dataset card before training, redistribution, or publication.

## Citation

```bibtex
@article{liu2025instella,
  title={Instella: Fully Open Language Models with Stellar Performance},
  author={Liu, Jiang and Wu, Jialian and Yu, Xiaodong and Su, Yusheng and Mishra, Prakamya and Ramesh, Gowtham and Ranjan, Sudhanshu and Manem, Chaitanya and Sun, Ximeng and Wang, Ze and Brahma, Pratik Prabhanjan and Liu, Zicheng and Barsoum, Emad},
  journal={arXiv preprint arXiv:2511.10628},
  year={2025}
}
```
