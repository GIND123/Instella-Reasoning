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
|   |-- METHODOLOGY.md            # Research method and metrics
|   `-- proposal/                 # User-provided proposal/reference material
|-- examples/                     # Tiny smoke-test benchmark and corpus
|-- scripts/                      # Shell entrypoints for local/cluster runs
|-- src/instella_reasoning/       # Python package and CLI
|-- tests/                        # Unit tests for the boilerplate
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

## Research Pipeline

1. Build a benchmark item file in JSONL.
2. Build or stream a corpus manifest for Instella training data.
3. Search each benchmark item against the corpus for exact, near-duplicate, and
   paraphrase-candidate matches.
4. Generate Instella answers for original and perturbed benchmark variants.
5. Score exact answer accuracy and grouped answer consistency.
6. Combine contamination hits and evaluation scores into a report.
7. Optionally run retrieval-style attribution over the top corpus hits.

The core exchange format is JSONL so long jobs can be restarted, sharded, and
merged without a database.

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
