# AMD Instella Data & Model Download Procedure

Instella is one of the very few competitive model families that is **fully open** —
weights, training code, data recipe, **and the training data itself** are public. That
is what makes the contamination-and-attribution study possible: for every benchmark
problem we can search the *actual* training corpus.

This guide lists the exact HuggingFace repositories, how to pull them into the
project's JSONL contract, and the licensing you must respect.

> **License note.** Instella checkpoints and the AMD GSM8K-synthetic dataset are
> released for academic/research use under AMD's ResearchRAIL terms. Confirm the
> license on each model and dataset card before training, redistribution, or
> publication. The web corpora (DCLM/OLMoE-mix, dolmino, smollm-corpus, TxT360)
> carry their own upstream licenses (mostly ODC-BY / source-specific).

## 0. Prerequisites

```bash
pip install -e ".[hf]"          # datasets + transformers + accelerate + safetensors
huggingface-cli login           # only needed for gated/private repos; most are public
```

Set a token in the environment if a repo is gated (see `.env.example`):

```bash
export HF_TOKEN=hf_xxx
```

## 1. Models (subject of the study)

| Model | HuggingFace ID |
|---|---|
| AMD OLMo-1B (scale baseline) | `amd/AMD-OLMo-1B` |
| Instella-3B-Stage1 (pre-stage-2) | `amd/Instella-3B-Stage1` |
| Instella-3B (base) | `amd/Instella-3B` |
| Instella-3B-SFT | `amd/Instella-3B-SFT` |
| Instella-3B-Instruct (pre-RL) | `amd/Instella-3B-Instruct` |
| Instella-3B-Math (post-RL math) | `amd/Instella-3B-Math` |
| Instella-3B-Long-Instruct (128K) | `amd/Instella-3B-Long-Instruct` |

Models are pulled automatically by `transformers` on first use (e.g. through
`instella-reasoning generate --model amd/Instella-3B`). Instella uses a custom
architecture, so keep `trust_remote_code` on (the CLI default).

## 2. Training data (contamination ground truth)

These are the datasets Instella was actually trained on. **Index them in priority
order** — start with the small, high-risk sets where contamination is most likely,
then sample the giant web corpora.

| Priority | Dataset | HuggingFace ID | Role | Strategy |
|---|---|---|---|---|
| 1 | Instella GSM8K-synthetic | `amd/Instella-GSM8K-synthetic` | Stage-2 math synthetic (smoking gun for GSM8K) | **full index** |
| 2 | OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | Instella-Math SFT source | sample ~2M |
| 3 | dolmino-mix-1124 | `allenai/dolmino-mix-1124` | Stage-2 enrichment (math/science/code) | sample ~2M |
| 4 | smollm-corpus | `HuggingFaceTB/smollm-corpus` | Stage-2 (python-edu, etc.) | sample ~1M |
| 5 | TxT360 (dm_math subset) | `LLM360/TxT360` | Stage-2 formal math (`dm_math` subdir) | full / sampled |
| 6 | OLMoE-mix-0924 (DCLM/FineWeb-Edu) | `allenai/OLMoE-mix-0924` | Stage-1 general web pretraining | sample 3-5M passages |

(Stage assignments follow the AMD Instella model card and training report.)

## 3. Pull data into the project JSONL contract

### Benchmarks

```bash
instella-reasoning load-benchmark --benchmark gsm8k         --output data/processed/gsm8k.jsonl --limit 200
instella-reasoning load-benchmark --benchmark arc_challenge --output data/processed/arc.jsonl   --limit 200
instella-reasoning load-benchmark --benchmark math          --output data/processed/math.jsonl  --limit 200
# built-in benchmark loaders: gsm8k, math, arc_challenge, logiqa2, bbh
```

### Corpus shards

```bash
# Priority 1 — full index of the AMD synthetic set (contamination smoking gun):
instella-reasoning load-corpus \
  --hf-path amd/Instella-GSM8K-synthetic \
  --source instella-gsm8k-synthetic \
  --output data/processed/instella_gsm8k_synth.jsonl

# Priority 2 — sample OpenMathInstruct-2 (streamed; --limit triggers streaming):
instella-reasoning load-corpus \
  --hf-path nvidia/OpenMathInstruct-2 \
  --source openmathinstruct-2 \
  --output data/processed/openmathinstruct2.jsonl --limit 2000000

# Web corpus sample (choose a subset/config with --hf-name as needed):
instella-reasoning load-corpus \
  --hf-path allenai/OLMoE-mix-0924 \
  --source olmoe-mix-0924 \
  --output data/processed/olmoe_sample.jsonl --limit 5000000
```

`--limit` switches `load-corpus` into **streaming** mode, so you sample a giant
dataset without downloading all of it. Drop `--limit` only for the small,
fully-indexable sets. `load-corpus` auto-joins `problem`/`question`/`solution`/
`answer` fields when a dataset has no plain `text` column.

Or run the bundled helper for a first pass:

```bash
bash scripts/download_data.sh 200 5000     # BENCH_LIMIT CORPUS_LIMIT (0 = no limit)
```

## 4. Direct HuggingFace access (alternative)

If you prefer to pull raw files and convert yourself:

```python
from datasets import load_dataset
ds = load_dataset("amd/Instella-GSM8K-synthetic", split="train")           # full
ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)  # sample
```

Then write rows to the corpus contract — `{"id","text","source","metadata"}` — see
[`DATA_MANIFEST.md`](DATA_MANIFEST.md).

## 5. Approximate sizes

| Dataset | Approx size | Indexed passages (plan) |
|---|---|---|
| Instella-GSM8K-synthetic | ~50K problems | full |
| OpenMathInstruct-2 | 14M pairs | ~2M |
| dolmino-mix-1124 | ~100B tokens | ~2M |
| smollm-corpus | tens of B tokens | ~1M |
| TxT360 (dm_math) | ~10B tokens | full/sampled |
| OLMoE-mix-0924 | trillions of tokens | 3-5M passages |

**Total indexed ~13-15M passages → FAISS ~2 GB (IVF+PQ) to ~6 GB (flat).** See
[`COMPUTE.md`](COMPUTE.md) for index sizing and the embedding time budget.

## 6. Reference links

- Instella model: <https://huggingface.co/amd/Instella-3B>
- Instella-GSM8K-synthetic: <https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic>
- Instella training code: <https://github.com/AMD-AGI/Instella>
- Instella report: <https://arxiv.org/abs/2511.10628>
- Instella launch blog: <https://rocm.blogs.amd.com/artificial-intelligence/instella/README.html>
