# Compute Requirements

The study is designed to run on **a single NVIDIA T4 (16 GB VRAM) plus CPU** — no
GPU cluster, no paid APIs, no external LLMs beyond Instella. Every non-model stage
(contamination search, embedding, FAISS, metrics, statistics, plots) runs on CPU,
and the whole analysis layer runs **dependency-free on a laptop** at smoke/sample
scale.

## Tiers at a glance

| Tier | Hardware | Installs | What runs | Time |
|---|---|---|---|---|
| **Smoke / CI** | Any CPU (Windows/Linux/Mac) | `.[dev]` | Full pipeline on bundled examples with the hashing embedder + brute-force index; whole test suite | ~1 min |
| **Sampled study** | Colab T4 (16 GB) | `.[hf,retrieval,viz]` | Real MiniLM + FAISS contamination scan, Instella-3B 4-bit generation, Atlas + figures on a few hundred items | hours |
| **Full study** | T4 (persistent) or MI300X | `.[all]` | All 4 model variants x ~10K benchmark items, TracIn attribution, emergence | ~25-35 GPU-days |

## Per-model VRAM (inference)

| Model | HF ID | FP16 | 4-bit (NF4) | T4 |
|---|---|---|---|---|
| AMD OLMo-1B | `amd/AMD-OLMo-1B` | ~2 GB | ~1 GB | full precision |
| Instella-3B | `amd/Instella-3B` | ~6 GB | ~3 GB | 4-bit |
| Instella-3B-Instruct | `amd/Instella-3B-Instruct` | ~6 GB | ~3 GB | 4-bit |
| Instella-3B-Math | `amd/Instella-3B-Math` | ~6 GB | ~3 GB | 4-bit |

Pass `--load-in-4bit` to `generate` (needs `bitsandbytes` + CUDA) for the 3B models.
The 1B model runs comfortably in FP16.

## Stage-by-stage budget (full study)

| Stage | Hardware | Time | Peak memory |
|---|---|---|---|
| Embed ~13M training passages (MiniLM) | CPU | 18-24 h | 8 GB RAM |
| Build FAISS index (IVF+PQ) | CPU | 2-3 h | 16 GB RAM |
| Contamination search (~10K items x top-20) | CPU | 15-30 min | 4 GB RAM |
| Instella-1B over ~10K items | T4 | 8-12 h | 4 GB VRAM |
| Each Instella-3B variant (4-bit) over ~10K | T4 | 20-30 h | 6 GB VRAM |
| Consistency suite (4 models x ~3.2K variants) | T4 | 4-6 days | 6 GB VRAM |
| TracIn-CP (500 items x ~100 candidates) | T4 | 3-5 days | ~14 GB VRAM |
| Concept Influence (100 items, forward-only) | T4 | 1-2 days | ~10 GB VRAM |

**Totals:** ~25-35 GPU-days, ~30-40 CPU-hours, 16 GB RAM peak, ~50 GB disk
(embeddings + index + outputs). Model-inference stages run unattended overnight.

## Disk

- Embeddings + FAISS index: ~2 GB (IVF+PQ) to ~6 GB (flat, 384-dim, 10M vectors)
- Model weights cache: ~6-12 GB per model
- JSONL artifacts + figures: < 1 GB

## FAISS index sizing (10M x 384-dim)

| Index | RAM | Notes |
|---|---|---|
| Flat (exact) | ~15 GB | ground-truth; validate critical hits here |
| IVF4096 + PQ48 | ~2 GB | recommended for development speed |
| HNSW32 | ~20 GB | best recall, heaviest |

## Attribution fallback ladder (if TracIn on 3B won't fit a T4)

1. Gradient checkpointing + micro-batches on Instella-3B (the default in
   `attribution_gradient.TracInAttributor`).
2. Run TracIn on **Instella-1B** (fits in FP16).
3. **Concept Influence** — forward-only representation similarity (`ConceptAttributor`).
4. **Embedding attribution only** (`attribution_embedding`, Tier 1) — always available,
   CPU-friendly, and the default in the orchestrator.

## Cost estimate (cloud)

- Preemptible cloud T4 (~$0.35/h): ~35 days -> ~$294.
- Colab Pro T4 (~$10/mo) with session management: ~$50.
- University / owned GPU: **$0**.
