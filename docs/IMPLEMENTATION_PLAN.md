# Implementation Plan — Reasoning or Remembering?

This is the execution roadmap for the study described in
[`docs/proposal/reasoning-or-remembering-proposal.md`](proposal/reasoning-or-remembering-proposal.md).
It maps every research phase onto concrete modules, data sources, tests, analyses,
and plots. It is designed to be **cloned and run on either a Colab T4 or a Windows
machine**; heavy model runs are optional and gated behind extras.

## Central idea (one paragraph)

When Instella answers a reasoning problem correctly, accuracy alone cannot tell us
whether it **reasoned** or **remembered**. Because Instella is fully open (weights +
training data + recipe), we can, for every benchmark item, search the actual training
corpus for near-duplicates, then measure **Reliability = Accuracy × Consistency**
across semantically equivalent variants, then **attribute** correct answers to the
training documents that caused them. The deliverable is the **Reasoning Reliability
Atlas**: a per-sub-skill map of which reasoning is genuine, fragile, or absent, and
what training data drives each.

## Four stages

1. **Filter** — contamination search → label each item Contaminated / Partial / Clean.
2. **Diagnose** — variant generation + Reliability Score.
3. **Attribute** — embedding attribution → TracIn-CP → Concept Influence (full tiered stack).
4. **Prescribe** — correlate reliability with training-data properties → curation guidelines.

---

## Data

### Subject models (HuggingFace)
| Model | HF ID | Role | T4 |
|---|---|---|---|
| AMD OLMo-1B | `amd/AMD-OLMo-1B` | scale baseline | FP16 |
| Instella-3B | `amd/Instella-3B` | base | 4-bit |
| Instella-3B-Instruct | `amd/Instella-3B-Instruct` | pre-RL | 4-bit |
| Instella-3B-Math | `amd/Instella-3B-Math` | post-RL math | 4-bit |
| Instella-3B-Long-Instruct | `amd/Instella-3B-Long-Instruct` | optional | 4-bit |

### Reasoning benchmarks (~9.7K items, 8 sub-skills)
GSM8K (1319) · MATH L1-3 (~2500) · LogiQA 2.0 (1600) · ARC-Challenge (1172) ·
BBH selected subtasks (~1500) · CLUTRR held-out rules (~500) · ReClor (500) ·
HumanEval (164) · TTT-Bench (~500).

### Robustness suites (reuse, don't rebuild)
GSM-Symbolic, GSM-Plus, MATH-Perturb, Functional-MATH, Putnam-AXIOM.

### Instella training corpus (ground truth for contamination) — index in this priority
1. `amd/Instella-GSM8K-synthetic` — **full index** (smoking gun for GSM8K)
2. `nvidia/OpenMathInstruct-2` — sample ~2M (Instella-Math SFT source)
3. `dm_math`, `python-edu` — full / sampled
4. `dolmino-mix-1124` — sample ~2M
5. `DCLM`, `FineWeb-Edu` — sample 3-5M passages each

Total ~13-15M passages; FAISS ~2 GB (IVF+PQ) to ~6 GB (flat). Start small/high-risk.

### Utility models
`sentence-transformers/all-MiniLM-L6-v2` (primary) · `thenlper/gte-small` (alt) ·
FAISS (CPU, GPU optional) · `transformers` + `bitsandbytes` (4-bit).

---

## Module plan (`src/instella_reasoning/`)

| Phase | Module(s) | Status |
|---|---|---|
| 0 | `datasets/loaders.py`, `prompting.py`, upgrade `evaluation.py` (4-bit, batching, CoT) | **done** |
| 1 | `embedding.py`, `faiss_index.py`, upgrade `contamination.py` (embedding+13-gram, C/PC/N) | **done** |
| 2 | `analysis/accuracy_gap.py` — baseline eval disaggregated by contamination label + z-test | **done** |
| 3 | `perturbations.py` (numeric/entity/reorder/irrelevant/rephrase), `analysis/atlas.py` Atlas builder | **done** |
| 4 | `attribution_embedding.py` (Tier 1), `attribution_gradient.py` (TracIn-CP + Concept Influence) | **done** |
| 5 | `emergence.py` (transition classes, Schaeffer test, CoT divergence, RL effect) | **done** |
| 6 | `analysis/stats.py`, `analysis/plots.py`, extended `reporting.py` (atlas/gap/attribution) | **done** |
| — | `pipeline.py` orchestrator (`run-all`), Colab notebook, `full.yaml`, download/run scripts | **done** |

The base scaffold provides: lexical contamination, exact-match scoring, Reliability
metric, retrieval-attribution proxy, reporting, torchrun builder, CLI. The full build
adds every phase above plus a single-command end-to-end orchestrator and a Colab
runner, all portable to CPU-only machines through dependency-free fallbacks.

---

## Tests

**Unit:** perturbations preserve answers · answer extractors on gold cases ·
contamination threshold boundaries · FAISS recall vs brute force on tiny set ·
metric edge cases · loader schema validation.

**Research analyses / statistics:**
1. Contamination rate + accuracy gap (contaminated vs clean): two-proportion z-test /
   χ², difficulty as covariate (logistic regression).
2. Reliability decomposition by contamination: Mann-Whitney + bootstrap 95% CI + effect size.
3. GSM8K→GSM-Symbolic drop vs same-size open models; treat as hypothesis (Ivanova critique);
   Benjamini-Hochberg for multiple comparisons.
4. Attribution concentration: near-duplicate share of top-influence docs (≥30% ⇒ memorization).
5. Emergence: smooth vs discontinuous scaling; Schaeffer metric-artifact test.
6. RL effect: consistency vs accuracy gain; contamination-masking check (arXiv:2510.02386).
7. Metric robustness: threshold / embedding-model / 4-bit-vs-FP16 sensitivity.

---

## Plots (use the `dataviz` skill before writing chart code)

**Contamination:** per-benchmark C/PC/N bars · similarity-score histogram/KDE with cutoffs · validation confusion matrix.
**Accuracy gap:** contaminated-vs-clean grouped bars · effect-size forest plot · difficulty-stratified accuracy lines.
**Reliability (Atlas):** sub-skill × {acc,cons,rel} heatmap per contamination level (money figure) · Accuracy-vs-Consistency quadrant scatter · reliability slope plot · per-problem reliability violins.
**Attribution:** top-source stacked bars per sub-skill · influence-concentration Lorenz/Gini · case-study cards.
**Scale/emergence:** reliability-vs-size lines · capability-transition heatmap · raw-accuracy-vs-reliability scaling (Schaeffer test) · CoT-divergence Sankey.

---

## Compute & feasibility

Single T4 (16 GB) + CPU · ~25-35 GPU-days total · ~50 GB disk · ~$50-$300 cloud or $0
on owned hardware. FAISS/embedding on CPU; inference overnight. TracIn fallback ladder:
gradient-checkpoint 3B → run on 1B → Concept Influence → embedding-attribution only.

## Portability contract

- Core pipeline (loaders, embedding, FAISS, contamination, metrics, plots) runs on
  **CPU-only Windows** at smoke/sample scale with no GPU.
- Heavy pieces (model generation, TracIn) are gated behind `[hf]` / `[train]` extras
  and degrade gracefully with clear install messages.
- All paths, configs, and CLI commands work identically on Colab and Windows.
