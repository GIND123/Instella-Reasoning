# Methodology

This repository implements the execution layer for a four-part study.

## 1. Contamination Search

Each benchmark item is searched against a manifest of known Instella training
documents. The dependency-free index in `src/instella_reasoning` uses lexical
overlap and character n-gram overlap for smoke tests. At production scale, use
the same JSONL contracts with a sharded FAISS or distributed embedding backend.

Labels:

- `exact`: normalized prompt text is contained in, or effectively identical to,
  a training document.
- `near_duplicate`: high lexical or character n-gram overlap.
- `paraphrase_candidate`: moderate overlap that should be manually inspected or
  sent to an embedding/paraphrase verifier.

## 2. Robustness Evaluation

Benchmark items should include a stable `parent_id`. Variants of the same item
share a parent and set `variant_type`, for example `original`,
`irrelevant_clause`, `entity_rename`, or `numeric_perturbation`.

## 3. Reliability Metric

For each parent group:

```text
accuracy = correct_variants / total_variants
answer_consistency = count(most_common_normalized_answer) / total_variants
reliability = accuracy * answer_consistency
```

This intentionally penalizes cases where a model answers one version correctly
but changes its answer under semantics-preserving perturbations.

## 4. Attribution Triage

The included attribution module joins contamination hits with correctness to
produce a retrieval-correctness influence proxy. This is a queue builder, not a
causal influence estimate. Use it to select examples for TracIn, TRAK, EK-FAC,
manual inspection, or checkpoint-based ablation.
