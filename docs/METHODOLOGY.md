# Methodology

This repository implements the execution layer for a four-part study: **filter**
(contamination) → **diagnose** (consistency/reliability) → **attribute** (training data)
→ **prescribe**. The controls below exist to make each stage survive peer review; see
[`REVIEW.md`](REVIEW.md) for the concern each one answers.

## 1. Contamination Search

Each benchmark item is searched against a manifest of known Instella training documents
(embedding + directional 13-gram overlap → C / PC / N labels). Two things are stated
explicitly rather than assumed:

- **Thresholds are calibrated, not asserted.** `validation.calibrate_cosine_threshold`
  sweeps the cosine cutoff against a hand-labeled set and reports precision/recall/F1 so
  the C/PC/N thresholds are chosen. (`instella-reasoning calibrate-contamination`.)
- **"Clean" means "not detected", a lower bound.** Only a sample of the trillion-token web
  corpora is indexed, so a "clean" label is *not-detected-in-sample*, not
  provably-uncontaminated. Reports label the axis accordingly and contamination is read as
  "at least this much".

## 2. Robustness Evaluation (consistency clusters)

Each item is expanded into a cluster sharing a `parent_id`, with a `variant_type` tag.
Two kinds of variant, treated differently on purpose:

- **Answer-preserving** (`entity_substitution`, `premise_reordering`, `irrelevant_context`,
  `rephrasing`) — the gold answer is unchanged; these drive the *consistency* term.
- **Answer-changing** (`gsm_symbolic`) — GSM-Symbolic-lite numeric resampling with the
  answer *recomputed* from the item's own `<<a op b=c>>` calc chain and conservatively
  round-trip-validated (`gsm_symbolic.build_template`). Per GSM-Symbolic (arXiv:2410.05229),
  numeric perturbation is the decisive probe, so a rigorous run sets `numeric_variants > 0`.

`validate_variants` audits a suite for answer-preservation rate and text well-formedness.

## 3. Reliability Metric

For each parent cluster:

```text
accuracy                    = correct_variants / total_variants
answer_consistency          = modal_answer_share over ANSWER-PRESERVING variants only
reliability                 = accuracy * answer_consistency
accuracy_under_perturbation = correct / total over ANSWER-CHANGING (numeric) variants
```

Excluding answer-changing variants from the consistency term fixes an incoherence: a good
reasoner *should* answer numeric variants differently, so counting that as
"inconsistency" would penalize correct reasoning. `accuracy_under_perturbation` is reported
alongside as the GSM-Symbolic-standard drop metric.

**Construct validity.** `validation.validate_reliability_metric` checks the metric actually
ranks *genuine* clusters above *fragile* ones on a labeled set, giving a number to cite
rather than asserting the metric measures what it claims.

## 4. Contamination Gap (the headline), with confound control

`accuracy_gap` reports contaminated-vs-clean accuracy with a two-proportion z-test and
**Benjamini-Hochberg FDR** across all scopes. Because contaminated items may simply be
*easier*, the naive gap is confounded (Ivanova et al.'s critique of GSM-Symbolic). So
`compute_stratified_accuracy_gap` also reports a **difficulty-adjusted** gap:

- `difficulty.estimate_difficulty` scores each item model-independently (GSM8K reasoning
  steps / MATH level / length-and-quantity fallback);
- items are placed in equal-frequency strata, the gap is pooled *within* strata
  (Mantel-Haenszel), and a **cluster-robust bootstrap** (resampling parent clusters, not
  individual variants) gives the CI. If the adjusted gap collapses toward 0, the naive gap
  was difficulty, not contamination.

## 5. Attribution Triage

The embedding attribution joins contamination hits with correctness to produce a
retrieval-correctness proxy. This is a **queue builder and associational signal**, not a
causal influence estimate — influence functions and TracIn are unreliable at LLM scale
(arXiv:2409.19998), so gradient methods (TracIn-CP, Concept Influence) are validation-only.

## 6. Emergence / Scale

Comparisons are labeled controlled vs confounded (`emergence.comparison_confound`). The
AMD-OLMo-1B ↔ Instella-3B axis is **confounded** (shared codebase but different data and
token counts); prefer Instella's own checkpoints (Stage-1/Stage-2, SFT/DPO) for causal
claims. `schaeffer_artifact_test` guards against metric-induced "emergence".
