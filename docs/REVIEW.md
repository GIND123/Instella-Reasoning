# Methodology Review & Response

A strong-reviewer pass over *Reasoning or Remembering?* against the literature it builds
on, and how the codebase now answers each concern. Load-bearing papers were checked
against the actual method, not summaries:

- GSM-Symbolic — Mirzadeh et al., ICLR 2025, [arXiv:2410.05229](https://arxiv.org/abs/2410.05229)
- Do Influence Functions Work on LLMs? — Li et al., [arXiv:2409.19998](https://arxiv.org/abs/2409.19998)
- Ivanova et al. statistical critique of GSM-Symbolic (as discussed in 2410.05229 App. A.3)

## Verdict

Framing and engineering are strong; the deficit is empirical execution and statistical
rigor, not vision. The controls below convert the reviewer's "path to accept" into code.

## Concern → response map

| # | Concern | Response in code |
|---|---|---|
| **M1** | Default perturbations are the *surface* changes GSM-Symbolic showed models are robust to; the decisive *numeric* change was excluded. | [`gsm_symbolic.py`](../src/instella_reasoning/gsm_symbolic.py) auto-derives validated numeric variants; `make-variants --numeric-k`, `numeric_variants` in the pipeline/`rigorous.yaml`. |
| **M2** | Perturbations not validated as answer-preserving (GSM-Symbolic's validity comes from that). | `perturbations.validate_variants` (answer-preservation rate + well-formedness); numeric variants are round-trip-validated at construction (`build_template`). CLI `validate-variants`. |
| **M3** | Headline gap confounded by difficulty; no FDR; variants not independent. | `accuracy_gap.compute_stratified_accuracy_gap` — Mantel-Haenszel difficulty pooling (`difficulty.py`) + **cluster** bootstrap CI; BH-FDR q-values on every scope. `--stratified-output`. |
| **M4** | "Clean" is unfalsifiable at sample scale; thresholds asserted. | Reporting reframes clean as **not-detected (lower bound)**; `validation.calibrate_cosine_threshold` gives PR/F1. CLI `calibrate-contamination`. |
| **M5** | Reliability metric conflates "stable" with "reasoning"; incoherent with numeric variants; unvalidated. | `metrics.cluster_consistency` excludes answer-changing variants; adds `accuracy_under_perturbation` (GSM-Symbolic standard); `validation.validate_reliability_metric` provides construct validity (genuine > fragile). |
| **M6** | OLMo-1B↔3B "scale control" is confounded (shared code, different data/tokens). | `emergence.comparison_confound` labels axes; docstring steers causal claims to Instella's own checkpoints; `schaeffer_artifact_test` guards metric-induced emergence. |
| **M7** | Single greedy sample can't separate perturbation-consistency from decoding noise. | *Protocol* documented; add temp>0 resampling for a sampling-consistency baseline. Not yet automated — see Open items. |

## What lands a workshop accept (reviewer's own bar), now supported

1. **One real result, honestly analyzed.** `rigorous.yaml` runs numeric-variant clusters +
   difficulty-adjusted, FDR-corrected, cluster-robust gap on the real model in bf16. Even a
   *null* result (gap ≈ 0 after difficulty control) is publishable.
2. **Validated perturbations** (`validate-variants`) and **calibrated thresholds**
   (`calibrate-contamination`).
3. **Contamination as a lower bound** (reporting + docs).

## Recently closed in code

- **MATH answer equivalence** — [`answer_equivalence.py`](../src/instella_reasoning/answer_equivalence.py)
  compares string → numeric → SymPy symbolic, so `\frac{1}{2}` / `0.5` / `1/2` are one answer.
  Exact string match was silently under-counting correct MATH answers and biasing the gap.
  Wired into `score_generations`; conservative (a parse failure never invents a match).
- **Atlas uncertainty + verdict gate** — every `AtlasCell` now carries a cluster-bootstrap 95%
  CI on reliability, and cells with fewer than `MIN_CLUSTERS_FOR_VERDICT` (3) clusters are
  labelled `insufficient_data` instead of GENUINE/FRAGILE/GAP. Kills the `n=1` verdict.
- **Run provenance** — [`provenance.py`](../src/instella_reasoning/provenance.py) writes
  `manifest.json` (git commit + dirty flag, package versions, resolved config) beside every run;
  the report shows a Reproducibility block. Answers "which checkpoint/versions produced this?".
- **Degeneracy hard-stop** — `generate` auto-runs the quality gate and `--fail-degenerate`
  exits non-zero, so a broken run (wrong chat template / transformers 5.x) cannot reach the Atlas.

## Open items (protocol, not yet code)

- **M7 sampling-consistency baseline** (temp>0, k samples) alongside perturbation-consistency.
- **Human validation** of a numeric-variant sample (the auto-templater is high-precision but
  not human-verified) and of contamination labels for the calibration set.
- **Cross-check accuracy** against `lm-eval-harness` to bound residual extraction error (the
  math-equivalence checker narrows this but does not replace an external cross-check).
- **Indirect-detection baseline** (Min-K%++) to *quantify* the ground-truth-vs-indirect claim.
- **Math-specialized embedder** for contamination (MiniLM makes distinct math problems look
  near-duplicate → false positives); use the calibration curve to pick the operating point.
- **Multi-checkpoint TracIn** with gradient normalization + random projection (the current
  gradient attributor is single-checkpoint and unprojected — demote to a validation tier).
