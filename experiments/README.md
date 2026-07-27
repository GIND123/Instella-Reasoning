# Experiments — canonical plan, scope, and status

This directory is the single home for the study's experiments: the **planned suite**
(from `docs/IMPLEMENTATION_PLAN.md` §Tests, the ambitious full scope), the **executed
runs** (labeled, under `runs/`), and the turnkey launchers. Supersedes the earlier
`docs/EXPERIMENTS.md` sketch.

> ## ⚠️ Start here: the current suite is [`FULLSCALE.md`](FULLSCALE.md)
>
> The `reliability-B*` runs under `runs/` are **superseded pilots**, kept for reference.
> [`docs/AUDIT_2026-07-26.md`](../docs/AUDIT_2026-07-26.md) shows that four of their five
> headline findings are artifacts — a truncated long-CoT checkpoint, an empty contamination
> treatment group, 15% wrong numeric labels, and a 2.61× magnitude confound. **Do not cite
> those numbers.**
>
> The replacement is the seen/unseen memorisation design in
> [`run_fullscale_suite.sh`](run_fullscale_suite.sh): GSM8K train (verifiably in Instella's
> stage-2 data) vs GSM8K test (verifiably not), across the full Instella-3B checkpoint
> trajectory, with every defect above closed and gated. Rationale:
> [`docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md).
>
> ```bash
> python scripts/preflight_fullscale.py --smoke   # must print SAFE TO LAUNCH
> bash experiments/run_fullscale_suite.sh
> ```
>
> `GPU_SUITE.md` and `run_reliability_suite.sh` remain for reproducing the pilots.

Status legend: ✅ done · 🟡 runnable now (CPU) · 🔴 needs GPU · ⛔ blocked (data/label)

---

## The planned suite (7 research analyses × full scope)

Verbatim from the implementation plan. Each analysis runs across the scope matrix below.

| # | Analysis | Method | Needs |
|---|----------|--------|-------|
| A1 | Contamination rate + accuracy gap (contaminated vs clean) | two-proportion z-test / χ²; difficulty as covariate | contamination scan + generations |
| A2 | Reliability decomposition by contamination | Mann-Whitney + bootstrap CI + effect size | variants + generations |
| A3 | GSM8K→GSM-Symbolic drop vs same-size open models | numeric-variant accuracy-under-perturbation; BH-FDR | numeric variants + generations |
| A4 | Attribution concentration | near-duplicate share of top-influence docs (≥30% ⇒ memorization) | embedding attribution |
| A5 | Emergence: smooth vs discontinuous scaling | transition classes + Schaeffer artifact test | score files across models |
| A6 | RL effect: consistency vs accuracy gain | 3B vs 3B-Math; contamination-masking check | generations for both variants |
| A7 | Metric robustness | threshold / embedding-model / 4-bit-vs-FP16 sensitivity | repeated runs |

### Scope matrix (the ambitious target)

- **Models (5):** AMD-OLMo-1B · Instella-3B · Instella-3B-Instruct · Instella-3B-Math · (Long-Instruct optional)
- **Benchmarks (9 planned):** gsm8k · math · logiqa2 · arc_challenge · bbh · reclor · **clutrr⛔** · **humaneval (contam-only)** · **ttt_bench⛔**
- **Corpus sources (6):** Instella-GSM8K-synthetic (full) · OpenMathInstruct-2 (sample) · dm_math · python-edu · dolmino-mix · DCLM/FineWeb-Edu (sample)

---

## Status matrix

### Benchmarks (loaders)
| Benchmark | Loader | Data pulled | Note |
|-----------|--------|-------------|------|
| gsm8k, math, arc_challenge, bbh | ✅ pre-existing | ✅ 200-item slices | |
| logiqa2 | ✅ **fixed** (nested-JSON schema) | ✅ | wrote blank prompts before the fix |
| reclor | ✅ **added** | ✅ 200 | scoring/prompting layers already knew "reclor" |
| humaneval | ✅ **added** (contam-only) | ✅ 164 | code benchmark; no exact-match answer |
| clutrr | ⛔ | — | script-based dataset, unsupported by `datasets` 5.0; no Hub mirror |
| ttt_bench | ⛔ | — | not on the Hub |

### Analyses
| Analysis | CPU-runnable part | Status | GPU part | Status |
|----------|-------------------|--------|----------|--------|
| A1 | contamination scan (C/PC/N) | ✅ gsm8k, math | accuracy gap + z-test | 🔴 |
| A2 | variant build + M2 validation | ✅ 6 benchmarks | reliability decomposition | 🔴 |
| A3 | numeric-variant build | ✅ gsm8k, math | accuracy-under-perturbation | 🔴 |
| A4 | embedding attribution (Gini/concentration) | ✅ gsm8k, math | TracIn cross-check | 🔴 |
| A5 | — | — | scale emergence | 🔴 |
| A6 | — | — | 3B vs 3B-Math | 🔴 |
| A7 | threshold sensitivity | 🟡 | 4-bit vs FP16 | 🔴 |
| calibration | cosine/13-gram threshold (A1 support) | ⛔ needs labels | | |
| reliability construct-validity | genuine>fragile check | ⛔ needs labels | | |

**Bottom line:** all CPU-side work that has data is done or runnable; every *headline*
result (A1 gap, A2/A3 reliability, A5/A6 scale/RL) is gated on **model generation**, which
needs a GPU — see `GPU_SUITE.md`. Two calibrations are gated on a small hand-labeled set.

---

## How to run

- **CPU experiments:** `bash experiments/run_cpu_suite.sh` reproduces the whole CPU side
  (data → variants → validation → contamination → attribution) into a dated run dir.
  `runs/2026-07-17_cpu-batch/` is the committed **reference run** (seed 6198, 200/5000
  limits). Verify any re-run against it with:

  ```bash
  python experiments/compare_runs.py experiments/runs/2026-07-17_cpu-batch <new_run_dir>
  ```

- **GPU experiments:** follow `GPU_SUITE.md` — one Colab block generates + scores + atlases
  every (model × benchmark), then you copy the score files back here for A5/A6.

### Re-run the CPU suite on Colab (no GPU needed)

Paste into one Colab cell (any runtime — CPU is fine):

```python
!git clone -b agent/cpu-experiment-suite https://github.com/GIND123/Instella-Reasoning
%cd Instella-Reasoning
!pip install -e ".[hf,retrieval,stats]"

!bash experiments/run_cpu_suite.sh 200 5000
import glob
!python experiments/compare_runs.py experiments/runs/2026-07-17_cpu-batch {glob.glob("experiments/runs/*_cpu-suite")[-1]}
```

The final line prints `Result: runs agree — everything looks good.` if the Colab re-run
matches the committed reference values (variant counts and validation rates must be
identical — they are seed-deterministic; attribution cosines are compared at 1e-3
tolerance to absorb hardware float noise).

## Environment caveat

`faiss` **segfaults (exit 139)** under this local stack (Python 3.14 / numpy 2.5) and fails
silently (empty output). Locally, always pass `--index-backend bruteforce`. On Colab
(Python 3.11) faiss works; the GPU suite uses it. Rebuild the local venv on Python 3.11–3.12
before any faiss-dependent scale run.
