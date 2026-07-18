# Worklog — CPU experiment suite, loader fixes, and experiment infrastructure

Branch: `agent/cpu-experiment-suite`. Base: `main` @ `7e5ff19`.
Sessions: 2026-07-17 (execution + fixes), 2026-07-18 (packaging + docs).
Everything below is verified by the test suite (**123 passed, 2 skipped**) and by the
committed reference artifacts in `experiments/runs/2026-07-17_cpu-batch/`.

---

## 1. Why this branch exists

The experiment plan (`docs/IMPLEMENTATION_PLAN.md` §Tests) defines 7 research analyses
(A1–A7) but the repo had (a) no tracker of what has actually been *run*, (b) four of the
nine planned benchmarks missing loaders, and (c) two data-loading bugs that silently
produced empty/blank inputs — either of which would have poisoned the study downstream.
This branch fixes the code, executes every CPU-runnable experiment stage on real data at
smoke scale, and packages both the CPU and GPU sides as turnkey scripts with a
reference-run comparison harness.

## 2. Code changes (`src/`)

### 2.1 Corpus loader ignored chat-format datasets (bug, fixed)
- **Symptom:** `load-corpus` on `amd/Instella-GSM8K-synthetic` wrote **0 documents** —
  the single most important contamination ground-truth source was unloadable.
- **Cause:** `loaders.py::load_corpus_dataset_to_jsonl` only understood a flat string
  `text` field (plus problem/question/solution/answer fallbacks). This dataset stores
  content as a `messages` list of `{role, content}` chat turns.
- **Fix:** new `_coerce_corpus_text()` flattens strings, lists, and chat-message lists;
  the fallback join now also covers `messages`.
- **Proof:** corpus now loads 5,000/5,000 docs; doc #0 is verbatim GSM8K-train "Natalia
  sold clips…". Tests: `test_loaders.py::test_coerce_*` (5 tests).

### 2.2 LogiQA2 loader produced blank prompts (bug, fixed)
- **Symptom:** every loaded LogiQA2 item had `prompt: ""`, `answer: "None"`;
  `make-variants` then crashed with "has no prompt/question/input field". Any GPU run on
  the unfixed loader would have scored garbage.
- **Cause:** `datatune/LogiQA2.0` packs each example as a **JSON string inside a single
  `text` column** (inner keys: `text`/`question`/`options`/`answer`); `_load_logiqa`
  expected flat columns and read empty strings.
- **Fix:** `_load_logiqa` now detects the nested form, `json.loads` it, and maps the inner
  `text` as context; the flat-column path is preserved for other mirrors.
- **Proof:** 200 real items with prompts + letter answers; variants build and validate at
  rate 1.000. Tests: `test_load_logiqa_nested_json_text`, `test_load_logiqa_flat_columns`.

### 2.3 ReClor + HumanEval loaders (missing, added)
- The plan's benchmark list includes both; `prompting.py`/`answer_equivalence.py` already
  special-cased `"reclor"` — only the loader was missing.
- `_load_reclor`: multiple-choice with letter answers; hidden test-split labels
  (`label < 0`) map to `answer=None`.
- `_load_humaneval`: **contamination-search only** — code correctness needs execution
  against unit tests, so `answer=None`, `metadata.scoring="execution"`; excluded from
  generate/score in the GPU suite.
- Registry: 5 → **7** benchmarks. Tests: 4 new in `test_loaders.py`.

### 2.4 Benchmarks that remain impossible (not code gaps)
- **CLUTRR:** upstream HF dataset is script-based; `datasets` 5.0 removed script support
  and no maintained Hub mirror exists (probed `CLUTRR/v1` + 4 alternates).
- **TTT-Bench:** not on the HF Hub (probed 2 IDs).
These stay ⛔ in the plan with this rationale.

## 3. Tests

`tests/test_loaders.py` — new file, **12 tests** (corpus coercion ×5, logiqa2 ×2,
reclor ×3, humaneval ×1, registry ×1). Fills one of the six module-coverage gaps.
Suite: 111 → **123 passed** (2 pre-existing skips), zero regressions.

## 4. Experiments executed (reference run: `runs/2026-07-17_cpu-batch/`)

Smoke scale: 200 items/benchmark, 5,000 corpus docs, seed 6198, real MiniLM embeddings,
exact (bruteforce) nearest-neighbour search. Full numbers + interpretation in the run's
`LOG.md`; headline:

| Stage | Serves | Result |
|---|---|---|
| Variants + M2 validation, 6 benchmarks | A2/A3 | answer-preservation **rate 1.000** everywhere; 130 numeric (answer-changing) variants for gsm8k; degenerate-text flags: gsm8k 163, math 90, logiqa2 17 (inspect before the reliability run) |
| Contamination scan gsm8k, math | A1 | gsm8k: 0 C / 2 PC / 198 not-detected (top-1 cosine max 0.78, 13-gram 0.0) — a legitimate lower bound (test-split items vs train-derived corpus); math: 0 hits (different domain) |
| Embedding attribution gsm8k, math | A4 | 200 profiles each (gini, near-dup share, top1_cosine, source shares); mean top1 cosine 0.606 (gsm8k) / 0.372 (math) |

**Status vs the 7 planned analyses:** none fully complete (all headline results need model
generation → GPU). Approximate completion: A4 ~80%, A1 ~50%, A2/A3 ~40%, A7 ~20%,
A5/A6 0%. The `README.md` status matrix tracks per-cell state.

## 5. Infrastructure added (`experiments/`)

| File | Purpose |
|---|---|
| `README.md` | Canonical plan: A1–A7 × scope matrix, status, how-to-run (incl. Colab CPU cell) |
| `run_cpu_suite.sh` | Turnkey CPU suite (data → variants → contamination → attribution), mirrors the reference run |
| `compare_runs.py` | Diffs two run dirs (validation counts/rates exact; cosines at 1e-3); exit 0 = agree. Self-test: 10/10 agreements on the reference run |
| `run_gpu_suite.sh` + `GPU_SUITE.md` | Turnkey GPU side: generate → score → atlas per (4 models × 6 benchmarks), then A5 scale + A6 RL emergence; single Colab cell documented |
| `runs/2026-07-17_cpu-batch/` + `LOG.md` | The committed reference artifacts every re-run is compared against |

## 6. Environment findings (read before running anything at scale)

1. **faiss segfaults locally** (exit 139) on Python 3.14 / numpy 2.5 — and the scan then
   fails *silently*, writing an empty output that looks like "0 hits". All scripts pin
   `--index-backend bruteforce` (exact search, identical results at ≤10k docs). Before a
   full-corpus scan, rebuild the venv on Python 3.11–3.12 or run on Colab.
2. **The hashing-embedder fallback is unusable for contamination** — it is
   semantic-blind and ranks unrelated documents nearest (verified: it missed a verbatim
   near-duplicate). Real `sentence-transformers` MiniLM is mandatory for any real scan;
   the fallback exists only so the pipeline runs dependency-free in CI/smoke.

## 7. What remains (the actual study)

- **GPU (headline results):** generation for 4 models × 6 benchmarks → accuracy gap (A1),
  reliability (A2/A3), atlas verdicts, A5 scale + A6 RL emergence. Turnkey:
  `GPU_SUITE.md`. Smoke pass ≈ hours on a T4; the full plan budgets ~25–35 GPU-days.
- **Hand labels (CPU, small):** contamination-threshold calibration and reliability
  construct-validity both need small human-labeled sets (REVIEW.md "human validation").
- **Scale-up:** current artifacts are smoke-scale; the study wants full benchmarks
  (GSM8K 1,319) and the multi-source corpus (~13–15M passages, faiss required).
- **M7** sampling-consistency baseline (temp>0) — protocol only, not yet automated.

## 8. How to verify this branch

```bash
pip install -e ".[dev,hf,retrieval]"
pytest                                       # expect: 123 passed, 2 skipped
bash experiments/run_cpu_suite.sh 200 5000   # re-run the CPU suite (~15 min CPU)
python experiments/compare_runs.py \
  experiments/runs/2026-07-17_cpu-batch experiments/runs/<today>_cpu-suite
# expect: "Result: runs agree — everything looks good."
```
