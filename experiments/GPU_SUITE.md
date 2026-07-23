# GPU suite — the experiments that need a T4

Model generation is the only thing standing between the CPU-side artifacts (contamination,
variants, attribution — already in `runs/`) and the headline results (A1 accuracy gap,
A2/A3 reliability, A5 scale, A6 RL). It needs a GPU. There are **two** turnkey paths:

| Script | Generates over | Measures | Use when |
|---|---|---|---|
| `run_reliability_suite.sh` | **variant clusters** | Reliability = Acc × **Consistency** (the real memorisation signal) | **the paper** — this is the recommended path |
| `run_gpu_suite.sh` | base benchmarks only | accuracy only (atlas cells report `ACCURACY-ONLY`) | breadth sweep across 6 benchmarks |

> **Why the reliability suite exists.** The base suite generates on *original* items only,
> so every cluster is a singleton and consistency is trivially 1.0 — the atlas can only
> report accuracy, and now honestly labels those cells **ACCURACY-ONLY (consistency
> untested)**. The study's core claim (reasoning vs memorising) is *unmeasurable* from that.
> `run_reliability_suite.sh` generates each model over the answer-preserving + numeric
> **variant clusters**, so consistency is genuinely probed and GENUINE/FRAGILE verdicts are
> earned. It also builds a higher-coverage contamination index from the full
> Instella-GSM8K-synthetic set, so A1 is not limited to the earlier 5,000-document scan.
> The full scan determines whether contaminated test items exist; it does not assume hits.

## Recommended: the 10-GPU-hour reliability run (one Colab cell, T4 first)

```python
!git clone https://github.com/GIND123/Instella-Reasoning
%cd Instella-Reasoning
!pip install -e ".[hf,retrieval,viz,stats]"
!pip install "bitsandbytes>=0.46.1"      # 4-bit loading; not in the pip extras

import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # HF token in Colab Secrets

# Contamination -> variants -> generate-over-variants -> score -> atlas -> gap -> emergence.
# Prints a GPU-hour estimate up front. Defaults target 10 T4-hours, but actual time depends
# on output length and hardware. Time the sanity pass and set SECONDS_PER_ITEM accordingly.
!bash experiments/run_reliability_suite.sh
```

**Budget knobs** (env vars): `BASE_ITEMS` (originals per benchmark, default 120),
`NUMERIC_K` (numeric variants per item, default 5), `BENCHMARKS` (default `gsm8k`;
`"gsm8k math"` for two skills), `SUITE_MODELS` (subset to split across Colab sessions,
e.g. `SUITE_MODELS="instruct"`), and `SECONDS_PER_ITEM` (estimate assumption, default 6).
The script warns if the estimated configuration exceeds ~9h. **Sanity pass first:**
`BASE_ITEMS=15 bash experiments/run_reliability_suite.sh` proves the whole chain end-to-end
and gives you a measured seconds/item value before you commit the budget.

Artifacts land in `experiments/runs/reliability-B<items>-K<numeric>/`: `base/`, `variants/`, `scores/`,
`atlas/` (consistency-probed verdicts), `accuracy_gap_stratified.json`, `report.md`,
`figures/`, `emergence_scale.json`, `emergence_rl.json`.

## Breadth alternative: the base accuracy suite

```python
# same install as above, then:
!bash experiments/run_gpu_suite.sh 200
```

- **generate → score** for 4 models × 6 answer-scored benchmarks (humaneval is
  contamination-only, excluded from scoring).
- **atlas** per model — accuracy cells only (`ACCURACY-ONLY`, since no variants).
- **emergence** (A5 scale: OLMo-1B vs Instella-3B; A6 RL: 3B-Instruct vs 3B-Math) —
  the emergence JSON now carries `consistency_probed: false` to flag these as
  accuracy-only, not measured reasoning emergence.

## Auto-save to Hugging Face + cross-machine resume (reliability suite)

`run_reliability_suite.sh` mirrors its run directory to a private HF **dataset** repo (default
`GOVINDFROM/Instella-Reasoning`) via [`hf_sync.py`](hf_sync.py), so no GPU hours are ever
re-spent — by you across Colab timeouts, or by a teammate on another machine.

**Lifecycle**

1. **Pull at start** — `hf_sync.py pull` snapshots the run dir back down. The remote path
   mirrors the local path (`experiments/runs/<name>`), so several runs coexist.
2. **Resume markers** — each finished `(model, benchmark)` writes `scores/<model>__<bench>.jsonl`;
   the loop `[[ -s "$sco" ]] && skip`, so restored work is skipped and generation continues
   at the first gap.
3. **Push as it runs** — after the CPU prep (contamination + variants), after every
   `(model, benchmark)`, after each atlas, and once at the end. Worst-case loss on a crash is
   the single benchmark in flight.

**GPU-hour budget** (single T4, conservative 6 s/item default): the script prints an estimate
on startup and warns past ~9 h. `gens = base × (1+4+numeric_k) × models × benchmarks`.
Defaults (GSM8K, 120 items, 4 models) ≈ 8 h; GSM8K+MATH ≈ 16 h — both inside a 20-h
budget before setup overhead. Time a `BASE_ITEMS=15` sanity pass first, then set
`SECONDS_PER_ITEM` to your measured rate.

**Env knobs:** `HF_RESULTS_REPO`, `HF_SYNC=0` (disable), `HF_INCLUDE_GENERATIONS=1` (also
upload raw generations — bulky; scores alone suffice to resume), `BASE_ITEMS`, `NUMERIC_K`,
`BENCHMARKS`, `SUITE_MODELS` (split models across sessions), `SECONDS_PER_ITEM`, `CORPUS_LIMIT`.
Auth: `HF_TOKEN` (from the Colab `hf` secret) must have **write** access.

**Manual sync** (bring a finished run onto a laptop for analysis, or seed the repo):

```bash
python experiments/hf_sync.py pull --repo GOVINDFROM/Instella-Reasoning \
  --path experiments/runs/reliability-B120-K5
python experiments/hf_sync.py push --repo GOVINDFROM/Instella-Reasoning \
  --path experiments/runs/reliability-B120-K5 --message "manual checkpoint"
```

## Then bring results back for analysis (base suite)

Or copy `experiments/runs/<date>_gpu-suite/scores/` and `atlas/` back into the repo, and the
CPU-side analysis (accuracy-gap, difficulty-stratified gap, report, figures) can run on a
laptop:

```bash
instella-reasoning accuracy-gap --scores <merged>.jsonl --contamination <contam>.jsonl \
  --stratified-output gap_stratified.json --output gap.json
instella-reasoning report --scores <merged>.jsonl --contamination <contam>.jsonl --output report.md
```

## Scale-up knobs

- `bash scripts/download_data.sh 0 0` — full benchmarks + full corpus (needs disk + the AMD
  ResearchRAIL terms in `docs/DATA_DOWNLOAD.md`).
- `bash experiments/run_gpu_suite.sh 0` — no per-benchmark cap.
- Add corpus sources (OpenMathInstruct-2, dm_math, DCLM/FineWeb-Edu) to the contamination
  scan for benchmarks beyond GSM8K.
