# GPU suite — the experiments that need a T4

Model generation is the only thing standing between the CPU-side artifacts (contamination,
variants, attribution — already in `runs/`) and the headline results (A1 accuracy gap,
A2/A3 reliability, A5 scale, A6 RL). It needs a GPU. This is the turnkey path.

## One Colab cell (Runtime → T4 GPU first)

```python
!git clone https://github.com/GIND123/Instella-Reasoning
%cd Instella-Reasoning
!pip install -e ".[hf,retrieval,viz,stats]"

import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # HF token in Colab Secrets

# real benchmark + corpus data (200/5000 smoke; raise for the full study)
!bash scripts/download_data.sh 200 5000
# generate + score every (model x benchmark), then atlas + emergence
!bash experiments/run_gpu_suite.sh 200
```

Artifacts land in `experiments/runs/<date>_gpu-suite/`:
`generations/`, `scores/`, `atlas/`, `emergence_scale.json`, `emergence_rl.json`.

## What it runs

- **generate → score** for 4 models × 6 answer-scored benchmarks (humaneval is
  contamination-only, excluded from scoring).
- **atlas** per model (A1/A2): reliability cells from that model's scores + the CPU-side
  contamination hits.
- **emergence** (A5 scale: OLMo-1B vs Instella-3B; A6 RL: 3B-Instruct vs 3B-Math).

## Then bring results back for analysis

Copy `experiments/runs/<date>_gpu-suite/scores/` and `atlas/` back into the repo, and the
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
