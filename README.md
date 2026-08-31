# Instella Reasoning

Measurement infrastructure for three questions that require knowing, rather than guessing,
whether a model was trained on the item it is being evaluated on.

Contamination work normally has to estimate membership from a proxy — embedding similarity,
perplexity, a rephrasing attack. Instella is released with its complete pretraining mixture,
and its stage-two component is generated from the **GSM8K train** split and not from **test**.
Membership of a benchmark item in the model's corpus is therefore a measured property, fixed by
exact 13-gram containment against a published corpus. Every study in this repository is built on
that one fact.

Two workshop manuscripts and one measurement scaffold live here. `FINDINGS.md` is the
authoritative ledger of what currently holds; where a claim was withdrawn it is kept with the
reason attached, because the same idea gets re-proposed otherwise.

---

## Studies

| Study | Question | Headline | Source |
|---|---|---|---|
| **Judge reliability** | Does an LLM judge grade problems from its own training corpus less reliably? | Balanced-accuracy gap of **−0.243** [−0.282, −0.202] with the reference answer withheld, **−0.074** [−0.116, −0.033] with it supplied | `paper/judge/` |
| **Premise verification across the checkpoint axis** | When in training does a model learn that a problem does not determine an answer, and what takes it away? | Discrimination against an answerable control rises **+1.84 → +10.10 → +38.08 → +52.63 pp** across the pipeline, then decays **−5.11 pp** per extra reasoning step [−6.36, −3.82] | `paper/mathai/`, `experiments/runs/{ckpt-axis-v1, abstain-v1}/` |
| **Reasoning Reliability Atlas** | Scaffold: contamination scan, perturbation, generation, scoring, attribution, statistics | `src/instella_reasoning/`, `configs/pipeline/` |

Inference throughout is a cluster bootstrap over **parent problems**, 4000 draws, seed 6198. The
measured intraclass correlation is ≈0.48, so resampling rows instead of parents would contract
intervals by a design effect near 2.4.

---

## Reproducing the reported numbers

### 1. Audit the manuscript — CPU, no downloads, under a minute

Every number in the judge manuscript is recomputed from the released verdict files by a script
that **deliberately does not import the analysis module**. It reimplements the rates, Cohen's
kappa and the cluster bootstrap from their stated definitions, at the same seed and draw count,
so agreement is a check rather than a tautology.

```bash
python paper/judge/audit_reported_numbers.py
```

```
TOTAL 60 checks | PASS 60 | FAIL 0
```

The verdict files it reads are tracked in `paper/judge/data/` — 12 files, 2017 graded rows each,
one per (judge × target checkpoint × condition). No GPU, no network, no credentials.

### 2. Test suite

```bash
make install-dev && make test        # 171 passed, 2 skipped
```

The suite runs the full pipeline end-to-end on bundled examples with a hashing embedder and a
brute-force index, so it exercises real code paths without a model download.

### 3. CPU smoke pipeline

```bash
make smoke                           # scripts/run_smoke_pipeline.sh
```

### 4. Full study — GPU

```bash
python scripts/preflight_fullscale.py --smoke     # must print SAFE TO LAUNCH
bash experiments/run_fullscale_suite.sh
```

The preflight checks GPU presence, dataset reachability, disk headroom and credentials, and runs
a four-item generation smoke test. It is not optional: a CPU-only runtime produces degenerate
completions that look like a result.

---

## Installation

Python ≥ 3.10. Optional extras are separated so that the analysis layer stays installable on a
laptop with no CUDA.

```bash
pip install -e .                     # core: pipeline, containment, scoring
pip install -e ".[dev]"              # + pytest, ruff
pip install -e ".[hf]"               # + transformers, datasets, accelerate, sympy
pip install -e ".[retrieval]"        # + faiss-cpu, sentence-transformers
pip install -e ".[viz,stats]"        # + matplotlib, scipy
pip install -e ".[all]"              # everything
```

**The `transformers>=4.44,<5` upper bound is load-bearing.** Instella ships custom remote
modeling code (`modeling_instella.py`, `trust_remote_code=True`) written against the 4.4x
attention-mask / `cache_position` API. transformers 5.x removed that API, which breaks the
forward pass *silently* — generation succeeds and returns degenerate text. Do not relax the pin
without re-verifying output coherence.

`sympy` is optional at runtime but not in practice: `answer_equivalence` degrades to
string/numeric match without it, which badly under-counts MATH accuracy.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/instella_reasoning/` | Library and CLI. Containment, perturbations, generation, scoring, attribution, statistics, figures. |
| `experiments/` | Turnkey launchers and per-run analysis scripts. `runs/` holds labelled run directories. |
| `paper/judge/` | JUDGe manuscript, verdict files, figure scripts, independent audit. |
| `paper/mathai/` | MATH-AI manuscript and figures. |
| `docs/` | Methodology, design documents, corpus corrections, figure provenance, compute budgets. |
| `configs/pipeline/` | End-to-end YAML configs: `full`, `rigorous`, `contamination`, `evaluation`. |
| `configs/training/` | Upstream Instella training configs (single- and multi-node). |
| `scripts/` | Data download, preflight, smoke and full pipeline drivers. |
| `tests/` | 173 tests, CPU-only, no network. |
| `FINDINGS.md` | Claims of record, including withdrawn claims and the reason for withdrawal. |
| `STATE.md` | Current handoff state: what is complete, what is outstanding. |

Command surface:

```bash
instella-reasoning --help
```

Subcommands cover `scan-contamination`, `make-variants`, `generate`, `score-generations`,
`accuracy-gap`, `memorization`, `attribute`, `atlas`, `plots`, `figures`,
`calibrate-contamination`, `validate-variants`, `check-termination` and `run-all` (whole
pipeline from a YAML config).

---

## How membership is measured

Item text is lowercased, reduced to alphanumeric tokens, and decomposed into every window of 13
consecutive tokens. Containment against a corpus is the fraction of an item's windows found in
**a single corpus document**, maximised over documents — not the fraction found anywhere in the
union of the corpus. The distinction is not cosmetic: the union definition reports 15.12% of
GSM8K train above 0.80 where the per-document definition reports 7.56%, and only the
per-document figure means "this problem appeared in a training document."

The scan is one streaming pass with an inverted index from window to source item. Cost is linear
in corpus size and independent of how many item sets are indexed together.

Two calibration facts make the detector auditable rather than assumed:

- Because the corpus derives from GSM8K **train**, every GSM8K **test** item is a known negative.
  The permissive any-match rule flags 3 of 1,000 known negatives (0.3%); the adopted threshold
  flags none.
- Exposure is a property of *(item, corpus, checkpoint)*, not of the item. Instella-3B stage two
  consumed `amd/Instella-GSM8K-synthetic[train_119K]` (119,014 rows). Instella-MoE-16B-A3B
  consumed the full `train` split (1,367,882 rows) at long-context extension phase 2. A
  containment number is meaningless without naming which.

**Identifier namespacing is load-bearing.** GSM8K train and test share an index scheme; loading
both without namespacing collides items and yields containment above 1.0, which is impossible for
a fraction. Anyone reimplementing this will hit the same collision.

---

## Reproducibility controls

These are deliberate, not incidental:

- **Pinned transformers version.** A version *range* places different checkpoints on different
  minor versions, and that difference enters the estimate as though it were a model difference.
- **Fixed batch size across all generation.** Greedy decoding in reduced precision is sensitive
  to padding; changing batch size alters tie-breaking.
- **Per-(item, perturbation) seeding.** Variant text is stable regardless of call order, so
  generations stay valid when the item set changes.
- **Flush-and-resume.** Every generation batch is written to disk; resume is keyed on item
  identifier, so an interrupted run continues rather than restarting.
- **Raw generations retained,** not only scores. A scoring bug found after the fact once
  destroyed an entire run for which only scores had been kept.
- **Judge temperature 0.** Same-verdict rates fall from above 95% to roughly 70% between T=0 and
  T=1, so a sampled judge would confound the contrast with its own noise.

---

## Results of record

### Judge reliability

2017 candidate solutions (988 seen / 1029 unseen, descending from 500 parent problems), each
graded twice by five judges across three families — once with the gold answer in the prompt, once
without. Balanced accuracy, seen minus unseen:

| Judge | Reference given | Reference withheld | Ratio |
|---|---|---|--:|
| Instella 3B Instruct | −0.054* | −0.081* | 1.5 |
| Llama 3.1 8B Instruct | −0.074* | **−0.243*** | 3.3 |
| Qwen2.5 7B Instruct | −0.036* | −0.200* | 5.6 |
| Qwen2.5 14B Instruct | −0.023 | −0.184* | 8.0 |
| Qwen2.5 32B Instruct | −0.062* | −0.191* | 3.1 |

<sub>* interval excludes zero. Argue from the within-judge ratio, not from a trend across the
ladder: with five judges the control gaps are not monotone in capability.</sub>

The effect is **specificity, not sensitivity**. For Llama 3.1 8B with the reference withheld,
sensitivity moves +0.011 [−0.005, +0.035] while specificity falls −0.496 [−0.572, −0.416]: it
rejects 52.3% of wrong solutions to unseen problems and 2.7% of wrong solutions to seen ones.
Familiarity buys credulity, not confusion — which falsifies the intuitive account in which the
judge compares against a remembered answer, since that predicts a sensitivity loss.

Cohen's kappa reports −0.485 on the same verdicts where balanced accuracy reports −0.243. The
inflation is traceable to a base-rate difference (0.772 seen against 0.526 unseen), not to any
change in agreement. Chance-corrected statistics should not be used to size a membership effect
where arms differ in base rate.

### Premise deletion across the checkpoint axis

3,179 items each have one premise the reference solution consumes removed, so declining is the
only correct response. Under a prompt naming `#### unanswerable`, declining is read against an
answerable control on the untouched parents, where declining is an error. Discrimination runs
**+1.84 → +10.10 → +38.08 → +52.63 pp** across `Stage1 → Instella-3B → SFT → Instruct`, so the
capability is built almost entirely by post-training, while wrong refusals stay under 3.5%. It
then **decays with reasoning depth**: −5.22 pp per extra calculator step against a flat control,
a difference of **−5.11 pp [−6.36, −3.82]**, steepening as the checkpoint improves. Reproduced on
an independent MATH probe and on three instruct checkpoints from other families.

The exposure question the study started from is answered in the negative and kept: deletion
recall rises across the stage-two data intervention (+1.70 pp, McNemar p = 4.6e-4) but rises at
least as much on items provably absent from the corpus (+1.95 pp), difference-in-differences
**−0.26 pp, 95% CI [−1.66, +1.14]**.

The null is **bounded, not bare**. Controlled injection at 0/1/4/16/64× verbatim repetition,
under a fixed 8,388,608-token budget, shows the probe detects recall only at 64× (verbatim
reproduction 0.946, p = 0.035). At 16× the model reproduces 49% of an injected document verbatim
and the probe detects nothing. Minimum detectable effect is +1.89 pp at 80% power against an
observed −0.26 pp.

---

## Data and artifacts

Tracked in-repo: judge verdict files, containment scan results, analysis JSON, figure sources,
and the deletion item sets that define what the study measured.

Not tracked, and re-derivable: raw generation text (bulky), the ~1.5 GB corpus snapshot
(re-downloadable from the Hub), and the 48 MB injection-mixture filler
(`experiments/build_injection_mixture.py`). `.gitignore` names each exclusion with its reason.

Bulk artifacts are mirrored to a Hugging Face dataset repository configured through
`HF_RESULTS_REPO`; the identifier is omitted here while the judge manuscript is under
double-blind review.

Credentials are read from the environment (`HF_TOKEN`, `GITHUB_TOKEN`) and never from the
repository. `.env.example` documents the variables; no token file is tracked.

---

## Compute

| Tier | Hardware | Extras | Scope | Time |
|---|---|---|---|---|
| Smoke / CI | any CPU | `.[dev]` | bundled examples, hashing embedder, full test suite | ~1 min |
| Sampled study | Colab T4 (16 GB) | `.[hf,retrieval,viz]` | real MiniLM + FAISS scan, 4-bit generation, a few hundred items | hours |
| Full study | persistent T4 or MI300X | `.[all]` | four checkpoints × ~10K items, attribution, emergence | ~25–35 GPU-days |

The corrected full-scale run consumed ≈4.9 h on an A100-40GB. Every non-model stage —
containment, embedding, indexing, scoring, statistics, plotting — runs on CPU. `docs/COMPUTE.md`
gives per-stage budgets and per-model VRAM.

---

## Corrections and superseded results

Findings that did not survive audit are documented rather than deleted, and the runs that
produced them are kept so the correction is checkable.

- **The `reliability-B*` pilot runs are superseded.** `docs/AUDIT_2026-07-26.md` shows four of
  their five headline findings to be artifacts: a truncated long-CoT checkpoint, an empty
  contamination treatment group, 15% wrong numeric labels, and a 2.61× magnitude confound. Those
  numbers must not be cited.
- **The item-axis design is superseded.** It contrasted high- against low-containment GSM8K
  *train* items. `allenai/tulu-3-sft-mixture` is in Instella's stage-two mixture and carries
  ~97% of GSM8K train at containment ≥0.999, so both arms of that contrast are exposed and no
  difference between them is attributable to membership. The checkpoint axis replaces it.
  See `docs/CORPUS_CORRECTION.md`.
- **`fullscale-S250` is superseded by `fullscale-S250-v2`.** The earlier run is complete and
  reproducible but its arms were built from a corrupted containment measurement. Both run
  directories live on the artifact mirror rather than in the repository; the tracked run
  directories under `experiments/runs/` are the checkpoint-axis, replication and pilot runs.
- **A withdrawn two-corpus comparison** is documented in `docs/EXPOSURE_TWO_CORPORA.md`: two
  scans read different text (flattened `text` field against joined `messages`), producing a pair
  of numbers that were impossible rather than merely inconsistent. Both corpora were rescanned
  through one code path.

---

## Documentation

| Document | Contents |
|---|---|
| `FINDINGS.md` | Claims of record; withdrawn claims with reasons |
| `STATE.md` | Handoff state: complete, outstanding, owned-by-human |
| `docs/METHODOLOGY.md` | Estimators, inference, design rationale |
| `docs/CHECKPOINT_AXIS_STUDY.md` | Design and pipeline for the active study |
| `docs/CORPUS_CORRECTION.md` | What containment was measured against, and what it invalidates |
| `docs/EXPOSURE_TWO_CORPORA.md` | Exposure under the two corpora AMD actually consumed |
| `docs/FIGURES.md` | Provenance of every figure, by filename |
| `docs/AUDIT_2026-07-26.md` | Defect audit of the superseded pilots |
| `docs/PHASE2_DESIGN.md` | Controlled-injection dose design |
| `docs/COMPUTE.md` | Compute budgets and VRAM |
| `docs/DATA_MANIFEST.md` | What each artifact file contains |

---

## License and citation

MIT (`LICENSE`). `CITATION.cff` carries the citation metadata for this repository and for the
Instella paper it builds on:

> Liu et al. *Instella: Fully Open Language Models with Stellar Performance.*
> arXiv:2511.10628, 2025.

Generation, containment and analysis code was implemented with assistance from an AI assistant
under the direction of the authors, who verified every reported number against the underlying
generation and verdict files. The assistant did not design the experimental conditions or select
which results to report.
