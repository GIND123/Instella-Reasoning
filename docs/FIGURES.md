# Figures and numbers — provenance and regeneration

Everything the MATH-AI paper claims, with the command that produced it and the file it
came from. If a number appears in the paper and not in this file, it is not sourced and
should not be trusted.

Written to be read cold: six weeks from now, by a co-author or a reviewer, with no
memory of the run.

---

## 0. One rule that governs every figure

**Rows carry the engine that generated them, and rows from different engines are never
pooled inside one series.** Phase 0 ran on HuggingFace `transformers` 4.56.0 (static
padded batches, batch size 8). Everything else ran on `vLLM` 0.8.5.post1 (continuous
batching). Greedy decoding at T=0 is not bit-identical across the two — a 40-item
comparison found 72.5% byte-identical completions, 100% agreement on the extracted
answer class, and 97.5% on the termination flag.

Consequence: **F1, F2 and F3 use vLLM rows only.** Phase 0's 200 transformers rows appear
in the paper solely as the truncation diagnostic that gated the run, never in a rate that
is compared against another checkpoint.

Every generation row stores `metadata.engine`, so a mixed pool is detectable after the
fact:

```bash
jq -r '.metadata.engine' experiments/runs/ckpt-axis-v1/generations/*.jsonl | sort | uniq -c
```

---

## 1. Provenance

### Run directory

All artifacts live under `experiments/runs/ckpt-axis-v1/`.

| subdirectory | holds | rows |
|---|---|---|
| `base/` | deletion item sets and their parents | see below |
| `generations/` | every model completion | 36,896 |
| `analysis/` | scored rows, summaries, `RESULTS.md` | — |
| `phase2/` | injection mixture, per-dose training reports | — |
| `logs/` | stdout from every run | — |

Two directories sit outside the run tree:

| path | holds |
|---|---|
| `outputs/corpus_scan/` | Phase 3 — 9 `*_summary.json`, 9 `*_items.jsonl`, 2 `*_routes.json` |
| `paper/figures/` | F1/F2/F3 as PDF (vector, for LaTeX) and PNG (preview) |

### Item sets — `experiments/runs/ckpt-axis-v1/base/`

| file | rows | what it is |
|---|---|---|
| `gsm8k_train_deletion.jsonl` | 1,591 | deletion variants of 891 GSM8K **train** parents |
| `gsm8k_train_parents.jsonl` | 891 | the parents, intact |
| `gsm8k_test_deletion.jsonl` | 1,588 | deletion variants of 888 GSM8K **test** parents |
| `gsm8k_test_parents.jsonl` | 888 | the parents, intact |

Built by:

```bash
python experiments/build_deletion_probe.py \
    --benchmark gsm8k_train --n-parents 900 --k 3 --seed 6198 \
    --out-dir experiments/runs/ckpt-axis-v1/base
python experiments/build_deletion_probe.py \
    --benchmark gsm8k --n-parents 900 --k 3 --seed 6198 \
    --out-dir experiments/runs/ckpt-axis-v1/base --tag gsm8k_test
```

`--k 3` requests three removals per parent; the realised mean is **1.786** (train) and
**1.788** (test). The shortfall is the answer-leakage guard in
`structural._premise_removal_at`, which drops a removal when the parent's gold answer
still appears elsewhere in the remaining text — such an item stays solvable, and a model
emitting the parent answer would be reasoning correctly rather than recalling. 900
parents were requested; 891 and 888 yielded at least one variant.

The two arms landing within 0.002 of each other on removals-per-parent is a property of
the generator, not of tuning.

### Generation files — `experiments/runs/ckpt-axis-v1/generations/`

| file | rows | engine |
|---|---|---|
| `phase0__stage1__deletion200.jsonl` | 200 | **transformers-4.56.0** |
| `phase1__{stage1,stage2,sft,instruct}__train__T0.0__k1__vllm.jsonl` | 1,591 each | vllm-0.8.5.post1 |
| `phase1__{stage1,stage2,sft,instruct}__test__T0.0__k1__vllm.jsonl` | 1,588 each | vllm-0.8.5.post1 |
| `phase1__{...}__train__T0.7__k5__vllm.jsonl` | 2,000 each | vllm-0.8.5.post1 |
| `phase1__{...}__train__T1.0__k5__vllm.jsonl` | 2,000 each | vllm-0.8.5.post1 |
| `phase2__dose{0,1,4,16,64}__test__T0.0__k1__vllm.jsonl` | 1,588 each | vllm-0.8.5.post1 |
| `smoke__vllm__stage1__40.jsonl` | 40 | vllm-0.8.5.post1 (engine-equivalence check only) |
| **total** | **36,896** | |

The temperature-sweep files are 2,000 rows = the first **400** deletion rows x **k=5**
draws. A single draw at T>0 is not a rate.

### Phase 2 mixture — `experiments/runs/ckpt-axis-v1/phase2/mixture/`

| file | rows |
|---|---|
| `injected.jsonl` | 200 GSM8K test parents, injected |
| `heldout.jsonl` | 200 GSM8K test parents, never injected |
| `injected_docs.jsonl` | 200, rendered in Stage-2 concatenated form |
| `filler.jsonl` | 30,413 Tulu-3 rows, ~10M tokens |

```bash
python experiments/build_injection_mixture.py \
    --parents experiments/runs/ckpt-axis-v1/base/gsm8k_test_parents.jsonl \
    --out-dir experiments/runs/ckpt-axis-v1/phase2/mixture \
    --n-injected 200 --n-heldout 200 --seed 6198
```

31,411 Tulu-3 rows streamed, 30,413 kept, **0 rejected for containment** — no filler row
shared a single 13-gram with any of the 400 measured items.

### What is where

| location | contents |
|---|---|
| HF `GOVINDFROM/Instella-Reasoning` (**private**) | the whole run directory, ~105 MB, including all generations and `analysis/RESULTS.md` |
| laptop `experiments/runs/ckpt-axis-v1/` | identical copy |
| git branch `agent/ckpt-axis-deletion-probe` | code, docs, analysis, item sets. **Excludes** `generations/` and `phase2/mixture/filler.jsonl` (`.gitignore`), both re-derivable |
| **nowhere** | the five 5.9 GB fp16 dose checkpoints. Destroyed with the GPU instance; exactly reproducible from `inject_pretrain.py` with seed 6198 |

---

## 2. Figures

All three: `figsize=(3.5, ...)`, single-column NeurIPS width. Palette is `SERIES` from
`src/instella_reasoning/analysis/figures.py`, validated with the `dataviz` skill's
checker:

```bash
node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4" --mode light
# ALL CHECKS PASS — worst adjacent dE 9.1 (protan), normal-vision floor 19.6
# WARN: contrast vs surface below 3:1 -> relief required. Satisfied by direct
#       count labels on every mark in all three figures.
```

Error bars everywhere are **95% cluster bootstrap over parent problems**, not over rows.
Deletion variants of one problem are correlated; resampling rows would draw intervals
narrower than the design supports.

---

### F1 — `paper/figures/F1_abstention.{pdf,png}`

**Claim it supports:** on provably underdetermined items, models emit a confident number
97.5–100% of the time. DPO buys a little of this capability and nothing else does.

```bash
python src/instella_reasoning/analysis/mathai_f1_abstention.py \
    experiments/runs/ckpt-axis-v1 paper/figures
```

**Reads:** `analysis/phase1_train_rows.jsonl`, `analysis/phase1_test_rows.jsonl`
(produced by `experiments/score_deletion_probe.py`, which reads the `phase1__*__T0.0__k1__vllm.jsonl`
generations and the two deletion item sets).

**Filtering:** none. Every row in both files is plotted. n = 1,591 (train) and 1,588
(test) per checkpoint.

**Engine:** vLLM 0.8.5.post1, greedy, 2048-token budget.

| checkpoint | train: abstain / n | test: abstain / n |
|---|---|---|
| Stage1 | 1 / 1591 (0.06%) | 0 / 1588 (0.00%) |
| Instella-3B | 0 / 1591 (0.00%) | 0 / 1588 (0.00%) |
| SFT | 6 / 1591 (0.38%) | 9 / 1588 (0.57%) |
| Instruct | 34 / 1591 (2.14%) | 40 / 1588 (2.52%) |

**Abstention is detected** by `score_deletion_probe._ABSTAIN`, a deliberately narrow
regex ("cannot be determined", "not enough information", …) plus the case of no
extractable answer at all. Narrow on purpose: a false positive here counts as reasoning
and deflates the finding.

**Axis note:** y is truncated at 5%. At full scale every bar vanishes; the subtitle
carries the ceiling ("the correct rate is 100%") so the truncation cannot mislead.

---

### F2 — `paper/figures/F2_dose_response.{pdf,png}`

**Claim it supports:** the probe detects injected memorisation only at 64x, far above the
exposure `train_119K` actually delivers. This is what bounds the Phase 1 null.

```bash
python src/instella_reasoning/analysis/mathai_f2_dose.py \
    experiments/runs/ckpt-axis-v1 paper/figures
```

**Reads:** `generations/phase2__dose{0,1,4,16,64}__test__T0.0__k1__vllm.jsonl`,
`base/gsm8k_test_deletion.jsonl`, `phase2/mixture/{injected,heldout}.jsonl`,
`phase2/dose{N}_report.json`.

**Filtering:** each 1,588-row probe file is split by parent membership —
**351 rows** whose parent is in the injected set, **371 rows** in the held-out set,
**866 rows** in neither (plotted in neither series). 351 + 371 + 866 = 1,588; no rows are
silently dropped.

**Engine:** vLLM 0.8.5.post1, greedy, 2048 tokens, `--n-shot 4 --no-chat-template`
(these checkpoints derive from `amd/Instella-3B`, a **base** model; probing them at
n_shot=0 would have made dose 0 incomparable to Phase 1's Instella-3B number).

| dose | injected recall | events / n | McNemar vs 0x (b/c) | p | verbatim after | held-out recall |
|---|---|---|---|---|---|---|
| 0x | 4.27% | 15 / 351 | — | — | 0.133 | 2.70% |
| 1x | 5.41% | 19 / 351 | 9 / 5 | 0.424 | 0.153 | 3.77% |
| 4x | 5.98% | 21 / 351 | 9 / 3 | 0.146 | 0.180 | 4.31% |
| 16x | 5.13% | 18 / 351 | 12 / 9 | 0.664 | 0.492 | 3.77% |
| **64x** | **7.41%** | **26 / 351** | **17 / 6** | **0.035** | **0.946** | 4.31% |

**Test:** exact two-sided McNemar, paired within item against the 0x arm. `b` = recall
present at this dose and absent at 0x; `c` = the reverse.

**The 0x validation:** 0x recall over all 1,588 rows is **0.0365**; Phase 1's
`amd/Instella-3B` on the same item set is **0.0359**. Agreement to 6e-4 through an
independent path (8.4M tokens of continue-pretraining on pure filler, then a fresh probe).

**The shaded 4x–16x band** is `train_119K`'s real exposure, spanning both defensible
accountings:

* **4x** — measured maximum verbatim repeat of any single document in the corpus.
  Source: 119,014 rows, **88,179** distinct by exact MD5, max multiplicity **4**,
  18,092 documents appearing more than once.
* **16x** — 119,014 / 7,473 documents per GSM8K seed.

The upper figure is conservative: those ~16 documents are numeric re-instantiations
carrying **different gold answers** (verified on the "Natalia clips" seed — 20 documents
with 36/48/52/54/60 clips and half/40%/60%/quarter/three-quarters), so they are not
answer-level exposure. Both readings fall left of the detection threshold, which is the
point of drawing a band rather than a line.

Reproduce the corpus statistics:

```bash
python - <<'PY'
import json, hashlib, collections
c = collections.Counter()
for line in open("experiments/runs/fullscale-S250-v3/contamination/synthetic_corpus.jsonl"):
    c[hashlib.md5((json.loads(line).get("text") or "").encode()).hexdigest()] += 1
print(len(c), "distinct;", max(c.values()), "max repeats;", sum(v > 1 for v in c.values()), "repeated")
PY
```

---

### F3 — `paper/figures/F3_trajectory_null.{pdf,png}`  *(cut first if space is tight)*

**Claim it supports:** recall rises across the Stage-2 data intervention, and it rises
just as much on items provably absent from the corpus.

```bash
python src/instella_reasoning/analysis/mathai_f3_trajectory.py \
    experiments/runs/ckpt-axis-v1 paper/figures
```

**Reads:** `analysis/phase1_{train,test}_rows.jsonl`, `analysis/ckpt_axis_results.json`.

**Filtering:** none. n = 1,591 / 1,588 per checkpoint.

**Engine:** vLLM 0.8.5.post1, greedy.

| checkpoint | train: recall / n | test: recall / n | train trunc | test trunc |
|---|---|---|---|---|
| Stage1 | 18 / 1591 (1.13%) | 26 / 1588 (1.64%) | 17.60% | 18.01% |
| Instella-3B | 45 / 1591 (2.83%) | 57 / 1588 (3.59%) | 2.89% | 2.27% |
| SFT | 49 / 1591 (3.08%) | 53 / 1588 (3.34%) | 2.14% | 2.46% |
| Instruct | 57 / 1591 (3.58%) | 50 / 1588 (3.15%) | 0.19% | 0.06% |

**The intervention, paired within item (train arm), Stage1 -> Instella-3B:**

* all rows: n=1591, gained **42**, lost **15**, exact McNemar **p = 4.60e-4**
* restricted to rows where **both** checkpoints terminated: n=1282,
  0.0125 -> 0.0265, gained **31**, lost **13**, **p = 9.56e-3**

Quote the restricted figure. Stage1 truncates 17.6% against Instella-3B's 2.9%, and a
truncated row cannot recall, so the unrestricted contrast credits the data intervention
for Stage1's looping. The residual truncation is a repetition loop, not reasoning cut
short: unfinished rows have median **6,941** characters against **148** for finished ones,
and contribute **0 / 34** recall events.

**Difference-in-differences:**

```
train change  +0.01697
test  change  +0.019521
DiD           -0.002551      95% CI [-0.016615, +0.011448]
```

4,000 bootstrap replicates, clusters = parent problems, seed 6198. The interval crosses
zero.

---

## 3. Numbers in the paper that are not in a figure

### Temperature sweep — is the contrast an argmax artefact?

`analysis/ckpt_axis_results.json` -> `temperature_sweep`. Restricted to the **first 400**
deletion rows, which is the subset the sweep was run on; T=0 is re-measured on that same
subset so the three columns are comparable.

| checkpoint | T=0 (k=1, n=400) | T=0.7 (k=5, n=2000) | T=1.0 (k=5, n=2000) |
|---|---|---|---|
| Stage1 | 0.0075 | 0.0160 | 0.0120 |
| Instella-3B | 0.0350 | 0.0335 | 0.0280 |
| SFT | 0.0425 | 0.0435 | 0.0435 |
| Instruct | 0.0500 | 0.0420 | 0.0415 |

Stage1 -> Instella-3B rise: **+0.0275** (T=0), **+0.0175** (T=0.7), **+0.0160** (T=1.0).
The rise survives sampling. Report the sampled figures alongside greedy: the effect is
smaller than T=0 alone suggests, and Stage1's rate roughly doubles under sampling while
every later checkpoint falls slightly.

### Phase 0 — the truncation gate  *(transformers rows; never pooled with the above)*

```bash
instella-reasoning generate \
  --benchmark experiments/runs/ckpt-axis-v1/base/gsm8k_train_deletion.jsonl \
  --model amd/Instella-3B-Stage1 \
  --output experiments/runs/ckpt-axis-v1/generations/phase0__stage1__deletion200.jsonl \
  --limit 200 --max-new-tokens 2048 --temperature 0.0 --batch-size 8 \
  --dtype bf16 --n-shot 4 --no-chat-template --min-termination-rate 0.0
```

Termination 83.0%, i.e. **truncation 17.0%** against a <15% gate. Failed by 2 points; the
residual is looping, not truncated reasoning (see F3 note above).

### Phase 3 — corpus containment  *(appendix material)*

```bash
python experiments/scan_corpora.py --corpus <name> --hf-path <repo> \
    --split train --text-field <field> --limit <N> --out outputs/corpus_scan
# --hf-glob 'data/<subset>/'        OLMoE-mix: shards disagree on columns even within one
#                                   subset, so `datasets` raises CastError before scanning
# --revision refs/convert/parquet   dm_math: script-based, and datasets>=3 dropped script
#                                   loading; the auto-generated parquet mirror works
# --local-jsonl <path>              train_119K: already materialised on disk
```

All three item sets are indexed in a single pass — GSM8K train 7,473, GSM8K test 1,319,
MATH train 7,500 (16,292 items, 510,227 distinct 13-grams) — so adding MATH costs nothing.

| corpus | rows scanned | GSM8K train ≥0.999 | GSM8K **test** ≥0.999 | MATH train ≥0.999 |
|---|---|---|---|---|
| smoltalk | 300,000 | **3,453** (46.2%) | 0 | **1,947** (26.0%) |
| train_119K *(Stage-2, as consumed)* | 119,014 | 21 (0.3%) | **0** | 0 |
| **openhermes-2.5** | 300,000 | 1 | **2** | 1 |
| webinstruct | 300,000 | 0 | 0 | 1 |
| dolmino-mix | 200,000 | 0 | 0 | 0 |
| ultrachat_200k | 200,000 | 0 | 0 | 0 |
| dm_math | 300,000 | 0 | 0 | 0 |
| OLMoE-mix / open-web-math | 200,000 | 0 | 0 | **126** |
| OLMoE-mix / algebraic-stack | 200,000 | 0 | 0 | **25** |

**Sampling.** Every row above is a sampled prefix except `train_119K`, which is a full
pass. Containment is a *lower* bound: a zero means "absent from the rows scanned", never
"absent from the corpus". Each `outputs/corpus_scan/<name>_summary.json` records
`rows_scanned`, `sampled`, `rows_skipped_too_short`, `extraction_suspect` and a
`coverage_note`.

**Three things this establishes.**

*The control arm holds against the corpus that matters.* GSM8K test is **0 at ≥0.8 with
median 0.0** against `train_119K` — the Stage-2 data as actually consumed, scanned in
full, not a proxy.

*MATH is contaminated in Stage-1 pretraining.* 151 MATH-train items at ≥0.999 across
OLMoE-mix's math subsets, before any intervention this study measures. That is why MATH
cannot serve as a clean second reasoning axis, and it is a stronger reason than "we ran
out of time".

*J1 is now a measurement, not an assumption.* **0 of 7,473 GSM8K train items and 0 of
1,319 test items reached containment 0.3 in 400,000 rows of OLMoE-mix's math-bearing
subsets.** The supported claim is exactly that sentence — not "Stage1 never saw GSM8K".

#### Two GSM8K **test** items are contaminated, and here is the exposure

`openhermes-2.5` contains `gsm8k_00018` and `gsm8k_00269` at containment 1.0. Their reach
into the study:

| set | contaminated members |
|---|---|
| the 888 probed test parents | **1** |
| the 1,588 test deletion rows' parents | 1 |
| the 200 Phase 2 **injected** items | **0** |
| the 200 Phase 2 **held-out** items | **0** |

Phase 2 is untouched. Sensitivity of the headline DiD to dropping the affected parent:

```
as reported                    n_test_parents=888   DiD=-0.002551  CI[-0.016615, +0.011448]
contaminated parent dropped    n_test_parents=887   DiD=-0.002563  CI[-0.016800, +0.011418]
```

Disclose it in the paper. It changes nothing.

### Route attribution — which subset carries the leak

```bash
python experiments/route_attribution.py --corpus smoltalk \
    --hf-path HuggingFaceTB/smoltalk --config all --split train \
    --text-field messages --source-field source --limit 300000 --out outputs/corpus_scan
```

Counts are of **distinct items**, not documents: one item matched by fifty documents from
one source is one leaked item.

| corpus | source subset | GSM8K train | MATH train | from rows |
|---|---|---|---|---|
| smoltalk | `metamathqa-50k` | **2,276** (30.5%) | **1,530** (20.4%) | 13,495 |
| smoltalk | `openhermes-100k` | 1,268 (17.0%) | 172 (2.3%) | 27,356 |
| smoltalk | `numina-cot-100k` | 523 (7.0%) | 228 (3.0%) | 30,537 |
| smoltalk | `smol-magpie-ultra` | — | 120 (1.6%) | 117,774 |
| tulu3 | `ai2-adapt-dev/flan_v2_converted` | **6,021** (80.6%) | — | 89,982 |
| tulu3 | `ai2-adapt-dev/tulu_v3.9_wildchat_100k` | 4 (0.1%) | — | 100,000 |

**Coverage caveat, and it is a real one.** A sampled prefix is not a representative sample
of a mixture's *source* composition. The tulu3 scan reached only **6** of that mixture's
subsets in 300,000 rows and never saw `tulu_v3.9_open_math_2_gsm8k_50k`, which the
teammate's full-corpus attribution documents. Their Tulu-3 numbers are the better ones;
these are a lower bound. The smoltalk table is not subject to the same doubt — 13 sources
were seen and the dense route was among them.

### Dose regression — recall against Stage-2 containment

```bash
python experiments/dose_regression.py \
    --run experiments/runs/ckpt-axis-v1 \
    --containment outputs/corpus_scan/train119k_items.jsonl
```

**Reads:** `analysis/phase1_train_rows.jsonl`, `outputs/corpus_scan/train119k_items.jsonl`.
**Writes:** `analysis/dose_regression.json`.

Containment against `train_119K` is continuous across the train arm (median 0.35, range
to 1.0), giving a graded exposure axis *inside one split*. If the Stage-2 rise were
memorisation, recall should climb with it. Recall at `amd/Instella-3B`:

| containment bin | parents | rows | recall events | rate | 95% CI |
|---|---|---|---|---|---|
| [0.00, 0.10) | 219 | 407 | 15 | 0.0369 | [0.0186, 0.0587] |
| [0.10, 0.30) | 177 | 326 | 5 | 0.0153 | [0.0031, 0.0307] |
| [0.30, 0.50) | 197 | 346 | 3 | 0.0087 | [0.0000, 0.0196] |
| [0.50, 0.80) | 240 | 415 | 18 | 0.0434 | [0.0253, 0.0639] |
| [0.80, 1.01) | 58 | 97 | 4 | 0.0412 | [0.0100, 0.0811] |

Non-monotonic, every interval overlapping; the least-contained bin (3.69%) matches the
most-contained (4.12%). **This is the one null that does not use the train/test contrast**,
so the generalisation-gap objection does not reach it.

Parents with a recorded containment value: 722 of 891. The remaining 169 have containment
0 and fall in the first bin; 219 + 177 + 197 + 240 + 58 = 891, so no parent is dropped.

### Minimum detectable effect

Base rate **0.0283** over **891** parent clusters (mean 1.786 rows each). An increase to
**0.0472** — **+1.89 pp** — would be detected at 80% power, two-sided α=0.05. The observed
DiD is **−0.26 pp**.

That is the *statistical* floor. The *empirical* floor from Phase 2 is stricter and is the
one to quote: the probe separated from baseline at 64× (verbatim 0.946) and not at 16×
(verbatim 0.491), so near-verbatim reproduction is required before the instrument responds
at all.

---

## 3b. Two silent bugs found during Phase 3

Both produced plausible-looking wrong answers rather than errors, and both were caught by
a contradiction rather than by a traceback. Recorded because the failure mode generalises.

**Message-payload extraction.** `scan_corpora.row_text` read `m["content"]` from
chat-style rows. OpenHermes-2.5 uses `m["value"]` (ShareGPT convention), so all 300,000
rows yielded an empty string, fell below the 40-character floor, and were skipped — and
the corpus was reported as **0% contaminated**. Caught only because smoltalk's route table
attributed 1,268 GSM8K items to its `openhermes-100k` subset, contradicting the standalone
scan.

Fixed to try `content` / `value` / `text` in order, and a guard now prints a warning and
sets `extraction_suspect: true` in the manifest when more than 50% of rows fall under the
length floor. Verified afterwards that ultrachat and smoltalk both use `content`, so their
earlier scans were sound; only OpenHermes was affected.

**Uninitialised counter.** The `skipped_short` counter added with that fix was initialised
in `scan_corpora.py` but not in `route_attribution.py`, where the initialiser pattern
differed. The first tulu3 route run died with `UnboundLocalError` after six minutes. Fixed.

---

## 4. Regenerating everything

```bash
# all scored summaries, the DiD, the sweep table and the dose table
python experiments/analyze_ckpt_axis.py --run experiments/runs/ckpt-axis-v1
#   -> analysis/ckpt_axis_results.json, analysis/RESULTS.md

# the three figures
python src/instella_reasoning/analysis/mathai_f1_abstention.py experiments/runs/ckpt-axis-v1 paper/figures
python src/instella_reasoning/analysis/mathai_f2_dose.py       experiments/runs/ckpt-axis-v1 paper/figures
python src/instella_reasoning/analysis/mathai_f3_trajectory.py experiments/runs/ckpt-axis-v1 paper/figures
```

Requires `matplotlib` (`pip install matplotlib` into the project venv; it is not in the
default dependency set). All three scripts print the paths they wrote and exit non-zero
never — they print a message and return `None` if matplotlib is missing.

Seeds are fixed at **6198** throughout: item selection, mixture construction, training,
and every bootstrap.
