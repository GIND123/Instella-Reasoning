# The checkpoint-axis study — design, pipeline, and what each choice buys

The question, unchanged from the first draft:

> Does the large accuracy drop under GSM-Symbolic perturbation reflect recall of training
> material, or failure of inference?

What changed is how it is asked. This document explains the design, names every script in
the order it runs, and records the decisions that would otherwise have to be reverse-
engineered from code.

Companion documents: [`FIGURES.md`](FIGURES.md) for the provenance of every number,
[`PHASE2_DESIGN.md`](PHASE2_DESIGN.md) for the injection mixture,
[`CORPUS_CORRECTION.md`](CORPUS_CORRECTION.md) for why the previous design was abandoned,
and `experiments/runs/ckpt-axis-v1/analysis/RESULTS.md` for the results themselves.

---

## 1. Why the item axis had to go

The original design compared high-containment against low-containment GSM8K *train* items
and read the difference as memorisation. Two findings killed it:

* `train_119K` — not the full released pool — is the split Instella-3B Stage 2 actually
  consumed. Measuring containment against the pool over-counts exposure; only 50.8% of the
  original high-containment arm survives the correction.
* `allenai/tulu-3-sft-mixture` is *also* in the Stage-2 mixture and carries ~97% of GSM8K
  train at containment ≥0.999, through several independent routes.

The second is fatal in a way the first is not. If both arms are ~97% exposed, the
within-train contrast is not confounded — it is **unidentifiable**. No amount of matching,
arm-widening, or re-running recovers an estimand that does not exist.

What survives is the *checkpoint* axis. Tulu-3 enters at Stage 2, so it is part of the
treatment, not a leak into the control. The comparison becomes: the same items, before and
after the data intervention.

## 2. The probe

A **premise-deletion** variant removes one sentence the solution's calculator chain
actually consumes, making the item underdetermined — the original answer is no longer
derivable from the prompt.

* A model reasoning from the text should report that the problem cannot be solved.
* A model reproducing a memorised item emits the original answer regardless.

The rate of the latter is the memorisation estimate. Implementation lives in
`src/instella_reasoning/structural.py`; two guards keep it honest:

1. **Necessity.** The deleted sentence must carry an integer the rationale consumes, and
   that value must be gone from the *whole* remaining prompt. If it survives elsewhere the
   problem stays solvable and a correct answer is reasoning, not recall.
2. **Answer leakage.** If the gold answer itself appears elsewhere in the remaining text,
   emitting it is copying rather than recall, and the variant is dropped.

The second guard is why `--k 3` yields **1.786** removals per parent rather than 3. It also
matters that leakage correlates with containment — on the previously measured arms it
occurred in 6.9% of high-containment items against 3.1% of low — so dropping those items
removes a bias pointing in the direction the hypothesis predicts.

## 3. The three arms of evidence

| | what it rules out | result |
|---|---|---|
| **Trajectory + clean control** | that the Stage-2 rise is item-specific | DiD −0.26 pp, CI [−1.66, +1.14] |
| **Dose regression within the train arm** | the generalisation-gap objection to the above | non-monotonic in containment |
| **Controlled injection** | that the probe is simply insensitive | detects at 64×, not at 16× |

The third is what makes the first reportable. A null means nothing without knowing what
would have been detected.

## 4. Pipeline, in order

Every step is reproducible from the repository; seeds are **6198** throughout.

```bash
# --- item sets ------------------------------------------------------------------
python experiments/build_deletion_probe.py --benchmark gsm8k_train \
    --n-parents 900 --k 3 --out-dir experiments/runs/ckpt-axis-v1/base
python experiments/build_deletion_probe.py --benchmark gsm8k \
    --n-parents 900 --k 3 --out-dir experiments/runs/ckpt-axis-v1/base --tag gsm8k_test

# --- Phase 0: does the control arm terminate? -----------------------------------
instella-reasoning generate --benchmark .../gsm8k_train_deletion.jsonl \
    --model amd/Instella-3B-Stage1 --limit 200 --max-new-tokens 2048 \
    --batch-size 8 --dtype bf16 --n-shot 4 --no-chat-template \
    --min-termination-rate 0.0 --output .../phase0__stage1__deletion200.jsonl

# --- Phase 1: the probe across the trajectory -----------------------------------
ARM=train ENGINE=vllm bash experiments/run_phase1.sh
ARM=test  ENGINE=vllm bash experiments/run_phase1.sh
LIMIT=400 TEMP=0.7 NSAMPLES=5 ARM=train ENGINE=vllm bash experiments/run_phase1.sh
LIMIT=400 TEMP=1.0 NSAMPLES=5 ARM=train ENGINE=vllm bash experiments/run_phase1.sh

# --- Phase 2: controlled injection ----------------------------------------------
python experiments/build_injection_mixture.py --parents .../gsm8k_test_parents.jsonl \
    --out-dir .../phase2/mixture --n-injected 200 --n-heldout 200
DOSES="0 64"    bash experiments/run_phase2_gonogo.sh    # go/no-go pair first
DOSES="1 4 16"  SKIP_SMOKE=1 bash experiments/run_phase2_gonogo.sh

# --- Phase 3: corpus scans (CPU, parallel) --------------------------------------
python experiments/scan_corpora.py --corpus <name> --hf-path <repo> ...
python experiments/route_attribution.py --corpus <name> ...

# --- Phase 4: analysis and figures ----------------------------------------------
python experiments/analyze_ckpt_axis.py --run experiments/runs/ckpt-axis-v1
python experiments/dose_regression.py
python src/instella_reasoning/analysis/mathai_f1_abstention.py  <run> paper/figures
python src/instella_reasoning/analysis/mathai_f2_dose.py        <run> paper/figures
python src/instella_reasoning/analysis/mathai_f3_trajectory.py  <run> paper/figures
```

## 5. Decisions worth not relitigating

**Two engines, two virtualenvs.** vLLM 0.8.5.post1 pins its own torch. Installing it over
the transformers environment would have made Phase 0 irreproducible on the same machine.
`experiments/lambda_vllm_bootstrap.sh` provisions `~/.venvs/vllm` separately, and every
generated row carries `metadata.engine`. **Rows from different engines are never pooled in
one series.** Equivalence was checked on 40 shared items: 72.5% byte-identical
completions, 100% agreement on the extracted answer class, 97.5% on the termination flag —
with the honest caveat that the sample contained zero recall events, so it says nothing
about recall agreement specifically. What protects the contrast is that all four
checkpoints use the same engine, so any engine effect is common and differences out.

**vLLM for the measurement, transformers for the gate.** Continuous batching gave ~10×
throughput on this workload (67,000/hr vs 450/hr on the chat checkpoints), which is what
turned Phase 1 from a projected 14 GPU-hours into 38 minutes. Under continuous batching
there is no fixed batch, so the padding/tie-breaking concern that fixes `--batch-size 8`
on the transformers path does not arise.

**Held fixed across checkpoints:** token budget (2048) and decoding. **Allowed to vary:**
`n_shot` and the chat template, because those follow the checkpoint's own interface — base
checkpoints loop without few-shot exemplars. That is a property of the model, not a knob of
the measurement.

**fp32 parameters with a bf16 autocast forward** in `inject_pretrain.py`. bf16 carries 8
mantissa bits; at lr 1e-5 the update is ~1e-3 of the weight magnitude and rounds away
*inside the parameter*, so the optimiser appears to run while the weights barely move. The
~24 GB for master weights removes that failure mode rather than monitoring for it.

**Positive controls inside every injection arm.** Injected-document loss and verbatim
continuation, measured before and after. A training loop that silently fails produces a
flat dose–response indistinguishable from a true null. At 64× the controls are emphatic:
injected loss 1.265 → 0.156, verbatim 0.136 → 0.946, while held-out loss *rose*.

**The 0× arm.** The plan as written had 1/4/16/64. Without 0× there is no baseline for
"recall without exposure". Its measured recall (0.0365) also reproduces Phase 1's
Instella-3B figure (0.0359) through an entirely independent path, which validates the
harness end to end.

**F2's train_119K band spans 4×–16×** rather than marking a single point. 4× is the
measured maximum verbatim repeat in the corpus; 16× is documents-per-seed. The ~16 are
numeric re-instantiations carrying *different* gold answers, so they are not answer-level
exposure — but both readings fall left of the detection threshold, so the conclusion does
not depend on which a reader prefers.

## 6. Limitations, stated before a reviewer states them

* **Scope of "clean".** Every containment claim is bounded by the corpora scanned, and all
  but `train_119K` were sampled prefixes. The supported statement about Stage-1 is "0 of
  7,473 GSM8K train items and 0 of 1,319 test items reached containment 0.3 in 400,000 rows
  of OLMoE-mix's math-bearing subsets" — not "Stage1 never saw GSM8K".
* **Two contaminated test items.** `openhermes-2.5` contains `gsm8k_00018` and
  `gsm8k_00269` at containment 1.0. One is among the 888 probed test parents; none are in
  the Phase 2 injected or held-out sets. Dropping it moves the DiD from −0.002551 to
  −0.002563.
* **Effect sizes are small.** Recall rates run 1–3.5%, with a ceiling of 7.4% at maximum
  dose. Every figure prints event counts alongside rates for this reason.
* **Abstention is near zero everywhere**, so the probe separates "the parent's number" from
  "some other number", not "reasoning" from "recall" in the abstract. The paper should
  claim the former.
* **One model family.** Instella is the only 3B-class model with a fully published training
  corpus, which is what makes the checkpoint axis measurable at all.
* **One benchmark.** MATH would be the natural second axis and cannot serve: 151 MATH-train
  items sit at containment ≥0.999 in OLMoE-mix's math subsets, i.e. it is contaminated in
  Stage-1 pretraining, before any intervention this study measures. The premise-deletion
  generator also requires integer answers and calculator annotations, neither of which MATH
  provides.
