# Phase 2 — controlled injection: mixture design

Settled before GPU time, because the failure it guards against is silent. If total
training tokens scale with dose, then "64x" is also "trained 64x longer", the two effects
are not separable after the fact, and the dose–response curve cannot be interpreted no
matter how clean the probe is.

## The estimand

How much exposure does it take to produce recall at all? The output is recall rate as a
function of repetition count, measured with the *identical* premise-deletion probe used in
Phase 1. That is what converts every Phase 1 number from a bare rate into a calibrated
one: "our probe detects recall at >= N exposures; the trajectory shows X."

## Fixed-budget design

**Total training tokens are held constant across every arm.** Dose is the share of that
fixed budget occupied by the injected items, not an addition to it.

| | value | why |
|---|---|---|
| base model | `amd/Instella-3B` | the Stage-2 checkpoint; the trajectory node the curve calibrates |
| injected items | 200 GSM8K **test** parents (drawn from the 888 with deletion variants) | verified clean: 0/1319 containment in every corpus scanned |
| held-out control items | 200 further GSM8K test parents, never injected | measures damage from continue-pretraining, not exposure |
| doc format | Stage-2 concatenated Q–rationale–`#### answer` | recall is format-sensitive; another shape would decalibrate the curve |
| tokens/doc | mean 160, median 152, p95 263 (measured, Instella tokenizer) | |
| one repetition of the injected set | **31,390 tokens** | measured, not estimated |
| **fixed mixture size N** | **8,388,608 tokens (8M)** | 64x fits with filler left over |
| doses | **0x, 64x first** as a go/no-go pair; 1x/4x/16x only if they separate | see "Sequencing" |
| filler | Stage-2-like corpus, GSM8K-test-free | see "why the filler is safe" |

Injected share of the fixed budget:

| dose | injected tokens | share of 8M | filler tokens |
|---|---|---|---|
| 0x | 0 | 0.00% | 8,388,608 |
| 1x | 31,390 | 0.37% | 8,357,218 |
| 4x | 125,560 | 1.50% | 8,263,048 |
| 16x | 502,240 | 5.99% | 7,886,368 |
| 64x | 2,008,960 | 23.95% | 6,379,648 |

Every arm sees 8,388,608 tokens, the same optimizer steps, the same LR schedule, and the
same seed. The only thing that varies is how much of that budget is the injected set.

At a fixed budget, raising exposure to X necessarily lowers exposure to something else.
That is not a confound, it is what dose *means* here: the displaced material is generic
filler that no measurement touches. The alternative — hold filler fixed and let the total
grow — is the confound, and it is the one this design exists to avoid.

## The 0x arm is mandatory, and the plan as written omits it

With arms at 1x/4x/16x/64x only, there is no baseline for "recall without exposure", so a
nonzero recall rate at 1x cannot be distinguished from the rate the probe already returns
on an untouched model. The 0x arm is the same 8M tokens of pure filler: it separates the
effect of *exposure* from the effect of *continue-pretraining at all*. It costs one extra
arm, roughly six minutes of H100 time.

## Why the filler is safe, and why only test items can be injected

J4 established that `allenai/tulu-3-sft-mixture` carries 97.07% of GSM8K **train** at
containment >= 0.999. A Stage-2-like filler therefore *cannot* be used with train items:
the filler would re-inject the very items being dosed, and every arm would silently
receive far more than its nominal dose.

GSM8K **test** sits at 0/1319 in every corpus scanned. That single fact is what makes the
injection design identifiable at all: the dose an item receives is exactly the dose the
mixture gives it, because nothing else in the mixture contains it. The item-set choice is
load-bearing, not a convenience.

Caveat inherited from J1: "clean" means clean with respect to the corpora scanned. The
Stage-1 web mix (OLMoE-mix-0924, ~1.3T tokens) has not been scanned, so the 0x arm is a
baseline for *this mixture's* exposure, not for all exposure the base model ever had.

## Sequencing: extremes first

0x and 64x bracket the largest effect obtainable. If the strongest dose cannot move the
probe, no intermediate dose will, so the pair is run first and the middle doses are run
only if it separates. The saving is not compute — three middle arms is roughly eighteen
minutes — it is time-to-information: whether Phase 2 works at all is known in about a
quarter of an hour instead of half a day.

**0x runs before 64x.** The 0x arm is the same fixed budget made entirely of filler, so its
probe recall should return Instella-3B's already-measured rate (2.83% on the train arm,
n=1591). That is a free end-to-end validation of the harness against a known quantity, at
identical cost. If 0x lands far from it, continue-pretraining itself damaged the model,
and that is learned before anything is spent on 64x.

## The stopping rule, and what would make it unsound

A flat 64x-vs-0x pair has three possible causes, and they are not distinguishable from the
pair alone:

1. the fixed budget is too small to induce recall — the intended reading
2. **the training harness silently failed** — no optimizer step, frozen parameters, data
   not reaching the loss, updates rounding away inside a low-precision parameter
3. the probe cannot detect recall even where it exists

On the first run of a harness this repo did not previously contain, (2) is likelier than
(1). Reporting a null caused by (2) would be the worst outcome available here, so the
stopping rule is conditional on two positive controls inside each arm, both costing no
extra GPU time:

* **injected-document loss**, measured before and after. At 64x the model sees these 200
  documents sixty-four times; their loss must fall.
* **verbatim continuation** — feed the first half of an injected document, greedily
  continue, count reproduced tokens. A 3B model that cannot reproduce text it saw 64 times
  did not fail to memorise; the run failed to train.

Only when both pass does a flat pair mean "exposure at this budget does not induce recall".
`inject_pretrain.py` prints `positive_controls_pass` and shouts when it is false.

**Threshold, pre-specified before the numbers are seen.** The arms are paired — the same
200 injected items measured under two models — so the test is McNemar rather than two
independent proportions, which at this base rate is far more powerful: 8 one-directional
discordant items already reaches exact p=0.039, whereas an unpaired test at n=200 needs
recall to reach ~7.5% for 74% power against a 2.8% base. **Go = McNemar p<0.05 on the
injected set, with the 0x rate consistent with Instella-3B's 2.83%.**

## Cost

Forward+backward at 6 x 3e9 params x 8.4e6 tokens ~= 1.5e17 FLOPs per arm. At a realistic
~400 TFLOP/s bf16 on one H100 that is ~6 minutes of compute per arm, so five arms land
near 30–40 minutes plus checkpoint I/O — well under the 4.5 GPU-h the plan budgeted. The
headroom is better spent on a larger N (more realistic filler share) than on more doses.

Memory, 3B in bf16 with Adam: ~6 GB weights + ~6 GB grads + ~24 GB optimizer moments in
fp32 + ~12 GB fp32 master weights ~= 48 GB before activations. Gradient checkpointing at
seq 2048 keeps activations near ~10 GB, so ~58 GB of 80 GB. Micro-batch 4 with gradient
accumulation 8 leaves margin; the 64x arm is the one to watch, because its packed
sequences are the least diverse and any memory drift shows there first.

## Measurement

The probe is byte-identical to Phase 1: same deletion items, same vLLM engine
(0.8.5.post1), same greedy decoding, same 2048-token budget. Two rates per arm:

* **injected items** — recall as a function of dose. This is the curve.
* **held-out clean items** — recall should stay flat. If it rises, the mixture is
  teaching GSM8K-shaped behaviour rather than the specific items, and the curve measures
  generalisation rather than memorisation.

Forgetting check: accuracy on the *intact* parents of both sets before and after. A model
damaged by continue-pretraining makes the probe measure the wrong thing, and that failure
is silent — a damaged model abstains more, which reads as less recall.

## Open

* N = 8M is a judgement call, not a derivation. It is the smallest round budget in which
  the 64x arm still leaves a filler majority. A larger N makes every share more realistic
  and costs proportionally more.
* Filler source settled: `allenai/tulu-3-sft-mixture`, scanned rather than assumed.
  31,411 rows streamed, 30,413 kept (~10M tokens), and **0 rejected for containment** —
  not one row shared a single 13-gram with any of the 400 items this study measures. That
  is the scan-what-you-use property: the check covers the exact rows used as filler, not a
  corpus-level claim about the dataset.
