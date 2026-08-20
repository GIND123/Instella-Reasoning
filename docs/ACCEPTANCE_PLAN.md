# Revised plan: the blocking check resolved to the pivot branch

The Tier-0 check ran and came back contaminated. `allenai/tulu-3-sft-mixture` — which is in
Instella-3B's **stage-2 pre-training mixture** — contains **97.07% of GSM8K train at
containment >= 0.999**, through at least five independent routes (see
[CORPUS_CORRECTION.md](CORPUS_CORRECTION.md) §2b).

**Consequence: the recall half of the paper is dead.** Not underpowered, not confounded —
unidentifiable. Every GSM8K train item entered Instella-3B at stage 2 regardless of its
Instella-GSM8K-synthetic containment, so the high/low arms differ in marginal synthetic dose
rather than exposure versus non-exposure. Expanding arms, matching harder, or re-running
generation cannot fix an estimand that does not exist. Stage1, the only unexposed checkpoint,
sits at floor accuracy with 93.7% truncation.

The replacement claim is stronger, more surprising, and needs almost no GPU.

---

## The paper this becomes

**Claim.** On the most openly documented model family available, the seen/unseen contrast that
the benchmark-contamination literature relies on cannot be constructed — because the standard
open instruction corpora re-ingest benchmark *train* splits verbatim, through multiple
independent routes, including routes labelled "decontaminated" and routes that are organic.

This is a positive, quotable result, not a null. It explains a live puzzle: why train-versus-
test contamination studies keep reporting weak effects. Their clean arms are not clean.

**Evidence already in hand** (all CPU, all reproducible via `scripts/verify_stage2_corpus.py`):

| corpus | enters Instella at | GSM8K train >= 0.999 | GSM8K test |
|---|---|---|---|
| `tulu-3-sft-mixture` | **stage 2** | **97.07%** | 0.08% |
| `OpenMathInstruct-2` (train_1M) | SFT | **99.34%** | 0.00% |
| `Instella-GSM8K-synthetic` train_119K | stage 2 | 15.12% (at >= 0.80) | 0.00% |

The test-split column is the control: near-zero everywhere, so these are real exposures and not
detector false positives. That contrast is what makes the finding airtight.

**The three details that make it land:**
1. `numinamath_tir_math_decontaminated` carries 4,608 GSM8K train problems. Decontamination as
   practiced targets *test* splits and is blind to train re-ingestion — which is exactly what a
   train-versus-test design depends on.
2. `flan_v2_converted` carries 3,787. GSM8K has been in FLAN since before the contamination
   discourse existed.
3. WildChat and Aya rows carry GSM8K train items pasted by *users*. Benchmark items enter
   training corpora through organic human usage, which no curation pipeline closes.

**Companion result:** GSM-Symbolic fragility (Table 2) is untouched — it uses no containment
labels. Fragility is large and robust; exposure-based explanations for it cannot be tested on
any current open family. That pairing is the paper.

---

## Work remaining for Aug 29

| # | Task | Cost |
|---|---|---|
| R1 | Scan the rest of the stage-2 mixture: Dolmino/math, OpenHermes-2.5, WebInstructSub, smoltalk, ultrachat, dm_math | 10-20 CPU-h |
| R2 | Repeat for **MATH** train, not just GSM8K — generalises past one benchmark | 5-10 CPU-h |
| R3 | Scan 2-3 corpora used by *other* families (Dolma, Pile subsets) so the claim is field-level, not Instella-level | 10-20 CPU-h |
| R4 | Per-route attribution table (which subset, how many rows) for every corpus scanned | CPU, hours |
| R5 | Formalise "stage-indexed exposure" as a named, citable protocol | writing |
| R6 | Keep and re-present the GSM-Symbolic fragility result | done |
| R7 | Rewrite: new title, abstract, framing; retire the recall tables | 3-4 days |
| R8 | Rebuild Figure 1 (currently a spreadsheet chart with colliding labels) | 1 day |

**GPU required: essentially zero.** The headline is a corpus measurement and the fragility
numbers already exist. This is the change that makes Aug 29 realistic.

**Total: ~40-60 CPU-hours and roughly a week of writing.** It fits.

---

## Revised acceptance estimate

| version | estimate |
|---|---|
| Current draft, submitted as-is | ~20% (two independent invalidating errors now known) |
| Corpus correction only | not available — the correction does not restore the estimand |
| **Pivoted paper, R1-R8 done by Aug 29** | **~75-80%** |

The pivot raises the ceiling *and* lowers the cost, because the expensive component (GPU
generation for the recall probe) is what got deleted.

**Main reviewer risk:** "we already knew training data contains benchmark data." The defence is
in the specifics — verbatim >= 0.999 rather than fuzzy overlap; the clean test-split control;
multiplicity of independent routes; a subset labelled *decontaminated*; organic leakage via
chat logs; and a concrete published design class shown to be unidentifiable as a result.
Lead with those, not with the general observation.
