GOAL: Checkpoint-axis premise-deletion study for the MATH-AI workshop paper (4 pages,
      NeurIPS dblblindworkshop template, non-archival, unlimited appendix).

================================================================================
STATUS AT HANDOFF: all machine work is DONE. Only the paper remains, and it is yours.
================================================================================

PHASES 0-4 -- WHAT IS COMPLETE
  Phase 0  Stage1 truncation gate .................. DONE (17.0%, gate <15%, failed by 2pp
           but the residual is a repetition loop, not truncated reasoning: unfinished rows
           median 6,941 chars vs 148, and contribute 0/34 recall events)
  Phase 1  deletion variants vs train_119K ......... DONE (891 train / 888 test parents)
           probe x 4 checkpoints x 2 arms .......... DONE (12,716 generations)
           GSM8K-test clean control ................ DONE
           temperature sweep T=0/0.7/1.0, k=5 ...... DONE (16,000 generations)
  Phase 2  injection, 5 doses + positive controls .. DONE (0/1/4/16/64x, fixed 8,388,608
           token budget, probes on all five)
  Phase 3  6 Stage-2 corpora ....................... DONE (dolmino, openhermes,
           webinstruct, smoltalk, ultrachat, dm_math)
           MATH train in every scan ................ DONE (free, same pass)
           route-attribution tables ................ DONE (smoltalk, tulu3)
           OLMoE-mix bound (J1) .................... DONE (open-web-math, algebraic-stack)
           train_119K containment .................. DONE
  Phase 4  figures F1/F2/F3 (PDF+PNG) .............. DONE
           docs/FIGURES.md reproducibility doc ..... DONE
           dose regression + MDE ................... DONE
           ** paper rewrite ........................ NOT STARTED (user owns) **
           ** email Jiang: findings + authorship ... NOT STARTED (user decision) **

THE THREE RESULTS THE PAPER RESTS ON
  1. Fragility is inference failure, not recall.
     DiD = -0.0026, 95% CI [-0.0166, +0.0114], cluster bootstrap on parent, 4000 reps.
     Train arm +0.0170, clean test arm +0.0195 -- the clean arm rises MORE.
  2. Models cannot detect premise insufficiency.
     Abstention on provably underdetermined items: 0.06% / 0.00% / 0.38% / 2.14% (train)
     across Stage1 -> Instella-3B -> SFT -> Instruct. DPO buys the only movement.
  3. The null is bounded, not bare.
     Phase 2: verbatim reproduction 0.133 -> 0.946 across doses, but deletion recall stays
     flat until 64x (p=0.035). At 16x the model reproduces 49% of injected tokens verbatim
     and the probe detects NOTHING. MDE: +1.89pp detectable at 80% power; observed -0.26pp.
  Supporting: dose regression within the train arm is non-monotonic in containment
     (0.0369 / 0.0153 / 0.0087 / 0.0434 / 0.0412 across five bins) -- a third null that does
     not use the train/test contrast and so is immune to the generalisation-gap objection.
  Supporting: the contrast survives sampling (+0.0275 at T=0, +0.0175 at T=0.7,
     +0.0160 at T=1.0), so it is not an argmax artefact.

PHASE 3 FINDINGS (appendix material -- 4 pages cannot hold this in main text)
  smoltalk    46.2% of GSM8K train, 26.0% of MATH train at >=0.999.
              Routes: metamathqa-50k 2,276 gsm8k + 1,530 math from only 13,495 rows;
              openhermes-100k 1,268; numina-cot-100k 523.
  tulu3       flan_v2_converted 6,021 gsm8k train (80.6%); wildchat 4.
  openhermes  1 gsm8k train, 1 math train, 2 GSM8K **TEST** at >=0.999.
  OLMoE-mix   0 GSM8K train and 0 GSM8K test at >=0.3 in 400,000 rows of the math-bearing
              subsets -- J1's assumption is now a measured bound.
              BUT 151 MATH train items at >=0.999: MATH is contaminated in Stage-1
              pretraining, before any intervention. That is why MATH cannot serve as a
              clean second reasoning axis.
  train_119K  GSM8K test 0 at >=0.8, median 0.0. Control arm clean against the corpus the
              model actually trained on.
  dolmino, ultrachat, webinstruct, dm_math: clean.

CAVEATS THAT MUST APPEAR IN THE PAPER
  * 2 GSM8K test items are contaminated in OpenHermes-2.5. One is among the 888 probed
    test parents; ZERO are in the 200 injected or 200 held-out. Sensitivity check:
    DiD -0.002551 -> -0.002563 with it dropped. Disclose; it changes nothing.
  * Every Phase 3 scan is a SAMPLED PREFIX. Containment is a lower bound. A zero means
    "absent from the rows scanned", never "absent from the corpus". Each summary JSON
    records rows_scanned / sampled / coverage_note / extraction_suspect.
  * My tulu3 route table saw only 6 sources in the first 300,000 rows and never reached
    tulu_v3.9_open_math_2_gsm8k_50k. A prefix is not a representative sample of a
    mixture's source composition. The teammate's full-corpus attribution is better.
  * Phase 0 rows are transformers-4.56.0; everything else is vllm-0.8.5.post1. NEVER pool
    them in one series. Engine is stamped in metadata.engine on every row.
  * F2's train_119K band spans 4x-16x deliberately: 4x is the measured max verbatim repeat
    (119,014 rows, 88,179 distinct, max multiplicity 4), 16x is 119,014/7,473 docs per
    seed. The ~16 are numeric re-instantiations with DIFFERENT gold answers, so they are
    not answer-level exposure. Both readings fall left of the detection threshold.

BUGS FOUND AND FIXED TONIGHT (both were silent)
  * scan_corpora.row_text read m["content"] only; OpenHermes-2.5 uses m["value"], so
    300,000 rows yielded empty text and the corpus scanned as 0% contaminated. Fixed to
    try content/value/text, and a guard now warns + sets extraction_suspect when >50% of
    rows fall under the 40-char floor.
  * route_attribution.py: skipped_short incremented but never initialised ->
    UnboundLocalError killed the first tulu3 run. Fixed.

WHERE THINGS LIVE
  laptop  experiments/runs/ckpt-axis-v1/   full run, 105 MB
          outputs/corpus_scan/             all Phase 3 summaries + route tables
          paper/figures/                   F1/F2/F3 as PDF + PNG
          docs/FIGURES.md                  per-figure provenance, exact n and counts
          docs/PHASE2_DESIGN.md            injection design + stopping rule
  HF      GOVINDFROM/Instella-Reasoning (PRIVATE) -- run dir as of the FINAL push.
          NOTE: does NOT yet include tonight's Phase 3 outputs, the figures, the dose
          regression, or the corrected OpenHermes scan. Needs one more push.
  git     branch agent/ckpt-axis-deletion-probe, pushed. Commit 166db28 predates
          tonight's Phase 3 + figures work -- those are UNCOMMITTED.
  gone    the five fp16 dose checkpoints (destroyed with the GPU box; exactly
          reproducible from inject_pretrain.py at seed 6198)

FIRST THINGS TOMORROW (suggested scoping, ~15 min of machine work)
  1. Commit tonight's work to the branch (figures, Phase 3 outputs, dose regression,
     scan_corpora/route_attribution fixes, docs/FIGURES.md updates).
  2. Push the updated run dir to HF so the three copies match again.
  3. Fold tonight's new numbers into docs/FIGURES.md and analysis/RESULTS.md -- both
     predate the OpenHermes correction, dm_math, train_119K, the route tables and the
     dose regression.
  Then the paper.

WORKING COMMANDS
  .venv/bin/python experiments/analyze_ckpt_axis.py --run experiments/runs/ckpt-axis-v1
  .venv/bin/python experiments/dose_regression.py
  .venv/bin/python src/instella_reasoning/analysis/mathai_f{1,2,3}_*.py \
      experiments/runs/ckpt-axis-v1 paper/figures
  .venv/bin/python experiments/scan_corpora.py --corpus X --hf-path Y --split train \
      --text-field Z --limit N --out outputs/corpus_scan
      # add --hf-glob for OLMoE-style mixed schemas, --revision refs/convert/parquet for
      # script-based datasets, --local-jsonl for a corpus already on disk
  Seeds are 6198 everywhere: item selection, mixture, training, every bootstrap.
