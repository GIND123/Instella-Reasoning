# Paper-writing briefing — context for whoever/whatever helps draft the papers

This is a reference document, not a source of new facts. It summarizes what the two paper
drafts contain and how the underlying experiment works, so a reader (human or AI) can orient
quickly. For exact numbers, tables, and citations, use the source files linked at the bottom —
this briefing may simplify or omit detail.

---

## The one idea everything rests on: verified seen vs. unseen

Normally you cannot know what a model was trained on — closed labs (OpenAI, Google, etc.) do
not publish their training data, so "did the model memorize this benchmark item?" is
unanswerable. AMD's Instella family is different: AMD publishes the exact training corpus. That
makes the question answerable by direct measurement instead of guesswork.

- **Seen** = a GSM8K **train** problem provably present in Instella's stage-2 training corpus
  (`amd/Instella-GSM8K-synthetic`, 1.45 GB) — measured, not inferred.
- **Unseen** = a GSM8K **test** problem provably absent from that corpus.
- Items with ambiguous overlap are **excluded from both groups** rather than guessed into one.

**Why this matters for the study:** if a model scores much higher on seen items than unseen
ones, that alone doesn't prove memorization — seen items might just be inherently easier. So
each problem is also **numerically perturbed** (same structure, different numbers). A model that
memorized the answer fails once the numbers change; a model that reasoned still gets it right.
This "does the seen-item advantage survive perturbation?" question, formalized as a
difference-in-differences (DiD), is the core instrument for both papers.

---

## Two papers, two different roles for the models

| | MATH-AI paper | JUDGe paper |
|---|---|---|
| What the models do | **Answer** math problems (subjects being studied) | **Grade** other models' answers (judges/instruments) |
| Core question | Does Instella's seen-item accuracy advantage collapse under perturbation? | Does an LLM judge grade memorized problems worse — and why? |
| Headline result | **Null** — the memorization DiD is flat (~0, every CI spans zero) across all 4 training checkpoints | **Positive** — judges grade memorized problems substantially worse, but *only* when not given the reference answer |
| Models involved | 4 Instella checkpoints (stage1 → stage2 → sft → instruct) | 3 judges of rising capability: Instella-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, grading Instella's answers |

**Do not conflate the two.** In the JUDGe paper, Qwen is never being evaluated as a "better
Instella" — it's a measuring instrument, brought in only because Instella-3B is too weak
(near chance) to be a trustworthy grader. The comparison across judges is a **capability
ladder**, used to prove the effect isn't just a weak-judge fluke.

---

## Section 2 (JUDGe paper) — "Verified Membership" — what it must establish

Section 2's only job is to prove the seen/unseen labels are trustworthy before anything is
built on top of them.

**2.1 — Containment method.** Each problem's text is lowercased, tokenized, and broken into
every window of 13 consecutive tokens ("13-grams"). Containment = the fraction of those windows
that appear anywhere in the 1.45 GB corpus. Threshold: ≥0.80 → seen, ≤0.10 → unseen, between →
excluded from both arms. Computed in one streaming pass (rolling hash, linear cost).

**2.2 — Detector false-positive rate.** Because the corpus is *derived from* GSM8K train, every
GSM8K test item is a guaranteed true negative — it cannot possibly be in the corpus. This lets
the detector's error rate be *measured* instead of argued: at the 0.80 threshold, **0 of 1,000**
known negatives are wrongly flagged (0.0% false positives); a looser any-overlap rule gives
0.3%. This preempts the reviewer objection "your split is just noisy."

**2.3 — A bug found and fixed.** GSM8K train and test share the same ID scheme
(`gsm8k_00001`, ...), so loading both without namespacing caused ID collisions — containment
scores exceeded 1.0 (impossible), reaching 7.78 in one case. Fixed by namespacing IDs by split.
Worth reporting honestly; it's a trap anyone replicating this design would also hit.

---

## Writing rules (from the team's own instructions — treat as binding)

1. **Every claim needs a source.** Either a citation to prior published work, or a pointer to
   the paper's own table/figure. No unsupported claims. ("Everything that is not referred [to a
   citation] in the paper is assumed to be proposed by us" — so anything uncited is an implicit
   claim of originality; be certain before leaving something uncited.)
2. **Page budgets are body-only.** MATH-AI = 4 pages, JUDGe = 6 pages. References and technical
   appendices do **not** count against the limit (NeurIPS template rule).
3. **No standalone "Related Work" section in the skeleton.** Related work is folded into the
   Introduction as citations; the full literature discussion goes in a free (uncounted)
   appendix. This is a deliberate space-saving choice — preserve it.
4. **Do not rewrite the Abstract or Introduction** — treat as already reviewed/locked unless
   explicitly asked to touch them (e.g., a broken reference).
5. **Write with page limits in mind from the start** — don't draft long and cut later; keep each
   subsection tight to its allotted space.
6. **Don't rush.** Understand each number/mechanism before writing the sentence that reports it.

---

## Known open items (not yet resolved as of this briefing)

- **JUDGe §3.4** (balanced accuracy vs. Cohen's kappa): the draft currently presents the
  "kappa paradox" (chance-corrected agreement is misleading under unequal base rates) as an
  in-house finding, but it is a known statistical phenomenon — needs a citation (e.g.,
  Feinstein & Cicchetti on kappa's paradoxes) or explicit framing as "a known issue we
  encountered," not a novel contribution.
- **JUDGe §6 limitation**: the "sft" checkpoint shows no membership gap in either judging
  condition, unlike stage2 and instruct, and the reason is not yet explained. Flagged as an
  open limitation in the draft; a reviewer will likely ask about it.
- **Anonymization**: both drafts currently reference the real GitHub repo and HF dataset
  (`GOVINDFROM/Instella-Reasoning`) by name in their appendices. These identify the authors and
  must be replaced with an anonymous mirror before submission (double-blind review).

---

## Source files for exact numbers, tables, and full text

- `MATHAI_2026.md` — full MATH-AI draft (repo root)
- `JUDGE_2026.md` — full JUDGe draft (repo root)
- `docs/proposal/reasoning-or-remembering-proposal.md` — original project proposal (for checking
  what was originally scoped vs. what was added later)
- `docs/AUDIT_2026-07-26.md` — the audit that found the original reliability-suite results were
  largely artifacts, and motivated the redesign into the verified seen/unseen DiD approach used
  in both current papers
- `docs/figures/` — all referenced figures (f1–f8 for MATH-AI, j1–j5 and s1–s4 shared/supplementary)
- Raw data/analysis artifacts: `experiments/runs/fullscale-S250-v2/` and the mirrored copy at
  the HF dataset `GOVINDFROM/Instella-Reasoning` (same path)
