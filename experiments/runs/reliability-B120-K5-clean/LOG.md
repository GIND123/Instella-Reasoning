# Reliability suite — B120-K5 (120 base items, GSM8K, consistency-probed)

The first **genuinely consistency-probed** reliability run (the L200 base suite could only
report accuracy). Generated each model over the GSM8K **variant clusters** (answer-preserving
surface variants + GSM-Symbolic-style numeric variants), so Reliability = Accuracy ×
Consistency is a real memorization-vs-reasoning signal (A2/A3).

## Configuration
- **Benchmark:** gsm8k · **base items:** 120 · **numeric_k:** 5 → 556 variant items
- **Models:** AMD-OLMo-1B, Instella-3B, Instella-3B-Instruct, Instella-3B-Math-hf (556 generations each)
- **Contamination:** full Instella-GSM8K-synthetic (1,367,882 docs) → 451 partial hits
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (96 GB); transformers 4.x (<5); seed per suite default
- Raw `generations/` and the 1.45 GB `contamination/synthetic_corpus.jsonl` are gitignored / not committed.

## Precision note (differs from the script default — for the record)
Run **fully in bf16**, not the script's mixed bf16(Instruct)/4-bit(others). Reason:
bitsandbytes 4-bit produces ~90%+ degenerate output on this Blackwell GPU (sm_120 / CUDA 12.8);
bf16 is both correct and higher-precision, and the 96 GB VRAM makes quantization unnecessary.
The degenerate hard-gate was relaxed (`--fail-degenerate 1.0`) so the weak 1B base model's
naturally-loopy output warns instead of aborting — it does not change any generated text or score.

## Results — Reliability = Accuracy × Consistency

| Model | Accuracy | Consistency | Reliability |
|---|---:|---:|---:|
| AMD-OLMo-1B (1B base) | 0.007 | 0.514 | 0.003 |
| Instella-3B (base) | 0.503 | 0.690 | 0.431 |
| **Instella-3B-Instruct** | **0.659** | **0.785** | **0.588** |
| Instella-3B-Math | 0.227 | 0.492 | 0.130 |

- **A5 scale (OLMo-1B → Instella-3B):** reliability 0.004 → 0.410 (Δ +0.406), `amplified`.
- **A6 RL/math post-training (Instruct → Math):** reliability 0.590 → 0.129 (Δ **−0.461**),
  `regression` — both accuracy (−0.434) and consistency (−0.289) fall. Not math-domain-specific.
- **Instruction tuning** yields the only model that is both accurate and consistent (cons 0.785).

## Caveats
- Instella-3B-Math showed 26% degeneracy (assembled HF checkpoint); part of its low score is
  answer-extraction noise, not pure capability loss — spot-check before citing as a clean finding.
- Contamination labels are `partial` retrieval-similarity candidates, not proof of memorization
  (see `report.md` notes). A1 gap is a triage signal pending manual/gradient validation.
