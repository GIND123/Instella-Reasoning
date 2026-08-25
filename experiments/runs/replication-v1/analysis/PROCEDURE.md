# Procedure reproduction against dose

Fraction of the parent's intermediate calculator results appearing in the generation, excluding the final answer. Difference in differences against the never injected control, each dose measured against the 0x arm. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| model | dose | injected | held out | vs 0x | 95% CI | excludes 0 |
|---|--:|--:|--:|--:|---|---|
| Instella 3B | 0x | 0.3005 | 0.2898 | - | reference | no |
| Instella 3B | 1x | 0.2909 | 0.2767 | +0.34 pp | [-2.86, +3.67] | no |
| Instella 3B | 4x | 0.3002 | 0.2822 | +0.73 pp | [-2.27, +3.90] | no |
| Instella 3B | 16x | 0.3217 | 0.2832 | +2.78 pp | [-0.79, +6.42] | no |
| Instella 3B | 64x | 0.3474 | 0.2822 | +5.44 pp | [+1.46, +9.38] | **yes** |
| OLMo-2-1B | 0x | 0.2671 | 0.2579 | - | reference | no |
| OLMo-2-1B | 1x | 0.2577 | 0.2625 | -1.40 pp | [-3.73, +0.93] | no |
| OLMo-2-1B | 4x | 0.2610 | 0.2570 | -0.52 pp | [-3.32, +2.30] | no |
| OLMo-2-1B | 16x | 0.2514 | 0.2397 | +0.26 pp | [-3.02, +3.49] | no |
| OLMo-2-1B | 64x | 0.2448 | 0.2387 | -0.30 pp | [-4.29, +3.61] | no |
| Qwen2.5-1.5B | 0x | 0.3055 | 0.2833 | - | reference | no |
| Qwen2.5-1.5B | 1x | 0.2970 | 0.2965 | -2.17 pp | [-5.05, +0.65] | no |
| Qwen2.5-1.5B | 4x | 0.2994 | 0.2842 | -0.70 pp | [-3.69, +2.32] | no |
| Qwen2.5-1.5B | 16x | 0.3045 | 0.2740 | +0.83 pp | [-2.65, +4.38] | no |
| Qwen2.5-1.5B | 64x | 0.3062 | 0.2808 | +0.33 pp | [-3.49, +4.16] | no |
