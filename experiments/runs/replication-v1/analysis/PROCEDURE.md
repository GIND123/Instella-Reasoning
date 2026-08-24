# Procedure reproduction against dose

Fraction of the parent's intermediate calculator results appearing in the generation, excluding the final answer. Difference in differences against the never injected control, each dose measured against the 0x arm. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| model | dose | injected | held out | gap | 95% CI | vs 0x |
|---|--:|--:|--:|--:|---|--:|
| Instella 3B | 0x | 0.3005 | 0.2898 | +0.00 pp | reference | - |
| Instella 3B | 1x | 0.2909 | 0.2767 | +1.42 pp | [-3.55, +6.47] | +0.34 pp |
| Instella 3B | 4x | 0.3002 | 0.2822 | +1.80 pp | [-3.18, +6.97] | +0.73 pp |
| Instella 3B | 16x | 0.3217 | 0.2832 | +3.85 pp | [-1.12, +8.96] | +2.78 pp |
| Instella 3B | 64x | 0.3474 | 0.2822 | +6.52 pp | [+1.13, +12.06] | +5.44 pp |
| OLMo-2-1B | 0x | 0.2671 | 0.2579 | +0.00 pp | reference | - |
| OLMo-2-1B | 1x | 0.2577 | 0.2625 | -0.48 pp | [-5.43, +4.45] | -1.40 pp |
| OLMo-2-1B | 4x | 0.2610 | 0.2570 | +0.40 pp | [-4.60, +5.54] | -0.52 pp |
| OLMo-2-1B | 16x | 0.2514 | 0.2397 | +1.18 pp | [-3.70, +5.96] | +0.26 pp |
| OLMo-2-1B | 64x | 0.2448 | 0.2387 | +0.62 pp | [-4.27, +5.55] | -0.30 pp |
| Qwen2.5-1.5B | 0x | 0.3055 | 0.2833 | +0.00 pp | reference | - |
| Qwen2.5-1.5B | 1x | 0.2970 | 0.2965 | +0.05 pp | [-4.94, +5.17] | -2.17 pp |
| Qwen2.5-1.5B | 4x | 0.2994 | 0.2842 | +1.52 pp | [-3.64, +6.78] | -0.70 pp |
| Qwen2.5-1.5B | 16x | 0.3045 | 0.2740 | +3.05 pp | [-1.77, +8.22] | +0.83 pp |
| Qwen2.5-1.5B | 64x | 0.3062 | 0.2808 | +2.54 pp | [-2.45, +7.59] | +0.33 pp |
