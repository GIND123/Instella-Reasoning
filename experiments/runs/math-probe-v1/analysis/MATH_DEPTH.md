# Depth decay on the MATH probe

Slope of declining on premise sentence count, percentage points per sentence, on underdetermined items. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | parents | rows | slope | 95% CI | excludes 0 |
|---|--:|--:|--:|---|---|
| Stage 1 | 818 | 1091 | **-0.20 pp** | [-1.61, +1.54] | no |
| Instella 3B | 818 | 1091 | **-1.89 pp** | [-3.99, +0.23] | no |
| SFT | 818 | 1091 | **-5.56 pp** | [-7.39, -3.91] | **yes** |
| Instruct | 818 | 1091 | **-6.41 pp** | [-8.43, -4.49] | **yes** |

**No control slope is available on MATH.** The answerable control arm carries `n_sentences` as null on every row, so depth cannot be regressed on that arm from the released generations. The GSM8K control slope (−0.11, spanning zero) is the only control evidence for depth decay; the MATH figure stands as an uncontrolled replication of the gradient.
