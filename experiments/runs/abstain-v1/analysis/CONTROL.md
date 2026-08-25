# Abstention, with the answerable control

Identical prompt licensing `#### unanswerable`, identical decoding. Pruned items are underdetermined so declining is correct; parent problems are answerable so declining is an error. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | abstains, pruned | abstains, answerable | discrimination | 95% CI | excludes 0 |
|---|--:|--:|--:|---|---|
| Stage 1 | 0.043 | 0.025 | **+1.84 pp** | [+0.97, +2.70] | **yes** |
| Instella 3B | 0.110 | 0.008 | **+10.10 pp** | [+8.94, +11.30] | **yes** |
| SFT | 0.416 | 0.035 | **+38.08 pp** | [+36.06, +40.02] | **yes** |
| Instruct | 0.538 | 0.011 | **+52.63 pp** | [+50.77, +54.57] | **yes** |

A large positive discrimination means abstention tracks whether the problem is actually answerable. A value near zero would mean the rate measures willingness to use the offered option and nothing more.
