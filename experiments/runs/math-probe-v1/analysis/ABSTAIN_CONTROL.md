# Premise verification on the MATH probe

Identical prompt licensing `#### unanswerable`, identical decoding. Pruned items are underdetermined so declining is correct; parent problems are answerable so declining is an error. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | abstains, pruned | abstains, answerable | discrimination | 95% CI | balanced accuracy |
|---|--:|--:|--:|---|--:|
| Stage 1 | 0.071 | 0.038 | **+3.27 pp** | [+1.45, +5.11] | 0.516 |
| Instella 3B | 0.192 | 0.077 | **+11.46 pp** | [+8.57, +14.30] | 0.557 |
| SFT | 0.248 | 0.027 | **+22.15 pp** | [+19.43, +25.17] | 0.611 |
| Instruct | 0.420 | 0.070 | **+35.01 pp** | [+31.61, +38.36] | 0.675 |

A large positive discrimination means abstention tracks whether the problem is actually answerable. A value near zero would mean the rate measures willingness to use the offered option and nothing more.
