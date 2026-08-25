# Premise verification on the MATH probe

Identical prompt licensing `#### unanswerable`, identical decoding. Pruned items are underdetermined so declining is correct; parent problems are answerable so declining is an error. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | abstains, pruned | abstains, answerable | discrimination | 95% CI | balanced accuracy |
|---|--:|--:|--:|---|--:|
| Stage 1 | 0.071 | 0.038 | **+3.27 pp** | [+1.45, +5.11] | 0.516 |
| Instella 3B | 0.192 | 0.077 | **+11.46 pp** | [+8.57, +14.30] | 0.557 |

A large positive discrimination means abstention tracks whether the problem is actually answerable. A value near zero would mean the rate measures willingness to use the offered option and nothing more.
