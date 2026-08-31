# Abstention under a reworded licence, with the answerable control

Identical prompt licensing `#### unanswerable`, identical decoding. Pruned items are underdetermined so declining is correct; parent problems are answerable so declining is an error. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | abstains, pruned | abstains, answerable | discrimination | 95% CI | balanced accuracy |
|---|--:|--:|--:|---|--:|
| Stage 1 | 0.015 | 0.007 | **+0.80 pp** | [+0.29, +1.33] | 0.504 |
| Instella 3B | 0.075 | 0.008 | **+6.64 pp** | [+5.65, +7.70] | 0.533 |
| SFT | 0.343 | 0.008 | **+33.41 pp** | [+31.66, +35.19] | 0.667 |
| Instruct | 0.444 | 0.006 | **+43.77 pp** | [+41.98, +45.63] | 0.719 |

A large positive discrimination means abstention tracks whether the problem is actually answerable. A value near zero would mean the rate measures willingness to use the offered option and nothing more.
