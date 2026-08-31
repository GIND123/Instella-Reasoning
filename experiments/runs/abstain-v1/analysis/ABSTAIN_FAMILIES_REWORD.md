# Premise verification across families, under a reworded licence

Identical prompt licensing `#### unanswerable`, identical decoding. Pruned items are underdetermined so declining is correct; parent problems are answerable so declining is an error. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | abstains, pruned | abstains, answerable | discrimination | 95% CI | balanced accuracy |
|---|--:|--:|--:|---|--:|
| Instella 3B Instruct | 0.444 | 0.006 | **+43.77 pp** | [+41.98, +45.63] | 0.719 |
| OLMo-2-1B Instruct | 0.283 | 0.191 | **+9.17 pp** | [+6.85, +11.36] | 0.546 |
| Qwen2.5-1.5B Instruct | 0.460 | 0.061 | **+39.83 pp** | [+37.80, +41.98] | 0.699 |
| Qwen2.5-3B Instruct | 0.641 | 0.188 | **+45.33 pp** | [+42.91, +47.76] | 0.727 |

A large positive discrimination means abstention tracks whether the problem is actually answerable. A value near zero would mean the rate measures willingness to use the offered option and nothing more.
