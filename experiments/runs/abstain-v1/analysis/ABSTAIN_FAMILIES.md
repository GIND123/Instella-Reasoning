# Premise verification across model families

Identical prompt licensing `#### unanswerable`, identical decoding. Pruned items are underdetermined so declining is correct; parent problems are answerable so declining is an error. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| checkpoint | abstains, pruned | abstains, answerable | discrimination | 95% CI | balanced accuracy |
|---|--:|--:|--:|---|--:|
| OLMo-2-1B Instruct | 0.439 | 0.292 | **+14.68 pp** | [+12.13, +17.23] | 0.573 |
| Qwen2.5-1.5B Instruct | 0.609 | 0.090 | **+51.88 pp** | [+49.59, +54.10] | 0.759 |

A large positive discrimination means abstention tracks whether the problem is actually answerable. A value near zero would mean the rate measures willingness to use the offered option and nothing more.
