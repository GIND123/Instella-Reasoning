# Exposure of GSM8K under the two corpora AMD actually consumed

Instella-3B stage two consumed `amd/Instella-GSM8K-synthetic[train_119K]` (119,014 rows).
Instella-MoE-16B-A3B consumed the full `train` split (1,367,882 rows) at long-context
extension phase 2. The same benchmark item therefore carries different exposure in the two
model families, and a claim about one corpus does not transfer to the other.

## Why the earlier comparison was withdrawn

An initial pass compared `outputs/corpus_scan/train119k_summary.json` against a fresh scan
of the full pool. It reported 93.1% of items at containment >=0.1 for `train_119K` and
77.1% for the full pool. `train_119K` is a subset of `train`, so a per-document maximum
cannot fall when the corpus grows: the pair was impossible and one of the two was measuring
something else.

Cause: the two scans read different text. The older one consumed a locally materialised
JSONL carrying a flattened `text` field; the newer one streams the Hub dataset and joins
the `messages` field. Different characters in, different 13-grams out. The item sets also
differed (722 parents against 891). Neither scan was wrong on its own terms; they were not
comparable.

Both corpora were therefore rescanned through one code path
(`experiments/modal_boost.py::moe_containment`), on identical items, with the per-document
maximum used everywhere else in the paper.

## Result

13-gram containment, maximum over individual documents, one row per parent problem.

### GSM8K train parents (n = 891)

| containment | `train_119K` (Instella-3B) | full `train` (Instella-MoE) | ratio |
|---|--:|--:|--:|
| >= 0.999 | 86 (9.65%) | 165 (18.52%) | 1.92x |
| >= 0.9 | 91 (10.21%) | 173 (19.42%) | 1.90x |
| >= 0.8 | 112 (12.57%) | 208 (23.34%) | 1.86x |
| >= 0.5 | 319 (35.80%) | 457 (51.29%) | 1.43x |
| >= 0.3 | 461 (51.74%) | 580 (65.10%) | 1.26x |
| >= 0.1 | 597 (67.00%) | 687 (77.10%) | 1.15x |
| median | 0.333 | 0.500 | |

Every band is monotone, as a subset relation requires. An 11.5x larger corpus buys 1.92x
more verbatim-contained train items, so coverage saturates well before the corpus does.

### GSM8K test parents (n = 888)

| containment | `train_119K` | full `train` |
|---|--:|--:|
| >= 0.5 | 0 (0.00%) | 0 (0.00%) |
| >= 0.3 | 1 (0.11%) | 1 (0.11%) |
| max observed | 0.333 | 0.487 |
| median | 0.000 | 0.000 |

No test item reaches half containment against either corpus, and the single item above 0.3
shares fewer than half its 13-grams with any document in 1.37M rows. The released synthetic
corpus is train-derived and does not carry the GSM8K test split, which is what licenses
using the train/test split as an exposure contrast.

## Provenance

| | `train_119K` | full `train` |
|---|---|---|
| rows scanned | 119,014 | 1,367,882 |
| artefact | `outputs/corpus_scan/containment_train_119K_matched.json` | `outputs/corpus_scan/containment_train_full_matched.json` |

`outputs/corpus_scan/train119k_summary.json` predates this and uses the other extraction
path; it is retained for provenance and must not be compared against a Hub-streamed scan.
