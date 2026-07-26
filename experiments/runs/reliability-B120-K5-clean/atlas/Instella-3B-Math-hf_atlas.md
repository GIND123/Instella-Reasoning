## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 120 | 0.227 | 0.497 | 0.129 | [0.096, 0.167] | GAP |
| &nbsp;&nbsp;arithmetic | partial | 33 | 0.229 | 0.476 | 0.131 | [0.065, 0.217] | GAP |
| &nbsp;&nbsp;arithmetic | clean | 87 | 0.226 | 0.504 | 0.129 | [0.092, 0.168] | GAP |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld), ACCURACY-ONLY (no answer-preserving variants generated, so consistency is trivially 1.0 and reliability == accuracy — the memorisation signal is untested for this cell).

