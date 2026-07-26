## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 120 | 0.661 | 0.786 | 0.590 | [0.519, 0.661] | GENUINE |
| &nbsp;&nbsp;arithmetic | partial | 33 | 0.653 | 0.783 | 0.581 | [0.433, 0.712] | GENUINE |
| &nbsp;&nbsp;arithmetic | clean | 87 | 0.663 | 0.787 | 0.593 | [0.511, 0.679] | GENUINE |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld), ACCURACY-ONLY (no answer-preserving variants generated, so consistency is trivially 1.0 and reliability == accuracy — the memorisation signal is untested for this cell).

