## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 120 | 0.483 | 0.686 | 0.410 | [0.341, 0.479] | PARTIAL |
| &nbsp;&nbsp;arithmetic | partial | 33 | 0.583 | 0.706 | 0.514 | [0.377, 0.657] | GENUINE |
| &nbsp;&nbsp;arithmetic | clean | 87 | 0.445 | 0.679 | 0.370 | [0.294, 0.453] | PARTIAL |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld), ACCURACY-ONLY (no answer-preserving variants generated, so consistency is trivially 1.0 and reliability == accuracy — the memorisation signal is untested for this cell).

