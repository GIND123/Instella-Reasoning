## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 200 | 0.010 | 1.000 | 0.010 | [0.000, 0.025] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;arithmetic | partial | 2 | 0.000 | 1.000 | 0.000 | [0.000, 0.000] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;arithmetic | clean | 198 | 0.010 | 1.000 | 0.010 | [0.000, 0.025] | ACCURACY-ONLY (consistency untested) |
| commonsense_science | all | 200 | 0.180 | 1.000 | 0.180 | [0.130, 0.235] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;commonsense_science | clean | 200 | 0.180 | 1.000 | 0.180 | [0.130, 0.235] | ACCURACY-ONLY (consistency untested) |
| logical_deduction | all | 200 | 0.145 | 1.000 | 0.145 | [0.100, 0.200] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;logical_deduction | clean | 200 | 0.145 | 1.000 | 0.145 | [0.100, 0.200] | ACCURACY-ONLY (consistency untested) |
| logical_reading | all | 200 | 0.220 | 1.000 | 0.220 | [0.165, 0.275] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;logical_reading | clean | 200 | 0.220 | 1.000 | 0.220 | [0.165, 0.275] | ACCURACY-ONLY (consistency untested) |
| mathematical | all | 200 | 0.020 | 1.000 | 0.020 | [0.005, 0.040] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;mathematical | clean | 200 | 0.020 | 1.000 | 0.020 | [0.005, 0.040] | ACCURACY-ONLY (consistency untested) |
| multi_step | all | 200 | 0.070 | 1.000 | 0.070 | [0.040, 0.105] | ACCURACY-ONLY (consistency untested) |
| &nbsp;&nbsp;multi_step | clean | 200 | 0.070 | 1.000 | 0.070 | [0.040, 0.105] | ACCURACY-ONLY (consistency untested) |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld), ACCURACY-ONLY (no answer-preserving variants generated, so consistency is trivially 1.0 and reliability == accuracy — the memorisation signal is untested for this cell).

