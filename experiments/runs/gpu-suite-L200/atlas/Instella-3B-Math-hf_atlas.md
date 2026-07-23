## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 200 | 0.300 | 1.000 | 0.300 | [0.235, 0.365] | PARTIAL |
| &nbsp;&nbsp;arithmetic | partial | 2 | 0.500 | 1.000 | 0.500 | [0.000, 1.000] | n/a (too few clusters) |
| &nbsp;&nbsp;arithmetic | clean | 198 | 0.298 | 1.000 | 0.298 | [0.232, 0.364] | GAP |
| commonsense_science | all | 200 | 0.575 | 1.000 | 0.575 | [0.505, 0.640] | GENUINE |
| &nbsp;&nbsp;commonsense_science | clean | 200 | 0.575 | 1.000 | 0.575 | [0.505, 0.640] | GENUINE |
| logical_deduction | all | 200 | 0.330 | 1.000 | 0.330 | [0.265, 0.395] | PARTIAL |
| &nbsp;&nbsp;logical_deduction | clean | 200 | 0.330 | 1.000 | 0.330 | [0.265, 0.395] | PARTIAL |
| logical_reading | all | 200 | 0.325 | 1.000 | 0.325 | [0.260, 0.390] | PARTIAL |
| &nbsp;&nbsp;logical_reading | clean | 200 | 0.325 | 1.000 | 0.325 | [0.260, 0.390] | PARTIAL |
| mathematical | all | 200 | 0.090 | 1.000 | 0.090 | [0.050, 0.130] | GAP |
| &nbsp;&nbsp;mathematical | clean | 200 | 0.090 | 1.000 | 0.090 | [0.050, 0.130] | GAP |
| multi_step | all | 200 | 0.385 | 1.000 | 0.385 | [0.320, 0.460] | PARTIAL |
| &nbsp;&nbsp;multi_step | clean | 200 | 0.385 | 1.000 | 0.385 | [0.320, 0.460] | PARTIAL |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld).

