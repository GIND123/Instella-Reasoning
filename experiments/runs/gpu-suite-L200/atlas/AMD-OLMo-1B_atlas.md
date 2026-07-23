## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 200 | 0.010 | 1.000 | 0.010 | [0.000, 0.025] | GAP |
| &nbsp;&nbsp;arithmetic | partial | 2 | 0.000 | 1.000 | 0.000 | [0.000, 0.000] | n/a (too few clusters) |
| &nbsp;&nbsp;arithmetic | clean | 198 | 0.010 | 1.000 | 0.010 | [0.000, 0.025] | GAP |
| commonsense_science | all | 200 | 0.180 | 1.000 | 0.180 | [0.130, 0.235] | GAP |
| &nbsp;&nbsp;commonsense_science | clean | 200 | 0.180 | 1.000 | 0.180 | [0.130, 0.235] | GAP |
| logical_deduction | all | 200 | 0.145 | 1.000 | 0.145 | [0.100, 0.200] | GAP |
| &nbsp;&nbsp;logical_deduction | clean | 200 | 0.145 | 1.000 | 0.145 | [0.100, 0.200] | GAP |
| logical_reading | all | 200 | 0.220 | 1.000 | 0.220 | [0.165, 0.275] | GAP |
| &nbsp;&nbsp;logical_reading | clean | 200 | 0.220 | 1.000 | 0.220 | [0.165, 0.275] | GAP |
| mathematical | all | 200 | 0.020 | 1.000 | 0.020 | [0.005, 0.040] | GAP |
| &nbsp;&nbsp;mathematical | clean | 200 | 0.020 | 1.000 | 0.020 | [0.005, 0.040] | GAP |
| multi_step | all | 200 | 0.070 | 1.000 | 0.070 | [0.040, 0.105] | GAP |
| &nbsp;&nbsp;multi_step | clean | 200 | 0.070 | 1.000 | 0.070 | [0.040, 0.105] | GAP |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld).

