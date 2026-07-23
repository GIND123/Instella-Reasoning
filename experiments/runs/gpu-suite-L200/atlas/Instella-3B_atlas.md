## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 200 | 0.520 | 1.000 | 0.520 | [0.455, 0.585] | GENUINE |
| &nbsp;&nbsp;arithmetic | partial | 2 | 1.000 | 1.000 | 1.000 | [1.000, 1.000] | n/a (too few clusters) |
| &nbsp;&nbsp;arithmetic | clean | 198 | 0.515 | 1.000 | 0.515 | [0.449, 0.581] | GENUINE |
| commonsense_science | all | 200 | 0.325 | 1.000 | 0.325 | [0.260, 0.390] | PARTIAL |
| &nbsp;&nbsp;commonsense_science | clean | 200 | 0.325 | 1.000 | 0.325 | [0.260, 0.390] | PARTIAL |
| logical_deduction | all | 200 | 0.275 | 1.000 | 0.275 | [0.210, 0.335] | GAP |
| &nbsp;&nbsp;logical_deduction | clean | 200 | 0.275 | 1.000 | 0.275 | [0.210, 0.335] | GAP |
| logical_reading | all | 200 | 0.315 | 1.000 | 0.315 | [0.250, 0.380] | PARTIAL |
| &nbsp;&nbsp;logical_reading | clean | 200 | 0.315 | 1.000 | 0.315 | [0.250, 0.380] | PARTIAL |
| mathematical | all | 200 | 0.090 | 1.000 | 0.090 | [0.050, 0.130] | GAP |
| &nbsp;&nbsp;mathematical | clean | 200 | 0.090 | 1.000 | 0.090 | [0.050, 0.130] | GAP |
| multi_step | all | 200 | 0.040 | 1.000 | 0.040 | [0.015, 0.070] | GAP |
| &nbsp;&nbsp;multi_step | clean | 200 | 0.040 | 1.000 | 0.040 | [0.015, 0.070] | GAP |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld).

