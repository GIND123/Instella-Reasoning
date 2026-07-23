## Reasoning Reliability Atlas

Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level. The 95% CI is a cluster bootstrap on the cell's mean reliability.

| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| arithmetic | all | 200 | 0.750 | 1.000 | 0.750 | [0.690, 0.810] | GENUINE |
| &nbsp;&nbsp;arithmetic | partial | 2 | 1.000 | 1.000 | 1.000 | [1.000, 1.000] | n/a (too few clusters) |
| &nbsp;&nbsp;arithmetic | clean | 198 | 0.747 | 1.000 | 0.747 | [0.682, 0.808] | GENUINE |
| commonsense_science | all | 200 | 0.765 | 1.000 | 0.765 | [0.705, 0.820] | GENUINE |
| &nbsp;&nbsp;commonsense_science | clean | 200 | 0.765 | 1.000 | 0.765 | [0.705, 0.820] | GENUINE |
| logical_deduction | all | 200 | 0.485 | 1.000 | 0.485 | [0.415, 0.555] | PARTIAL |
| &nbsp;&nbsp;logical_deduction | clean | 200 | 0.485 | 1.000 | 0.485 | [0.415, 0.555] | PARTIAL |
| logical_reading | all | 200 | 0.535 | 1.000 | 0.535 | [0.465, 0.605] | GENUINE |
| &nbsp;&nbsp;logical_reading | clean | 200 | 0.535 | 1.000 | 0.535 | [0.465, 0.605] | GENUINE |
| mathematical | all | 200 | 0.195 | 1.000 | 0.195 | [0.145, 0.255] | GAP |
| &nbsp;&nbsp;mathematical | clean | 200 | 0.195 | 1.000 | 0.195 | [0.145, 0.255] | GAP |
| multi_step | all | 200 | 0.060 | 1.000 | 0.060 | [0.030, 0.095] | GAP |
| &nbsp;&nbsp;multi_step | clean | 200 | 0.060 | 1.000 | 0.060 | [0.030, 0.095] | GAP |

**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent), n/a (fewer than 3 clusters — verdict withheld).

