# Injection replication on base models outside AMD

Same 200 injected documents, same 200 held-out controls, same fixed 8,388,608 token budget, same seed 6198, same four-shot no-chat-template probe. Only the base checkpoint changes.

## Memorisation against dose

| model | dose | injected share | verbatim after | injected loss after | held-out loss after | controls |
|---|--:|--:|--:|--:|--:|---|
| OLMo-2-1B | 0x | 0.000 | 0.126 | 1.241 | 1.294 | fail (0x, by design) |
| OLMo-2-1B | 1x | 0.004 | 0.128 | 1.098 | 1.183 | pass |
| OLMo-2-1B | 4x | 0.017 | 0.172 | 1.006 | 1.156 | pass |
| OLMo-2-1B | 16x | 0.067 | 0.207 | 0.769 | 1.152 | pass |
| OLMo-2-1B | 64x | 0.268 | 0.727 | 0.322 | 1.373 | pass |
| Qwen2.5-1.5B | 0x | 0.000 | 0.243 | 1.254 | 1.277 | fail (0x, by design) |
| Qwen2.5-1.5B | 1x | 0.005 | 0.250 | 0.851 | 0.979 | pass |
| Qwen2.5-1.5B | 4x | 0.019 | 0.255 | 0.693 | 1.001 | pass |
| Qwen2.5-1.5B | 16x | 0.077 | 0.406 | 0.255 | 1.264 | pass |
| Qwen2.5-1.5B | 64x | 0.310 | 0.989 | 0.047 | 1.465 | pass |

## PRIMARY - difference in differences on answer recall, each dose against 0x

Injected minus held-out, so the 8.4M tokens every arm trains on cancel. Cluster bootstrap over parent problems, 4000 draws, seed 6198.

| model | dose | injected delta | held-out delta | difference in differences | 95% CI | excludes 0 |
|---|--:|--:|--:|--:|---|---|
| OLMo-2-1B | 1x | -0.85 pp | -0.54 pp | **-0.32 pp** | [-2.38, +1.67] | no |
| OLMo-2-1B | 4x | +0.00 pp | +0.00 pp | **+0.00 pp** | [-1.90, +1.77] | no |
| OLMo-2-1B | 16x | +1.43 pp | +0.81 pp | **+0.62 pp** | [-1.56, +2.83] | no |
| OLMo-2-1B | 64x | +0.28 pp | +0.00 pp | **+0.28 pp** | [-2.34, +2.73] | no |
| Qwen2.5-1.5B | 1x | +0.28 pp | +0.00 pp | **+0.28 pp** | [-1.99, +2.60] | no |
| Qwen2.5-1.5B | 4x | +1.14 pp | -0.81 pp | **+1.95 pp** | [-0.85, +4.84] | no |
| Qwen2.5-1.5B | 16x | +0.57 pp | -0.27 pp | **+0.84 pp** | [-2.13, +3.71] | no |
| Qwen2.5-1.5B | 64x | +0.00 pp | +0.00 pp | **+0.00 pp** | [-2.46, +2.52] | no |

Read the primary table only where controls pass. A model that did not memorise the dosed documents cannot inform a claim about what memorisation does to recall, which is why the 0x arm is the reference and not a row.
