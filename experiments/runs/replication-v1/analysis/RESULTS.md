# Injection replication on base models outside AMD

Same 200 injected documents, same 200 held-out controls, same fixed 8,388,608 token budget, same seed 6198, same four-shot no-chat-template probe. Only the base checkpoint changes. Exact two-sided McNemar, paired by item.

## Memorisation actually happened

| model | dose | injected share | verbatim after | injected loss after | held-out loss after | controls |
|---|--:|--:|--:|--:|--:|---|
| OLMo-2-1B | 0x | 0.000 | 0.126 | 1.241 | 1.294 | **FAIL** |
| OLMo-2-1B | 64x | 0.268 | 0.727 | 0.322 | 1.373 | pass |
| Qwen2.5-1.5B | 0x | 0.000 | 0.243 | 1.254 | 1.277 | **FAIL** |
| Qwen2.5-1.5B | 64x | 0.310 | 0.989 | 0.047 | 1.465 | pass |

## PRIMARY - difference in differences on answer recall, 64x against 0x

Injected minus held-out, so the 8.4M tokens every arm trains on cancel.

| model | injected delta | held-out delta | difference in differences | 95% CI | excludes 0 |
|---|--:|--:|--:|---|---|
| OLMo-2-1B | +0.28 pp | +0.00 pp | **+0.28 pp** | [-2.34, +2.73] | no |
| Qwen2.5-1.5B | +0.00 pp | +0.00 pp | **+0.00 pp** | [-2.46, +2.52] | no |

## The single differences behind it

| model | set | n | recall 0x | recall 64x | delta | gained | lost | McNemar p |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| OLMo-2-1B | injected | 351 | 0.0285 | 0.0313 | +0.28 pp | 5 | 4 | 1.0000 |
| OLMo-2-1B | heldout | 371 | 0.0162 | 0.0162 | +0.00 pp | 5 | 5 | 1.0000 |
| Qwen2.5-1.5B | injected | 351 | 0.0456 | 0.0456 | +0.00 pp | 6 | 6 | 1.0000 |
| Qwen2.5-1.5B | heldout | 371 | 0.0296 | 0.0296 | +0.00 pp | 7 | 7 | 1.0000 |

Read the primary table only where controls pass. A model that did not memorise the dosed documents cannot inform a claim about what memorisation does to recall.
