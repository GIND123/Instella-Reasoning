# Data Manifests

The pipeline uses JSONL files so large runs can be sharded and resumed.

## Benchmark JSONL

Required fields:

```json
{"id":"gsm_0001","prompt":"...","answer":"42"}
```

Recommended fields:

```json
{
  "id": "gsm_0001_irrelevant_clause",
  "parent_id": "gsm_0001",
  "variant_type": "irrelevant_clause",
  "prompt": "...",
  "answer": "42",
  "metadata": {"benchmark": "gsm8k", "skill": "arithmetic"}
}
```

## Corpus JSONL

Required fields:

```json
{"id":"doc_0001","text":"..."}
```

Recommended fields:

```json
{
  "id": "olmoe_mix_0924_openwebmath_0001",
  "source": "allenai/OLMoE-mix-0924",
  "text": "...",
  "metadata": {
    "license": "ODC-BY-1.0",
    "split": "train",
    "shard": "open-web-math/part-00"
  }
}
```

## Generation JSONL

```json
{"benchmark_id":"gsm_0001","completion":"... #### 42","model":"amd/Instella-3B-Instruct"}
```
