"""Recompute 13-gram containment against the corpora each checkpoint actually consumed.

Motivated by two corrections from J. Liu (AMD, personal communication, 2026-08-19):

1. `amd/Instella-GSM8K-synthetic` ships a `train` split (1,367,882 rows) and a
   `train_119K` split (119,014 rows). Only `train_119K` was used in Instella-3B stage-2
   pre-training. Containment measured against `train` over-counts exposure.

2. `nvidia/OpenMathInstruct-2` enters at the SFT stage and is inherited by Instruct and
   Math. It is generated from the GSM8K and MATH train sets, so containment against the
   stage-2 corpus alone does not describe GSM8K-derived exposure for post-training
   checkpoints.

The n-gram rule, normalisation and rolling hash are identical to
``instella_reasoning.datasets.splits`` so the outputs are directly comparable to the
published containment numbers.

Usage:
    python scripts/verify_stage2_corpus.py --corpus stage2      # train_119K
    python scripts/verify_stage2_corpus.py --corpus pool        # full train split
    python scripts/verify_stage2_corpus.py --corpus omi2        # OpenMathInstruct-2 train_1M
    python scripts/verify_stage2_corpus.py --crosstab           # after stage2 + pool
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

NGRAM_N = 13
_MOD = (1 << 61) - 1
_BASE = 1_000_003
_WORD = re.compile(r"[a-z0-9]+")

OUT = Path("outputs/corpus_verification")

CORPORA = {
    "stage2": ("amd/Instella-GSM8K-synthetic", ["data/train_119K-00000-of-00001.parquet"]),
    "pool": (
        "amd/Instella-GSM8K-synthetic",
        [f"data/train-0000{i}-of-00003.parquet" for i in range(3)],
    ),
    "omi2": (
        "nvidia/OpenMathInstruct-2",
        [f"data/train_1M-0000{i}-of-00003.parquet" for i in range(3)],
    ),
}


def normalize_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _token_ids(tokens, vocab: dict[str, int]) -> list[int]:
    out = []
    for tok in tokens:
        tid = vocab.get(tok)
        if tid is None:
            tid = len(vocab) + 1
            vocab[tok] = tid
        out.append(tid)
    return out


def _gram_hashes(ids: list[int], n: int = NGRAM_N):
    if len(ids) < n:
        return
    high = pow(_BASE, n - 1, _MOD)
    h = 0
    for tid in ids[:n]:
        h = (h * _BASE + tid) % _MOD
    yield h
    for i in range(n, len(ids)):
        h = ((h - ids[i - n] * high) * _BASE + ids[i]) % _MOD
        yield h


def load_gsm8k():
    from huggingface_hub import hf_hub_download

    items = []
    for split in ("train", "test"):
        path = hf_hub_download(
            "openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset"
        )
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            items.append((f"{split}_{i}", row["question"]))
    return items


def corpus_texts(repo: str, files: list[str]):
    from huggingface_hub import hf_hub_download

    for f in files:
        path = hf_hub_download(repo, f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        field = "messages" if "messages" in cols else "problem"
        for batch in pf.iter_batches(batch_size=2000, columns=[field]):
            for value in batch.column(field).to_pylist():
                if isinstance(value, list):
                    yield " ".join(m["content"] for m in value)
                else:
                    yield value
        print(f"  scanned {f}", file=sys.stderr, flush=True)


def containment(items, texts) -> dict[str, float]:
    vocab: dict[str, int] = {}
    gram_owners: dict[int, set[str]] = defaultdict(set)
    item_grams: dict[str, set[int]] = {}
    for iid, prompt in items:
        grams = set(_gram_hashes(_token_ids(normalize_tokens(prompt), vocab)))
        item_grams[iid] = grams
        for g in grams:
            gram_owners[g].add(iid)

    matched: dict[str, set[int]] = defaultdict(set)
    n_docs = 0
    for text in texts:
        n_docs += 1
        for g in _gram_hashes(_token_ids(normalize_tokens(text), vocab)):
            owners = gram_owners.get(g)
            if owners:
                for o in owners:
                    matched[o].add(g)
        if n_docs % 200_000 == 0:
            print(f"  ... {n_docs} docs", file=sys.stderr, flush=True)
    print(f"  corpus documents scanned: {n_docs}", file=sys.stderr)
    return {
        iid: (len(matched[iid]) / len(grams) if grams else 0.0)
        for iid, grams in item_grams.items()
    }


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre + half


def summarise(name: str, values: dict[str, float]) -> None:
    for split in ("train", "test"):
        vals = [v for k, v in values.items() if k.startswith(f"{split}_")]
        n = len(vals)
        flagged = sum(1 for v in vals if v >= 0.80)
        low = sum(1 for v in vals if v <= 0.10)
        print(f"\nGSM8K {split} (n={n}) vs {name}")
        print(f"  containment >= 0.80 : {flagged:5d} ({100 * flagged / n:5.2f}%)")
        print(f"  containment <= 0.10 : {low:5d} ({100 * low / n:5.2f}%)")
        print(f"  mean containment    : {sum(vals) / n:.4f}")
        if split == "test":
            print(f"  false-positive rate : {flagged}/{n}, Wilson 95% upper "
                  f"{100 * wilson_upper(flagged, n):.3f}%")


def crosstab() -> None:
    pool = json.loads((OUT / "containment_pool.json").read_text())
    stage2 = json.loads((OUT / "containment_stage2.json").read_text())
    train = [k for k in pool if k.startswith("train_")]

    hi_pool = {k for k in train if pool[k] >= 0.80}
    hi_s2 = {k for k in train if stage2[k] >= 0.80}
    lo_s2 = {k for k in train if stage2[k] <= 0.10}

    print(f"\nGSM8K train items: {len(train)}")
    print(f"high containment vs released pool : {len(hi_pool)}")
    print(f"  still high vs stage-2 corpus    : {len(hi_pool & hi_s2)} "
          f"({100 * len(hi_pool & hi_s2) / len(hi_pool):.1f}%)")
    print(f"  falls to the low arm            : {len(hi_pool & lo_s2)} "
          f"({100 * len(hi_pool & lo_s2) / len(hi_pool):.1f}%)")
    print(f"  falls into the excluded band    : "
          f"{len(hi_pool) - len(hi_pool & hi_s2) - len(hi_pool & lo_s2)}")
    print("\nPLACEBO ARM (generated by the same procedure, never trained on):")
    print(f"  pool >= 0.80 and stage-2 <= 0.10: {len(hi_pool & lo_s2)}")
    print("\nCorrected arms available vs the stage-2 corpus:")
    print(f"  high (>= 0.80): {len(hi_s2)}    low (<= 0.10): {len(lo_s2)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    ap.add_argument("--crosstab", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.crosstab:
        crosstab()
        return
    if not args.corpus:
        ap.error("pass --corpus or --crosstab")

    repo, files = CORPORA[args.corpus]
    items = load_gsm8k()
    print(f"items: {len(items)}", file=sys.stderr)
    values = containment(items, corpus_texts(repo, files))
    (OUT / f"containment_{args.corpus}.json").write_text(json.dumps(values))
    summarise(f"{repo} [{args.corpus}]", values)


if __name__ == "__main__":
    main()
