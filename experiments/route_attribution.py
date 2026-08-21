#!/usr/bin/env python
"""Which subset of a mixture carries the contaminated items?

A corpus-level containment number says a benchmark leaked; it does not say *how*. The
route matters because the routes are not equivalent: a dedicated math-instruction subset
leaking GSM8K is a documented design choice, whereas the same problems arriving through a
general chat log means the leak is not something a curator could have filtered by name.

Records, per (source, item-set) pair, how many distinct benchmark items reach containment
>= --threshold in at least one document from that source. Counts are of *items*, not
documents: one item matched by fifty documents from the same source is one leaked item.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from instella_reasoning.records import read_benchmark
from instella_reasoning.text import word_ngrams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="messages")
    ap.add_argument("--source-field", default="source")
    ap.add_argument("--limit", type=int, default=300_000)
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--ngram-n", type=int, default=13)
    ap.add_argument("--items-dir", default="experiments/runs/ckpt-axis-v1/base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sets = {}
    for tag, fname in (("gsm8k_train", "gsm8k_train_all.jsonl"),
                       ("gsm8k_test", "gsm8k_test_all.jsonl"),
                       ("math_train", "math_train_all.jsonl")):
        p = Path(args.items_dir) / fname
        if p.exists():
            sets[tag] = read_benchmark(p)

    keys, grams = [], []
    index: dict[str, list[int]] = collections.defaultdict(list)
    for tag, items in sets.items():
        for it in items:
            g = word_ngrams(it.prompt, args.ngram_n)
            if not g:
                continue
            i = len(keys)
            keys.append((tag, it.id))
            grams.append(g)
            for gram in g:
                index[gram].append(i)
    print(f"[{args.corpus}] indexed {len(keys)} items", flush=True)

    from datasets import load_dataset
    ds = load_dataset(args.hf_path, args.config, split=args.split, streaming=True)

    # (source, item_set) -> set of item ids reaching the threshold in >=1 doc.
    hits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    per_source_rows: collections.Counter = collections.Counter()
    scanned = skipped_short = 0
    for row in ds:
        if scanned >= args.limit:
            break
        scanned += 1
        src = str(row.get(args.source_field) or "unknown")
        per_source_rows[src] += 1
        v = row.get(args.text_field)
        if isinstance(v, list):
            text = "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in v)
        else:
            text = str(v or "")
        if len(text) < 40:
            skipped_short += 1
            continue
        counts: collections.Counter = collections.Counter()
        for gram in word_ngrams(text, args.ngram_n):
            for i in index.get(gram, ()):
                counts[i] += 1
        for i, c in counts.items():
            if c / len(grams[i]) >= args.threshold:
                tag, iid = keys[i]
                hits[(src, tag)].add(iid)
        if scanned % 50_000 == 0:
            print(f"[{args.corpus}] {scanned:,}/{args.limit:,}", flush=True)

    table = []
    for (src, tag), ids in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        table.append({"source": src, "item_set": tag, "items_contained": len(ids),
                      "rows_from_source": per_source_rows[src],
                      "share_of_item_set": round(len(ids) / len(sets[tag]), 6)})
    report = {
        "corpus": args.corpus, "hf_path": args.hf_path, "config": args.config,
        "rows_scanned": scanned, "threshold": args.threshold, "ngram_n": args.ngram_n,
        "wall_clock_sec": round(time.time() - t0, 1),
        "n_sources_seen": len(per_source_rows),
        "coverage_note": "Sampled prefix; counts are lower bounds per source.",
        "routes": table,
    }
    (out / f"{args.corpus}_routes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:1800], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
