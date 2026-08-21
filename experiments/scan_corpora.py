#!/usr/bin/env python
"""Scan one training corpus for containment of benchmark items. One process per corpus.

Phase 3 of the study, and under a 4-page limit this is appendix material: it documents
what the checkpoint-axis design was built to work around, rather than being the result.

Method. Containment is the fraction of an item's word 13-grams that appear in a single
corpus document (`text.word_ngram_overlap`), maximised over documents. Computing that
naively is |items| x |docs| comparisons. Instead one inverted index is built from
13-gram -> item, and each streamed document is looked up once: the per-document match
counts fall out of that single pass, so cost is linear in corpus size and independent of
how many benchmark item sets are indexed. Adding MATH alongside GSM8K is therefore free,
which is why all three item sets are scanned together rather than in separate passes.

Sampling. Several Stage-2 corpora are far too large to scan whole, so --limit takes a
streamed prefix and the manifest records exactly how many rows were seen. A sampled scan
bounds containment from below: it can prove an item IS present, never that it is absent.
Every summary therefore reports coverage, and "0 found" must be read as "none in the rows
scanned" — the distinction J1 insisted on for OLMoE-mix.

  python experiments/scan_corpora.py --corpus tulu3 --hf-path allenai/tulu-3-sft-mixture \
      --split train --text-field messages --limit 300000 --out outputs/corpus_scan
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from instella_reasoning.records import read_benchmark
from instella_reasoning.text import word_ngrams

BANDS = [0.999, 0.90, 0.80, 0.50, 0.30, 0.10]


def _msg_text(m) -> str:
    """Pull the payload out of one chat message.

    ShareGPT-style corpora use `value`, OpenAI-style use `content`, and some use `text`.
    Reading only `content` silently yields an empty string on the others - which is not a
    parse error, just a document that then falls below the length floor and is skipped.
    OpenHermes-2.5 scanned as 0% contaminated that way, across 300,000 rows.
    """
    if not isinstance(m, dict):
        return str(m)
    for k in ("content", "value", "text"):
        if m.get(k):
            return str(m[k])
    return ""


def row_text(row: dict, field: str) -> str:
    """Corpora disagree on where the text lives; normalise the shapes we actually meet."""
    v = row.get(field)
    if isinstance(v, list):  # chat-style, but the payload key is not standardised
        return "\n".join(_msg_text(m) for m in v)
    if v:
        return str(v)
    for k in ("text", "content", "problem", "question", "conversations", "messages"):
        alt = row.get(k)
        if isinstance(alt, list):
            return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in alt)
        if alt:
            return str(alt)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="short name for output files")
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--local-jsonl", default=None,
                    help="Scan a JSONL already on disk instead of streaming from the Hub. "
                         "Used for the Stage-2 corpus as actually consumed (train_119K), "
                         "which was materialised locally by the repo loader.")
    ap.add_argument("--revision", default=None,
                    help="HF revision. 'refs/convert/parquet' reaches the auto-generated\n                         parquet mirror of a script-based dataset, which datasets>=3 can\n                         no longer load from its original loading script.")
    ap.add_argument("--hf-glob", default=None,
                    help="Repo-file prefix, e.g. 'data/open-web-math/'. Streams the shards "
                         "directly as gzipped JSON lines instead of going through "
                         "`datasets`. OLMoE-mix needs this: its shards disagree on columns "
                         "even inside one subset, so any attempt to unify them into a "
                         "single table raises CastError before a byte is scanned.")
    ap.add_argument("--data-files", default=None,
                    help="glob within the repo. Needed for mixtures whose shards carry "
                         "different schemas: streaming the whole repo then raises CastError "
                         "because one table cannot cover them all (OLMoE-mix does this).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--limit", type=int, default=300_000)
    ap.add_argument("--ngram-n", type=int, default=13)
    ap.add_argument("--items-dir", default="experiments/runs/ckpt-axis-v1/base")
    ap.add_argument("--math-limit", type=int, default=7500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- item sets -------------------------------------------------------------
    sets: dict[str, list] = {}
    for tag, path in (("gsm8k_train", "gsm8k_train_all.jsonl"),
                      ("gsm8k_test", "gsm8k_test_all.jsonl")):
        p = Path(args.items_dir) / path
        if p.exists():
            sets[tag] = read_benchmark(p)
    try:
        from instella_reasoning.datasets.loaders import load_benchmark_to_jsonl
        mp = Path(args.items_dir) / "math_train_all.jsonl"
        if not mp.exists():
            load_benchmark_to_jsonl("math", mp, split="train", limit=args.math_limit)
        sets["math_train"] = read_benchmark(mp)
    except Exception as exc:  # noqa: BLE001 - MATH is a bonus set, never a blocker
        print(f"[{args.corpus}] MATH unavailable ({type(exc).__name__}: {exc}); continuing", flush=True)

    keys: list[tuple[str, str]] = []
    grams: list[set[str]] = []
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
    print(f"[{args.corpus}] indexed {len(keys)} items from {list(sets)} "
          f"({len(index):,} distinct {args.ngram_n}-grams) in {time.time()-t0:.0f}s", flush=True)

    # ---- stream the corpus ------------------------------------------------------
    best = [0.0] * len(keys)
    scanned = skipped_short = 0
    def _raw_shard_stream(repo: str, prefix: str):
        """Yield rows from gzipped JSON-line shards, one file at a time.

        Each shard is internally consistent even when the set of shards is not, so
        reading them independently sidesteps the schema unification entirely.
        """
        import gzip
        from huggingface_hub import HfApi, hf_hub_download
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                 if f.startswith(prefix) and f.endswith((".json.gz", ".jsonl.gz"))]
        files.sort()
        print(f"[{args.corpus}] {len(files)} shards under {prefix}", flush=True)
        for fname in files:
            local = hf_hub_download(repo, fname, repo_type="dataset")
            with gzip.open(local, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def _local_stream(path: str):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    try:
        if args.local_jsonl:
            ds = _local_stream(args.local_jsonl)
        elif args.hf_glob:
            ds = _raw_shard_stream(args.hf_path, args.hf_glob)
        else:
            from datasets import load_dataset
            if args.data_files:
                ds = load_dataset(args.hf_path, data_files=args.data_files,
                                  split=args.split, streaming=True,
                                  revision=args.revision)
            else:
                ds = load_dataset(args.hf_path, args.config, split=args.split,
                                  streaming=True)
    except Exception as exc:  # noqa: BLE001
        (out / f"{args.corpus}_FAILED.json").write_text(
            json.dumps({"corpus": args.corpus, "hf_path": args.hf_path,
                        "error": f"{type(exc).__name__}: {exc}"}, indent=2), encoding="utf-8")
        print(f"[{args.corpus}] LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2

    t1 = time.time()
    for row in ds:
        if scanned >= args.limit:
            break
        scanned += 1
        text = row_text(row, args.text_field)
        if len(text) < 40:
            skipped_short += 1
            continue
        hits: collections.Counter = collections.Counter()
        for gram in word_ngrams(text, args.ngram_n):
            for i in index.get(gram, ()):
                hits[i] += 1
        for i, c in hits.items():
            v = c / len(grams[i])
            if v > best[i]:
                best[i] = v
        if scanned % 25_000 == 0:
            el = time.time() - t1
            print(f"[{args.corpus}] {scanned:,}/{args.limit:,} rows  "
                  f"{scanned/max(el,1e-9)*3600:,.0f}/hr  {el/60:.1f}m", flush=True)

    # ---- summarise --------------------------------------------------------------
    per_set: dict[str, dict] = {}
    for tag in sets:
        idx = [i for i, (t, _) in enumerate(keys) if t == tag]
        vals = sorted((best[i] for i in idx), reverse=True)
        n = len(idx)
        per_set[tag] = {
            "n_items": n,
            "bands": {f">={b}": sum(1 for v in vals if v >= b) for b in BANDS},
            "band_frac": {f">={b}": round(sum(1 for v in vals if v >= b) / n, 6) for b in BANDS},
            "max": round(vals[0], 6) if vals else None,
            "median": round(vals[n // 2], 6) if vals else None,
        }
    short_frac = skipped_short / max(scanned, 1)
    if short_frac > 0.5:
        print("\n" + "!" * 72, flush=True)
        print(f"WARNING: {short_frac:.1%} of rows yielded <40 chars of text. The text field "
              f"({args.text_field!r}) is probably wrong for this corpus, and a containment "
              f"of zero here means 'nothing was read', not 'nothing is present'.", flush=True)
        print("!" * 72 + "\n", flush=True)
    manifest = {
        "corpus": args.corpus,
        "rows_skipped_too_short": skipped_short,
        "frac_skipped_too_short": round(short_frac, 6),
        "extraction_suspect": bool(short_frac > 0.5),
        "hf_path": args.hf_path,
        "config": args.config,
        "data_files": args.data_files,
        "hf_glob": args.hf_glob,
        "local_jsonl": args.local_jsonl,
        "revision": args.revision,
        "split": args.split,
        "rows_scanned": scanned,
        "row_limit": args.limit,
        "sampled": scanned >= args.limit,
        "ngram_n": args.ngram_n,
        "wall_clock_sec": round(time.time() - t0, 1),
        "rows_per_hour": round(scanned / max(time.time() - t1, 1e-9) * 3600),
        "coverage_note": (
            "Sampled prefix: containment is a LOWER bound. A zero count means 'absent "
            "from the rows scanned', never 'absent from the corpus'."
            if scanned >= args.limit else "Full pass over the split."),
        "per_item_set": per_set,
    }
    (out / f"{args.corpus}_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out / f"{args.corpus}_items.jsonl").open("w", encoding="utf-8") as fh:
        for (tag, iid), v in zip(keys, best, strict=True):
            if v > 0:
                fh.write(json.dumps({"item_set": tag, "id": iid, "containment": round(v, 6)}) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
