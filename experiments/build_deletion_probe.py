#!/usr/bin/env python
"""Build the premise-deletion probe item set — the study's recall measurement.

The CLI's ``make-variants`` covers only the answer-preserving perturbations; the
structural ops live in :mod:`instella_reasoning.structural` and were reachable only
through the Modal app, which runs on vLLM. This driver exposes them for the
transformers path used on the Lambda box, so Phase 0/1 generation goes through the
same ``instella-reasoning generate`` entry point every other measurement uses.

Parents are chosen by a seeded shuffle over the items that actually admit a removal
(the generator skips items whose calculator chain does not pin a premise value, and
skips again when the gold answer survives elsewhere in the text). Selection is
therefore stable across runs and independent of how many items are scanned.

  python experiments/build_deletion_probe.py \
      --benchmark gsm8k_train --n-parents 900 --k 3 --out-dir experiments/runs/<run>/base
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from instella_reasoning.datasets.loaders import load_benchmark_to_jsonl
from instella_reasoning.records import read_benchmark, write_jsonl
from instella_reasoning.structural import make_structural_variants, removable_premises


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="gsm8k_train", help="gsm8k_train | gsm8k (test)")
    ap.add_argument("--n-parents", type=int, default=900)
    ap.add_argument("--k", type=int, default=3, help="premise removals per parent")
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None, help="filename stem (default: the benchmark name)")
    ap.add_argument("--cache", default=None, help="reuse an already-downloaded benchmark JSONL")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.benchmark

    cache = Path(args.cache) if args.cache else out_dir / f"{tag}_all.jsonl"
    if not cache.exists():
        n = load_benchmark_to_jsonl(args.benchmark, cache)
        print(f"[build] downloaded {args.benchmark}: {n} items -> {cache}", flush=True)
    items = read_benchmark(cache)
    print(f"[build] loaded {len(items)} items", flush=True)

    # Eligible = admits at least one removal. Checked before sampling so n-parents is the
    # count actually delivered rather than an upper bound thinned by generator skips.
    eligible = [it for it in items if removable_premises(it)]
    print(f"[build] eligible parents: {len(eligible)}/{len(items)}", flush=True)

    rng = random.Random(args.seed)
    order = list(eligible)
    rng.shuffle(order)
    parents = order[: args.n_parents]

    result = make_structural_variants(
        parents, kinds=("premise_removal",), seed=args.seed, premise_removal_k=args.k
    )
    variants = result.variants
    kept_parent_ids = {v.parent_id for v in variants}
    parents_kept = [p for p in parents if p.id in kept_parent_ids]

    del_path = out_dir / f"{tag}_deletion.jsonl"
    par_path = out_dir / f"{tag}_parents.jsonl"
    write_jsonl(del_path, variants)
    write_jsonl(par_path, parents_kept)

    per_parent: dict[str, int] = {}
    for v in variants:
        per_parent[v.parent_id] = per_parent.get(v.parent_id, 0) + 1
    manifest = {
        "benchmark": args.benchmark,
        "seed": args.seed,
        "k": args.k,
        "n_items_total": len(items),
        "n_eligible": len(eligible),
        "n_parents_requested": args.n_parents,
        "n_parents_with_variants": len(parents_kept),
        "n_deletion_rows": len(variants),
        "mean_removals_per_parent": round(len(variants) / max(len(parents_kept), 1), 3),
        "n_skipped": len(result.skipped),
        "deletion_path": str(del_path),
        "parents_path": str(par_path),
    }
    (out_dir / f"{tag}_deletion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
