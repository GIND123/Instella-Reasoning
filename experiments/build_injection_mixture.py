#!/usr/bin/env python
"""Assemble the Phase 2 injection mixture: injected items, held-out controls, clean filler.

Three outputs, and the third is the one that takes work:

* ``injected.jsonl``  200 GSM8K *test* parents rendered in Stage-2 concatenated form.
* ``heldout.jsonl``   200 further test parents, never injected. Their recall must stay
                      flat; if it rises with dose, the mixture taught GSM8K-shaped
                      behaviour rather than the specific items, and the curve measures
                      generalisation instead of memorisation.
* ``filler.jsonl``    Stage-2-like documents verified to contain neither set.

Why the filler must be scanned rather than trusted: `allenai/tulu-3-sft-mixture` is in
Instella's Stage-2 mixture and carries 97% of GSM8K *train* at containment >= 0.999. A
filler that silently re-injected the dosed items would give every arm far more than its
nominal dose and flatten the curve — which would then be misread as "the budget is too
small". So the rule is scan-what-you-use: every filler document kept here has been checked
against the exact 400 items this study measures, using the repo's own 13-gram containment
rule, rather than against a corpus-level claim.

The scan is done with one set intersection per document. The 13-grams of all 400 items are
unioned once; a document is clean iff it shares none of them. That is strictly more
conservative than per-item containment thresholds (any single shared 13-gram rejects the
document), which is the right direction for a filler.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from instella_reasoning.records import read_benchmark, write_jsonl
from instella_reasoning.text import word_ngrams


def stage2_doc(item) -> str:
    """Stage-2 concatenated Q-rationale-answer form.

    Recall is format-sensitive, so the injected documents must match the shape GSM8K-derived
    text actually takes in the pretraining corpus. Anything else produces a dose-response
    curve that is not calibrated to the trajectory it is meant to explain.
    """
    rationale = (item.metadata or {}).get("rationale", "")
    body = rationale if rationale else str(item.answer)
    return f"{item.prompt}\n{body}\n#### {item.answer}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", required=True, help="gsm8k_test_parents.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-injected", type=int, default=200)
    ap.add_argument("--n-heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--ngram-n", type=int, default=13)
    ap.add_argument("--filler-hf", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--filler-split", default="train")
    ap.add_argument("--filler-scan-limit", type=int, default=120_000,
                    help="rows to stream and scan; kept rows stop at --filler-target-chars")
    ap.add_argument("--filler-target-chars", type=int, default=40_000_000,
                    help="~4 chars/token, so 40M chars ~ 10M tokens of clean filler")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parents = read_benchmark(args.parents)
    rng = random.Random(args.seed)
    order = list(parents)
    rng.shuffle(order)
    need = args.n_injected + args.n_heldout
    if len(order) < need:
        raise SystemExit(f"only {len(order)} parents available, need {need}")
    injected = order[: args.n_injected]
    heldout = order[args.n_injected : need]

    write_jsonl(out / "injected.jsonl", injected)
    write_jsonl(out / "heldout.jsonl", heldout)
    (out / "injected_docs.jsonl").write_text(
        "".join(json.dumps({"id": i.id, "text": stage2_doc(i)}) + "\n" for i in injected),
        encoding="utf-8",
    )
    print(f"[mix] injected={len(injected)} heldout={len(heldout)}", flush=True)

    # One union of 13-grams over every item this study measures. A filler document sharing
    # even one of them is dropped.
    banned: set[str] = set()
    for it in injected + heldout:
        banned |= word_ngrams(stage2_doc(it), args.ngram_n)
        banned |= word_ngrams(it.prompt, args.ngram_n)
    print(f"[mix] banned 13-gram set: {len(banned):,}", flush=True)

    from datasets import load_dataset

    ds = load_dataset(args.filler_hf, split=args.filler_split, streaming=True)

    def row_text(row) -> str:
        msgs = row.get("messages")
        if isinstance(msgs, list):
            return "\n".join(str(m.get("content", "")) for m in msgs)
        for k in ("text", "content", "prompt"):
            if row.get(k):
                return str(row[k])
        return ""

    kept: list[dict] = []
    chars = scanned = rejected = 0
    for row in ds:
        scanned += 1
        if scanned > args.filler_scan_limit or chars >= args.filler_target_chars:
            break
        text = row_text(row)
        if len(text) < 200:
            continue
        if word_ngrams(text, args.ngram_n) & banned:
            rejected += 1
            continue
        kept.append({"id": f"filler_{len(kept):06d}", "text": text})
        chars += len(text)
        if len(kept) % 2000 == 0:
            print(f"[mix] scanned={scanned:,} kept={len(kept):,} chars={chars:,} "
                  f"rejected={rejected}", flush=True)

    with (out / "filler.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    manifest = {
        "n_injected": len(injected),
        "n_heldout": len(heldout),
        "seed": args.seed,
        "ngram_n": args.ngram_n,
        "filler_source": args.filler_hf,
        "filler_rows_scanned": scanned,
        "filler_rows_kept": len(kept),
        "filler_rows_rejected_for_containment": rejected,
        "filler_chars": chars,
        "filler_est_tokens": chars // 4,
    }
    (out / "mixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
