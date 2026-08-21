#!/usr/bin/env python
"""Score a premise-deletion generation set: recall, abstention, truncation.

These rows have no gold answer — the premise the answer depended on has been deleted, so
the item is underdetermined by construction. ``score-generations`` therefore cannot be
used: it scores against ``item.answer``, which is ``None`` here on purpose.

Three quantities, and the distinction between the second and third is the whole probe:

* **recall**      the model emits the *parent's* answer to a question that no longer
                  determines it. This is the memorisation estimate.
* **abstention**  the model produces no extractable number, or says outright that the
                  problem cannot be solved. This is what reasoning from the text looks
                  like when the text is insufficient.
* **other**       some other number — an attempt at inference on partial information.
                  Neither recall nor abstention, and it must not be folded into either.

Truncation is reported alongside because a completion cut off at the token cap can score
as abstention for purely mechanical reasons, which would read as reasoning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from instella_reasoning.answer_equivalence import numeric_equal
from instella_reasoning.evaluation import termination_report
from instella_reasoning.prompting import extract_answer
from instella_reasoning.records import GenerationRecord, read_benchmark, read_jsonl

# Phrases a model uses when it recognises the problem is underdetermined. Deliberately
# narrow: a false positive here would be counted as reasoning and would deflate recall.
_ABSTAIN = re.compile(
    r"\b(cannot|can't|can not|unable to|impossible to|not possible to)\s+"
    r"(be\s+)?(determined|solved|answered|calculated|computed|known)"
    r"|\bnot enough information\b|\binsufficient information\b"
    r"|\bmissing information\b|\bunderdetermined\b|\bno(t)? (enough|sufficient) data\b",
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="the deletion item JSONL")
    ap.add_argument("--generations", required=True, nargs="+")
    ap.add_argument("--output", required=True, help="summary JSON")
    ap.add_argument("--rows-out", default=None, help="optional per-row JSONL")
    args = ap.parse_args()

    items = {it.id: it for it in read_benchmark(args.benchmark)}
    summaries = []
    all_rows = []

    for gen_path in args.generations:
        gens = [GenerationRecord.from_dict(r) for r in read_jsonl(gen_path)]
        if not gens:
            print(f"[score] {gen_path}: empty, skipped", flush=True)
            continue
        term = termination_report(gens)

        n = recall = abstain = other = unmatched = 0
        for g in gens:
            item = items.get(g.benchmark_id)
            if item is None:
                unmatched += 1
                continue
            parent = (item.metadata or {}).get("parent_answer")
            if parent is None:
                unmatched += 1
                continue
            n += 1
            pred = extract_answer(item, g.completion)
            hit = pred is not None and str(pred).strip() != "" and numeric_equal(
                str(parent), str(pred)
            )
            if hit:
                recall += 1
                label = "recall"
            elif pred is None or str(pred).strip() == "" or _ABSTAIN.search(g.completion or ""):
                abstain += 1
                label = "abstain"
            else:
                other += 1
                label = "other"
            all_rows.append({
                "model": g.model,
                "benchmark_id": g.benchmark_id,
                "parent_id": item.parent_id,
                "parent_answer": parent,
                "predicted": pred,
                "label": label,
                "finished": bool(g.metadata.get("finished", True)),
                "chars": len(g.completion or ""),
            })

        s = {
            "generations": gen_path,
            "model": gens[0].model,
            "n_scored": n,
            "n_unmatched": unmatched,
            "recall": recall,
            "recall_rate": round(recall / n, 4) if n else None,
            "abstain": abstain,
            "abstain_rate": round(abstain / n, 4) if n else None,
            "other": other,
            "other_rate": round(other / n, 4) if n else None,
            "termination_rate": round(term.termination_rate, 4),
            "truncation_rate": round(1.0 - term.termination_rate, 4),
            "marker_rate": round(term.marker_rate, 4),
            "median_chars": term.median_chars,
        }
        summaries.append(s)
        print(
            f"[score] {s['model']:28s} n={n:5d} recall={s['recall_rate']} "
            f"abstain={s['abstain_rate']} other={s['other_rate']} "
            f"trunc={s['truncation_rate']}",
            flush=True,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if args.rows_out:
        Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rows_out, "w", encoding="utf-8") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
