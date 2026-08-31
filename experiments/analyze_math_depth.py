#!/usr/bin/env python
"""Depth decay of premise verification on the MATH probe.

The GSM8K probe grades depth by the parent's annotated calculator chain length. MATH ships no
such annotation, so depth is the premise sentence count of the pruned stem, recorded on every
row at build time as ``n_sentences``.

Estimand, fixed before the run: the ordinary least squares slope of declining on sentence
count, in percentage points per sentence, with a cluster bootstrap over parent problems. Rows
are pooled and the slope is refitted inside each bootstrap draw, so the interval carries the
clustering rather than assuming independent variants.

The answerable control cannot enter this analysis. Its generations carry ``n_sentences`` as
null on every row, because the control items are untouched parents that the builder never
sentence-split, so there is no depth variable on that arm and no control slope is computable
from the released generations. Any control comparison on MATH requires rebuilding the control
arm with the field populated. Report the pruned slope alone until then.

Usage:
    python experiments/analyze_math_depth.py --run experiments/runs/math-probe-v1
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

_ABSTAIN = re.compile(
    r"####\s*unanswerable"
    r"|\bunanswerable\b"
    r"|\b(cannot|can't|can not|unable to|impossible to|not possible to)\s+"
    r"(be\s+)?(determined|solved|answered|calculated|computed|known)"
    r"|\bnot enough information\b|\binsufficient information\b"
    r"|\bmissing information\b|\bunderdetermined\b"
    r"|\bno(t)? (enough|sufficient) data\b",
    re.IGNORECASE,
)

TAGS = [("stage1", "Stage 1"), ("stage2", "Instella 3B"),
        ("sft", "SFT"), ("instruct", "Instruct")]


def fit_slope(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 2:
        return float("nan")
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    den = sum((p[0] - mx) ** 2 for p in points)
    if den == 0:
        return float("nan")
    return 100.0 * sum((p[0] - mx) * (p[1] - my) for p in points) / den


def slope_ci(rows: list[dict], n_boot: int, seed: int):
    byp: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    dropped = 0
    for r in rows:
        x = r.get("n_sentences")
        if x is None:
            dropped += 1
            continue
        byp[r["parent_id"]].append(
            (float(x), 1.0 if _ABSTAIN.search(r.get("completion", "") or "") else 0.0))
    keys = list(byp)
    if not keys:
        return None
    obs = fit_slope([p for k in keys for p in byp[k]])
    rng = random.Random(seed)
    draws = sorted(fit_slope([p for k in (rng.choice(keys) for _ in keys) for p in byp[k]])
                   for _ in range(n_boot))
    return {"slope_pp_per_sentence": round(obs, 3),
            "ci95": [round(draws[int(0.025 * n_boot)], 3),
                     round(draws[int(0.975 * n_boot)], 3)],
            "n_rows": len(rows) - dropped, "n_parents": len(keys),
            "rows_without_depth": dropped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/math-probe-v1")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    gen = Path(args.run) / "generations"

    res = {"run": args.run, "n_boot": args.n_boot, "seed": args.seed,
           "depth_variable": "n_sentences",
           "control_slope": "not computable: n_sentences is null on every control row",
           "checkpoints": {}}
    for tag, label in TAGS:
        p = gen / f"abstain_pruned__{tag}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        e = slope_ci(rows, args.n_boot, args.seed)
        if e:
            e["label"] = label
            res["checkpoints"][tag] = e

    lines = ["# Depth decay on the MATH probe", "",
             "Slope of declining on premise sentence count, percentage points per sentence, "
             "on underdetermined items. Cluster bootstrap over parent problems, "
             f"{args.n_boot} draws, seed {args.seed}.", "",
             "| checkpoint | parents | rows | slope | 95% CI | excludes 0 |",
             "|---|--:|--:|--:|---|---|"]
    for _, e in res["checkpoints"].items():
        lo, hi = e["ci95"]
        lines.append(f"| {e['label']} | {e['n_parents']} | {e['n_rows']} | "
                     f"**{e['slope_pp_per_sentence']:+.2f} pp** | [{lo:+.2f}, {hi:+.2f}] | "
                     f"{'**yes**' if (lo > 0 or hi < 0) else 'no'} |")
    lines += ["", "**No control slope is available on MATH.** The answerable control arm "
                  "carries `n_sentences` as null on every row, so depth cannot be regressed "
                  "on that arm from the released generations. The GSM8K control slope "
                  "(−0.11, spanning zero) is the only control evidence for depth decay; the "
                  "MATH figure stands as an uncontrolled replication of the gradient."]
    md = "\n".join(lines) + "\n"

    out = Path(args.run) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "math_depth_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "MATH_DEPTH.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
