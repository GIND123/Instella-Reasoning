#!/usr/bin/env python
"""Is the chain length decay in abstention really about depth, or about difficulty?

Abstention on underdetermined problems falls as the reference solution's calculator chain
gets longer. Longer chains are also harder problems, so the observed decay has two candidate
explanations that the answerable control cannot separate:

  depth        tracking whether the premises still suffice gets harder as the chain the model
               must hold gets longer, which is a claim about the structure of its reasoning.
  difficulty   the model is simply worse at hard problems in every respect, and abstention
               inherits that, which is not a claim about premise verification at all.

Difficulty is measured here by the model itself rather than assumed: a parent problem counts
as solvable when the model answers the untouched version correctly, taken from the answerable
control generations that already exist. The chain length slope is then estimated separately
within each stratum.

PRIMARY ENDPOINT, fixed before the split
    slope of abstention on chain length, computed within solvable parents and within
    unsolvable parents. Surviving inside both strata means depth is doing work that
    difficulty does not explain. Collapsing inside either means the decay was difficulty.

Usage:
    python experiments/analyze_depth_confound.py --checkpoint instruct
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CALC = re.compile(r"<<([^>]+)>>")
ABSTAIN = re.compile(
    r"####\s*unanswerable|\bunanswerable\b"
    r"|\b(cannot|can't|can not|unable to|impossible to|not possible to)\s+"
    r"(be\s+)?(determined|solved|answered|calculated|computed|known)"
    r"|\bnot enough information\b|\binsufficient information\b"
    r"|\bmissing information\b|\bunderdetermined\b", re.IGNORECASE)


def to_f(x):
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def predicted(completion: str):
    hits = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", completion or "")
    if hits:
        return to_f(hits[-1])
    all_n = NUM.findall(completion or "")
    return to_f(all_n[-1]) if all_n else None


def slope(byp: dict, sample: list, want: bool) -> float:
    xs, ys = [], []
    for q in sample:
        for steps, solvable, ab in byp.get(q, ()):
            if solvable is want:
                xs.append(steps)
                ys.append(ab)
    if len(xs) < 10:
        return 0.0
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return 100 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/runs/ckpt-axis-v1/base")
    ap.add_argument("--gen", default="experiments/runs/abstain-v1/generations")
    ap.add_argument("--checkpoint", default="instruct")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    base, gen = Path(args.base), Path(args.gen)

    steps, gold = {}, {}
    for fn in ("gsm8k_test_parents.jsonl", "gsm8k_train_parents.jsonl"):
        for line in (base / fn).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                steps[r["id"]] = len(CALC.findall((r.get("metadata") or {}).get("rationale") or ""))
                gold[r["id"]] = to_f(r.get("answer"))

    solvable = {}
    ctrl = gen / f"abstain_control__{args.checkpoint}.jsonl"
    for line in ctrl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            g, p = gold.get(r["parent_id"]), predicted(r.get("completion", ""))
            solvable[r["parent_id"]] = bool(g is not None and p is not None and g == p)

    byp: dict[str, list] = collections.defaultdict(list)
    for line in (gen / f"abstain__{args.checkpoint}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r["parent_id"]
        if pid not in solvable or not steps.get(pid):
            continue
        byp[pid].append((steps[pid], solvable[pid],
                         1.0 if ABSTAIN.search(r.get("completion", "") or "") else 0.0))

    parents = sorted(byp)
    rng = random.Random(args.seed)
    obs = {w: slope(byp, parents, w) for w in (True, False)}
    draws = {True: [], False: []}
    for _ in range(args.n_boot):
        smp = [rng.choice(parents) for _ in parents]
        for w in (True, False):
            draws[w].append(slope(byp, smp, w))

    res = {"checkpoint": args.checkpoint, "n_parents": len(parents),
           "solvable_rate": round(sum(solvable.values()) / len(solvable), 4),
           "n_boot": args.n_boot, "seed": args.seed, "strata": {}}
    for w, name in ((True, "solvable"), (False, "unsolvable")):
        d = sorted(draws[w])
        lo, hi = d[int(0.025 * args.n_boot)], d[int(0.975 * args.n_boot)]
        res["strata"][name] = {"slope_pp_per_step": round(obs[w], 3),
                               "ci95": [round(lo, 3), round(hi, 3)],
                               "excludes_zero": bool(lo > 0 or hi < 0)}

    lines = ["# Chain length against difficulty", "",
             "Abstention on underdetermined problems, slope per additional calculator step, "
             "estimated separately within problems the model can and cannot solve when they "
             "are intact. Solvability is measured from the answerable control generations, so "
             "difficulty is the model's own rather than assumed. Cluster bootstrap over "
             f"parent problems, {args.n_boot} draws, seed {args.seed}.", "",
             f"Parents labelled: {len(parents)}. Solved intact: "
             f"{res['solvable_rate']:.1%}.", "",
             "| stratum | slope per step | 95% CI | excludes 0 |", "|---|--:|---|---|"]
    for name, e in res["strata"].items():
        lines.append(f"| {name} | {e['slope_pp_per_step']:+.2f} pp | "
                     f"[{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}] | "
                     f"{'**yes**' if e['excludes_zero'] else 'no'} |")
    lines += ["", "A slope surviving inside both strata means chain depth carries information "
                  "that problem difficulty does not account for."]
    md = "\n".join(lines) + "\n"

    out = Path("experiments/runs/abstain-v1/analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth_confound.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "DEPTH_CONFOUND.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
