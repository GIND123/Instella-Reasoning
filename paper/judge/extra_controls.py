#!/usr/bin/env python
"""Two checks a reviewer asked for, both computed from the committed verdict files.

PLACEBO SPLIT
    The unseen arm is randomly halved and one half relabelled seen, then the contrast is
    recomputed. Nothing distinguishes the two halves, so an effect here would mean the
    pipeline manufactures gaps from arm labels alone. Repeated over many random splits to
    give a null distribution rather than a single draw.

CONDITION BY MEMBERSHIP INTERACTION
    The manuscript compares the reference given and reference withheld gaps by inspecting
    two intervals, which is not a test of their difference. Both conditions grade the same
    solutions, so their difference removes every property of those solutions, including how
    subtle their errors are. That difference is the quantity the specificity account rests
    on and it is reported here with its own interval.

    python paper/judge/extra_controls.py
"""

from __future__ import annotations

import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
SEED, NBOOT, NPLACEBO = 6198, 4000, 2000
JUDGES = [("llama-3.1-8b-instruct", "Llama 3.1 8B"), ("qwen2.5-32b-instruct", "Qwen2.5 32B")]


def load(judge: str, cond: str) -> list[dict]:
    p = DATA / f"j_{judge}_instruct_{cond}.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("judge_verdict") is not None]


def rates(rows: list[dict]) -> dict:
    tp = fn = tn = fp = 0
    for r in rows:
        g, v = bool(r["ground_truth"]), bool(r["judge_verdict"])
        if g and v:
            tp += 1
        elif g and not v:
            fn += 1
        elif not g and not v:
            tn += 1
        else:
            fp += 1
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"ba": (sens + spec) / 2, "sens": sens, "spec": spec,
            "n_wrong": tn + fp, "n_rejected": tn}


def by_parent(rows):
    d = {}
    for r in rows:
        d.setdefault(r["parent_id"], []).append(r)
    return d


def gap(rows, stat, seen_ids=None):
    if seen_ids is None:
        a = [r for r in rows if r["arm"] == "seen"]
        b = [r for r in rows if r["arm"] == "unseen"]
    else:
        a = [r for r in rows if r["parent_id"] in seen_ids]
        b = [r for r in rows if r["parent_id"] not in seen_ids]
    if not a or not b:
        return 0.0
    return rates(a)[stat] - rates(b)[stat]


def interaction(judge: str, stat: str):
    """Withheld gap minus given gap, on identical solutions, resampling parents once."""
    giv, wit = load(judge, "reference_based"), load(judge, "reference_free")
    gp, wp = by_parent(giv), by_parent(wit)
    parents = sorted(set(gp) | set(wp))

    def stat_fn(sample):
        g = [r for q in sample for r in gp.get(q, ())]
        w = [r for q in sample for r in wp.get(q, ())]
        return gap(w, stat) - gap(g, stat)

    obs = stat_fn(parents)
    rng = random.Random(SEED)
    d = sorted(stat_fn([rng.choice(parents) for _ in parents]) for _ in range(NBOOT))
    return obs, d[int(0.025 * NBOOT)], d[int(0.975 * NBOOT)]


def placebo(judge: str, cond: str, stat: str):
    """Relabel half the unseen arm as seen. Any effect here is manufactured by the pipeline."""
    rows = [r for r in load(judge, cond) if r["arm"] == "unseen"]
    parents = sorted({r["parent_id"] for r in rows})
    rng = random.Random(SEED)
    draws = []
    for _ in range(NPLACEBO):
        shuffled = parents[:]
        rng.shuffle(shuffled)
        fake_seen = set(shuffled[: len(shuffled) // 2])
        draws.append(gap(rows, stat, fake_seen))
    draws.sort()
    mean = sum(draws) / len(draws)
    return mean, draws[int(0.025 * NPLACEBO)], draws[int(0.975 * NPLACEBO)]


def main():
    out = {"seed": SEED, "n_boot": NBOOT, "n_placebo": NPLACEBO,
           "interaction": {}, "placebo": {}, "counts": {}}

    print("CONDITION BY MEMBERSHIP INTERACTION (withheld gap minus given gap)\n")
    for jid, lab in JUDGES:
        for stat in ("ba", "spec", "sens"):
            o, lo, hi = interaction(jid, stat)
            out["interaction"].setdefault(lab, {})[stat] = {
                "estimate": round(o, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {lab:14} {stat:5} {o:+7.4f} [{lo:+7.4f}, {hi:+7.4f}] {star}")

    print("\nPLACEBO SPLIT (unseen arm randomly halved, one half relabelled seen)\n")
    for jid, lab in JUDGES:
        for cond, cl in (("reference_based", "given"), ("reference_free", "withheld")):
            m, lo, hi = placebo(jid, cond, "spec")
            out["placebo"].setdefault(lab, {})[cl] = {
                "mean_spec_gap": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            print(f"  {lab:14} {cl:9} specificity gap {m:+7.4f} [{lo:+7.4f}, {hi:+7.4f}]")

    print("\nRAW COUNTS behind the specificity numbers\n")
    for jid, lab in JUDGES:
        for cond, cl in (("reference_based", "given"), ("reference_free", "withheld")):
            rows = load(jid, cond)
            e = {}
            for arm in ("seen", "unseen"):
                r = rates([x for x in rows if x["arm"] == arm])
                e[arm] = {"wrong": r["n_wrong"], "rejected": r["n_rejected"],
                          "specificity": round(r["spec"], 4)}
                print(f"  {lab:14} {cl:9} {arm:6} rejected {r['n_rejected']:4d} "
                      f"of {r['n_wrong']:4d} wrong  ({r['spec']:.4f})")
            out["counts"].setdefault(lab, {})[cl] = e

    dest = HERE / "extra_controls.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest.name}")


if __name__ == "__main__":
    main()
