#!/usr/bin/env python
"""G1b: does an insertion that announces its own irrelevance cost less than one that does not?

The crossed design in ``analyze_quantity_ablation.py`` found that an inserted unused number
costs nothing beyond the insertion itself, and that any twenty-word irrelevant sentence
costs about two points. Two points is small next to the losses Shi et al. report for
irrelevant context on grade-school arithmetic, and there is a candidate explanation in their
own paper: they find that instructing a model to ignore irrelevant information mitigates the
loss. The frame used here ends with "and has no bearing on this question", which is such an
instruction carried inside the inserted sentence.

This compares that frame against one identical in every respect except the marker:

    marked    ... was filed in another office and has no bearing on this question.
    unmarked  ... was filed in another office on the afternoon of the same day.

Both are twenty words for the whole sentence, both sit in the same position, both carry the
same topic noun and the same unused integer. Only the self-labelling differs.

PRIMARY ENDPOINT, fixed before the run
    marker effect = mean(unmarked cells) - mean(marked cells), paired within parent problem.

    Negative means removing the marker costs additional accuracy, which would place the
    modest cost measured earlier inside Shi et al.'s mitigated regime and make the two
    results consistent. Null means the marker is not doing the work and the small effect
    needs another explanation, which the paper would then have to state as open.

Usage:
    python experiments/analyze_marker.py --run experiments/runs/qty-ablation-v1
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

MARKED = ("offtopic_qty", "domain_qty")
UNMARKED = ("offtopic_qty_unmarked", "domain_qty_unmarked")
TAGS = [("stage2", "Instella 3B"), ("sft", "SFT"), ("instruct", "Instruct")]

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_float(x):
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def extract(completion: str):
    if not completion:
        return None
    hits = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", completion)
    if hits:
        return to_float(hits[-1])
    hits = _NUM.findall(completion)
    return to_float(hits[-1]) if hits else None


def load_items(*paths: Path) -> dict:
    items = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                items[r["id"]] = r
    return items


def load_gen(items: dict, path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        it = items.get(g.get("benchmark_id"))
        if it is None:
            continue
        gold = to_float(it.get("answer"))
        pred = extract(g.get("completion") or "")
        out.append({"parent_id": it["parent_id"],
                    "condition": it["metadata"]["condition"],
                    "correct": bool(gold is not None and pred is not None and gold == pred)})
    return out


def boot(parents, stat, n, seed):
    obs = stat(parents)
    rng = random.Random(seed)
    draws = sorted(stat([rng.choice(parents) for _ in parents]) for _ in range(n))
    return obs, draws[int(0.025 * n)], draws[int(0.975 * n)]


def contrast(table, a, b, n_boot, seed):
    usable = [p for p, c in table.items() if all(k in c for k in tuple(a) + tuple(b))]
    if not usable:
        return None

    def stat(sample):
        xa = [table[p][k] for p in sample for k in a]
        xb = [table[p][k] for p in sample for k in b]
        return sum(xa) / len(xa) - sum(xb) / len(xb)

    obs, lo, hi = boot(usable, stat, n_boot, seed)
    return {"effect_pp": round(obs * 100, 3),
            "ci95_pp": [round(lo * 100, 3), round(hi * 100, 3)],
            "n_parents": len(usable),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def analyse(run: Path, n_boot: int, seed: int) -> dict:
    items = load_items(run / "base" / "qty_ablation.jsonl",
                       run / "base" / "qty_marker.jsonl")
    res = {"run": str(run), "seed": seed, "n_boot": n_boot, "checkpoints": {}}
    for tag, name in TAGS:
        rows = (load_gen(items, run / "generations" / f"qty__{tag}.jsonl")
                + load_gen(items, run / "generations" / f"marker__{tag}.jsonl"))
        if not rows:
            continue
        table: dict[str, dict[str, bool]] = collections.defaultdict(dict)
        for r in rows:
            table[r["parent_id"]][r["condition"]] = r["correct"]
        table = {p: c for p, c in table.items() if "orig" in c}
        if not any(u in c for c in table.values() for u in UNMARKED):
            continue
        acc = {}
        for cond in ("orig",) + MARKED + UNMARKED:
            v = [c[cond] for c in table.values() if cond in c]
            if v:
                acc[cond] = {"accuracy": round(sum(v) / len(v), 4), "n": len(v)}
        res["checkpoints"][tag] = {
            "label": name,
            "accuracy_by_condition": acc,
            "PRIMARY_marker_effect": contrast(table, UNMARKED, MARKED, n_boot, seed),
            "marked_vs_orig": contrast(table, MARKED, ("orig",), n_boot, seed),
            "unmarked_vs_orig": contrast(table, UNMARKED, ("orig",), n_boot, seed),
        }
    return res


def render(res: dict) -> str:
    L = ["# Irrelevance marker ablation", "",
         "Both insertions are twenty words, same position, same topic noun, same unused "
         "integer. They differ only in whether the sentence states that it has no bearing "
         "on the question. Paired within parent problem, cluster bootstrap over parents, "
         f"{res['n_boot']} draws, seed {res['seed']}.", "",
         "## PRIMARY - effect of removing the marker (unmarked minus marked)", "",
         "| checkpoint | effect | 95% CI | parents | excludes 0 |", "|---|--:|---|--:|---|"]
    for _, e in res["checkpoints"].items():
        p = e.get("PRIMARY_marker_effect")
        if p:
            L.append(f"| {e['label']} | {p['effect_pp']:+.2f} pp | "
                     f"[{p['ci95_pp'][0]:+.2f}, {p['ci95_pp'][1]:+.2f}] | "
                     f"{p['n_parents']} | {'**yes**' if p['excludes_zero'] else 'no'} |")
    L += ["", "## Each insertion against the untouched problem", "",
          "| checkpoint | marked | unmarked |", "|---|--:|--:|"]
    for _, e in res["checkpoints"].items():
        m, u = e.get("marked_vs_orig") or {}, e.get("unmarked_vs_orig") or {}
        L.append(f"| {e['label']} | {m.get('effect_pp', float('nan')):+.2f} pp "
                 f"[{m.get('ci95_pp',[0,0])[0]:+.2f}, {m.get('ci95_pp',[0,0])[1]:+.2f}] | "
                 f"{u.get('effect_pp', float('nan')):+.2f} pp "
                 f"[{u.get('ci95_pp',[0,0])[0]:+.2f}, {u.get('ci95_pp',[0,0])[1]:+.2f}] |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/qty-ablation-v1")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    run = Path(args.run)
    res = analyse(run, args.n_boot, args.seed)
    if not res["checkpoints"]:
        print(f"[marker] no paired marked/unmarked generations under {run / 'generations'}")
        return 1
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "marker_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = render(res)
    (out / "MARKER.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
