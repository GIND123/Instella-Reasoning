#!/usr/bin/env python
"""Pre-registered analysis for the quantity ablation. Written before the generations exist.

The estimands are fixed here, in advance, so that reading the results cannot influence
which contrast gets reported. Everything is paired within parent problem against that
problem's own ``orig`` condition, and every interval is a cluster bootstrap over parent
problems because a parent contributes one row per condition and those rows are not
independent.

The design is a 2x2 crossed on the inserted sentence:

                        no number ("several")     number (a free two-digit integer)
    off-topic noun      offtopic_noqty            offtopic_qty
    domain noun         domain_noqty              domain_qty

All four inserted sentences are the same twenty-word frame with two words swapped, so
length, syntax, and insertion position are held exactly fixed.

PRIMARY ENDPOINT
    main effect of NUMBER = mean(offtopic_qty, domain_qty) - mean(offtopic_noqty, domain_noqty)

    This is the paper's claim: an unused quantity costs accuracy where an equally long,
    equally placed sentence without one does not. It is the only contrast the paper will
    headline, and it is stated before any number is seen.

SECONDARY, pre-specified
    main effect of TOPICALITY = mean(domain_*) - mean(offtopic_*)
    interaction               = (domain_qty - domain_noqty) - (offtopic_qty - offtopic_noqty)
    per-condition drop        = condition - orig, for each of the four cells

EXPLORATORY, not reportable as an estimate
    reorder_safe - orig. The order-safety filter is a surface heuristic and manual
    inspection found violations that survive it, so this arm is a spot-check pending a
    human audit and is printed under a separate heading with that caveat attached.

Decision rule, fixed in advance:
    * number effect significant, topicality not  -> the quantity is what costs accuracy
    * both significant                           -> report both main effects; the claim
                                                    narrows to "any relevant-looking
                                                    insertion costs accuracy"
    * neither significant                        -> the earlier 6-8 pp finding was
                                                    length or topicality, and the paper
                                                    says so

Usage:
    python experiments/analyze_quantity_ablation.py --run experiments/runs/qty-ablation-v1
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

CELLS = ("offtopic_noqty", "domain_noqty", "offtopic_qty", "domain_qty")
WITH_NUMBER = ("offtopic_qty", "domain_qty")
DOMAIN = ("domain_noqty", "domain_qty")

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_float(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(completion: str) -> str | None:
    """`#### N` if present, else the last number. Mirrors the repo's arithmetic extractor."""
    if not completion:
        return None
    hits = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", completion)
    if hits:
        return hits[-1]
    hits = _NUM.findall(completion)
    return hits[-1] if hits else None


def load_rows(items_path: Path, gen_path: Path) -> list[dict]:
    items = {}
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items[r["id"]] = r
    out = []
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        bid = g.get("benchmark_id") or g.get("id")
        it = items.get(bid)
        if it is None:
            continue
        gold = to_float(it.get("answer"))
        pred = to_float(extract_answer(g.get("completion") or ""))
        out.append({
            "parent_id": it["parent_id"],
            "condition": it["metadata"]["condition"],
            "correct": bool(gold is not None and pred is not None and gold == pred),
            "finished": bool((g.get("metadata") or {}).get("finished", True)),
        })
    return out


def paired_table(rows: list[dict]) -> dict[str, dict[str, bool]]:
    """parent_id -> condition -> correct. Parents missing `orig` are dropped."""
    t: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    for r in rows:
        t[r["parent_id"]][r["condition"]] = r["correct"]
    return {p: c for p, c in t.items() if "orig" in c}


def _boot(parents: list[str], stat, n: int, seed: int) -> tuple[float, float, float]:
    obs = stat(parents)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        draws.append(stat([rng.choice(parents) for _ in parents]))
    draws.sort()
    lo = draws[int(0.025 * n)]
    hi = draws[int(0.975 * n)]
    return obs, lo, hi


def contrast(table, group_a: tuple[str, ...], group_b: tuple[str, ...],
             n_boot: int, seed: int) -> dict | None:
    """mean(accuracy over group_a) - mean(accuracy over group_b), paired within parent."""
    usable = [p for p, c in table.items() if all(k in c for k in group_a + group_b)]
    if not usable:
        return None

    def stat(sample: list[str]) -> float:
        a = [table[p][k] for p in sample for k in group_a]
        b = [table[p][k] for p in sample for k in group_b]
        return sum(a) / len(a) - sum(b) / len(b)

    obs, lo, hi = _boot(usable, stat, n_boot, seed)
    return {"effect_pp": round(obs * 100, 3), "ci95_pp": [round(lo * 100, 3), round(hi * 100, 3)],
            "n_parents": len(usable), "excludes_zero": bool(lo > 0 or hi < 0)}


def analyse(run: Path, n_boot: int, seed: int) -> dict:
    items_path = run / "base" / "qty_ablation.jsonl"
    gen_dir = run / "generations"
    results: dict = {"run": str(run), "seed": seed, "n_boot": n_boot, "checkpoints": {}}

    for gen_path in sorted(gen_dir.glob("qty__*.jsonl")):
        tag = gen_path.stem.replace("qty__", "")
        rows = load_rows(items_path, gen_path)
        if not rows:
            continue
        table = paired_table(rows)
        acc = {}
        for cond in ("orig",) + CELLS + ("reorder_safe",):
            vals = [c[cond] for c in table.values() if cond in c]
            if vals:
                acc[cond] = {"accuracy": round(sum(vals) / len(vals), 4), "n": len(vals)}

        entry = {
            "n_rows": len(rows),
            "truncation_rate": round(1 - sum(r["finished"] for r in rows) / len(rows), 4),
            "accuracy_by_condition": acc,
            "PRIMARY_number_main_effect": contrast(table, WITH_NUMBER,
                                                   ("offtopic_noqty", "domain_noqty"),
                                                   n_boot, seed),
            "secondary_topicality_main_effect": contrast(table, DOMAIN,
                                                         ("offtopic_noqty", "offtopic_qty"),
                                                         n_boot, seed),
            "secondary_drop_vs_orig": {
                cond: contrast(table, (cond,), ("orig",), n_boot, seed) for cond in CELLS
            },
            "EXPLORATORY_reorder_safe": contrast(table, ("reorder_safe",), ("orig",),
                                                 n_boot, seed),
        }
        # interaction: does the number cost more when the sentence is on-topic?
        usable = [p for p, c in table.items() if all(k in c for k in CELLS)]
        if usable:
            def inter(sample: list[str]) -> float:
                d = sum(table[p]["domain_qty"] - table[p]["domain_noqty"] for p in sample)
                o = sum(table[p]["offtopic_qty"] - table[p]["offtopic_noqty"] for p in sample)
                return (d - o) / len(sample)
            obs, lo, hi = _boot(usable, inter, n_boot, seed)
            entry["secondary_interaction"] = {
                "effect_pp": round(obs * 100, 3),
                "ci95_pp": [round(lo * 100, 3), round(hi * 100, 3)],
                "n_parents": len(usable),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        results["checkpoints"][tag] = entry
    return results


def render(results: dict) -> str:
    out = ["# Quantity ablation - pre-registered analysis", ""]
    out.append("Paired within parent problem against that problem's own `orig`. "
               "Cluster bootstrap over parents, "
               f"{results['n_boot']} draws, seed {results['seed']}.")
    out.append("")
    out.append("## PRIMARY - main effect of an inserted number")
    out.append("")
    out.append("| checkpoint | effect | 95% CI | parents | excludes 0 |")
    out.append("|---|--:|---|--:|---|")
    for tag, e in results["checkpoints"].items():
        p = e.get("PRIMARY_number_main_effect")
        if p:
            out.append(f"| {tag} | {p['effect_pp']:+.2f} pp | "
                       f"[{p['ci95_pp'][0]:+.2f}, {p['ci95_pp'][1]:+.2f}] | "
                       f"{p['n_parents']} | {'**yes**' if p['excludes_zero'] else 'no'} |")
    out.append("")
    out.append("## Secondary")
    out.append("")
    out.append("| checkpoint | topicality | interaction | trunc |")
    out.append("|---|--:|--:|--:|")
    for tag, e in results["checkpoints"].items():
        t = e.get("secondary_topicality_main_effect") or {}
        i = e.get("secondary_interaction") or {}
        out.append(f"| {tag} | {t.get('effect_pp', float('nan')):+.2f} pp | "
                   f"{i.get('effect_pp', float('nan')):+.2f} pp | {e['truncation_rate']:.1%} |")
    out.append("")
    out.append("## Accuracy by condition")
    out.append("")
    conds = ("orig",) + CELLS
    out.append("| checkpoint | " + " | ".join(conds) + " |")
    out.append("|---" * (len(conds) + 1) + "|")
    for tag, e in results["checkpoints"].items():
        cells = [f"{e['accuracy_by_condition'].get(c, {}).get('accuracy', float('nan')):.3f}"
                 for c in conds]
        out.append(f"| {tag} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## EXPLORATORY - order-safe reordering")
    out.append("")
    out.append("> The order-safety filter is a surface heuristic. Manual inspection found "
               "violations that survive it, so these are spot-checks pending a human audit, "
               "not estimates. Do not report them as a result.")
    out.append("")
    out.append("| checkpoint | effect | 95% CI | parents |")
    out.append("|---|--:|---|--:|")
    for tag, e in results["checkpoints"].items():
        r = e.get("EXPLORATORY_reorder_safe")
        if r:
            out.append(f"| {tag} | {r['effect_pp']:+.2f} pp | "
                       f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] | {r['n_parents']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/qty-ablation-v1")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()

    run = Path(args.run)
    results = analyse(run, args.n_boot, args.seed)
    out_dir = run / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qty_ablation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    md = render(results)
    (out_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    print(md)
    if not results["checkpoints"]:
        print("[qty-ablation] no generation files found under "
              f"{run / 'generations'} (expected qty__<tag>.jsonl)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
