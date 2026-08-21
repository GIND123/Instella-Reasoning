#!/usr/bin/env python
"""Phase 4 - does deletion-recall track how heavily an item was reproduced in Stage 2?

Two questions the paper has to answer before a reviewer asks them.

**Dose regression.** Containment against `train_119K` is continuous across the train arm,
so within that arm alone there is a graded exposure axis. If the Stage-2 recall rise were
memorisation, recall should climb with containment. This bins parents by containment and
reports recall per bin, at Stage1 (before the data) and Instella-3B (after). A flat
profile after the intervention is the same null the DiD reports, measured a second way and
without relying on the train/test contrast at all - which matters because the train/test
comparison invites a generalisation-gap objection and this one does not.

**Minimum detectable effect.** A null needs a floor. Phase 2 supplies the empirical one -
the probe separates from baseline at 64x (verbatim 0.946) and not at 16x (verbatim 0.491)
- and this adds the statistical one: given the observed base rate and the number of parent
clusters, what absolute increase in recall would have been detected at 80% power?
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

BINS = [(0.0, 0.10), (0.10, 0.30), (0.30, 0.50), (0.50, 0.80), (0.80, 1.01)]


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mde(p0: float, n_clusters: int, icc_rows: float, power: float = 0.80,
        alpha_z: float = 1.96) -> float:
    """Smallest p1 > p0 detectable at `power`, for a clustered proportion.

    n_clusters is parents, not rows: the design effect from correlated deletion variants
    is already absorbed by treating the parent as the unit.
    """
    z_beta = 0.8416 if power == 0.80 else 1.2816
    n = n_clusters * icc_rows
    lo, hi = p0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        se = math.sqrt((p0 * (1 - p0) + mid * (1 - mid)) / n)
        se0 = math.sqrt(2 * ((p0 + mid) / 2) * (1 - (p0 + mid) / 2) / n)
        got = 1 - _phi((alpha_z * se0 - (mid - p0)) / se)
        if got < power:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/ckpt-axis-v1")
    ap.add_argument("--containment", default="outputs/corpus_scan/train119k_items.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    run = Path(args.run)

    cont: dict[str, float] = {}
    for line in open(args.containment, encoding="utf-8"):
        r = json.loads(line)
        if r["item_set"] == "gsm8k_train":
            cont[r["id"]] = r["containment"]

    rows = [json.loads(x) for x in
            (run / "analysis" / "phase1_train_rows.jsonl").read_text().splitlines() if x.strip()]
    by_model: dict[str, dict[str, list[bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in rows:
        by_model[r["model"]][r["parent_id"]].append(r["label"] == "recall")

    S1, S2 = "amd/Instella-3B-Stage1", "amd/Instella-3B"
    rng = random.Random(args.seed)

    def binned(model: str) -> list[dict]:
        out = []
        for lo, hi in BINS:
            keys = [p for p in by_model[model]
                    if lo <= cont.get(p, 0.0) < hi]
            if not keys:
                out.append({"bin": f"[{lo:.2f},{hi:.2f})", "n_parents": 0})
                continue
            hit = sum(sum(by_model[model][k]) for k in keys)
            tot = sum(len(by_model[model][k]) for k in keys)
            draws = []
            for _ in range(args.n_boot):
                pick = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
                h = sum(sum(by_model[model][k]) for k in pick)
                t = sum(len(by_model[model][k]) for k in pick)
                draws.append(h / t if t else 0.0)
            draws.sort()
            out.append({
                "bin": f"[{lo:.2f},{hi:.2f})", "n_parents": len(keys), "n_rows": tot,
                "recall_events": hit, "recall_rate": round(hit / tot, 6),
                "ci95": [round(draws[int(.025 * args.n_boot)], 6),
                         round(draws[int(.975 * args.n_boot)], 6)],
            })
        return out

    n_parents = len(by_model[S2])
    rows_per_parent = sum(len(v) for v in by_model[S2].values()) / max(n_parents, 1)
    p0 = sum(sum(v) for v in by_model[S2].values()) / sum(len(v) for v in by_model[S2].values())
    report = {
        "containment_source": args.containment,
        "n_parents_with_containment": sum(1 for p in by_model[S2] if p in cont),
        "n_parents_total": n_parents,
        "stage1_by_containment": binned(S1),
        "instella3b_by_containment": binned(S2),
        "mde": {
            "base_rate_p0": round(p0, 6),
            "n_parent_clusters": n_parents,
            "mean_rows_per_parent": round(rows_per_parent, 3),
            "detectable_p1_at_80pct_power": round(mde(p0, n_parents, rows_per_parent), 6),
            "absolute_increase_pp": round(
                (mde(p0, n_parents, rows_per_parent) - p0) * 100, 3),
            "note": (
                "Statistical floor only. The empirical floor from Phase 2 is stricter: the "
                "probe separated from baseline at 64x exposure (verbatim 0.946) and not at "
                "16x (verbatim 0.491), so near-verbatim memorisation is required before the "
                "instrument responds at all."),
        },
    }
    out = run / "analysis" / "dose_regression.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"parents with containment: {report['n_parents_with_containment']}/{n_parents}")
    for name, key in (("Stage1", "stage1_by_containment"),
                      ("Instella-3B", "instella3b_by_containment")):
        print(f"\n{name} recall by train_119K containment:")
        for b in report[key]:
            if not b["n_parents"]:
                continue
            print(f"   {b['bin']:14s} parents={b['n_parents']:>4}  rows={b['n_rows']:>5}  "
                  f"recall={b['recall_events']:>3}/{b['n_rows']:<5} = {b['recall_rate']:.4f}  "
                  f"CI[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]")
    m = report["mde"]
    print(f"\nMDE: base {m['base_rate_p0']:.4f}, {m['n_parent_clusters']} parent clusters -> "
          f"detectable at {m['detectable_p1_at_80pct_power']:.4f} "
          f"(+{m['absolute_increase_pp']:.2f} pp) at 80% power")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
