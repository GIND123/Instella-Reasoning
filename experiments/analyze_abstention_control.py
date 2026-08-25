#!/usr/bin/env python
"""Does the model detect insufficiency, or does it just take an offered escape hatch?

Every item in the abstention run has a premise removed, so declining is correct on all of
them. A high abstention rate under a prompt that names ``#### unanswerable`` is therefore
consistent with two incompatible readings:

  detection    the model notices the remaining premises no longer determine an answer, which
               is a claim about reasoning.
  compliance   the model takes an option the prompt offered, which is a claim about
               instruction following and carries no information about reasoning.

The control separates them. The same prompt, the same decoding and the same checkpoints run
over the untouched parent problems, which are answerable, so abstaining there is an error.

PRIMARY ENDPOINT, fixed before the control was run
    discrimination = abstention on pruned items minus abstention on answerable parents,
    per checkpoint, with a cluster bootstrap over parent problems.

    Large and positive means the model discriminates and the abstention result is about
    reasoning. Near zero means the rate measures willingness to use the escape hatch, and the
    abstention finding cannot be reported as insufficiency detection.

Usage:
    python experiments/analyze_abstention_control.py --run experiments/runs/abstain-v1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

TAGS = [("stage1", "Stage 1"), ("stage2", "Instella 3B"),
        ("sft", "SFT"), ("instruct", "Instruct")]

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


def abstained(completion: str) -> bool:
    return bool(_ABSTAIN.search(completion or ""))


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def by_parent(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["parent_id"], []).append(
            1.0 if abstained(r.get("completion", "")) else 0.0)
    return out


def boot_diff(a: dict, b: dict, n_boot: int, seed: int):
    parents = sorted(set(a) | set(b))

    def stat(sample):
        x = [v for q in sample for v in a.get(q, ())]
        y = [v for q in sample for v in b.get(q, ())]
        if not x or not y:
            return 0.0
        return 100 * (sum(x) / len(x) - sum(y) / len(y))

    obs = stat(parents)
    rng = random.Random(seed)
    draws = sorted(stat([rng.choice(parents) for _ in parents]) for _ in range(n_boot))
    return (round(obs, 3), round(draws[int(0.025 * n_boot)], 3),
            round(draws[int(0.975 * n_boot)], 3))


def rate(d: dict) -> float:
    flat = [v for vs in d.values() for v in vs]
    return sum(flat) / len(flat) if flat else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/abstain-v1")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    gen = Path(args.run) / "generations"

    res = {"n_boot": args.n_boot, "seed": args.seed, "checkpoints": {}}
    for tag, label in TAGS:
        pruned = load(gen / f"abstain__{tag}.jsonl")
        control = load(gen / f"abstain_control__{tag}.jsonl")
        if not pruned or not control:
            continue
        p, c = by_parent(pruned), by_parent(control)
        obs, lo, hi = boot_diff(p, c, args.n_boot, args.seed)
        res["checkpoints"][tag] = {
            "label": label,
            "abstain_pruned": round(rate(p), 4), "n_pruned": len(pruned),
            "abstain_answerable": round(rate(c), 4), "n_answerable": len(control),
            "PRIMARY_discrimination_pp": obs,
            "PRIMARY_ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    if not res["checkpoints"]:
        print(f"[control] need both abstain__*.jsonl and abstain_control__*.jsonl under {gen}")
        return 1

    lines = ["# Abstention, with the answerable control", "",
             "Identical prompt licensing `#### unanswerable`, identical decoding. Pruned "
             "items are underdetermined so declining is correct; parent problems are "
             "answerable so declining is an error. Cluster bootstrap over parent problems, "
             f"{args.n_boot} draws, seed {args.seed}.", "",
             "| checkpoint | abstains, pruned | abstains, answerable | discrimination | "
             "95% CI | excludes 0 |", "|---|--:|--:|--:|---|---|"]
    for _, e in res["checkpoints"].items():
        ci = e["PRIMARY_ci95"]
        lines.append(
            f"| {e['label']} | {e['abstain_pruned']:.3f} | {e['abstain_answerable']:.3f} | "
            f"**{e['PRIMARY_discrimination_pp']:+.2f} pp** | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | "
            f"{'**yes**' if e['excludes_zero'] else 'no'} |")
    lines += ["", "A large positive discrimination means abstention tracks whether the "
                  "problem is actually answerable. A value near zero would mean the rate "
                  "measures willingness to use the offered option and nothing more."]
    md = "\n".join(lines) + "\n"

    out = Path(args.run) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "abstain_control_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "CONTROL.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
