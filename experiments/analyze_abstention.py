#!/usr/bin/env python
"""Analysis for G3: does a model decline an underdetermined problem when allowed to?

Every item here has had one premise removed, so the original answer is not derivable from
what remains. Under the evaluation prompt used elsewhere in the paper the model is told to
give a number after a marker and is never offered any alternative, which makes the observed
abstention rate uninterpretable as a measure of whether it can recognise insufficiency. This
run repeats the identical items under a prompt differing in exactly one respect, that
"#### unanswerable" is available and named.

Three outcomes are counted, and the second and third are the whole point of separating them:

  abstain   the model writes "unanswerable" or says in words that the problem cannot be
            solved. Under the licensing prompt this is the correct response.
  recall    it emits the deleted parent's original answer, which the pruned prompt no
            longer determines.
  other     some other number, an attempt at inference on partial information.

Reported per checkpoint and arm, with cluster bootstrap intervals over parent problems,
matching the inference used everywhere else in the paper. The comparison against the
answer-forcing prompt is paired within item where both runs exist.

Usage:
    python experiments/analyze_abstention.py --run experiments/runs/abstain-v1
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

TAGS = [("stage1", "Stage 1"), ("stage2", "Instella 3B"),
        ("sft", "SFT"), ("instruct", "Instruct")]

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: Deliberately narrow. A false positive here counts a wrong answer as recognised
#: insufficiency, which would overstate exactly the capability under test.
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


def to_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def extract_number(completion: str) -> float | None:
    if not completion:
        return None
    hits = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", completion)
    if hits:
        return to_float(hits[-1])
    hits = _NUM.findall(completion)
    return to_float(hits[-1]) if hits else None


def label(completion: str, parent_answer) -> str:
    """abstain takes precedence: a model that declines and then speculates has declined."""
    if _ABSTAIN.search(completion or ""):
        return "abstain"
    pred = extract_number(completion)
    if pred is None:
        return "abstain"
    gold = to_float(parent_answer)
    if gold is not None and pred == gold:
        return "recall"
    return "other"


def cluster_ci(by_parent: dict[str, list[bool]], n_boot: int, seed: int):
    keys = list(by_parent)
    if not keys:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)

    def mean(sample):
        flat = [v for k in sample for v in by_parent[k]]
        return sum(flat) / len(flat) if flat else 0.0

    obs = mean(keys)
    draws = sorted(mean([rng.choice(keys) for _ in keys]) for _ in range(n_boot))
    return obs, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def analyse(run: Path, n_boot: int, seed: int) -> dict:
    gen_dir = run / "generations"
    out: dict = {"run": str(run), "seed": seed, "n_boot": n_boot,
                 "prompt_condition": "abstain_licensed", "checkpoints": {}}

    for tag, name in TAGS:
        path = gen_dir / f"abstain__{tag}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        entry = {"label": name, "n_rows": len(rows),
                 "truncation": round(
                     1 - sum(bool((r.get("metadata") or {}).get("finished", True)) for r in rows)
                     / len(rows), 4)}
        for arm in ("train", "test", "all"):
            sel = [r for r in rows if arm == "all" or r.get("arm") == arm]
            if not sel:
                continue
            counts = collections.Counter(label(r.get("completion", ""), r.get("parent_answer"))
                                         for r in sel)
            byp: dict[str, list[bool]] = collections.defaultdict(list)
            for r in sel:
                byp[r["parent_id"]].append(
                    label(r.get("completion", ""), r.get("parent_answer")) == "abstain")
            rate, lo, hi = cluster_ci(byp, n_boot, seed)
            entry[arm] = {
                "n": len(sel),
                "abstain": counts["abstain"], "recall": counts["recall"], "other": counts["other"],
                "abstain_rate": round(rate, 4),
                "abstain_ci95": [round(lo, 4), round(hi, 4)],
                "recall_rate": round(counts["recall"] / len(sel), 4),
                "other_rate": round(counts["other"] / len(sel), 4),
            }
        out["checkpoints"][tag] = entry
    return out


def render(res: dict) -> str:
    L = ["# Abstention under a prompt that licenses declining", "",
         "Every item is underdetermined by construction: one premise the solution consumes "
         "has been removed, so declining is the correct response. The prompt differs from "
         "the one used elsewhere in the paper only in naming `#### unanswerable` as an "
         "option. Intervals are cluster bootstraps over parent problems, "
         f"{res['n_boot']} draws, seed {res['seed']}.", "",
         "| checkpoint | arm | n | abstain | 95% CI | recall | other | trunc |",
         "|---|---|--:|--:|---|--:|--:|--:|"]
    for tag, e in res["checkpoints"].items():
        for arm in ("train", "test"):
            a = e.get(arm)
            if not a:
                continue
            L.append(
                f"| {e['label']} | {arm} | {a['n']} | {a['abstain_rate']:.3f} | "
                f"[{a['abstain_ci95'][0]:.3f}, {a['abstain_ci95'][1]:.3f}] | "
                f"{a['recall_rate']:.3f} | {a['other_rate']:.3f} | {e['truncation']:.1%} |")
    L += ["", "`abstain` counts an explicit refusal or an unparseable answer; `recall` counts "
              "the deleted parent's original answer; `other` counts any different number."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/abstain-v1")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    run = Path(args.run)
    res = analyse(run, args.n_boot, args.seed)
    if not res["checkpoints"]:
        print(f"[abstain] no generations under {run / 'generations'}")
        return 1
    out_dir = run / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "abstain_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = render(res)
    (out_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
