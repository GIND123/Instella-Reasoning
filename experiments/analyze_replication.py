#!/usr/bin/env python
"""Does the injection dissociation hold on base models AMD did not train?

The Instella-3B injection arms showed the two things coming apart under a fixed token
budget: verbatim reproduction of the dosed documents climbs steeply with dose, while recall
of the deleted parent's answer on those same items does not follow. A single model family
cannot separate "this is how language models behave" from "this is how AMD's data pipeline
behaves", and the paper's own limitations say so.

This scores the same contrast on OLMo-2-1B and Qwen2.5-1.5B. Same 200 injected documents,
same 200 held-out controls, same fixed 8,388,608 token budget, same seed, same four-shot
no-chat-template probe. Only the base checkpoint changes.

What Instella-3B actually showed, which is the thing to be replicated
    From 0x to 64x, verbatim continuation of the dosed documents went 0.133 -> 0.946, while
    recall of the deleted parent's answer on those same items went 0.0427 -> 0.0741. That
    recall rise is real, not a null: exact McNemar gives p = 0.035, 17 gained against 6
    lost. The held-out contrast moved less and did not separate, 0.0270 -> 0.0431 at
    p = 0.238.

    So the claim is a dissociation of magnitude, not an absence of effect. A model that
    reproduces a document verbatim roughly nineteen times in twenty recovers the deleted
    answer three points more often than one that never saw it. Near-total memorisation buys
    almost no recall.

PRIMARY ENDPOINT, fixed before the arms were run
    Within each model, recall on injected parents at 64x minus recall at 0x, paired by item
    and tested with exact McNemar, with the held-out contrast as the control.

    The dissociation replicates if verbatim continuation at 64x is high while the injected
    recall gain stays in single digits of percentage points on a comparable base. It fails
    to replicate if injected recall climbs toward the verbatim rate, which would mean the
    Instella result understated what memorisation does and the gap was a property of that
    model rather than of the mechanism.

A read of this table is only meaningful where positive_controls_pass is true. An arm whose
injected loss did not fall has not memorised anything, and its flat recall measures a
broken training run.

Usage:
    python experiments/analyze_replication.py --run experiments/runs/replication-v1
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

MODELS = [("olmo2-1b", "OLMo-2-1B"), ("qwen2.5-1.5b", "Qwen2.5-1.5B")]
DOSES = [0, 64]


def _to_float(x):
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _extract(completion: str):
    import re
    if not completion:
        return None
    hits = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", completion)
    if hits:
        return _to_float(hits[-1])
    hits = re.findall(r"-?\d[\d,]*(?:\.\d+)?", completion)
    return _to_float(hits[-1]) if hits else None


def _recall(parent_answer, completion: str) -> bool:
    gold, pred = _to_float(parent_answer), _extract(completion)
    return bool(gold is not None and pred is not None and gold == pred)


def mcnemar(a_hit: dict, b_hit: dict, ids: list[str]) -> tuple[int, int, float]:
    """Exact two-sided McNemar. b = gained under B, c = lost."""
    b = sum(1 for i in ids if b_hit[i] and not a_hit[i])
    c = sum(1 for i in ids if a_hit[i] and not b_hit[i])
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return b, c, p


def _load_gens(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["benchmark_id"]] = r
    return out


def _members(mix: Path, name: str) -> set:
    return {json.loads(x)["id"] for x in open(mix / name, encoding="utf-8") if x.strip()}


def analyse(run: Path, mix: Path, items_path: Path) -> dict:
    items = {}
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items[r["id"]] = r
    inj, hel = _members(mix, "injected.jsonl"), _members(mix, "heldout.jsonl")

    res = {"run": str(run), "n_injected": len(inj), "n_heldout": len(hel), "models": {}}
    for alias, label in MODELS:
        base_dir = run / alias
        gens = {d: _load_gens(base_dir / f"probe__dose{d}.jsonl") for d in DOSES}
        if not all(gens[d] for d in DOSES):
            continue
        entry = {"label": label, "doses": {}}
        for d in DOSES:
            rp = base_dir / f"dose{d}_report.json"
            if rp.exists():
                rep = json.loads(rp.read_text())
                entry["doses"][d] = {
                    "injected_share": rep.get("injected_share"),
                    "verbatim_post": rep.get("post", {}).get("injected_verbatim"),
                    "injected_loss_post": rep.get("post", {}).get("injected_loss"),
                    "heldout_loss_post": rep.get("post", {}).get("heldout_loss"),
                    "positive_controls_pass": rep.get("positive_controls_pass"),
                }
        base = gens[0]
        for name, member in (("injected", inj), ("heldout", hel)):
            sel = [b for b in gens[64]
                   if b in base and (items.get(b, {}).get("parent_id")) in member]
            if not sel:
                continue
            pa = {b: (items[b].get("metadata") or {}).get("parent_answer") for b in sel}
            cur = {b: _recall(pa[b], gens[64][b].get("completion", "")) for b in sel}
            ref = {b: _recall(pa[b], base[b].get("completion", "")) for b in sel}
            gained, lost, p = mcnemar(ref, cur, sel)
            entry[name] = {
                "n": len(sel),
                "recall_dose0": round(sum(ref.values()) / len(sel), 6),
                "recall_dose64": round(sum(cur.values()) / len(sel), 6),
                "delta_pp": round(100 * (sum(cur.values()) - sum(ref.values())) / len(sel), 3),
                "gained": gained, "lost": lost, "mcnemar_p": round(p, 6),
            }
        res["models"][alias] = entry
    return res


def render(res: dict) -> str:
    L = ["# Injection replication on base models outside AMD", "",
         "Same 200 injected documents, same 200 held-out controls, same fixed "
         "8,388,608 token budget, same seed 6198, same four-shot no-chat-template probe. "
         "Only the base checkpoint changes. Exact two-sided McNemar, paired by item.", "",
         "## Memorisation actually happened", "",
         "| model | dose | injected share | verbatim after | injected loss after | "
         "held-out loss after | controls |", "|---|--:|--:|--:|--:|--:|---|"]
    for _, e in res["models"].items():
        for d, c in e["doses"].items():
            L.append(f"| {e['label']} | {d}x | {c['injected_share']:.3f} | "
                     f"{c['verbatim_post']:.3f} | {c['injected_loss_post']:.3f} | "
                     f"{c['heldout_loss_post']:.3f} | "
                     f"{'pass' if c['positive_controls_pass'] else '**FAIL**'} |")
    L += ["", "## PRIMARY - recall of the deleted parent's answer, 64x against 0x", "",
          "| model | set | n | recall 0x | recall 64x | delta | gained | lost | McNemar p |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for _, e in res["models"].items():
        for name in ("injected", "heldout"):
            c = e.get(name)
            if c:
                L.append(f"| {e['label']} | {name} | {c['n']} | {c['recall_dose0']:.4f} | "
                         f"{c['recall_dose64']:.4f} | {c['delta_pp']:+.2f} pp | "
                         f"{c['gained']} | {c['lost']} | {c['mcnemar_p']:.4f} |")
    L += ["", "Read the primary table only where controls pass. A model that did not "
              "memorise the dosed documents cannot inform a claim about what memorisation "
              "does to recall."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/replication-v1")
    ap.add_argument("--mixture", default="experiments/runs/ckpt-axis-v1/phase2/mixture")
    ap.add_argument("--items", default="experiments/runs/ckpt-axis-v1/base/gsm8k_test_deletion.jsonl")
    args = ap.parse_args()
    run = Path(args.run)
    res = analyse(run, Path(args.mixture), Path(args.items))
    if not res["models"]:
        print(f"[replication] no complete dose pair under {run}")
        return 1
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = render(res)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
