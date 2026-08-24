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

What Instella-3B showed, and why the controlled contrast is the only readable one
    From 0x to 64x, verbatim continuation of the dosed documents went 0.133 -> 0.946.
    Recall of the deleted parent's answer on those same injected items went
    0.0427 -> 0.0741, which taken alone is significant at exact McNemar p = 0.035.

    That within-arm number must not be read as an injection effect. Every arm trains on
    8,388,608 tokens, so a model at 64x differs from one at 0x by all of that training and
    not only by the injection. The never-injected held-out parents move too, 0.0270 ->
    0.0431, and subtracting them leaves +1.52 points on [-1.93, +5.08]: the injection
    specific effect on answer recall does not separate from zero. Held-out controls exist
    precisely to absorb the shared training, and the uncontrolled contrast is what they
    correct.

PRIMARY ENDPOINT, fixed before the arms were run
    Difference in differences within each model: (injected recall at 64x minus injected
    recall at 0x) minus (held-out recall at 64x minus held-out recall at 0x). This matches
    the estimand the paper reports for Instella-3B and is the contrast the design supports.

    The dissociation replicates if verbatim continuation at 64x is high while that
    difference in differences stays indistinguishable from zero. It fails to replicate if
    answer recall tracks the verbatim rate once the shared training is removed, which would
    mean the Instella null was a property of that model rather than of the mechanism.

    Both single differences are reported alongside it, because a reader who sees only the
    difference in differences cannot tell a genuine null from two large effects cancelling.

A read of this table is only meaningful where positive_controls_pass is true. An arm whose
injected loss did not fall has not memorised anything, and its flat recall measures a
broken training run.

Usage:
    python experiments/analyze_replication.py --run experiments/runs/replication-v1
"""

from __future__ import annotations

import argparse
import json
import random
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


def cluster_did(inj_pairs: dict, hel_pairs: dict, n_boot: int, seed: int):
    """Cluster bootstrap of the difference in differences, resampling parent problems.

    Items sharing a parent are not independent: one deletion can be probed several ways and
    the same underlying problem drives all of them. Resampling items would understate the
    interval, so parents are the unit, matching the inference used for the checkpoint axis.
    """
    parents = sorted(set(inj_pairs) | set(hel_pairs))

    def stat(sample):
        di = [d for q in sample for d in inj_pairs.get(q, ())]
        dh = [d for q in sample for d in hel_pairs.get(q, ())]
        if not di or not dh:
            return 0.0
        return 100 * (sum(di) / len(di) - sum(dh) / len(dh))

    obs = stat(parents)
    rng = random.Random(seed)
    draws = sorted(stat([rng.choice(parents) for _ in parents]) for _ in range(n_boot))
    return (round(obs, 3), round(draws[int(0.025 * n_boot)], 3),
            round(draws[int(0.975 * n_boot)], 3))


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


def analyse(run: Path, mix: Path, items_path: Path,
            n_boot: int = 4000, seed: int = 6198) -> dict:
    items = {}
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items[r["id"]] = r
    inj, hel = _members(mix, "injected.jsonl"), _members(mix, "heldout.jsonl")

    res = {"run": str(run), "n_injected": len(inj), "n_heldout": len(hel),
       "n_boot": n_boot, "seed": seed, "models": {}}
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
        per_parent = {"injected": {}, "heldout": {}}
        for name, member in (("injected", inj), ("heldout", hel)):
            sel = [b for b in gens[64]
                   if b in base and (items.get(b, {}).get("parent_id")) in member]
            if not sel:
                continue
            pa = {b: (items[b].get("metadata") or {}).get("parent_answer") for b in sel}
            cur = {b: _recall(pa[b], gens[64][b].get("completion", "")) for b in sel}
            ref = {b: _recall(pa[b], base[b].get("completion", "")) for b in sel}
            gained, lost, p = mcnemar(ref, cur, sel)
            for b in sel:
                per_parent[name].setdefault(items[b]["parent_id"], []).append(
                    int(cur[b]) - int(ref[b]))
            entry[name] = {
                "n": len(sel),
                "recall_dose0": round(sum(ref.values()) / len(sel), 6),
                "recall_dose64": round(sum(cur.values()) / len(sel), 6),
                "delta_pp": round(100 * (sum(cur.values()) - sum(ref.values())) / len(sel), 3),
                "gained": gained, "lost": lost, "mcnemar_p": round(p, 6),
            }
        if "injected" in entry and "heldout" in entry:
            obs, lo, hi = cluster_did(per_parent["injected"], per_parent["heldout"],
                                      n_boot, seed)
            entry["PRIMARY_did_pp"] = obs
            entry["PRIMARY_did_ci95"] = [lo, hi]
            entry["PRIMARY_excludes_zero"] = bool(lo > 0 or hi < 0)
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
    L += ["", "## PRIMARY - difference in differences on answer recall, 64x against 0x", "",
          "Injected minus held-out, so the 8.4M tokens every arm trains on cancel.", "",
          "| model | injected delta | held-out delta | difference in differences | "
          "95% CI | excludes 0 |", "|---|--:|--:|--:|---|---|"]
    for _, e in res["models"].items():
        if "PRIMARY_did_pp" in e:
            ci = e["PRIMARY_did_ci95"]
            L.append(f"| {e['label']} | {e['injected']['delta_pp']:+.2f} pp | "
                     f"{e['heldout']['delta_pp']:+.2f} pp | "
                     f"**{e['PRIMARY_did_pp']:+.2f} pp** | "
                     f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                     f"{'**yes**' if e['PRIMARY_excludes_zero'] else 'no'} |")
    L += ["", "## The single differences behind it", "",
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
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    args = ap.parse_args()
    run = Path(args.run)
    res = analyse(run, Path(args.mixture), Path(args.items), args.n_boot, args.seed)
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
