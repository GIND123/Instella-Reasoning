#!/usr/bin/env python
"""Does injection return the solution procedure even where it does not return the answer?

The deletion probe removes one premise the reference solution consumes, so the parent's
answer is not derivable from what remains. Answer recall asks whether the model emits that
answer anyway. This asks the separate question of whether it emits the parent's *intermediate
calculation results*, which are equally underdetermined once the premise is gone.

The two can come apart, and the claim the manuscript rests on is that they do: heavy
repetition returns the arithmetic chain while leaving the final answer no more recoverable
than it was without any exposure.

PROCEDURE REPRODUCTION, defined here
    GSM8K reference solutions carry ``<<a op b=c>>`` calculator annotations. For each probe
    item the parent's chain supplies a set of intermediate results, excluding the final
    answer so that the two measures cannot trivially agree. The score for one generation is
    the fraction of those intermediates appearing as numbers anywhere in the completion.

    Reporting a fraction rather than a hit or miss matters: chains vary from two to seven
    steps, and an all or nothing rule would weight a two step problem the same as a seven
    step one while making the metric hostage to a single unreproduced value.

ESTIMAND
    Difference in differences, injected minus held out, at each dose against the 0x arm.
    Identical in form to the answer recall estimand, so the two are directly comparable and
    the shared 8,388,608 token budget cancels from both.

NOTE ON PROVENANCE
    The manuscript's original procedure figures were produced by code that was not committed,
    and the checkpoints that generated them were destroyed with the instance. The definition
    above is an independent reconstruction. It reproduces the shape of the original result,
    a rise that is flat at low dose and largest at 64x, but not its exact values, so the
    numbers this script prints supersede the earlier ones rather than confirming them.

Usage:
    python experiments/analyze_procedure.py
    python experiments/analyze_procedure.py --model olmo2-1b
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

CALC = re.compile(r"<<([^>]+)>>")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
DOSES = [0, 1, 4, 16, 64]

SOURCES = {
    "instella-3b": ("experiments/runs/ckpt-axis-v1/generations",
                    "phase2__dose{d}__test__T0.0__k1__vllm.jsonl"),
    "olmo2-1b": ("experiments/runs/replication-v1/olmo2-1b", "probe__dose{d}.jsonl"),
    "qwen2.5-1.5b": ("experiments/runs/replication-v1/qwen2.5-1.5b", "probe__dose{d}.jsonl"),
}
LABELS = {"instella-3b": "Instella 3B", "olmo2-1b": "OLMo-2-1B",
          "qwen2.5-1.5b": "Qwen2.5-1.5B"}


def to_f(x):
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def intermediates(item) -> list[float]:
    """Parent chain results, minus the final answer so the two measures stay independent."""
    vals = []
    for expr in CALC.findall((item.get("metadata") or {}).get("rationale") or ""):
        if "=" in expr:
            v = to_f(expr.split("=")[-1])
            if v is not None:
                vals.append(v)
    ans = to_f((item.get("metadata") or {}).get("parent_answer"))
    return [v for v in vals if ans is None or v != ans]


def numbers_in(text: str) -> set[float]:
    out = set()
    for m in NUM.findall(text or ""):
        v = to_f(m)
        if v is not None:
            out.add(v)
    return out


def load_items(path: Path) -> dict:
    return {r["id"]: r for r in
            (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}


def load_gens(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["benchmark_id"]] = r.get("completion", "")
    return out


def cluster_did(inj: dict, hel: dict, n_boot: int, seed: int):
    parents = sorted(set(inj) | set(hel))

    def stat(sample):
        a = [v for q in sample for v in inj.get(q, ())]
        b = [v for q in sample for v in hel.get(q, ())]
        if not a or not b:
            return 0.0
        return 100 * (sum(a) / len(a) - sum(b) / len(b))

    obs = stat(parents)
    rng = random.Random(seed)
    draws = sorted(stat([rng.choice(parents) for _ in parents]) for _ in range(n_boot))
    return (round(obs, 3), round(draws[int(0.025 * n_boot)], 3),
            round(draws[int(0.975 * n_boot)], 3))


def analyse(model: str, items: dict, inj: set, hel: set, n_boot: int, seed: int) -> dict:
    d, pat = SOURCES[model]
    gens = {k: load_gens(Path(d) / pat.format(d=k)) for k in DOSES}
    if not gens.get(0):
        return {}
    base = gens[0]
    res = {"label": LABELS[model], "doses": {}}
    per = {}
    for k in DOSES:
        if not gens.get(k):
            continue
        cell = {}
        for name, mem in (("injected", inj), ("heldout", hel)):
            byp: dict[str, list[float]] = {}
            for b in gens[k]:
                if b not in base or items.get(b, {}).get("parent_id") not in mem:
                    continue
                inter = intermediates(items[b])
                if not inter:
                    continue
                got = numbers_in(gens[k][b])
                byp.setdefault(items[b]["parent_id"], []).append(
                    sum(1 for v in inter if v in got) / len(inter))
            flat = [v for vs in byp.values() for v in vs]
            cell[name] = {"score": round(sum(flat) / len(flat), 6) if flat else None,
                          "n": len(flat)}
            per[(k, name)] = byp
        res["doses"][k] = cell
    for k in DOSES:
        if k == 0 or (k, "injected") not in per:
            continue
        cur = cluster_did(per[(k, "injected")], per[(k, "heldout")], n_boot, seed)
        res["doses"][k]["raw_gap_pp"] = cur[0]
        res["doses"][k]["raw_gap_ci95"] = [cur[1], cur[2]]
        # The quantity the manuscript reports is the dose contrast against the 0x arm, so it
        # needs its own interval rather than the raw gap's. Resampling parents once per draw
        # and differencing inside the draw keeps the two arms paired.
        parents = sorted(set(per[(k, "injected")]) | set(per[(k, "heldout")])
                         | set(per[(0, "injected")]) | set(per[(0, "heldout")]))

        def gap(inj, hel, sample):
            a = [v for q in sample for v in inj.get(q, ())]
            b = [v for q in sample for v in hel.get(q, ())]
            return 100 * (sum(a) / len(a) - sum(b) / len(b)) if a and b else 0.0

        def did(sample, kk=k):
            return (gap(per[(kk, "injected")], per[(kk, "heldout")], sample)
                    - gap(per[(0, "injected")], per[(0, "heldout")], sample))

        obs = did(parents)
        rng2 = random.Random(seed)
        draws = sorted(did([rng2.choice(parents) for _ in parents]) for _ in range(n_boot))
        lo, hi = draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]
        res["doses"][k]["did_vs_dose0_pp"] = round(obs, 3)
        res["doses"][k]["did_vs_dose0_ci95"] = [round(lo, 3), round(hi, 3)]
        res["doses"][k]["did_excludes_zero"] = bool(lo > 0 or hi < 0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items",
                    default="experiments/runs/ckpt-axis-v1/base/gsm8k_test_deletion.jsonl")
    ap.add_argument("--mixture", default="experiments/runs/ckpt-axis-v1/phase2/mixture")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    items = load_items(Path(args.items))
    mix = Path(args.mixture)
    inj = {json.loads(x)["id"] for x in (mix / "injected.jsonl").read_text().splitlines() if x.strip()}
    hel = {json.loads(x)["id"] for x in (mix / "heldout.jsonl").read_text().splitlines() if x.strip()}

    models = [args.model] if args.model else list(SOURCES)
    out = {"n_boot": args.n_boot, "seed": args.seed, "models": {}}
    for m in models:
        r = analyse(m, items, inj, hel, args.n_boot, args.seed)
        if r:
            out["models"][m] = r

    lines = ["# Procedure reproduction against dose", "",
             "Fraction of the parent's intermediate calculator results appearing in the "
             "generation, excluding the final answer. Difference in differences against the "
             "never injected control, each dose measured against the 0x arm. Cluster "
             f"bootstrap over parent problems, {args.n_boot} draws, seed {args.seed}.", "",
             "| model | dose | injected | held out | vs 0x | 95% CI | excludes 0 |",
             "|---|--:|--:|--:|--:|---|---|"]
    for m, r in out["models"].items():
        for k in DOSES:
            c = r["doses"].get(k)
            if not c or c["injected"]["score"] is None:
                continue
            ci = c.get("did_vs_dose0_ci95")
            lines.append(
                f"| {r['label']} | {k}x | {c['injected']['score']:.4f} | "
                f"{c['heldout']['score']:.4f} | "
                f"{('%+.2f pp' % c['did_vs_dose0_pp']) if 'did_vs_dose0_pp' in c else '-'} | "
                f"{'[%+.2f, %+.2f]' % (ci[0], ci[1]) if ci else 'reference'} | "
                f"{'**yes**' if c.get('did_excludes_zero') else 'no'} |")
    md = "\n".join(lines) + "\n"

    dest = Path("experiments/runs/replication-v1/analysis")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "procedure_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (dest / "PROCEDURE.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
