#!/usr/bin/env python
"""The abstention result and its control, computed from generations rather than constants.

Panel (a) is the developmental ladder: how well each checkpoint separates problems that
cannot be answered from problems that can, under a prompt that merely permits declining.
Panel (b) is the graded limit: detection against the length of the solution chain, with the
answerable control on the same axes so that a reader can see the control is flat.

Both panels are derived from experiments/runs/abstain-v1/generations, so neither can drift
from the tables.

    python paper/figures/make_abstention_fig.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GEN = ROOT / "experiments/runs/abstain-v1/generations"
BASE = ROOT / "experiments/runs/ckpt-axis-v1/base"

BLUE, RED, GREY = "#1f5fa9", "#b3202c", "#8a8a8a"
SEED, NBOOT = 6198, 4000
TAGS = [("stage1", "Stage 1"), ("stage2", "Instella 3B"), ("sft", "SFT"), ("instruct", "Instruct")]

ABSTAIN = re.compile(
    r"####\s*unanswerable|\bunanswerable\b"
    r"|\b(cannot|can't|can not|unable to|impossible to|not possible to)\s+"
    r"(be\s+)?(determined|solved|answered|calculated|computed|known)"
    r"|\bnot enough information\b|\binsufficient information\b"
    r"|\bmissing information\b|\bunderdetermined\b", re.IGNORECASE)
CALC = re.compile(r"<<([^>]+)>>")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def chain_lengths() -> dict:
    steps = {}
    for fn in ("gsm8k_test_parents.jsonl", "gsm8k_train_parents.jsonl"):
        for x in (BASE / fn).open(encoding="utf-8"):
            r = json.loads(x)
            steps[r["id"]] = len(CALC.findall((r.get("metadata") or {}).get("rationale") or ""))
    for fn in ("gsm8k_test_deletion.jsonl", "gsm8k_train_deletion.jsonl"):
        for x in (BASE / fn).open(encoding="utf-8"):
            r = json.loads(x)
            if not steps.get(r["parent_id"]):
                steps[r["parent_id"]] = len(
                    CALC.findall((r.get("metadata") or {}).get("rationale") or ""))
    return steps


def load(path: pathlib.Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def by_parent(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["parent_id"]].append(1.0 if ABSTAIN.search(r.get("completion", "") or "") else 0.0)
    return d


def boot(a, b, n_boot=NBOOT, seed=SEED):
    parents = sorted(set(a) | set(b))

    def stat(sample):
        x = [v for q in sample for v in a.get(q, ())]
        y = [v for q in sample for v in b.get(q, ())]
        if not x or not y:
            return 0.0
        return 100 * (sum(x) / len(x) - sum(y) / len(y))

    obs = stat(parents)
    rng = random.Random(seed)
    d = sorted(stat([rng.choice(parents) for _ in parents]) for _ in range(n_boot))
    return obs, d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]


def main():
    steps = chain_lengths()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    labels, est, lo, hi = [], [], [], []
    for tag, label in TAGS:
        p, c = load(GEN / f"abstain__{tag}.jsonl"), load(GEN / f"abstain_control__{tag}.jsonl")
        if not p or not c:
            continue
        o, l, h = boot(by_parent(p), by_parent(c))
        labels.append(label)
        est.append(o)
        lo.append(o - l)
        hi.append(h - o)
    xs = range(len(labels))
    a.errorbar(xs, est, yerr=[lo, hi], color=BLUE, marker="o", markersize=4, capsize=2.6,
               elinewidth=0.85, markerfacecolor="white", markeredgewidth=0.9)
    a.axhline(0, color=GREY, linewidth=0.7)
    a.set_xticks(list(xs))
    a.set_xticklabels(labels)
    a.set_ylabel("Discrimination\n(underdetermined minus answerable)")
    a.set_xlim(-0.4, len(labels) - 0.6)
    a.text(-0.02, 1.06, "(a)", transform=a.transAxes, fontweight="bold", fontsize=9)

    tag = "instruct" if (GEN / "abstain__instruct.jsonl").exists() else "sft"
    lab = dict(TAGS)[tag]
    for path, colour, marker, ls, name in (
            (GEN / f"abstain__{tag}.jsonl", RED, "s", "-", "Underdetermined"),
            (GEN / f"abstain_control__{tag}.jsonl", BLUE, "o", "--", "Answerable control")):
        rows = load(path)
        d = collections.defaultdict(list)
        for r in rows:
            s = steps.get(r["parent_id"])
            if s:
                d[s].append(1.0 if ABSTAIN.search(r.get("completion", "") or "") else 0.0)
        ks = sorted(k for k in d if len(d[k]) >= 40)
        b.plot(ks, [sum(d[k]) / len(d[k]) for k in ks], color=colour, marker=marker,
               linestyle=ls, markersize=4, markerfacecolor="white", markeredgewidth=0.9,
               label=name)
    b.set_xlabel("Solution chain length (calculator steps)")
    b.set_ylabel("Abstention rate")
    b.set_ylim(bottom=-0.02)
    b.yaxis.set_major_locator(MultipleLocator(0.1))
    b.legend(loc="upper right", handlelength=1.7)
    b.text(-0.02, 1.06, "(b)", transform=b.transAxes, fontweight="bold", fontsize=9)
    b.set_title(lab, fontsize=8)

    fig.subplots_adjust(wspace=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"mathai_fig_abstention.{ext}")
    print("wrote mathai_fig_abstention")


if __name__ == "__main__":
    main()
