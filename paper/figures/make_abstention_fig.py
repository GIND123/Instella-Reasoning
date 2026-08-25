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
MATHGEN = ROOT / "experiments/runs/math-probe-v1/generations"
BASE = ROOT / "experiments/runs/ckpt-axis-v1/base"

BLUE, RED, GREY = "#1f5fa9", "#b3202c", "#8a8a8a"
SEED, NBOOT = 6198, 4000
TAGS = [("stage1", "Stage 1"), ("stage2", "Instella 3B"), ("sft", "SFT"), ("instruct", "Instruct")]
FAMILIES = [("instruct", "Instella\n3B", GEN, "abstain__{t}.jsonl"),
            ("qwen2.5-3b-it", "Qwen2.5\n3B", GEN, "abstain_pruned__{t}.jsonl"),
            ("qwen2.5-1.5b-it", "Qwen2.5\n1.5B", GEN, "abstain_pruned__{t}.jsonl"),
            ("olmo2-1b-it", "OLMo-2\n1B", GEN, "abstain_pruned__{t}.jsonl")]

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


def boot_bal(a, b, n_boot=NBOOT, seed=SEED):
    """Balanced accuracy of the decline decision, which a raw difference of rates hides."""
    parents = sorted(set(a) | set(b))

    def stat(sample):
        x = [v for q in sample for v in a.get(q, ())]
        y = [v for q in sample for v in b.get(q, ())]
        if not x or not y:
            return 0.0
        return (sum(x) / len(x) + (1 - sum(y) / len(y))) / 2

    obs = stat(parents)
    rng = random.Random(seed)
    d = sorted(stat([rng.choice(parents) for _ in parents]) for _ in range(n_boot))
    return obs, d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]


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
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 2.4))

    # (a) the ladder, in both domains
    for gendir, pat, colour, marker, ls, name in (
            (GEN, "abstain__{t}.jsonl", BLUE, "o", "-", "GSM8K"),
            (MATHGEN, "abstain_pruned__{t}.jsonl", RED, "s", "--", "MATH")):
        xs, ys, lo, hi = [], [], [], []
        for i, (tag, _) in enumerate(TAGS):
            p_ = load(gendir / pat.format(t=tag))
            c_ = load(gendir / f"abstain_control__{tag}.jsonl")
            if not p_ or not c_:
                continue
            o, l, h = boot_bal(by_parent(p_), by_parent(c_))
            xs.append(i)
            ys.append(o)
            lo.append(o - l)
            hi.append(h - o)
        if xs:
            a.errorbar(xs, ys, yerr=[lo, hi], color=colour, marker=marker, linestyle=ls,
                       markersize=4, capsize=2.4, elinewidth=0.8, markerfacecolor="white",
                       markeredgewidth=0.9, label=name)
    a.axhline(0.5, color=GREY, linewidth=0.7, linestyle=":")
    a.text(3.42, 0.507, "chance", fontsize=7, color=GREY, style="italic", ha="right")
    a.set_xticks(range(len(TAGS)))
    a.set_xticklabels([lab for _, lab in TAGS], fontsize=7)
    a.set_ylabel("Balanced accuracy\nof the decline decision")
    a.set_ylim(0.46, 0.82)
    a.yaxis.set_major_locator(MultipleLocator(0.1))
    a.legend(loc="upper left", handlelength=1.6)
    a.text(-0.02, 1.06, "(a)", transform=a.transAxes, fontweight="bold", fontsize=9)

    # (b) the depth decay, with its flat control
    tag = "instruct"
    for path, colour, marker, ls, name in (
            (GEN / f"abstain__{tag}.jsonl", RED, "s", "-", "Underdetermined"),
            (GEN / f"abstain_control__{tag}.jsonl", BLUE, "o", "--", "Answerable control")):
        rows = load(path)
        d = collections.defaultdict(list)
        for r in rows:
            st = steps.get(r["parent_id"])
            if st:
                d[st].append(1.0 if ABSTAIN.search(r.get("completion", "") or "") else 0.0)
        ks = sorted(k for k in d if len(d[k]) >= 40)
        b.plot(ks, [sum(d[k]) / len(d[k]) for k in ks], color=colour, marker=marker,
               linestyle=ls, markersize=4, markerfacecolor="white", markeredgewidth=0.9,
               label=name)
    b.set_xlabel("Solution chain length (calculator steps)")
    b.set_ylabel("Decline rate")
    b.set_ylim(bottom=-0.02)
    b.yaxis.set_major_locator(MultipleLocator(0.2))
    b.legend(loc="upper right", handlelength=1.6)
    b.text(-0.02, 1.06, "(b)", transform=b.transAxes, fontweight="bold", fontsize=9)

    # (c) across families
    labels, ys, lo, hi = [], [], [], []
    for tag, lab, gendir, pat in FAMILIES:
        p_ = load(gendir / pat.format(t=tag))
        c_ = load(gendir / f"abstain_control__{tag}.jsonl")
        if not p_ or not c_:
            continue
        o, l, h = boot_bal(by_parent(p_), by_parent(c_))
        labels.append(lab)
        ys.append(o)
        lo.append(o - l)
        hi.append(h - o)
    xs = range(len(labels))
    c.bar(list(xs), ys, 0.58, color=BLUE, edgecolor="none",
          yerr=[lo, hi], error_kw={"elinewidth": 0.85, "capsize": 2.6, "ecolor": "#333333"})
    c.axhline(0.5, color=GREY, linewidth=0.7, linestyle=":")
    c.text(len(labels) - 0.55, 0.507, "chance", fontsize=7, color=GREY, style="italic",
           ha="right")
    c.set_xticks(list(xs))
    c.set_xticklabels(labels, fontsize=7)
    c.set_ylabel("Balanced accuracy\nof the decline decision")
    c.set_ylim(0.46, 0.82)
    c.yaxis.set_major_locator(MultipleLocator(0.1))
    c.text(-0.02, 1.06, "(c)", transform=c.transAxes, fontweight="bold", fontsize=9)

    fig.subplots_adjust(wspace=0.38)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"mathai_fig_abstention.{ext}")
    print("wrote mathai_fig_abstention")


if __name__ == "__main__":
    main()
