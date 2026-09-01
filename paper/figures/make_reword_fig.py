#!/usr/bin/env python
"""Robustness of the premise-verification result to the wording of the licence.

Panel (a) is the developmental ladder under both wordings, so a reader can see the shape is
preserved while the level is not. Panel (b) is the cross-family sweep under both wordings,
showing the rank order survives. Panel (c) is the false-decline rate, the cell that actually
moves, drawn as a slope chart because it moves in opposite directions across models.

All values are read from the committed analysis JSON, so the figure cannot drift from the
tables it illustrates.

    python paper/figures/make_reword_fig.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AN = ROOT / "experiments/runs/abstain-v1/analysis"

BLUE, RED, GREY = "#1f5fa9", "#b3202c", "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(name: str) -> dict:
    return json.loads((AN / f"{name}.json").read_text(encoding="utf-8"))["checkpoints"]


def main() -> None:
    orig, rw = load("abstain_control_results"), load("abstain_reword_results")
    fam_o, fam_r = load("abstain_families_results"), load("abstain_families_reword_results")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 2.5))

    # (a) the ladder, both wordings, with cluster-bootstrap intervals
    tags = ["stage1", "stage2", "sft", "instruct"]
    labels = ["Stage 1", "Instella 3B", "SFT", "Instruct"]
    x = range(len(tags))
    for src, colour, marker, name in ((orig, BLUE, "o", "original licence"),
                                      (rw, RED, "s", "reworded licence")):
        y = [src[t]["PRIMARY_discrimination_pp"] for t in tags]
        lo = [y[i] - src[t]["PRIMARY_ci95"][0] for i, t in enumerate(tags)]
        hi = [src[t]["PRIMARY_ci95"][1] - y[i] for i, t in enumerate(tags)]
        a.errorbar(x, y, yerr=[lo, hi], color=colour, marker=marker, markersize=3.5,
                   capsize=2, elinewidth=0.7, label=name)
    a.axhline(0, color=GREY, lw=0.6, ls=":")
    a.set_xticks(list(x)); a.set_xticklabels(labels, rotation=20, ha="right")
    a.set_ylabel("discrimination (pp)")
    a.set_title("(a) the ladder survives rewording", fontsize=8, loc="left")
    a.legend(frameon=False, loc="upper left")

    # (b) cross-family decision quality, both wordings
    fam = [("qwen2.5-3b-it", "Qwen2.5\n3B"), ("instruct", "Instella\n3B"),
           ("qwen2.5-1.5b-it", "Qwen2.5\n1.5B"), ("olmo2-1b-it", "OLMo-2\n1B")]
    xs = range(len(fam))
    w = 0.36
    yo = [(fam_o.get(t) or orig["instruct"])["balanced_accuracy"] for t, _ in fam]
    yr = [fam_r[t]["balanced_accuracy"] for t, _ in fam]
    b.bar([i - w / 2 for i in xs], yo, w, color=BLUE, label="original")
    b.bar([i + w / 2 for i in xs], yr, w, color=RED, label="reworded")
    b.axhline(0.5, color=GREY, lw=0.6, ls=":")
    b.set_ylim(0.45, 0.85)
    b.set_xticks(list(xs)); b.set_xticklabels([l for _, l in fam])
    b.set_ylabel("decision quality")
    b.set_title("(b) rank order is preserved", fontsize=8, loc="left")
    b.legend(frameon=False, ncol=2, loc="upper right")

    # (c) false decline, the cell that moves, and not in one direction
    rows = [("Qwen2.5-3B", (fam_o["qwen2.5-3b-it"])["abstain_answerable"],
             fam_r["qwen2.5-3b-it"]["abstain_answerable"]),
            ("OLMo-2-1B", (fam_o["olmo2-1b-it"])["abstain_answerable"],
             fam_r["olmo2-1b-it"]["abstain_answerable"]),
            ("Qwen2.5-1.5B", (fam_o["qwen2.5-1.5b-it"])["abstain_answerable"],
             fam_r["qwen2.5-1.5b-it"]["abstain_answerable"]),
            ("Instella-3B", orig["instruct"]["abstain_answerable"],
             rw["instruct"]["abstain_answerable"])]
    # Endpoints can coincide (OLMo-2 and Qwen2.5-3B both land near 0.19), so labels are
    # pushed apart on the y axis before drawing rather than overplotted.
    order = sorted(range(len(rows)), key=lambda i: rows[i][2])
    label_y, min_gap = {}, 0.022
    for rank, i in enumerate(order):
        y = rows[i][2]
        if rank and y - label_y[order[rank - 1]] < min_gap:
            y = label_y[order[rank - 1]] + min_gap
        label_y[i] = y
    for i, (name, o_, r_) in enumerate(rows):
        colour = RED if r_ > o_ else BLUE
        c.plot([0, 1], [o_, r_], color=colour, marker="o", markersize=3)
        c.annotate(name, (1, label_y[i]), xytext=(5, 0), textcoords="offset points",
                   fontsize=6.5, va="center", color=colour)
    c.set_xlim(-0.15, 1.75); c.set_xticks([0, 1])
    c.set_xticklabels(["original", "reworded"])
    c.set_ylabel("false decline rate")
    c.set_title("(c) wrong refusals move both ways", fontsize=8, loc="left")

    fig.subplots_adjust(wspace=0.42)
    for ext in ("pdf", "png"):
        out = HERE / f"mathai_fig_reword.{ext}"
        fig.savefig(out)
        print("wrote", out)
    fig.savefig(ROOT / "paper/mathai/mathai_fig_reword.pdf")
    print("wrote paper/mathai/mathai_fig_reword.pdf")


if __name__ == "__main__":
    main()
