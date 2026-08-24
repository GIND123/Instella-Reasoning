"""Regenerate every figure for both manuscripts from the underlying run data.

Design rules, applied uniformly. No chart titles: the caption carries the description, and
a title duplicated inside the artwork is wasted column space. No interpretive annotation
inside the axes either, since a phrase like "crosses zero" states a conclusion the reader
should draw from the geometry. Axis labels carry units. Error bars are 95 percent cluster
bootstraps over parent problems everywhere, matching the inference used in the text.
Colours are print safe and separable in greyscale by also varying marker and line style.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

INK = "#1a1a1a"
BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#7f7f7f"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- MATH-AI Figure 1
def fig_trajectory():
    d = json.load(open("fig_data_trajectory.json"))
    labels = ["Stage 1", "Instella 3B", "SFT", "Instruct"]
    x = range(len(labels))
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(5.4, 1.55), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    for arm, colour, marker, ls, name in (
        ("train", BLUE, "o", "-", "GSM8K train (in stage-two corpus)"),
        ("test", RED, "s", "--", "GSM8K test (verified absent)"),
    ):
        y = [d[f"{arm}|{label}"]["rate"] for label in labels]
        lo = [y[i] - d[f"{arm}|{label}"]["lo"] for i, label in enumerate(labels)]
        hi = [d[f"{arm}|{label}"]["hi"] - y[i] for i, label in enumerate(labels)]
        ax.errorbar(
            x,
            y,
            yerr=[lo, hi],
            color=colour,
            marker=marker,
            linestyle=ls,
            markersize=4,
            linewidth=1.1,
            capsize=2.5,
            elinewidth=0.8,
            label=name,
        )

    ax.axvline(0.5, color=GREY, linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Deletion recall rate (%)")
    ax.set_ylim(0, 6.2)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="upper left", handlelength=1.8, bbox_to_anchor=(0.0, 1.02))
    ax.annotate(
        "stage-two corpus\nenters here",
        xy=(0.5, 0.45),
        xytext=(1.15, 0.62),
        fontsize=7,
        color=GREY,
        style="italic",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6),
    )
    ax.text(-0.02, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # difference in differences at each successive boundary, from the text
    bnames = ["Stage 1 to\nInstella 3B", "Stage 1\nto SFT", "Stage 1 to\nInstruct"]
    est = [-0.26, 0.25, 0.94]
    lo = [-1.66, -1.14, -0.46]
    hi = [1.14, 1.65, 2.41]
    ypos = range(len(bnames))
    ax2.axvline(0, color=INK, linewidth=0.8, zorder=0)
    ax2.errorbar(
        est,
        ypos,
        xerr=[
            [estimate - lower for estimate, lower in zip(est, lo)],
            [h - e for e, h in zip(est, hi)],
        ],
        fmt="D",
        color=INK,
        markersize=3.6,
        capsize=2.5,
        elinewidth=0.8,
        linestyle="none",
    )
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(bnames)
    ax2.set_ylim(-0.6, 2.6)
    ax2.invert_yaxis()
    ax2.set_xlabel("Difference in differences\n(percentage points)")
    ax2.set_xlim(-2.6, 2.9)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.text(-0.02, 1.04, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9)

    fig.subplots_adjust(wspace=0.42)
    save(fig, "mathai_fig1_trajectory")


# ---------------------------------------------------------------- MATH-AI Figure 2
def fig_dose():
    dose = [0, 1, 4, 16, 64]
    xi = range(len(dose))
    verbatim = [13.3, 15.3, 18.0, 49.1, 94.6]
    ans = [0.0, 0.06, 0.09, -0.22, 1.52]
    ans_lo = [0.0, -3.06, -2.42, -3.66, -1.93]
    ans_hi = [0.0, 3.20, 2.70, 3.07, 5.08]
    proc = [0.0, 0.48, 0.32, 2.10, 4.76]
    proc_lo = [0.0, -2.47, -2.33, -1.27, 1.30]
    proc_hi = [0.0, 3.29, 3.01, 5.33, 8.23]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(5.4, 1.55))

    ax.plot(xi, verbatim, color=INK, marker="o", markersize=4, linewidth=1.2)
    ax.set_xticks(list(xi))
    ax.set_xticklabels([f"{d}x" for d in dose])
    ax.set_ylabel("Verbatim reproduction (%)")
    ax.set_ylim(0, 104)
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.text(-0.02, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    off = 0.10
    ax2.axhline(0, color=GREY, linewidth=0.7, zorder=0)
    ax2.errorbar(
        [i - off for i in xi],
        proc,
        yerr=[
            [point - lower for point, lower in zip(proc, proc_lo)],
            [h - p for p, h in zip(proc, proc_hi)],
        ],
        fmt="o",
        color=BLUE,
        markersize=4,
        capsize=2.5,
        elinewidth=0.8,
        linestyle="none",
        label="Solution procedure",
    )
    ax2.errorbar(
        [i + off for i in xi],
        ans,
        yerr=[
            [point - lower for point, lower in zip(ans, ans_lo)],
            [h - a for a, h in zip(ans, ans_hi)],
        ],
        fmt="s",
        color=RED,
        markersize=4,
        capsize=2.5,
        elinewidth=0.8,
        linestyle="none",
        label="Final answer",
    )
    ax2.set_xticks(list(xi))
    ax2.set_xticklabels([f"{d}x" for d in dose])
    ax2.set_ylabel("Reproduction, injected minus\nheld out (percentage points)")
    ax2.set_ylim(-4.6, 9.2)
    ax2.yaxis.set_major_locator(MultipleLocator(3))
    ax2.legend(loc="upper left", handlelength=1.4)
    ax2.text(-0.02, 1.04, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9)

    fig.subplots_adjust(wspace=0.46, bottom=0.24)
    fig.supxlabel(
        "Repetitions of each injected document during continued pretraining", fontsize=9, y=0.015
    )
    save(fig, "mathai_fig2_dose")


# ---------------------------------------------------------------- JUDGe Figure 1
def fig_judge_dissociation():
    judges = ["Instella 3B\nInstruct", "Qwen2.5 7B\nInstruct", "Qwen2.5 14B\nInstruct"]
    xi = range(len(judges))
    ctrl = [-0.0541, -0.0359, -0.0230]
    ctrl_lo = [-0.083, -0.068, -0.057]
    ctrl_hi = [-0.026, -0.008, 0.006]
    free = [-0.0809, -0.2000, -0.1840]
    free_lo = [-0.109, -0.248, -0.231]
    free_hi = [-0.054, -0.149, -0.133]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhline(0, color=GREY, linewidth=0.7, zorder=0)
    off = 0.09
    ax.errorbar(
        [i - off for i in xi],
        ctrl,
        yerr=[
            [point - lower for point, lower in zip(ctrl, ctrl_lo)],
            [h - c for c, h in zip(ctrl, ctrl_hi)],
        ],
        fmt="o",
        color=BLUE,
        markersize=4.2,
        capsize=2.5,
        elinewidth=0.8,
        linestyle="-",
        linewidth=1.0,
        label="Reference given",
    )
    ax.errorbar(
        [i + off for i in xi],
        free,
        yerr=[
            [point - lower for point, lower in zip(free, free_lo)],
            [h - f for f, h in zip(free, free_hi)],
        ],
        fmt="s",
        color=RED,
        markersize=4.2,
        capsize=2.5,
        elinewidth=0.8,
        linestyle="--",
        linewidth=1.0,
        label="Reference withheld",
    )
    ax.set_xticks(list(xi))
    ax.set_xticklabels(judges)
    ax.set_ylabel("Balanced accuracy,\nseen minus unseen")
    ax.set_ylim(-0.29, 0.06)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(loc="lower left", handlelength=1.8)
    save(fig, "judge_fig1_dissociation")


# ---------------------------------------------------------------- JUDGe Figure 2
def fig_judge_ladder():
    judges = ["Instella 3B\nInstruct", "Qwen2.5 7B\nInstruct", "Qwen2.5 14B\nInstruct"]
    xi = range(len(judges))
    series = [
        ("Reference given, seen", [0.5137, 0.9370, 0.9459], BLUE, "o", "-"),
        ("Reference given, unseen", [0.5678, 0.9728, 0.9689], BLUE, "^", "--"),
        ("Reference withheld, seen", [0.5016, 0.5998, 0.6362], RED, "s", "-"),
        ("Reference withheld, unseen", [0.5825, 0.7998, 0.8202], RED, "v", "--"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhline(0.5, color=GREY, linewidth=0.7, linestyle=":", zorder=0)
    ax.text(2.42, 0.512, "chance", fontsize=7, color=GREY, style="italic", ha="right")
    for name, y, colour, marker, ls in series:
        ax.plot(
            xi,
            y,
            color=colour,
            marker=marker,
            linestyle=ls,
            markersize=4,
            linewidth=1.0,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=name,
        )
    ax.set_xticks(list(xi))
    ax.set_xticklabels(judges)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.45, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(loc="upper left", handlelength=1.9, ncol=1, labelspacing=0.25)
    save(fig, "judge_fig2_ladder")


# ---------------------------------------------------------------- JUDGe Figure 3
def fig_judge_kappa():
    """The kappa paradox: same data, two statistics, opposite conclusions."""
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    labels = ["Cohen's kappa", "Balanced accuracy"]
    seen = [0.011, 0.5137]
    unseen = [0.303, 0.5678]
    xi = [0, 1]
    w = 0.3
    ax.bar([i - w / 2 for i in xi], seen, w, color=BLUE, label="Seen arm", edgecolor="none")
    ax.bar([i + w / 2 for i in xi], unseen, w, color=RED, label="Unseen arm", edgecolor="none")
    for i, (s, u) in enumerate(zip(seen, unseen)):
        ax.text(i - w / 2, s + 0.015, f"{s:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, u + 0.015, f"{u:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xi)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Statistic value")
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="upper left", handlelength=1.2)
    save(fig, "judge_fig3_kappa")


if __name__ == "__main__":
    fig_trajectory()
    fig_dose()
    fig_judge_dissociation()
    fig_judge_ladder()
    fig_judge_kappa()


def fig_mathai_combined():
    """All four MATH-AI panels in one float: two on the trajectory, two on injection."""
    import json

    d = json.load(open("fig_data_trajectory.json"))
    labels = ["Stage 1", "Instella 3B", "SFT", "Instruct"]
    x = range(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.25))
    (ax, ax2), (ax3, ax4) = axes

    for arm, colour, marker, ls, name in (
        ("train", BLUE, "o", "-", "In stage-two corpus"),
        ("test", RED, "s", "--", "Verified absent"),
    ):
        y = [d[f"{arm}|{label}"]["rate"] for label in labels]
        lo = [y[i] - d[f"{arm}|{label}"]["lo"] for i, label in enumerate(labels)]
        hi = [d[f"{arm}|{label}"]["hi"] - y[i] for i, label in enumerate(labels)]
        ax.errorbar(
            x,
            y,
            yerr=[lo, hi],
            color=colour,
            marker=marker,
            linestyle=ls,
            markersize=3.6,
            linewidth=1.0,
            capsize=2.2,
            elinewidth=0.7,
            label=name,
        )
    ax.axvline(0.5, color=GREY, linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Deletion recall (%)")
    ax.set_ylim(0, 6.0)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.legend(loc="upper left", handlelength=1.6, fontsize=7)
    ax.text(-0.02, 1.06, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    bn = ["to Instella 3B", "to SFT", "to Instruct"]
    est = [-0.26, 0.25, 0.94]
    lo = [-1.66, -1.14, -0.46]
    hi = [1.14, 1.65, 2.41]
    yp = range(len(bn))
    ax2.axvline(0, color=INK, linewidth=0.8, zorder=0)
    ax2.errorbar(
        est,
        yp,
        xerr=[
            [estimate - lower for estimate, lower in zip(est, lo)],
            [h - e for e, h in zip(est, hi)],
        ],
        fmt="D",
        color=INK,
        markersize=3.2,
        capsize=2.2,
        elinewidth=0.7,
        linestyle="none",
    )
    ax2.set_yticks(list(yp))
    ax2.set_yticklabels(bn, fontsize=7)
    ax2.set_ylim(-0.6, 2.6)
    ax2.invert_yaxis()
    ax2.set_xlabel("Difference in differences (pp)", fontsize=8)
    ax2.set_xlim(-2.6, 2.9)
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    ax2.text(-0.02, 1.06, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=9)

    dose = [0, 1, 4, 16, 64]
    xi = range(len(dose))
    ax3.plot(
        xi, [13.3, 15.3, 18.0, 49.1, 94.6], color=INK, marker="o", markersize=3.6, linewidth=1.0
    )
    ax3.set_xticks(list(xi))
    ax3.set_xticklabels([f"{v}x" for v in dose], fontsize=7)
    ax3.set_ylabel("Verbatim reproduction (%)")
    ax3.set_xlabel("Repetitions of injected document", fontsize=8)
    ax3.set_ylim(0, 104)
    ax3.yaxis.set_major_locator(MultipleLocator(25))
    ax3.text(-0.02, 1.06, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=9)

    ans = [0.0, 0.06, 0.09, -0.22, 1.52]
    alo = [0.0, -3.06, -2.42, -3.66, -1.93]
    ahi = [0.0, 3.20, 2.70, 3.07, 5.08]
    pro = [0.0, 0.48, 0.32, 2.10, 4.76]
    plo = [0.0, -2.47, -2.33, -1.27, 1.30]
    phi = [0.0, 3.29, 3.01, 5.33, 8.23]
    off = 0.11
    ax4.axhline(0, color=GREY, linewidth=0.7, zorder=0)
    ax4.errorbar(
        [i - off for i in xi],
        pro,
        yerr=[
            [point - lower for point, lower in zip(pro, plo)],
            [h - p for p, h in zip(pro, phi)],
        ],
        fmt="o",
        color=BLUE,
        markersize=3.6,
        capsize=2.2,
        elinewidth=0.7,
        linestyle="none",
        label="Procedure",
    )
    ax4.errorbar(
        [i + off for i in xi],
        ans,
        yerr=[
            [point - lower for point, lower in zip(ans, alo)],
            [h - a for a, h in zip(ans, ahi)],
        ],
        fmt="s",
        color=RED,
        markersize=3.6,
        capsize=2.2,
        elinewidth=0.7,
        linestyle="none",
        label="Final answer",
    )
    ax4.set_xticks(list(xi))
    ax4.set_xticklabels([f"{v}x" for v in dose], fontsize=7)
    ax4.set_xlabel("Repetitions of injected document", fontsize=8)
    ax4.set_ylabel("Injected minus held out (pp)")
    ax4.set_ylim(-4.6, 9.2)
    ax4.yaxis.set_major_locator(MultipleLocator(4))
    ax4.legend(loc="upper left", handlelength=1.3, fontsize=7)
    ax4.text(-0.02, 1.06, "(d)", transform=ax4.transAxes, fontweight="bold", fontsize=9)

    fig.subplots_adjust(wspace=0.42, hspace=0.62)
    save(fig, "mathai_fig_combined")


fig_mathai_combined()
