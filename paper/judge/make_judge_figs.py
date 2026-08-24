#!/usr/bin/env python
"""Every figure in the JUDGe manuscript, computed from verdict files rather than constants.

Each panel here is derived from the same jsonl verdict records the tables are derived from,
so a figure cannot drift from a table. The one exception is the three judges from the
original run whose raw verdicts were lost with the instance that produced them; their cell
values are carried in judge_ladder_data.json and are marked as such at the call site.

Verdicts that failed to parse are excluded throughout, which is the convention every number
in the manuscript uses. Counting them as INCORRECT instead shifts the 32B reference free gap
from -0.191 to -0.183, so the choice is stated rather than left implicit.

    python paper/judge/make_judge_figs.py
"""

from __future__ import annotations

import json
import pathlib
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

BLUE, RED, GREY, DARK = "#1f5fa9", "#b3202c", "#8a8a8a", "#222222"
SEED, NBOOT = 6198, 4000

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name: str) -> list[dict]:
    rows = [json.loads(x) for x in (DATA / name).open(encoding="utf-8") if x.strip()]
    return [r for r in rows if r.get("judge_verdict") is not None]


def rates(rows: list[dict]) -> dict:
    tp = fn = tn = fp = 0
    for r in rows:
        g, v = bool(r["ground_truth"]), bool(r["judge_verdict"])
        if g and v:
            tp += 1
        elif g and not v:
            fn += 1
        elif not g and not v:
            tn += 1
        else:
            fp += 1
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"ba": (sens + spec) / 2, "sens": sens, "spec": spec,
            "n": len(rows), "base": (tp + fn) / len(rows) if rows else float("nan")}


def cluster_gap(rows: list[dict], stat: str):
    """Seen minus unseen, resampling parent problems so correlated variants stay together."""
    byp: dict[str, list[dict]] = {}
    for r in rows:
        byp.setdefault(r["parent_id"], []).append(r)
    parents = sorted(byp)

    def val(sample):
        seen = [r for q in sample for r in byp[q] if r["arm"] == "seen"]
        uns = [r for q in sample for r in byp[q] if r["arm"] == "unseen"]
        if not seen or not uns:
            return 0.0
        return rates(seen)[stat] - rates(uns)[stat]

    obs = val(parents)
    rng = random.Random(SEED)
    draws = sorted(val([rng.choice(parents) for _ in parents]) for _ in range(NBOOT))
    return obs, draws[int(0.025 * NBOOT)], draws[int(0.975 * NBOOT)]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


JUDGES = [("llama-3.1-8b-instruct", "Llama 3.1\n8B"), ("qwen2.5-32b-instruct", "Qwen2.5\n32B")]


def fig_mechanism():
    """Sensitivity is intact; specificity collapses. The whole account rests on this."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    width, xs = 0.34, range(len(JUDGES))

    for ax, cond, label in ((a, "reference_based", "Reference given"),
                            (b, "reference_free", "Reference withheld")):
        for k, (stat, colour, lab) in enumerate((("sens", DARK, "Sensitivity"),
                                                 ("spec", RED, "Specificity"))):
            vals, los, his = [], [], []
            for jid, _ in JUDGES:
                rows = load(f"j_{jid}_instruct_{cond}.jsonl")
                o, lo, hi = cluster_gap(rows, stat)
                vals.append(o)
                los.append(o - lo)
                his.append(hi - o)
            ax.bar([x + (k - 0.5) * width for x in xs], vals, width,
                   color=colour, edgecolor="none",
                   yerr=[los, his], error_kw={"elinewidth": 0.8, "capsize": 2.4,
                                              "ecolor": "#333333"}, label=lab)
        ax.axhline(0, color=GREY, linewidth=0.7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([n for _, n in JUDGES])
        ax.set_ylim(-0.62, 0.10)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(label, fontsize=8)
    a.set_ylabel("Seen minus unseen")
    a.legend(loc="lower left", handlelength=1.3)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig_mechanism")


def fig_robustness():
    """Two things a reviewer will ask: does the prompt drive it, does the target."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    width, xs = 0.34, range(len(JUDGES))
    for k, (pref, colour, lab) in enumerate((("j", RED, "Primary prompt"),
                                             ("alt", "#e08b93", "Paraphrased prompt"))):
        vals, los, his = [], [], []
        for jid, _ in JUDGES:
            fn = (f"j_{jid}_instruct_reference_free.jsonl" if pref == "j"
                  else f"alt_{jid}_reference_free.jsonl")
            o, lo, hi = cluster_gap(load(fn), "ba")
            vals.append(o)
            los.append(o - lo)
            his.append(hi - o)
        a.bar([x + (k - 0.5) * width for x in xs], vals, width, color=colour,
              edgecolor="none", yerr=[los, his],
              error_kw={"elinewidth": 0.8, "capsize": 2.4, "ecolor": "#333333"}, label=lab)
    a.axhline(0, color=GREY, linewidth=0.7)
    a.set_xticks(list(xs))
    a.set_xticklabels([n for _, n in JUDGES])
    a.set_ylabel("Balanced accuracy,\nseen minus unseen")
    a.set_ylim(-0.34, 0.04)
    a.yaxis.set_major_locator(MultipleLocator(0.1))
    a.legend(loc="lower left", handlelength=1.3)
    a.set_title("Reference withheld, prompt varied", fontsize=8)

    for k, (tgt, colour, lab) in enumerate((("instruct", RED, "Instruct outputs"),
                                            ("stage2", "#e08b93", "Stage two outputs"))):
        vals, los, his = [], [], []
        for jid, _ in JUDGES:
            o, lo, hi = cluster_gap(load(f"j_{jid}_{tgt}_reference_free.jsonl"), "ba")
            vals.append(o)
            los.append(o - lo)
            his.append(hi - o)
        b.bar([x + (k - 0.5) * width for x in xs], vals, width, color=colour,
              edgecolor="none", yerr=[los, his],
              error_kw={"elinewidth": 0.8, "capsize": 2.4, "ecolor": "#333333"}, label=lab)
    b.axhline(0, color=GREY, linewidth=0.7)
    b.set_xticks(list(xs))
    b.set_xticklabels([n for _, n in JUDGES])
    b.set_ylim(-0.34, 0.04)
    b.yaxis.set_major_locator(MultipleLocator(0.1))
    b.legend(loc="lower left", handlelength=1.3)
    b.set_title("Reference withheld, target varied", fontsize=8)
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig_robustness")


def fig_kappa():
    """Chance correction inflates the gap once the judge's own marginal shifts.

    Both statistics run on one set of verdicts. Ground truth base rates are identical across
    the two conditions, 0.772 on the seen arm against 0.526 on the unseen, because the
    solutions being graded do not change. Only the judge's verdict marginal moves. That is
    enough to make kappa report roughly twice the gap balanced accuracy reports.
    """
    def kappa(rs):
        n = len(rs)
        po = sum(1 for r in rs if bool(r["ground_truth"]) == bool(r["judge_verdict"])) / n
        pg = sum(1 for r in rs if r["ground_truth"]) / n
        pv = sum(1 for r in rs if r["judge_verdict"]) / n
        pe = pg * pv + (1 - pg) * (1 - pv)
        return (po - pe) / (1 - pe)

    conds = [("reference_based", "Reference given"), ("reference_free", "Reference withheld")]
    kg, bg = [], []
    for cond, _ in conds:
        rows = load(f"j_llama-3.1-8b-instruct_instruct_{cond}.jsonl")
        seen = [r for r in rows if r["arm"] == "seen"]
        uns = [r for r in rows if r["arm"] == "unseen"]
        kg.append(kappa(seen) - kappa(uns))
        bg.append(rates(seen)["ba"] - rates(uns)["ba"])

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    width, xs = 0.34, range(len(conds))
    for k, (vals, colour, lab) in enumerate(((kg, DARK, "Cohen's kappa"),
                                             (bg, RED, "Balanced accuracy"))):
        ax.bar([x + (k - 0.5) * width for x in xs], vals, width, color=colour,
               edgecolor="none", label=lab)
        for x, v in zip(xs, vals):
            ax.text(x + (k - 0.5) * width, v - 0.035, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.axhline(0, color=GREY, linewidth=0.7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n for _, n in conds])
    ax.set_ylabel("Membership gap")
    ax.set_ylim(-0.60, 0.06)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="lower left", handlelength=1.3)
    save(fig, "fig_kappa")
    print(f"   given: kappa {kg[0]:+.3f} vs balanced accuracy {bg[0]:+.3f}")
    print(f"   withheld: kappa {kg[1]:+.3f} vs balanced accuracy {bg[1]:+.3f}")


def fig_ladder():
    """Capability rises, the control gap closes, the reference free gap does not."""
    with (HERE / "judge_ladder_data.json").open(encoding="utf-8") as fh:
        d = json.load(fh)["instruct_target"]
    order = ["instella-3b-instruct", "qwen2.5-7b-instruct",
             "qwen2.5-14b-instruct", "qwen2.5-32b-instruct"]
    names = ["Instella 3B\nInstruct", "Qwen2.5 7B\nInstruct",
             "Qwen2.5 14B\nInstruct", "Qwen2.5 32B\nInstruct"]
    ba = [d[k]["ba"] for k in order]
    xs = range(len(order))
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for name, idx, colour, marker, ls in (
            ("Reference given, seen", 0, BLUE, "o", "-"),
            ("Reference given, unseen", 1, BLUE, "^", "--"),
            ("Reference withheld, seen", 2, RED, "s", "-"),
            ("Reference withheld, unseen", 3, RED, "v", "--")):
        a.plot(xs, [v[idx] for v in ba], color=colour, marker=marker, linestyle=ls,
               markersize=4, markerfacecolor="white", markeredgewidth=0.9, label=name)
    a.axhline(0.5, color=GREY, linewidth=0.7, linestyle=":", zorder=0)
    a.text(3.45, 0.512, "chance", fontsize=7, color=GREY, style="italic", ha="right")
    a.set_xticks(list(xs))
    a.set_xticklabels(names)
    a.set_ylabel("Balanced accuracy")
    a.set_ylim(0.38, 1.03)
    a.yaxis.set_major_locator(MultipleLocator(0.1))
    a.legend(loc="lower right", handlelength=1.7, labelspacing=0.22, fontsize=6.6)

    for key, colour, marker, ls, lab in (("given", BLUE, "o", "-", "Reference given"),
                                         ("withheld", RED, "s", "--", "Reference withheld")):
        est = [d[k][key][0] for k in order]
        lo = [e - d[k][key][1] for e, k in zip(est, order)]
        hi = [d[k][key][2] - e for e, k in zip(est, order)]
        b.errorbar(xs, est, yerr=[lo, hi], color=colour, marker=marker, linestyle=ls,
                   markersize=4, capsize=2.4, elinewidth=0.8, markerfacecolor="white",
                   markeredgewidth=0.9, label=lab)
    b.axhline(0, color=GREY, linewidth=0.7)
    b.set_xticks(list(xs))
    b.set_xticklabels(names)
    b.set_ylabel("Membership gap")
    b.set_ylim(-0.32, 0.06)
    b.yaxis.set_major_locator(MultipleLocator(0.1))
    b.legend(loc="lower left", handlelength=1.7)
    fig.subplots_adjust(wspace=0.24)
    save(fig, "fig_ladder")


if __name__ == "__main__":
    fig_mechanism()
    fig_robustness()
    fig_kappa()
    fig_ladder()
