#!/usr/bin/env python
"""The injection result: what heavy exposure returns, and what it does not.

Three panels, all computed from committed artefacts rather than constants.

  (a) memorisation took hold, and did so on every model. Without this the other two panels
      are uninterpretable, since a flat dose response from a model that never memorised
      anything measures a broken training run.
  (b) reproduction of the deleted answer against dose, injected minus a never injected
      control, on all four models. Flat everywhere.
  (c) the dissociation on Instella, where the memorised calculation chain returns while the
      answer does not.

    python paper/figures/make_dose_fig.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REP = ROOT / "experiments/runs/replication-v1"
CKPT = ROOT / "experiments/runs/ckpt-axis-v1"

BLUE, RED, GREY, DARK = "#1f5fa9", "#b3202c", "#8a8a8a", "#222222"
DOSES = [0, 1, 4, 16, 64]
MODELS = [("instella-3b", "Instella 3B", DARK, "o", "-"),
          ("qwen2.5-3b", "Qwen2.5 3B", BLUE, "^", "--"),
          ("qwen2.5-1.5b", "Qwen2.5 1.5B", "#4d8fd1", "v", "--"),
          ("olmo2-1b", "OLMo-2 1B", "#7f9bb5", "s", ":")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def verbatim(model: str) -> dict:
    """Post-training verbatim continuation rate per dose."""
    out = {}
    if model == "instella-3b":
        for d in DOSES:
            p = CKPT / "phase2" / f"dose{d}_report.json"
            if p.exists():
                out[d] = json.loads(p.read_text())["post"]["injected_verbatim"]
        return out
    for d in DOSES:
        p = REP / model / f"dose{d}_report.json"
        if p.exists():
            out[d] = json.loads(p.read_text())["post"]["injected_verbatim"]
    return out


def answer_did() -> dict:
    """Answer reproduction, injected minus held out, each dose against 0x."""
    out = {}
    rp = REP / "analysis" / "replication_results.json"
    if rp.exists():
        for m, e in json.loads(rp.read_text())["models"].items():
            out[m] = {int(k): (v["did_pp"], v["did_ci95"])
                      for k, v in e.get("contrasts", {}).items() if "did_pp" in v}
    ck = CKPT / "analysis" / "ckpt_axis_results.json"
    if ck.exists():
        rows = json.loads(ck.read_text()).get("phase2_dose_response", [])
        base = {r["dose"]: r for r in rows}
        b0 = base.get(0)
        if b0:
            ref = (b0["injected_recall"] - b0["heldout_recall"])
            out["instella-3b"] = {
                r["dose"]: (100 * ((r["injected_recall"] - r["heldout_recall"]) - ref), None)
                for r in rows if r["dose"]}
    return out


def procedure_did() -> dict:
    p = REP / "analysis" / "procedure_results.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())["models"].get("instella-3b", {})
    return {int(k): (v["did_vs_dose0_pp"], v.get("did_vs_dose0_ci95"))
            for k, v in d.get("doses", {}).items() if "did_vs_dose0_pp" in v}


def main():
    fig, (a, c) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    xi = range(len(DOSES))

    for key, label, colour, marker, ls in MODELS:
        v = verbatim(key)
        xs = [i for i, d in enumerate(DOSES) if d in v]
        if xs:
            a.plot(xs, [v[DOSES[i]] for i in xs], color=colour, marker=marker, linestyle=ls,
                   markersize=4, markerfacecolor="white", markeredgewidth=0.9, label=label)
    a.set_xticks(list(xi))
    a.set_xticklabels([f"{d}x" for d in DOSES])
    a.set_xlabel("Repetitions of each injected document")
    a.set_ylabel("Verbatim reproduction\nof injected documents")
    a.set_ylim(0, 1.05)
    a.yaxis.set_major_locator(MultipleLocator(0.25))
    a.legend(loc="upper left", handlelength=1.7)
    a.text(-0.02, 1.06, "(a)", transform=a.transAxes, fontweight="bold", fontsize=9)

    ans = answer_did()
    proc = procedure_did()
    ai = ans.get("instella-3b", {})
    for e, colour, marker, ls, label in ((proc, BLUE, "o", "-", "Solution procedure"),
                                         (ai, RED, "s", "--", "Final answer")):
        xs = [i for i, d in enumerate(DOSES) if d in e]
        if not xs:
            continue
        ys = [e[DOSES[i]][0] for i in xs]
        cis = [e[DOSES[i]][1] for i in xs]
        if all(ci for ci in cis):
            c.errorbar(xs, ys,
                       yerr=[[y - ci[0] for y, ci in zip(ys, cis)],
                             [ci[1] - y for y, ci in zip(ys, cis)]],
                       color=colour, marker=marker, linestyle=ls, markersize=4, capsize=2.2,
                       elinewidth=0.7, markerfacecolor="white", markeredgewidth=0.9,
                       label=label)
        else:
            c.plot(xs, ys, color=colour, marker=marker, linestyle=ls, markersize=4,
                   markerfacecolor="white", markeredgewidth=0.9, label=label)
    c.axhline(0, color=GREY, linewidth=0.7)
    c.set_xticks(list(xi))
    c.set_xticklabels([f"{d}x" for d in DOSES])
    c.set_xlabel("Repetitions of each injected document")
    c.set_ylabel("Reproduction, injected\nminus control (pp)")
    c.legend(loc="upper left", handlelength=1.7)
    c.text(-0.02, 1.06, "(b)", transform=c.transAxes, fontweight="bold", fontsize=9)

    fig.subplots_adjust(wspace=0.34)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"mathai_fig_dose.{ext}")
    print("wrote mathai_fig_dose")


if __name__ == "__main__":
    main()
