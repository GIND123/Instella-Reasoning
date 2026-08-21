"""F3 - the trajectory null: recall rises, but it rises on clean items too.

Two panels sharing one x-axis-free story.

Top: deletion-recall across the Instella trajectory, one line per arm. The train arm is
resident in the Stage-2 corpus; the test arm is verified absent from every corpus scanned
(0 items at containment >= 0.999 across seven scans). Both rise across the Stage-2 data
intervention, and they rise together. Lines rather than bars because the message is the
*parallelism* of the two trajectories, which paired bars make the reader reconstruct.

Bottom: the difference-in-differences with its interval, on the same percentage units, so
the reader can carry the top panel's gap straight down to the estimate. The zero line is
the reference; the interval crosses it.

Why this figure is cut first under a four-page limit: it makes the same point F1 and F2
already carry, and the DiD is a single number that survives fine in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

from instella_reasoning.analysis.figures import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    NEUTRAL,
    SERIES,
    SURFACE,
    _mpl,
    _style,
)
from instella_reasoning.analysis.mathai_f1_abstention import cluster_ci, save_both

ORDER = [
    ("amd/Instella-3B-Stage1", "Stage1"),
    ("amd/Instella-3B", "Instella-3B"),
    ("amd/Instella-3B-SFT", "SFT"),
    ("amd/Instella-3B-Instruct", "Instruct"),
]
ARMS = [("train", "GSM8K train (in corpus)", SERIES[0]),
        ("test", "GSM8K test (verified clean)", SERIES[1])]


def build(run: str | Path, out_dir: str | Path) -> list[str] | None:
    plt = _mpl()
    if plt is None:
        print("matplotlib not installed: pip install -e .[viz]")
        return None
    run, out_dir = Path(run), Path(out_dir)
    results = json.loads((run / "analysis" / "ckpt_axis_results.json").read_text())
    did = results["did"]

    from collections import defaultdict
    stats: dict[str, dict] = {}
    for arm, _, _ in ARMS:
        rows = [json.loads(x) for x in
                (run / "analysis" / f"phase1_{arm}_rows.jsonl").read_text().splitlines() if x.strip()]
        per_model = defaultdict(list)
        for r in rows:
            per_model[r["model"]].append(r)
        stats[arm] = {m: cluster_ci(v, "recall") for m, v in per_model.items()}

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(3.5, 3.6), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [3.1, 0.9], "hspace": 0.62})
    xs = list(range(len(ORDER)))

    for arm, label, colour in ARMS:
        above = arm == "test"
        pts, los, his, counts = [], [], [], []
        for model, _ in ORDER:
            r, lo, hi, hit = stats[arm].get(model, (0.0, 0.0, 0.0, 0))
            pts.append(r * 100)
            los.append((r - lo) * 100)
            his.append((hi - r) * 100)
            counts.append(hit)
        ax.plot(xs, pts, color=colour, linewidth=1.5, zorder=3)
        ax.errorbar(xs, pts, yerr=[los, his], fmt="o", markersize=4.2, color=colour,
                    ecolor=colour, elinewidth=1.0, capsize=2.0, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, label=label, zorder=4)
        for x, y, hi, lo_, c in zip(xs, pts, his, los, counts, strict=True):
            if above:
                ax.text(x, y + hi + 0.10, str(c), ha="center", va="bottom",
                        fontsize=5.8, color=colour, zorder=5)
            else:
                ax.text(x, y - lo_ - 0.10, str(c), ha="center", va="top",
                        fontsize=5.8, color=colour, zorder=5)

    _style(ax, plt, ylabel="deletion-recall (%)",
           title="Recall rises after Stage 2 - on clean items too",
           subtitle="same items, same decoding, four checkpoints")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in ORDER], fontsize=7.5, color=INK)
    ax.set_ylim(0, 6.4)
    leg = ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2,
                    borderpad=0.1, labelspacing=0.22)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    # The estimate, in the same units, directly beneath the gap it summarises.
    est, lo, hi = did["did"] * 100, did["ci95"][0] * 100, did["ci95"][1] * 100
    axd.axvline(0, color=NEUTRAL, linewidth=1.0, zorder=2)
    axd.errorbar([est], [0], xerr=[[est - lo], [hi - est]], fmt="o", markersize=5,
                 color=INK, ecolor=INK_SECONDARY, elinewidth=1.4, capsize=3, zorder=3)
    axd.set_yticks([])
    axd.set_ylim(-0.55, 0.55)
    axd.set_xlim(-2.2, 2.2)
    _style(axd, plt, xlabel="difference-in-differences (percentage points)")
    axd.grid(False)
    axd.spines["left"].set_visible(False)
    # Left of the zero line, not across it: text laid over a reference line reads as
    # if it were annotating the line rather than the estimate.
    axd.text(-2.1, -0.34, "crosses zero:\nno item-specific effect", fontsize=6.0,
             color=INK_SECONDARY, ha="left", va="bottom", linespacing=1.2)
    # Above the interval: below it lands on the tick labels at this panel height.
    axd.text(est, 0.30, f"{est:+.2f}  [{lo:+.2f}, {hi:+.2f}]", ha="center", va="bottom",
             fontsize=6.2, color=INK)

    fig.text(0.0, -0.13,
             "numerals: recall events (n=1591 train / 1588 test rows per checkpoint) · "
             "error bars and DiD interval: 95% cluster\nbootstrap over parent problems "
             "(4000 reps) · all rows generated with vLLM 0.8.5.post1, greedy",
             fontsize=5.4, color=INK_MUTED, linespacing=1.5)
    return save_both(fig, out_dir / "F3_trajectory_null", plt)


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "experiments/runs/ckpt-axis-v1"
    outd = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
    for p in build(run, outd) or []:
        print(f"wrote {p}")
