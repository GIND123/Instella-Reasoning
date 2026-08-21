"""F1 — premise-insufficiency detection across the Instella trajectory.

The headline figure. Every item plotted here is *provably underdetermined*: a premise the
solution consumes has been deleted and the answer is no longer derivable from the prompt.
The correct response is to say so. The measured response, at every checkpoint and on both
arms, is to emit a confident number instead.

Form: grouped bars. The job is magnitude across a categorical sequence (the trajectory)
split by a second category (arm), which is what grouped bars are for. Checkpoints run in
trajectory order, never alphabetical — alphabetical puts Stage1 last when it is
developmentally first and silently reverses the story.

Scale: the y-axis stops at 5%, not 100%. At full scale every bar is invisible and the
figure says nothing; at data scale the bars look large and the near-zero finding is lost.
The annotation carries the ceiling so the truncated axis cannot mislead.

Colour: SERIES[0]/SERIES[1] from figures.py — the validated CVD-safe slots (worst adjacent
dE 9.1 under protanopia). Those slots carry a contrast WARN against the surface, which
obligates relief; the direct count labels on every bar supply it, and they are wanted
anyway because a rate of 2.1% built from 34 events must not look like a rate built from
34,000.

Error bars are cluster-bootstrap CIs over *parent problems*, matching the analysis:
deletion variants of one problem are correlated, and treating them as independent would
draw intervals narrower than the design supports.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from instella_reasoning.analysis.figures import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    _mpl,
    _style,
)

ORDER = [
    ("amd/Instella-3B-Stage1", "Stage1"),
    ("amd/Instella-3B", "Instella-3B"),
    ("amd/Instella-3B-SFT", "SFT"),
    ("amd/Instella-3B-Instruct", "Instruct"),
]
ARMS = [("train", "GSM8K train (in corpus)", SERIES[0]),
        ("test", "GSM8K test (verified clean)", SERIES[1])]


def cluster_ci(rows: list[dict], label: str, n_boot: int = 2000, seed: int = 6198):
    """Bootstrap a rate over parent problems, resampling clusters rather than rows."""
    by_parent: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_parent[r["parent_id"]].append(r["label"] == label)
    keys = list(by_parent)
    if not keys:
        return 0.0, 0.0, 0.0, 0
    hit = sum(sum(by_parent[k]) for k in keys)
    tot = sum(len(by_parent[k]) for k in keys)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        h = sum(sum(by_parent[k]) for k in pick)
        t = sum(len(by_parent[k]) for k in pick)
        draws.append(h / t if t else 0.0)
    draws.sort()
    return hit / tot, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)], hit


def save_both(fig, stem: Path, plt) -> list[str]:
    """Vector PDF for LaTeX, PNG preview for humans."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        p = stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight", facecolor=SURFACE, **kw)
        out.append(str(p))
    plt.close(fig)
    return out


def build(run: str | Path, out_dir: str | Path) -> list[str] | None:
    plt = _mpl()
    if plt is None:
        print("matplotlib not installed: pip install -e .[viz]")
        return None
    run, out_dir = Path(run), Path(out_dir)

    data: dict[str, dict] = {}
    for arm, _, _ in ARMS:
        rows = [json.loads(x) for x in
                (run / "analysis" / f"phase1_{arm}_rows.jsonl").read_text().splitlines() if x.strip()]
        per_model = defaultdict(list)
        for r in rows:
            per_model[r["model"]].append(r)
        data[arm] = {m: cluster_ci(v, "abstain") for m, v in per_model.items()}

    fig, ax = plt.subplots(figsize=(3.5, 2.7), facecolor=SURFACE)
    width, xs = 0.38, list(range(len(ORDER)))
    for j, (arm, arm_label, colour) in enumerate(ARMS):
        off = (j - 0.5) * width
        rates, los, his, counts = [], [], [], []
        for model, _ in ORDER:
            rate, lo, hi, hit = data[arm].get(model, (0.0, 0.0, 0.0, 0))
            rates.append(rate * 100)
            los.append((rate - lo) * 100)
            his.append((hi - rate) * 100)
            counts.append(hit)
        ax.bar([x + off for x in xs], rates, width * 0.92, color=colour,
               edgecolor=SURFACE, linewidth=1.4, label=arm_label, zorder=3)
        ax.errorbar([x + off for x in xs], rates, yerr=[los, his], fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=1.0, capsize=2.0, zorder=4)
        # Direct count labels: the relief the contrast WARN requires, and the honesty a
        # percentage alone destroys — 2.1% here is 34 events, not a stable rate.
        for x, rate, hi, c in zip(xs, rates, his, counts, strict=True):
            ax.text(x + off, rate + hi + 0.16, str(c), ha="center", va="bottom",
                    fontsize=6.5, color=INK_SECONDARY, zorder=5)

    _style(ax, plt, xlabel="", ylabel="abstention rate (%)",
           title="Models almost never detect a missing premise",
           subtitle="answer not derivable from the prompt, so the correct rate is 100%")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in ORDER], fontsize=8, color=INK)
    ax.set_ylim(0, 5)
    # The ceiling lives in the subtitle, not in a floating annotation: at this figure
    # width a text box in the plot area collides with the legend.
    leg = ax.legend(frameon=False, fontsize=6.8, loc="upper left",
                    handlelength=1.1, borderpad=0.1, labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.text(0.0, -0.06, "bars: abstention rate · numerals: abstention events · "
             "error bars: 95% cluster bootstrap over parent problems (2000 reps)",
             fontsize=5.8, color=INK_MUTED)
    return save_both(fig, out_dir / "F1_abstention", plt)


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "experiments/runs/ckpt-axis-v1"
    outd = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
    for p in build(run, outd) or []:
        print(f"wrote {p}")
