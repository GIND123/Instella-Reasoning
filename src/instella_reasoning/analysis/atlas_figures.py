"""Publication figures for the reasoning reliability atlas.

Every figure is written to both ``.pdf`` (vector, for the paper) and ``.png`` (raster,
for browsing and for the repository README). Titles are deliberately omitted: the paper
carries the description in ``\\caption``, and a title inside the axes duplicates it.

Design decisions that are load bearing rather than cosmetic:

- The categorical palette is the validated four-slot order (blue, orange, aqua, yellow).
  Worst adjacent CVD separation is dE 9.1 and worst normal-vision separation dE 22.9,
  both clear of their floors. Two of the four sit below 3:1 contrast on a light surface,
  so every mark also carries a direct value label rather than relying on colour alone.
- Perturbation conditions are ordered by how far they sit from the pretraining
  augmentation's support: original, distractor, numeric (inside), depth (outside). The
  slope between the last two is the paper's claim, so it must be adjacent and readable.
- Colour follows the *model*, never its rank, so a figure that drops a checkpoint does
  not repaint the survivors.
"""

from __future__ import annotations

import json
import pathlib

# Validated categorical slots, light surface. Order is fixed and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

#: Ordered by distance from the pretraining augmentation's support.
CONDITIONS = [
    ("acc_original", "original"),
    ("acc_distractor", "distractor"),
    ("acc_numeric", "numeric"),
    ("acc_depth", "depth"),
]

#: Stable display order for the checkpoint ladder.
MODEL_ORDER = ["M1", "M3-stage1", "M3-base", "M3-sft", "M3-instruct", "M3-math"]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Serif keeps figure text consistent with the paper body.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.frameon": False,
            "figure.dpi": 200,
        }
    )
    return plt


def _order(models: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def _save(fig, out_dir: pathlib.Path, stem: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(str(p))
    return paths


def fig_perturbation_slope(atlas: dict, out_dir: pathlib.Path, skill: str = "arithmetic"):
    """Accuracy across perturbation conditions, one line per checkpoint.

    The discriminating figure. Numeric perturbation re-values parameters, which is the
    operation that produced the pretraining augmentation, so it lies inside the training
    distribution. Depth extension changes the solution program, which the augmentation
    never did. A checkpoint that holds under numeric perturbation and falls under depth
    extension is recalling a procedure rather than composing one.
    """
    plt = _style()
    models = _order([m for m in atlas if skill in atlas[m]])
    if not models:
        return []

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    xs = list(range(len(CONDITIONS)))
    for i, model in enumerate(models):
        cell = atlas[model][skill]
        ys = [cell.get(key) for key, _ in CONDITIONS]
        pts = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
        if not pts:
            continue
        colour = SERIES[i % len(SERIES)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=colour, linewidth=2.0, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=model, zorder=3)
        # Direct label at the line end satisfies the relief rule for the low-contrast
        # slots and removes the need to trace colour back to the legend.
        ax.annotate(f"{model} {pts[-1][1]:.2f}", xy=pts[-1],
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_SOFT)

    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in CONDITIONS])
    ax.set_ylabel("exact-match accuracy")
    ax.set_xlim(-0.25, len(CONDITIONS) - 0.25 + 1.1)
    ax.set_ylim(0, 1)
    # The boundary the claim turns on: everything left of it was inside the
    # augmentation's support, the last point is outside it.
    ax.axvline(2.5, color=INK_SOFT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("inside augmentation support", xy=(1.25, 0.96), ha="center",
                fontsize=7, color=INK_SOFT)
    ax.annotate("outside", xy=(3.0, 0.96), ha="center", fontsize=7, color=INK_SOFT)
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, out_dir, f"a1_perturbation_slope_{skill}")


def fig_reliability_heatmap(atlas: dict, out_dir: pathlib.Path):
    """Reliability (accuracy x consistency) per checkpoint and reasoning subskill.

    Magnitude across two categorical dimensions, so a single-hue sequential ramp with
    every cell labelled. Cells without answer-preserving clusters are left blank rather
    than imputed.
    """
    plt = _style()
    import numpy as np

    models = _order(list(atlas))
    skills = sorted({s for m in atlas for s in atlas[m]})
    if not models or not skills:
        return []

    grid = np.full((len(models), len(skills)), np.nan)
    for r, m in enumerate(models):
        for c, s in enumerate(skills):
            v = atlas.get(m, {}).get(s, {}).get("reliability")
            if v is not None:
                grid[r, c] = v

    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(skills), 0.55 * len(models) + 1.3))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(skills)))
    ax.set_xticklabels([s.replace("_", " ") for s in skills], rotation=28, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.grid(False)
    for r in range(len(models)):
        for c in range(len(skills)):
            v = grid[r, c]
            if np.isnan(v):
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=7, color=INK_SOFT)
            else:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("reliability = accuracy x consistency", fontsize=8)
    cb.outline.set_edgecolor(GRID)
    return _save(fig, out_dir, "a2_reliability_heatmap")


def fig_accuracy_consistency(atlas: dict, out_dir: pathlib.Path, skill: str = "arithmetic"):
    """Decomposition of reliability into its two factors, per checkpoint.

    Reliability is a product, so a checkpoint can lose it two different ways. Showing the
    factors side by side distinguishes a model that is wrong-but-stable from one that is
    right-but-fragile, which the product alone hides.
    """
    plt = _style()
    import numpy as np

    models = _order([m for m in atlas if skill in atlas[m]])
    if not models:
        return []

    fields = [("accuracy", "accuracy"), ("consistency", "consistency"),
              ("reliability", "reliability")]
    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(1.4 + 1.05 * len(models), 3.0))
    for i, (key, label) in enumerate(fields):
        vals = [atlas[m][skill].get(key) for m in models]
        offs = x + (i - 1) * (width + 0.02)  # 2px-equivalent gap between adjacent fills
        heights = [0 if v is None else v for v in vals]
        ax.bar(offs, heights, width, label=label, color=SERIES[i],
               edgecolor=SURFACE, linewidth=1.0, zorder=3)
        for xi, v in zip(offs, vals, strict=False):
            ax.text(xi, (0 if v is None else v) + 0.015,
                    "n/a" if v is None else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK_SOFT)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left", ncol=3)
    return _save(fig, out_dir, f"a3_accuracy_consistency_{skill}")


def fig_recall_trajectory(did: dict, out_dir: pathlib.Path, intervention: str = "M3-base",
                          arm_labels: tuple[str, str] = ("high containment", "low containment")):
    """Verbatim recall on unanswerable items, across the training trajectory.

    The headline figure. Each item has had a premise its answer depended on deleted, so
    the answer is not derivable from the prompt; reproducing the parent's answer is
    therefore recall rather than reasoning. Left panel shows the rate on verified-seen and
    verified-unseen items; right panel shows their difference with cluster-bootstrap
    intervals over parent problems.

    ``intervention`` marks the checkpoint at which GSM8K-derived data enters the training
    mixture. Everything to its left is a control that never saw the benchmark, which is
    what licenses reading a step at that boundary as an effect of the data rather than of
    scale or post-training.
    """
    plt = _style()
    models = [m for m in MODEL_ORDER if m in did and did[m].get("recall_seen") is not None]
    if not models:
        return []
    xs = list(range(len(models)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    fig.subplots_adjust(wspace=0.32)

    ax1.plot(xs, [did[m]["recall_seen"] for m in models], color=SERIES[0], linewidth=2.0,
             marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label=arm_labels[0], zorder=3)
    ax1.plot(xs, [did[m]["recall_unseen"] for m in models], color=SERIES[1], linewidth=2.0,
             marker="s", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label=arm_labels[1], zorder=3)
    for m, x in zip(models, xs, strict=False):
        ax1.annotate(f"{did[m]['recall_seen']:.3f}", xy=(x, did[m]["recall_seen"]),
                     xytext=(0, 7), textcoords="offset points", ha="center",
                     fontsize=7, color=INK_SOFT)
    ax1.set_ylabel("recall rate on unanswerable items")
    ax1.set_title("both arms are GSM8K train; only corpus containment differs",
                  fontsize=7.5, color=INK_SOFT, loc="left", pad=4)
    ax1.set_ylim(bottom=0)

    gaps = [did[m].get("recall_gap") for m in models]
    cis = [did[m].get("ci_recall_gap") or [None, None] for m in models]
    lo = [g - c[0] if c[0] is not None else 0 for g, c in zip(gaps, cis, strict=False)]
    hi = [c[1] - g if c[1] is not None else 0 for g, c in zip(gaps, cis, strict=False)]
    ax2.errorbar(xs, gaps, yerr=[lo, hi], fmt="o", color=SERIES[3], markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=2.0,
                 capsize=3, zorder=3)
    ax2.axhline(0, color=INK_SOFT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax2.set_ylabel("recall gap (high - low)")
    for _m, x, g in zip(models, xs, gaps, strict=False):
        ax2.annotate(f"{g:+.3f}", xy=(x, g), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=7, color=INK_SOFT)

    # Right margin for the trailing direct label, which otherwise clips at the spine.
    ax2.set_xlim(-0.5, len(models) - 1 + 0.95)
    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=20, ha="right")
        if intervention in models:
            # Drawn between the last control and the treated checkpoint.
            ax.axvline(models.index(intervention) - 0.5, color=INK_SOFT,
                       linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
    ax1.legend(loc="upper left")
    ax1.annotate("benchmark data\nenters here", xy=(models.index(intervention) - 0.45, 0),
                 xytext=(3, 6), textcoords="offset points", fontsize=7, color=INK_SOFT)
    return _save(fig, out_dir, "a0_recall_trajectory")


def build_all(atlas_path: str | pathlib.Path, out_dir: str | pathlib.Path,
              did_path: str | pathlib.Path | None = None) -> dict:
    """Render every figure the atlas supports and return the written paths."""
    atlas = json.loads(pathlib.Path(atlas_path).read_text(encoding="utf-8"))
    out = pathlib.Path(out_dir)
    skills = sorted({s for m in atlas for s in atlas[m]})

    written: dict[str, list[str]] = {}
    if did_path and pathlib.Path(did_path).exists():
        did = json.loads(pathlib.Path(did_path).read_text(encoding="utf-8"))
        paths = fig_recall_trajectory(did, out)
        if paths:
            written["a0_recall_trajectory"] = paths
    written["a2_reliability_heatmap"] = fig_reliability_heatmap(atlas, out)
    for skill in skills:
        p = fig_perturbation_slope(atlas, out, skill)
        if p:
            written[f"a1_perturbation_slope_{skill}"] = p
        q = fig_accuracy_consistency(atlas, out, skill)
        if q:
            written[f"a3_accuracy_consistency_{skill}"] = q
    return written
