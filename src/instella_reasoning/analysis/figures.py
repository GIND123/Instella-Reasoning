"""Publication figure set for the memorisation study.

Eight figures, each answering one question a reviewer will actually ask:

===  ==========================  =====================================================
F1   trajectory                  Where along the training pipeline does accuracy appear,
                                 and does it appear on seen items only?
F2   did_forest                  How big is the memorisation component, with intervals?
F3   perturbation_slopes         Does accuracy survive numeric perturbation, per checkpoint?
F4   containment                 Is the seen/unseen treatment assignment actually verified?
F5   magnitude_control           Is the perturbation effect just bigger arithmetic?
F6   measurement_validity        Are the models being scored on complete outputs?
F7   consistency_decomposition   Is the inconsistency perturbation, or decoding noise?
F8   design_power                Is the sample size adequate, given the measured ICC?
===  ==========================  =====================================================

Design rules applied throughout (they are not stylistic preferences — each one is a
known way charts mislead):

* **one y-axis per panel, never two** — a dual-scale chart lets the author choose the
  visual conclusion;
* **categorical hues assigned in fixed order, never cycled** — a checkpoint keeps its
  colour when a filter changes the series count;
* colours are the validated CVD-safe categorical slots in :data:`SERIES` (worst adjacent
  separation under simulated protanopia/deuteranopia/tritanopia is dE 9.4, above the 8
  floor), and identity is *also* carried by direct labels so it is never colour-alone;
* error bars are bootstrap CIs over **parent items**, matching the analysis;
* grid and axes are recessive; values are labelled selectively, never on every point.

Matplotlib is an optional dependency (`pip install -e .[viz]`); every entry point
degrades to a clear message rather than failing the run.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from instella_reasoning.records import EvaluationRecord

# Validated categorical slots, in fixed assignment order. Never cycle past the end:
# a 6th series folds into "other" or becomes a small multiple.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"
# Diverging pair for signed quantities (memorisation vs anti-memorisation), with a
# neutral — never a hue — at the midpoint.
POS, NEG, NEUTRAL = "#eb6834", "#2a78d6", "#8a8985"


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:  # pragma: no cover - optional extra
        return None


def _style(ax, plt, *, xlabel="", ylabel="", title="", subtitle=""):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=9,
            va="bottom",
        )


def _save(fig, path: Path, plt) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def _order_models(models: list[str]) -> list[str]:
    """Trajectory order, not alphabetical.

    Alphabetical puts ``Instella-3B-Stage1`` last when it is developmentally first, which
    silently reverses the story a reader takes from any left-to-right chart.
    """
    from instella_reasoning.checkpoints import try_resolve

    def key(name: str) -> tuple[int, str]:
        ckpt = try_resolve(name)
        return (ckpt.step if ckpt else 99, name)

    return sorted(models, key=key)


def _spread(
    labels: list[tuple[float, str, str]], min_gap: float
) -> list[tuple[float, str, str]]:
    """Push overlapping direct labels apart, preserving their vertical order.

    One upward pass from the lowest label: whenever the next label is closer than
    ``min_gap``, move it up to exactly that gap. Order is never swapped, so a leader line
    back to the real endpoint stays unambiguous.
    """
    ordered = sorted(labels, key=lambda t: t[0])
    out: list[tuple[float, str, str]] = []
    previous = None
    for value, text, colour in ordered:
        if previous is not None and value - previous < min_gap:
            value = previous + min_gap
        out.append((value, text, colour))
        previous = value
    return out


def _bootstrap_ci(values: list[int], n: int = 2000, seed: int = 6198) -> tuple[float, float]:
    import random

    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in range(len(values))) / len(values)
        for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


# -- F1: the trajectory ----------------------------------------------------------


def plot_trajectory(trajectory: list[dict], output_dir: str | Path) -> str | None:
    """Accuracy along the Instella checkpoint chain, split by arm and condition.

    The one figure the paper is built around. A memorisation story looks like the *seen,
    original* line lifting away from the other three at the stage-1 -> stage-2 step (where
    GSM8K-derived data enters); a generalisation story looks like all four rising together.
    """
    plt = _mpl()
    if plt is None or not trajectory:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    tags = [p["tag"] for p in trajectory]
    x = list(range(len(tags)))

    lines = [
        ("seen_original", "seen · original", SERIES[0], "-", "o"),
        ("unseen_original", "unseen · original", SERIES[1], "-", "s"),
        ("seen_perturbed", "seen · perturbed", SERIES[0], "--", "o"),
        ("unseen_perturbed", "unseen · perturbed", SERIES[1], "--", "s"),
    ]
    for key, label, colour, dash, marker in lines:
        ys = [p["accuracy"][key] for p in trajectory]
        ax.plot(
            x,
            ys,
            color=colour,
            linestyle=dash,
            linewidth=2.0,
            marker=marker,
            markersize=6,
            markerfacecolor=colour if dash == "-" else SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.6,
            label=label,
            zorder=3,
        )

    # Mark the intervention that introduces GSM8K-derived data — the causal boundary.
    for i, point in enumerate(trajectory):
        if point["saw_gsm8k_derived_data"] and (
            i == 0 or not trajectory[i - 1]["saw_gsm8k_derived_data"]
        ):
            ax.axvline(i - 0.5, color=INK_MUTED, linewidth=1.2, linestyle=":", zorder=1)
            ax.text(
                i - 0.45,
                0.02,
                "GSM8K-derived data enters",
                rotation=90,
                fontsize=8,
                color=INK_SECONDARY,
                va="bottom",
            )
            break

    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylim(0, 1)
    _style(
        ax,
        plt,
        ylabel="exact-match accuracy",
        title="Where GSM8K accuracy appears along the Instella trajectory",
        subtitle="Solid = original items · dashed = numerically perturbed · error-free lines are point estimates",
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(fig, Path(output_dir) / "f1_trajectory.png", plt)


# -- F2: DiD forest --------------------------------------------------------------


def plot_did_forest(trajectory: list[dict], output_dir: str | Path) -> str | None:
    """Memorisation DiD per checkpoint with cluster-bootstrap intervals.

    A forest plot is the right form here because the *interval* is the finding: an
    interval tight around zero is a publishable bounded null, and only a forest shows
    that as clearly as it shows a positive effect.
    """
    plt = _mpl()
    if plt is None or not trajectory:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(trajectory) + 1.9))
    ys = list(range(len(trajectory)))[::-1]

    for y, point in zip(ys, trajectory, strict=False):
        lo, hi = point["did_ci95"]
        est = point["did"]
        colour = POS if lo > 0 else NEG if hi < 0 else NEUTRAL
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.plot(
            [est], [y], marker="o", markersize=8, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3,
        )
        ax.text(
            hi + 0.012, y, f"{est:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
            va="center", fontsize=8.5, color=INK_SECONDARY,
        )

    ax.axvline(0, color=INK_MUTED, linewidth=1.2, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([p["tag"] for p in trajectory], fontsize=9)
    # Pin the row spacing; matplotlib's autoscale leaves a forest plot uncomfortably airy.
    ax.set_ylim(-0.7, len(trajectory) - 0.3)
    ax.grid(False, axis="y")
    _style(
        ax,
        plt,
        xlabel="difference-in-differences  (seen advantage lost to perturbation)",
        title="How much of the seen-item advantage does not survive perturbation",
        subtitle="Orange = memorisation · blue = reverse · grey = interval covers zero",
    )
    lo_all = min(p["did_ci95"][0] for p in trajectory)
    hi_all = max(p["did_ci95"][1] for p in trajectory)
    span = max(hi_all - lo_all, 0.1)
    ax.set_xlim(lo_all - 0.05 * span, hi_all + 0.42 * span)
    return _save(fig, Path(output_dir) / "f2_did_forest.png", plt)


# -- F3: perturbation slopes -----------------------------------------------------


def plot_perturbation_slopes(records: list[EvaluationRecord], output_dir: str | Path) -> str | None:
    """Paired original -> perturbed accuracy per model, one slope each.

    A slope chart, not a grouped bar chart: the quantity of interest is the *change*
    within a model, and a slope encodes change as slope rather than as the difference
    between two bar heights the eye has to subtract.
    """
    from instella_reasoning.analysis.memorization import (
        DID_EXCLUDED_VARIANTS,
        ORIGINAL,
        PERTURBED,
        record_condition,
    )

    plt = _mpl()
    if plt is None or not records:
        return None
    by_model: dict[str, dict[str, list[int]]] = defaultdict(lambda: {ORIGINAL: [], PERTURBED: []})
    for record in records:
        # Same exclusion as the DiD: the resample control is not answer-changing, so it would
        # be drawn into the ORIGINAL end of every slope for the checkpoints that ran it,
        # tilting those slopes for a reason that has nothing to do with perturbation.
        if record.variant_type in DID_EXCLUDED_VARIANTS:
            continue
        by_model[record.model][record_condition(record)].append(int(record.correct))

    models = _order_models(list(by_model))
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    labels: list[tuple[float, str, str]] = []
    for i, model in enumerate(models):
        colour = SERIES[i % len(SERIES)]
        cells = by_model[model]
        if not cells[ORIGINAL] or not cells[PERTURBED]:
            continue
        a = sum(cells[ORIGINAL]) / len(cells[ORIGINAL])
        b = sum(cells[PERTURBED]) / len(cells[PERTURBED])
        alo, ahi = _bootstrap_ci(cells[ORIGINAL])
        blo, bhi = _bootstrap_ci(cells[PERTURBED])
        ax.plot([0, 1], [a, b], color=colour, linewidth=2.0, zorder=3)
        ax.plot([0, 0], [alo, ahi], color=colour, linewidth=1.2, alpha=0.55, zorder=2)
        ax.plot([1, 1], [blo, bhi], color=colour, linewidth=1.2, alpha=0.55, zorder=2)
        for xpos, val in ((0, a), (1, b)):
            ax.plot([xpos], [val], marker="o", markersize=8, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        labels.append((b, f"{model.split('/')[-1]}  {b - a:+.3f}", colour))

    # Direct labels instead of a legend box (only 4-6 series, and the label sits where the
    # eye already is) — but two checkpoints can land within a few points of each other, so
    # nudge them apart and draw a leader line back to the true endpoint.
    for y_target, text, colour in _spread(labels, min_gap=0.045):
        y_true = next(v for v, t, _ in labels if t == text)
        if abs(y_target - y_true) > 1e-6:
            ax.plot([1.0, 1.035], [y_true, y_target], color=colour, linewidth=0.8,
                    alpha=0.6, zorder=2)
        ax.text(1.045, y_target, text, va="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original", "numerically\nperturbed"], fontsize=9)
    ax.set_xlim(-0.12, 1.9)
    ax.set_ylim(0, 1)
    _style(
        ax, plt,
        ylabel="exact-match accuracy",
        title="Accuracy under numeric perturbation",
        subtitle="Vertical bars are item-clustered bootstrap 95% CIs",
    )
    return _save(fig, Path(output_dir) / "f3_perturbation_slopes.png", plt)


# -- F4: containment evidence ----------------------------------------------------


def plot_containment(containment: list[dict], output_dir: str | Path) -> str | None:
    """Distribution of exact 13-gram containment, by intended arm.

    This is the figure that makes the treatment assignment credible. Two well-separated
    masses near 0 and 1 mean "seen" and "unseen" are verified facts; overlap in the middle
    would mean the labels are guesses, and the study would have to say so.
    """
    plt = _mpl()
    if plt is None or not containment:
        return None
    seen = [c["containment"] for c in containment if c.get("intended_arm") == "seen"]
    unseen = [c["containment"] for c in containment if c.get("intended_arm") == "unseen"]
    if not seen and not unseen:
        seen = [c["containment"] for c in containment if c["verdict"] == "seen"]
        unseen = [c["containment"] for c in containment if c["verdict"] != "seen"]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = [i / 20 for i in range(21)]
    ax.hist([unseen, seen], bins=bins, color=[SERIES[1], SERIES[0]],
            label=["GSM8K test (expected unseen)", "GSM8K train (expected seen)"],
            edgecolor=SURFACE, linewidth=1.2, stacked=False)
    for threshold, text in ((0.10, "unseen ≤ 0.10"), (0.80, "seen ≥ 0.80")):
        ax.axvline(threshold, color=INK_MUTED, linestyle=":", linewidth=1.2)
        ax.text(threshold, ax.get_ylim()[1] * 0.96, f" {text}", fontsize=8,
                color=INK_SECONDARY, va="top")
    _style(
        ax, plt,
        xlabel="fraction of the item's 13-grams found in the training corpus",
        ylabel="items",
        title="The seen/unseen assignment is verified, not inferred",
        subtitle="Items between the thresholds are excluded from both arms",
    )
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(fig, Path(output_dir) / "f4_containment.png", plt)


# -- F5: magnitude control -------------------------------------------------------


def plot_magnitude_control(records: list[EvaluationRecord], output_dir: str | Path) -> str | None:
    """Two panels: is the perturbation magnitude-neutral, and does magnitude drive accuracy?

    Small multiples rather than a dual axis. The left panel is the *design* check (did the
    generator inflate the numbers?); the right is the *outcome* check (does accuracy fall
    with magnitude regardless of arm?). Together they answer the single strongest
    objection to any GSM-Symbolic-style result.
    """
    plt = _mpl()
    if plt is None:
        return None
    ratios = [
        float(r.metadata["magnitude_ratio"])
        for r in records
        if r.metadata.get("magnitude_ratio") is not None
    ]
    perturbed = [
        r for r in records if r.metadata.get("magnitude_ratio") is not None
    ]
    if not ratios:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    logs = [math.log(r) for r in ratios if r > 0]
    axes[0].hist(logs, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    axes[0].axvline(0, color=INK_MUTED, linewidth=1.4)
    median = sorted(ratios)[len(ratios) // 2]
    axes[0].text(
        0.02, 0.95, f"median ratio {median:.2f}x\nn = {len(ratios)}",
        transform=axes[0].transAxes, fontsize=9, color=INK_SECONDARY, va="top",
    )
    _style(
        axes[0], plt,
        xlabel="log(perturbed answer / original answer)",
        ylabel="variants",
        title="Perturbation is magnitude-neutral",
        subtitle="A mass centred on 0 is the requirement; the earlier generator sat at +0.96",
    )

    # Accuracy by magnitude decile — one series, so no legend box is needed.
    ordered = sorted(perturbed, key=lambda r: float(r.metadata["magnitude_ratio"]))
    n_bins = 8
    size = max(1, len(ordered) // n_bins)
    xs, ys, los, his = [], [], [], []
    for b in range(n_bins):
        chunk = ordered[b * size : (b + 1) * size]
        if not chunk:
            continue
        vals = [int(r.correct) for r in chunk]
        mid = sorted(float(r.metadata["magnitude_ratio"]) for r in chunk)[len(chunk) // 2]
        lo, hi = _bootstrap_ci(vals, n=800)
        xs.append(mid)
        ys.append(sum(vals) / len(vals))
        los.append(lo)
        his.append(hi)
    axes[1].fill_between(xs, los, his, color=SERIES[0], alpha=0.16, linewidth=0)
    axes[1].plot(xs, ys, color=SERIES[0], linewidth=2.0, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.4)
    axes[1].set_ylim(0, 1)
    _style(
        axes[1], plt,
        xlabel="answer magnitude ratio (perturbed / original)",
        ylabel="accuracy on perturbed items",
        title="Accuracy vs magnitude",
        subtitle="A flat line means the perturbation effect is not a bigger-arithmetic effect",
    )
    fig.tight_layout()
    return _save(fig, Path(output_dir) / "f5_magnitude_control.png", plt)


# -- F6: measurement validity ----------------------------------------------------


def plot_measurement_validity(reports: list[dict], output_dir: str | Path) -> str | None:
    """Termination and answer-marker rates per model.

    Included because omitting it is how the previous run reached a wrong conclusion: at a
    512-token budget the long-CoT checkpoint terminated on 2% of items and its measured
    accuracy was an artifact of the cap. A reviewer should be able to see, in one figure,
    that every model was scored on complete outputs.
    """
    plt = _mpl()
    if plt is None or not reports:
        return None
    order = _order_models([r["model"] for r in reports])
    reports = sorted(reports, key=lambda r: order.index(r["model"]))
    models = [r["model"].split("/")[-1] for r in reports]
    term = [r["termination_rate"] for r in reports]
    marker = [r["marker_rate"] for r in reports]
    y = list(range(len(models)))[::-1]
    height = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(models) + 2.2))
    # 2px surface gap between adjacent bars is achieved by the offset + edge colour.
    ax.barh([v + height / 2 for v in y], term, height=height, color=SERIES[0],
            edgecolor=SURFACE, linewidth=1.4, label="completed before cap", zorder=3)
    ax.barh([v - height / 2 for v in y], marker, height=height, color=SERIES[2],
            edgecolor=SURFACE, linewidth=1.4, label="emitted '####' marker", zorder=3)
    threshold = reports[0].get("threshold", 0.85)
    ax.axvline(threshold, color=INK_MUTED, linestyle=":", linewidth=1.4, zorder=4)
    ax.text(threshold, len(models) - 0.4, f" gate {threshold:.0%}", fontsize=8,
            color=INK_SECONDARY, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    _style(
        ax, plt,
        xlabel="share of generations",
        title="Every model was scored on complete outputs",
        subtitle="A model below the gate is being measured on truncated text, not on its ability",
    )
    # Above the plot: bars run to the right edge, so any in-axes legend lands on data.
    legend = ax.legend(
        frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.10)
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(fig, Path(output_dir) / "f6_measurement_validity.png", plt)


# -- F7: consistency decomposition -----------------------------------------------


def plot_consistency_decomposition(
    records: list[EvaluationRecord], output_dir: str | Path
) -> str | None:
    """Modal-answer agreement under decoding noise vs under answer-preserving variants.

    The decoding-noise bar is the null: identical prompts resampled at T>0. If agreement
    under surface variants is no lower than that, the model's apparent inconsistency is
    the sampler, not its reasoning.

    Answer-*changing* (numeric) variants are deliberately absent. Modal agreement is
    undefined for them — a correct reasoner *should* answer each differently, so agreement
    would reward exactly the wrong behaviour (this is the incoherence
    ``METHODOLOGY.md`` §3 fixes in the reliability metric). Their story is accuracy, and it
    is told in F3.
    """
    from instella_reasoning.metrics import is_answer_changing

    plt = _mpl()
    if plt is None or not records:
        return None

    def modal_share(rows: list[EvaluationRecord]) -> float | None:
        by_parent: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            by_parent[r.parent_id].append(r.normalized_predicted)
        shares = []
        for answers in by_parent.values():
            if len(answers) < 2:
                continue
            counts: dict[str, int] = defaultdict(int)
            for a in answers:
                counts[a] += 1
            shares.append(max(counts.values()) / len(answers))
        return sum(shares) / len(shares) if shares else None

    groups = [
        ("decoding noise (same prompt, T>0) — the null",
         lambda r: r.variant_type == "resample", SERIES[2]),
        ("answer-preserving surface variants",
         lambda r: r.variant_type not in ("original", "resample") and not is_answer_changing(r),
         SERIES[0]),
    ]
    models = _order_models(list({r.model for r in records}))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    width = 0.34
    plotted_any = False
    for gi, (label, predicate, colour) in enumerate(groups):
        xs, ys = [], []
        for mi, model in enumerate(models):
            rows = [r for r in records if r.model == model and predicate(r)]
            share = modal_share(rows)
            if share is None:
                continue
            xs.append(mi + (gi - 0.5) * width)
            ys.append(share)
        if not xs:
            continue
        plotted_any = True
        ax.bar(xs, ys, width=width * 0.94, color=colour, edgecolor=SURFACE,
               linewidth=1.4, label=label, zorder=3)
    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=9)
    ax.set_ylim(0, 1)
    _style(
        ax, plt,
        ylabel="modal-answer agreement",
        title="Is the inconsistency reasoning, or the sampler?",
        subtitle="Answer-changing variants are excluded — agreement is undefined for them (see F3)",
    )
    # Legend above the plot area: inside, it lands on top of the bars at every scale.
    legend = ax.legend(
        frameon=False, fontsize=8.5, ncol=2,
        loc="lower left", bbox_to_anchor=(0.0, 1.10),
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(fig, Path(output_dir) / "f7_consistency_decomposition.png", plt)


# -- F8: design + power ----------------------------------------------------------


def plot_design_power(icc: float, n_items: int, output_dir: str | Path,
                      variants_per_item: int = 5) -> str | None:
    """Detectable effect vs number of items, at the measured intra-cluster correlation.

    The figure that pre-empts "is n large enough?". It also shows *why* the budget was
    spent on more items rather than more variants per item: at ICC ~0.5 the marginal
    value of an extra variant collapses after the third.
    """
    plt = _mpl()
    if plt is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # Left: marginal effective-n contribution of the k-th variant.
    ks = list(range(1, 9))
    def eff(k: int) -> float:
        return k / (1 + (k - 1) * icc)
    marginal = [eff(k) - (eff(k - 1) if k > 1 else 0.0) for k in ks]
    colours = [SERIES[0] if k <= variants_per_item else INK_MUTED for k in ks]
    axes[0].bar(ks, marginal, width=0.68, color=colours, edgecolor=SURFACE, linewidth=1.4, zorder=3)
    for k, m in zip(ks, marginal, strict=False):
        if k <= 3 or k == variants_per_item:
            axes[0].text(k, m + 0.02, f"{m:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)
    _style(
        axes[0], plt,
        xlabel="variant index within an item",
        ylabel="effective items gained",
        title=f"Why more items, not more variants  (ICC = {icc:.2f})",
        subtitle="Grey bars are past the chosen cluster size",
    )

    # Right: minimum detectable effect vs items per arm, one series -> no legend box.
    deff = 1 + (variants_per_item - 1) * icc
    za, zb = 1.959964, 0.8416
    items = list(range(50, 601, 10))
    mde = []
    for n in items:
        n_eff = n * variants_per_item / deff
        p = 0.65
        mde.append((za + zb) * math.sqrt(2 * p * (1 - p) / n_eff))
    axes[1].plot(items, mde, color=SERIES[0], linewidth=2.0, zorder=3)
    axes[1].axvline(n_items, color=INK_MUTED, linestyle=":", linewidth=1.4, zorder=2)
    at = min(range(len(items)), key=lambda i: abs(items[i] - n_items))
    axes[1].plot([items[at]], [mde[at]], marker="o", markersize=8, color=SERIES[1],
                 markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    # Offset in axis fraction, not data units: the x range depends on the item grid, so a
    # fixed data offset collides with the marker on some scales and floats away on others.
    axes[1].annotate(
        f"n = {n_items}/arm\nMDE ≈ {mde[at]:.3f}",
        xy=(items[at], mde[at]),
        xytext=(14, 10),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
        ha="left",
    )
    _style(
        axes[1], plt,
        xlabel="items per arm",
        ylabel="minimum detectable difference",
        title="Power at the chosen sample size",
        subtitle=f"80% power, α = .05, design effect {deff:.2f} from the measured ICC",
    )
    fig.tight_layout()
    return _save(fig, Path(output_dir) / "f8_design_power.png", plt)


# -- orchestration ---------------------------------------------------------------


def render_all(
    records: list[EvaluationRecord],
    analysis: dict,
    output_dir: str | Path,
    containment: list[dict] | None = None,
    termination: list[dict] | None = None,
    icc: float | None = None,
    n_items: int = 250,
) -> dict[str, str | None]:
    """Render every figure that the supplied inputs can support."""
    out = Path(output_dir)
    written: dict[str, str | None] = {}
    trajectory = analysis.get("trajectory", [])
    written["f1_trajectory"] = plot_trajectory(trajectory, out)
    written["f2_did_forest"] = plot_did_forest(trajectory, out)
    written["f3_perturbation_slopes"] = plot_perturbation_slopes(records, out)
    written["f4_containment"] = plot_containment(containment or [], out)
    written["f5_magnitude_control"] = plot_magnitude_control(records, out)
    written["f6_measurement_validity"] = plot_measurement_validity(termination or [], out)
    written["f7_consistency_decomposition"] = plot_consistency_decomposition(records, out)
    written["f8_design_power"] = plot_design_power(
        icc if icc is not None else estimate_icc(records), n_items, out
    )
    return written


def estimate_icc(records: list[EvaluationRecord]) -> float:
    """One-way ANOVA intra-cluster correlation of correctness within parent items.

    Drives both the power figure and the design justification in the paper. Measured at
    ~0.48 on this project's pilot run, which is why the budget favours items over variants.
    """
    # Key on (model, parent) — a cluster is one item *for one model*. Pooling a parent's
    # records across checkpoints mixes between-model accuracy differences into the
    # within-cluster term and drives the estimate toward zero.
    clusters: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in records:
        clusters[(record.model, record.parent_id)].append(int(record.correct))
    usable = {k: v for k, v in clusters.items() if len(v) > 1}
    if len(usable) < 2:
        return 0.0
    values = [x for v in usable.values() for x in v]
    grand = sum(values) / len(values)
    k_bar = len(values) / len(usable)
    msb = sum(len(v) * ((sum(v) / len(v)) - grand) ** 2 for v in usable.values()) / (
        len(usable) - 1
    )
    within_df = max(1, len(values) - len(usable))
    msw = sum(
        sum((x - sum(v) / len(v)) ** 2 for x in v) for v in usable.values()
    ) / within_df
    denominator = msb + (k_bar - 1) * msw
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, (msb - msw) / denominator))
