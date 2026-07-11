"""Publication figures for the Reasoning Reliability Atlas (Phase 6).

Matplotlib is imported lazily behind the ``viz`` extra (``pip install -e .[viz]``),
so importing this module and its pure helpers costs no heavy dependency. Every figure
uses a single validated design system: a colorblind-safe categorical palette, a
one-hue sequential blue ramp for magnitude (the Atlas heatmap), a blue<->red diverging
pair for signed quantities (accuracy gaps), and reserved status colors for the
GENUINE / PARTIAL / FRAGILE / GAP verdicts. Axes are recessive, marks are direct-
labelled, and a legend is present whenever more than one series is drawn.

The figures, following the implementation plan:

- ``plot_reliability_atlas``          — skill x {accuracy, consistency, reliability}
  heatmap (the money figure).
- ``plot_accuracy_consistency_quadrant`` — per-cluster acc-vs-consistency scatter.
- ``plot_accuracy_gap``               — contaminated-vs-clean accuracy bars.
- ``plot_contamination_breakdown``    — C/PC/N counts per source.
- ``plot_attribution_sources``        — training-source shares per sub-skill.
- ``plot_scale_reliability``          — reliability-vs-model-size lines per sub-skill.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

# Validated categorical palette (light mode), assigned in fixed order — never cycled.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {
    "genuine": "#0ca30c",   # good
    "partial": "#fab219",   # warning
    "fragile": "#d03b3b",   # critical
    "gap": "#898781",       # muted
}
CONTAMINATION_COLOR = {
    "contaminated": "#e34948",  # red
    "partial": "#eda100",       # yellow
    "clean": "#2a78d6",         # blue
}
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe (Colab, CI, servers)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Plotting needs the viz extra: `pip install -e .[viz]` (matplotlib)."
        ) from exc


def quadrant_of(accuracy: float, consistency: float, threshold: float = 0.5) -> str:
    """Label an (accuracy, consistency) point by quadrant — pure helper, testable."""
    high_acc = accuracy >= threshold
    high_cons = consistency >= threshold
    if high_acc and high_cons:
        return "genuine"
    if high_acc and not high_cons:
        return "fragile"
    if not high_acc and high_cons:
        return "consistent_gap"
    return "random"


def _style_axes(ax) -> None:  # pragma: no cover - needs matplotlib
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)


def plot_reliability_atlas(atlas_report, output_path: str | Path, level: str = "all"):  # pragma: no cover
    """Heatmap of accuracy / consistency / reliability per sub-skill (the money figure)."""
    plt = _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    cells = [c for c in atlas_report.cells if c.contamination_level == level]
    cells.sort(key=lambda c: c.reliability, reverse=True)
    skills = [c.skill for c in cells]
    metrics = ["accuracy", "consistency", "reliability"]
    matrix = [[c.accuracy, c.consistency, c.reliability] for c in cells]

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(skills) + 1.4))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(metrics)), [m.capitalize() for m in metrics])
    ax.set_yticks(range(len(skills)), skills)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ink = "#ffffff" if value > 0.55 else INK_PRIMARY
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=ink, fontsize=9)
    for i, cell in enumerate(cells):
        ax.text(
            len(metrics) - 0.35, i, cell.classification.upper(),
            ha="left", va="center", color=STATUS.get(cell.classification, INK_MUTED), fontsize=8,
        )
    ax.set_title("Reasoning Reliability Atlas", color=INK_PRIMARY, fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.16, label="score")
    return _save(fig, output_path, plt)


def plot_accuracy_consistency_quadrant(clusters, output_path: str | Path):  # pragma: no cover
    """Scatter of per-cluster accuracy vs consistency, coloured by contamination level."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _style_axes(ax)
    seen: set[str] = set()
    for cluster in clusters:
        color = CONTAMINATION_COLOR.get(cluster.contamination_level, INK_MUTED)
        label = cluster.contamination_level if cluster.contamination_level not in seen else None
        seen.add(cluster.contamination_level)
        # jitter identical points slightly so overlaps are visible
        ax.scatter(
            cluster.accuracy, cluster.consistency, s=42, color=color, alpha=0.75,
            edgecolor=SURFACE, linewidth=1.0, label=label, zorder=3,
        )
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(0.5, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Accuracy", color=INK_PRIMARY)
    ax.set_ylabel("Consistency", color=INK_PRIMARY)
    ax.set_title("Accuracy vs Consistency (per cluster)", color=INK_PRIMARY, fontsize=12)
    ax.text(0.75, 0.04, "FRAGILE", color=STATUS["fragile"], fontsize=9, ha="center")
    ax.text(0.75, 0.96, "GENUINE", color=STATUS["genuine"], fontsize=9, ha="center")
    if seen:
        ax.legend(frameon=False, fontsize=9, title="contamination")
    return _save(fig, output_path, plt)


def plot_accuracy_gap(gap_results, output_path: str | Path):  # pragma: no cover
    """Grouped bars of contaminated vs clean accuracy per scope, with a significance mark."""
    plt = _require_matplotlib()
    scopes = [r.scope for r in gap_results]
    contaminated = [r.by_label["contaminated"].accuracy for r in gap_results]
    clean = [r.by_label["clean"].accuracy for r in gap_results]

    fig, ax = plt.subplots(figsize=(1.6 * len(scopes) + 2.0, 4.6))
    _style_axes(ax)
    x = range(len(scopes))
    width = 0.38
    ax.bar([i - width / 2 for i in x], contaminated, width, color=CONTAMINATION_COLOR["contaminated"], label="contaminated", zorder=3)
    ax.bar([i + width / 2 for i in x], clean, width, color=CONTAMINATION_COLOR["clean"], label="clean", zorder=3)
    for i, r in enumerate(gap_results):
        if r.p_value < 0.05:
            top = max(contaminated[i], clean[i])
            ax.text(i, top + 0.03, "*", ha="center", color=INK_PRIMARY, fontsize=14)
    ax.set_xticks(list(x), scopes, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", color=INK_PRIMARY)
    ax.set_title("Contaminated vs Clean Accuracy", color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, output_path, plt)


def plot_contamination_breakdown(contamination_hits, output_path: str | Path):  # pragma: no cover
    """Stacked C/PC counts per training-data source."""
    plt = _require_matplotlib()
    by_source: dict[str, Counter] = defaultdict(Counter)
    for hit in contamination_hits:
        by_source[hit.source][_canonical(hit.label)] += 1
    sources = sorted(by_source)
    contaminated = [by_source[s].get("contaminated", 0) for s in sources]
    partial = [by_source[s].get("partial", 0) for s in sources]

    fig, ax = plt.subplots(figsize=(1.4 * len(sources) + 2.0, 4.4))
    _style_axes(ax)
    ax.bar(sources, contaminated, color=CONTAMINATION_COLOR["contaminated"], label="contaminated", zorder=3)
    ax.bar(sources, partial, bottom=contaminated, color=CONTAMINATION_COLOR["partial"], label="partial", zorder=3)
    ax.set_xticks(range(len(sources)), sources, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Benchmark items", color=INK_PRIMARY)
    ax.set_title("Contamination by Training Source", color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, output_path, plt)


def plot_attribution_sources(skill_attributions, output_path: str | Path, top_sources: int = 6):  # pragma: no cover
    """Stacked bars of training-source shares per sub-skill."""
    plt = _require_matplotlib()
    all_sources: Counter = Counter()
    for agg in skill_attributions:
        for source, share in agg.source_shares.items():
            all_sources[source] += share
    ranked = [s for s, _ in all_sources.most_common(top_sources)]
    color_for = {source: CATEGORICAL[i % len(CATEGORICAL)] for i, source in enumerate(ranked)}

    fig, ax = plt.subplots(figsize=(1.5 * len(skill_attributions) + 2.0, 4.6))
    _style_axes(ax)
    skills = [agg.skill for agg in skill_attributions]
    bottoms = [0.0] * len(skill_attributions)
    for source in ranked:
        heights = [agg.source_shares.get(source, 0.0) for agg in skill_attributions]
        ax.bar(skills, heights, bottom=bottoms, color=color_for[source], label=source, zorder=3)
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=False)]
    ax.set_xticks(range(len(skills)), skills, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Similarity-weighted source share", color=INK_PRIMARY)
    ax.set_title("Attribution Sources by Sub-skill", color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, output_path, plt)


def plot_scale_reliability(atlas_by_model: dict, output_path: str | Path, level: str = "all"):  # pragma: no cover
    """Reliability vs model (size order) as one line per sub-skill."""
    plt = _require_matplotlib()
    models = list(atlas_by_model)
    skills = sorted({c.skill for report in atlas_by_model.values() for c in report.cells})

    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 2.0, 4.8))
    _style_axes(ax)
    for i, skill in enumerate(skills):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        y = []
        for model in models:
            match = next(
                (c.reliability for c in atlas_by_model[model].cells
                 if c.skill == skill and c.contamination_level == level),
                0.0,
            )
            y.append(match)
        ax.plot(range(len(models)), y, marker="o", markersize=7, linewidth=2, color=color, label=skill, zorder=3)
        ax.text(len(models) - 0.95, y[-1], f" {skill}", color=color, fontsize=8, va="center")
    ax.set_xticks(range(len(models)), models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reliability", color=INK_PRIMARY)
    ax.set_title("Reliability across Scale / Post-training", color=INK_PRIMARY, fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, output_path, plt)


def _save(fig, output_path: str | Path, plt):  # pragma: no cover - needs matplotlib
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _canonical(label: str) -> str:
    if label in {"contaminated", "exact", "near_duplicate"}:
        return "contaminated"
    if label in {"partial", "paraphrase_candidate"}:
        return "partial"
    return "clean"
