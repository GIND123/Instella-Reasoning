"""F2 — dose-response: what exposure does the premise-deletion probe actually detect?

This figure is what makes the Phase 1 null reportable. A null only means something if the
instrument could have found the effect, and the only way to know that is to put a known
amount of memorisation into a model and ask the probe about it.

Two series. **Injected** items were repeated `dose` times inside a fixed 8,388,608-token
budget; **held-out** items are clean test items never injected, and they are the
specificity control — if they rise with dose, the mixture taught GSM8K-shaped behaviour
rather than these documents, and the curve would be measuring generalisation.

The x-axis uses equal categorical spacing rather than a linear or log dose scale. Doses
are 0/1/4/16/64 and 0 has no place on a log axis; equal spacing states plainly that these
are five measured conditions, not samples of a continuum. Thin segments connect adjacent
measured points for readability and imply no interpolation - there is no fit here.

The shaded band is the exposure train_119K actually delivers, and it spans two honest
accountings because a single number would be arguable either way: 4x is the measured
maximum verbatim repeat of any one document in the corpus (119,014 rows, 88,179 distinct),
and 16x is 119,014/7,473 documents per GSM8K seed. The upper reading is conservative -
those ~16 documents are numeric re-instantiations carrying *different* gold answers, so
they are not answer-level exposure at all. Both readings fall left of the detection
threshold, which is the point: the conclusion does not depend on which one a reader
prefers.
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
from instella_reasoning.analysis.mathai_f1_abstention import save_both
from instella_reasoning.answer_equivalence import numeric_equal
from instella_reasoning.prompting import extract_answer
from instella_reasoning.records import GenerationRecord, read_benchmark, read_jsonl

DOSES = [0, 1, 4, 16, 64]


def _recall(item, completion: str) -> bool:
    pred = extract_answer(item, completion)
    parent = (item.metadata or {}).get("parent_answer")
    return bool(pred) and str(pred).strip() != "" and numeric_equal(str(parent), str(pred))


def _rate_ci(pairs: list[tuple[str, bool]], n_boot: int = 2000, seed: int = 6198):
    """Rate plus 95% CI, resampling parent problems rather than rows."""
    by_parent: dict[str, list[bool]] = defaultdict(list)
    for parent, hit in pairs:
        by_parent[parent].append(hit)
    keys = list(by_parent)
    if not keys:
        return 0.0, 0.0, 0.0, 0, 0
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
    return hit / tot, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)], hit, tot


def build(run: str | Path, out_dir: str | Path) -> list[str] | None:
    plt = _mpl()
    if plt is None:
        print("matplotlib not installed: pip install -e .[viz]")
        return None
    run, out_dir = Path(run), Path(out_dir)

    items = {i.id: i for i in read_benchmark(run / "base" / "gsm8k_test_deletion.jsonl")}
    inj = {json.loads(x)["id"] for x in
           (run / "phase2" / "mixture" / "injected.jsonl").read_text().splitlines() if x.strip()}
    hel = {json.loads(x)["id"] for x in
           (run / "phase2" / "mixture" / "heldout.jsonl").read_text().splitlines() if x.strip()}

    stats: dict[int, dict] = {}
    for d in DOSES:
        p = run / "generations" / f"phase2__dose{d}__test__T0.0__k1__vllm.jsonl"
        if not p.exists():
            continue
        rows = [GenerationRecord.from_dict(r) for r in read_jsonl(p)]
        cur: dict[str, list[tuple[str, bool]]] = {"injected": [], "heldout": []}
        for g in rows:
            it = items.get(g.benchmark_id)
            if it is None:
                continue
            group = "injected" if it.parent_id in inj else "heldout" if it.parent_id in hel else None
            if group:
                cur[group].append((it.parent_id, _recall(it, g.completion)))
        stats[d] = {k: _rate_ci(v) for k, v in cur.items()}

    xs = list(range(len(DOSES)))
    fig, ax = plt.subplots(figsize=(3.5, 3.0), facecolor=SURFACE)
    # Headroom is computed from the data, never assumed: the injected series
    # reaches 7.4% and a hard-coded 5% ceiling silently clips it off the axes.
    top = max(s[g][2] for s in stats.values() for g in ("injected", "heldout")) * 100
    ymax = max(2.0, top * 1.35)

    # The corpus's real exposure, spanning both defensible accountings (4x .. 16x).
    ax.axvspan(2, 3, color=INK_MUTED, alpha=0.13, zorder=1, linewidth=0)
    ax.text(2.5, ymax * 0.985, "train_119K", ha="center", va="top", fontsize=6.0,
            color=INK_SECONDARY)
    ax.text(2.5, ymax * 0.915, "actual dose", ha="center", va="top", fontsize=5.6,
            color=INK_MUTED)

    base = stats[0]["injected"][0] * 100 if 0 in stats else 0.0
    # The baseline is identified in the legend rather than by a floating label: at this
    # width every in-plot position for it collides with either the line itself or a series.
    ax.axhline(base, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2,
               label="0x baseline")

    for series, colour, label in (("injected", SERIES[1], "injected items"),
                                  ("heldout", SERIES[0], "held-out (clean)")):
        pts, los, his, counts = [], [], [], []
        for d in DOSES:
            r, lo, hi, hit, tot = stats[d][series]
            pts.append(r * 100); los.append((r - lo) * 100); his.append((hi - r) * 100)
            counts.append((hit, tot))
        ax.plot(xs, pts, color=colour, linewidth=1.4, zorder=3)
        ax.errorbar(xs, pts, yerr=[los, his], fmt="o", markersize=4.5, color=colour,
                    ecolor=colour, elinewidth=1.0, capsize=2.0,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=label, zorder=4)
        if series == "injected":
            for x, y, hi, (h, t) in zip(xs, pts, his, counts):
                ax.text(x, y + hi + 0.16, f"{h}/{t}", ha="center", va="bottom",
                        fontsize=5.8, color=INK_SECONDARY, zorder=5)

    # Detection threshold: the probe separates from 0x only between 16x and 64x.
    ax.axvline(3.5, color=INK, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.55, zorder=2)
    ax.text(3.57, ymax * 0.97, "detection\nthreshold", ha="left", va="top",
            fontsize=6.0, color=INK, linespacing=1.15)

    _style(ax, plt, xlabel="verbatim repetitions of each injected document",
           ylabel="deletion-recall rate (%)",
           title="The probe detects only far above the corpus dose",
           subtitle="recall of the parent answer on underdetermined items")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}x" for d in DOSES], fontsize=8, color=INK)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.35, 4.35)
    leg = ax.legend(frameon=False, fontsize=6.8, loc="lower left",
                    handlelength=1.2, borderpad=0.1, labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    p64 = stats[64]["injected"]
    fig.text(0.0, -0.10,
             "points: measured conditions, segments imply no interpolation · "
             f"labels: recall events / rows · 64x is the only dose separating from 0x "
             f"(McNemar b=17 c=6, p=0.035)\nerror bars: 95% cluster bootstrap over parent "
             "problems (2000 reps) · all rows generated with vLLM 0.8.5.post1, greedy",
             fontsize=5.4, color=INK_MUTED, linespacing=1.5)
    return save_both(fig, out_dir / "F2_dose_response", plt)


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "experiments/runs/ckpt-axis-v1"
    outd = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
    for p in build(run, outd) or []:
        print(f"wrote {p}")
