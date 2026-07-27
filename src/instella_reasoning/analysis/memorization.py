"""The memorisation contrast — the study's headline analysis.

Design
------
GSM8K **train** items are verbatim in Instella's stage-2 training data; GSM8K **test**
items are not (verified exactly, not inferred — see
:mod:`instella_reasoning.datasets.splits`). Cross that verified treatment assignment with
the perturbation condition and you get a 2x2:

===============  ==================  ======================
                 original            numerically perturbed
===============  ==================  ======================
**seen**         A                   B
**unseen**       C                   D
===============  ==================  ======================

* ``A - C`` is the raw advantage on problems the model has provably seen.
* ``B - D`` is that same advantage after the surface numbers change.
* ``DiD = (A - C) - (B - D)`` is the part of the advantage that **does not survive
  perturbation** — the memorisation component.

The difference-in-differences form is what makes this robust. A plain ``A - C``
comparison is confounded by any residual difference between the train and test pools; a
plain ``A - B`` perturbation drop is confounded by the integer-magnitude shift that
arXiv:2605.28700 identifies. Differencing twice cancels both: the same perturbation
distribution is applied to both arms, so any magnitude effect enters ``A - B`` and
``C - D`` alike and subtracts out.

Predictions
-----------
* **Memorisation**: ``A >> C`` and ``DiD > 0`` — the advantage evaporates under perturbation.
* **Genuine reasoning**: ``A ~ C`` and ``DiD ~ 0`` — being trained on an item confers no
  test-time advantage beyond what generalisation already provides.

A tight interval around zero is a real result, not a failed experiment: it bounds how
much of the model's benchmark accuracy memorisation can explain.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from instella_reasoning.analysis.mixed_models import ModelFit, fit_glmm, fit_logistic_cluster_robust
from instella_reasoning.metrics import is_answer_changing
from instella_reasoning.records import EvaluationRecord

SEEN = "seen"
UNSEEN = "unseen"
ORIGINAL = "original"
PERTURBED = "perturbed"


def record_arm(record: EvaluationRecord) -> str | None:
    """``seen`` / ``unseen`` from the verified split, or None if unlabelled."""
    arm = record.metadata.get("arm")
    return str(arm) if arm in (SEEN, UNSEEN) else None


def record_condition(record: EvaluationRecord) -> str:
    """``perturbed`` for answer-changing variants, else ``original``.

    Surface (answer-preserving) variants are deliberately *not* counted as perturbed
    here: they belong to the consistency term, and folding them in would mix an
    answer-preserving probe with an answer-changing one inside a single cell.
    """
    if is_answer_changing(record):
        return PERTURBED
    return ORIGINAL if record.variant_type == "original" else ORIGINAL


@dataclass(slots=True)
class Cell:
    arm: str
    condition: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "condition": self.condition,
            "n": self.n,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
        }


@dataclass(slots=True)
class DiDResult:
    """Difference-in-differences memorisation estimate for one model."""

    model: str
    cells: dict[str, Cell] = field(default_factory=dict)
    seen_advantage_original: float = 0.0
    seen_advantage_perturbed: float = 0.0
    did: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    n_clusters_seen: int = 0
    n_clusters_unseen: int = 0
    status: str = "measured"

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "cells": {k: v.to_dict() for k, v in sorted(self.cells.items())},
            "seen_advantage_original": round(self.seen_advantage_original, 6),
            "seen_advantage_perturbed": round(self.seen_advantage_perturbed, 6),
            "difference_in_differences": round(self.did, 6),
            "cluster_bootstrap_ci95": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "n_clusters": {"seen": self.n_clusters_seen, "unseen": self.n_clusters_unseen},
            "excludes_zero": self.significant,
            "interpretation": _interpret(self),
        }


def _interpret(result: DiDResult) -> str:
    if result.status != "measured":
        return "Not estimable: one or more cells is empty."
    if result.significant and result.did > 0:
        return (
            "Memorisation: the advantage on provably-seen items does not survive numeric "
            "perturbation."
        )
    if result.significant and result.did < 0:
        return "Unexpected sign: seen items are *more* robust to perturbation than unseen."
    width = result.ci_high - result.ci_low
    return (
        f"No evidence of a memorisation component; 95% CI excludes effects outside "
        f"[{result.ci_low:.3f}, {result.ci_high:.3f}] (width {width:.3f})."
    )


def _key(arm: str, condition: str) -> str:
    return f"{arm}|{condition}"


def build_cells(records: list[EvaluationRecord]) -> dict[str, Cell]:
    cells: dict[str, Cell] = {
        _key(a, c): Cell(a, c, 0, 0) for a in (SEEN, UNSEEN) for c in (ORIGINAL, PERTURBED)
    }
    for record in records:
        arm = record_arm(record)
        if arm is None:
            continue
        cell = cells[_key(arm, record_condition(record))]
        cell.n += 1
        cell.correct += int(record.correct)
    return cells


def _did_from_groups(groups: dict[str, list[tuple[str, bool]]]) -> float | None:
    """DiD from {cell_key: [(parent, correct), ...]}; None if any cell is empty."""
    acc = {}
    for key, rows in groups.items():
        if not rows:
            return None
        acc[key] = sum(1 for _, ok in rows if ok) / len(rows)
    return (acc[_key(SEEN, ORIGINAL)] - acc[_key(UNSEEN, ORIGINAL)]) - (
        acc[_key(SEEN, PERTURBED)] - acc[_key(UNSEEN, PERTURBED)]
    )


def difference_in_differences(
    records: list[EvaluationRecord],
    model: str | None = None,
    n_bootstrap: int = 4000,
    seed: int = 6198,
) -> DiDResult:
    """Estimate the memorisation DiD with a cluster-robust bootstrap CI.

    The bootstrap resamples **parent items within each arm**, not individual generations.
    Resampling generations would treat the variants of one problem as independent draws
    and shrink the interval by roughly the design effect (measured ~2.4x on this data),
    producing a confidently wrong CI.
    """
    subset = [r for r in records if record_arm(r) is not None]
    if model is not None:
        subset = [r for r in subset if r.model == model]
    label = model or (subset[0].model if subset else "unknown")

    cells = build_cells(subset)
    result = DiDResult(model=label, cells=cells)
    if any(c.n == 0 for c in cells.values()):
        result.status = "insufficient_cells"
        return result

    result.seen_advantage_original = (
        cells[_key(SEEN, ORIGINAL)].accuracy - cells[_key(UNSEEN, ORIGINAL)].accuracy
    )
    result.seen_advantage_perturbed = (
        cells[_key(SEEN, PERTURBED)].accuracy - cells[_key(UNSEEN, PERTURBED)].accuracy
    )
    result.did = result.seen_advantage_original - result.seen_advantage_perturbed

    # cluster -> its rows, kept per arm so the bootstrap preserves the arm sizes
    by_arm: dict[str, dict[str, list[tuple[str, bool]]]] = {SEEN: {}, UNSEEN: {}}
    for record in subset:
        arm = record_arm(record)
        parent = record.parent_id or record.benchmark_id
        by_arm[arm].setdefault(parent, []).append((record_condition(record), record.correct))
    result.n_clusters_seen = len(by_arm[SEEN])
    result.n_clusters_unseen = len(by_arm[UNSEEN])

    rng = random.Random(f"{seed}:did:{label}")
    estimates: list[float] = []
    arms = {arm: list(clusters) for arm, clusters in by_arm.items()}
    for _ in range(n_bootstrap):
        groups: dict[str, list[tuple[str, bool]]] = {
            _key(a, c): [] for a in (SEEN, UNSEEN) for c in (ORIGINAL, PERTURBED)
        }
        for arm, cluster_ids in arms.items():
            if not cluster_ids:
                continue
            for _ in range(len(cluster_ids)):
                pick = cluster_ids[rng.randrange(len(cluster_ids))]
                for condition, ok in by_arm[arm][pick]:
                    groups[_key(arm, condition)].append((pick, ok))
        value = _did_from_groups(groups)
        if value is not None:
            estimates.append(value)

    if estimates:
        estimates.sort()
        result.ci_low = _percentile(estimates, 2.5)
        result.ci_high = _percentile(estimates, 97.5)
    return result


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return sorted_values[int(low)]
    weight = rank - low
    return sorted_values[int(low)] * (1 - weight) + sorted_values[int(high)] * weight


# -- regression form of the same contrast ----------------------------------------


def memorization_regression(
    records: list[EvaluationRecord],
    model: str | None = None,
    include_magnitude: bool = True,
) -> dict:
    """The DiD as a regression, with the magnitude confound entered as a covariate.

    Design matrix: ``correct ~ 1 + seen + perturbed + seen:perturbed [+ log_magnitude]``.
    The ``seen:perturbed`` interaction *is* the DiD, expressed on the log-odds scale, and
    its cluster-robust p-value is the headline inferential number.

    ``log_magnitude`` is the log ratio of a perturbed item's gold answer to its original.
    Including it is not optional politeness: arXiv:2605.28700 showed that controlling for
    exactly this shift removed the statistical significance of about half the perturbation
    effects in the original GSM-Symbolic analysis.
    """
    subset = [r for r in records if record_arm(r) is not None]
    if model is not None:
        subset = [r for r in subset if r.model == model]
    if not subset:
        return {"status": "no_labelled_records"}

    names = ["intercept", "seen", "perturbed", "seen_x_perturbed"]
    use_magnitude = include_magnitude and any(
        "log_magnitude_ratio" in r.metadata for r in subset
    )
    if use_magnitude:
        names.append("log_magnitude")

    design: list[list[float]] = []
    outcome: list[int] = []
    clusters: list[str] = []
    for record in subset:
        seen = 1.0 if record_arm(record) == SEEN else 0.0
        perturbed = 1.0 if record_condition(record) == PERTURBED else 0.0
        row = [1.0, seen, perturbed, seen * perturbed]
        if use_magnitude:
            row.append(float(record.metadata.get("log_magnitude_ratio", 0.0) or 0.0))
        design.append(row)
        outcome.append(int(record.correct))
        clusters.append(record.parent_id or record.benchmark_id)

    primary = fit_logistic_cluster_robust(design, outcome, clusters, names)
    payload: dict = {
        "status": "measured",
        "model": model or subset[0].model,
        "formula": "correct ~ seen * perturbed"
        + (" + log_magnitude" if use_magnitude else "")
        + " , cluster(parent_id)",
        "primary": primary.to_dict(),
        "did_term": (primary.term("seen_x_perturbed") or _empty_term()).to_dict(),
    }
    glmm: ModelFit | None = fit_glmm(design, outcome, clusters, names)
    if glmm is not None:
        payload["glmm"] = glmm.to_dict()
    else:
        payload["glmm"] = {
            "method": "unavailable",
            "note": "statsmodels not installed; cluster-robust fit is the primary analysis.",
        }
    return payload


def _empty_term():
    from instella_reasoning.analysis.mixed_models import Coefficient

    return Coefficient("seen_x_perturbed", 0.0, float("inf"), 0.0, 1.0, 0.0, 0.0)


# -- magnitude-matched sensitivity -----------------------------------------------


def magnitude_matched(
    records: list[EvaluationRecord], tolerance: float = 0.20
) -> list[EvaluationRecord]:
    """Keep only perturbed items whose gold answer is within ``tolerance`` of the original.

    The strongest possible answer to "your perturbation just made the arithmetic bigger":
    re-run the whole contrast on the subset where it demonstrably did not. Originals and
    answer-preserving variants are always kept.
    """
    keep: list[EvaluationRecord] = []
    lo, hi = 1.0 - tolerance, 1.0 + tolerance
    for record in records:
        if not is_answer_changing(record):
            keep.append(record)
            continue
        ratio = record.metadata.get("magnitude_ratio")
        if ratio is None:
            continue
        if lo <= float(ratio) <= hi:
            keep.append(record)
    return keep


# -- the checkpoint trajectory ----------------------------------------------------


@dataclass(slots=True)
class TrajectoryPoint:
    tag: str
    step: int
    model: str
    intervention: str
    saw_gsm8k_derived_data: bool
    accuracy_seen: float
    accuracy_unseen: float
    accuracy_seen_perturbed: float
    accuracy_unseen_perturbed: float
    did: float
    did_ci: tuple[float, float]
    n_obs: int

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "step": self.step,
            "model": self.model,
            "intervention": self.intervention,
            "saw_gsm8k_derived_data": self.saw_gsm8k_derived_data,
            "accuracy": {
                "seen_original": round(self.accuracy_seen, 6),
                "unseen_original": round(self.accuracy_unseen, 6),
                "seen_perturbed": round(self.accuracy_seen_perturbed, 6),
                "unseen_perturbed": round(self.accuracy_unseen_perturbed, 6),
            },
            "did": round(self.did, 6),
            "did_ci95": [round(self.did_ci[0], 6), round(self.did_ci[1], 6)],
            "n_obs": self.n_obs,
        }


def build_trajectory(
    records_by_model: dict[str, list[EvaluationRecord]],
    n_bootstrap: int = 2000,
) -> list[TrajectoryPoint]:
    """Per-checkpoint 2x2 accuracies and DiD, ordered along the Instella trajectory.

    The step from ``Instella-3B-Stage1`` to ``Instella-3B`` is the one that introduces
    GSM8K-derived data. A DiD that is ~0 at Stage 1 and positive at Stage 2 is direct
    causal evidence that stage-2 data bought memorisation rather than reasoning; a DiD
    that stays ~0 across the step is direct evidence that it did not.
    """
    from instella_reasoning.checkpoints import try_resolve

    points: list[TrajectoryPoint] = []
    for model_name, records in records_by_model.items():
        ckpt = try_resolve(model_name)
        did = difference_in_differences(records, n_bootstrap=n_bootstrap)
        cells = did.cells or build_cells(records)
        points.append(
            TrajectoryPoint(
                tag=ckpt.tag if ckpt else model_name,
                step=ckpt.step if ckpt else 99,
                model=model_name,
                intervention=ckpt.intervention if ckpt else "unknown",
                saw_gsm8k_derived_data=bool(ckpt.saw_gsm8k_derived_data) if ckpt else False,
                accuracy_seen=cells[_key(SEEN, ORIGINAL)].accuracy,
                accuracy_unseen=cells[_key(UNSEEN, ORIGINAL)].accuracy,
                accuracy_seen_perturbed=cells[_key(SEEN, PERTURBED)].accuracy,
                accuracy_unseen_perturbed=cells[_key(UNSEEN, PERTURBED)].accuracy,
                did=did.did,
                did_ci=(did.ci_low, did.ci_high),
                n_obs=len(records),
            )
        )
    return sorted(points, key=lambda p: p.step)


def trajectory_deltas(points: list[TrajectoryPoint]) -> list[dict]:
    """Change in each quantity across consecutive checkpoints — the intervention effects."""
    out = []
    for before, after in zip(points, points[1:], strict=False):
        out.append(
            {
                "transition": f"{before.tag} -> {after.tag}",
                "intervention": after.intervention,
                "introduces_gsm8k_data": after.saw_gsm8k_derived_data
                and not before.saw_gsm8k_derived_data,
                "delta_accuracy_seen": round(after.accuracy_seen - before.accuracy_seen, 6),
                "delta_accuracy_unseen": round(after.accuracy_unseen - before.accuracy_unseen, 6),
                "delta_did": round(after.did - before.did, 6),
            }
        )
    return out


def summarize(records: list[EvaluationRecord], n_bootstrap: int = 4000) -> dict:
    """Everything the headline analysis produces, for one scores file with several models."""
    by_model: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model].append(record)

    per_model = {}
    for model_name, rows in sorted(by_model.items()):
        did = difference_in_differences(rows, n_bootstrap=n_bootstrap)
        matched = magnitude_matched(rows)
        did_matched = difference_in_differences(
            matched, model=model_name, n_bootstrap=max(500, n_bootstrap // 4)
        )
        per_model[model_name] = {
            "did": did.to_dict(),
            "did_magnitude_matched": did_matched.to_dict(),
            "regression": memorization_regression(rows, model=model_name),
        }

    points = build_trajectory(dict(by_model), n_bootstrap=max(500, n_bootstrap // 4))
    return {
        "per_model": per_model,
        "trajectory": [p.to_dict() for p in points],
        "trajectory_deltas": trajectory_deltas(points),
    }
