"""Contamination-aware accuracy analysis — the study's headline number.

Given scored generations and contamination hits, this computes accuracy split by
contamination level (Contaminated / Partial / Clean) per benchmark and per model,
and tests whether the contaminated-vs-clean gap is statistically significant with a
two-proportion z-test. This is Phase 2 of the proposal: "If Instella scores 82% on
contaminated GSM8K but 55% on clean, that quantifies how much reasoning is memorised."

The z-test is implemented dependency-free (no scipy) so it runs anywhere; the code
falls back to scipy's exact survival function for the p-value when available and uses
a numerical normal-CDF approximation otherwise.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from instella_reasoning.records import ContaminationHit, EvaluationRecord

# Order of severity; a benchmark item takes its strongest contamination label.
_LABEL_RANK = {"contaminated": 3, "partial": 2, "near_duplicate": 3, "paraphrase_candidate": 2, "exact": 3}


def contamination_labels_by_benchmark_id(hits: list[ContaminationHit]) -> dict[str, str]:
    """Reduce many hits per item to one C/PC label (strongest wins); absent = clean."""
    best: dict[str, tuple[int, str]] = {}
    for hit in hits:
        rank = _LABEL_RANK.get(hit.label, 1)
        canonical = _canonical_label(hit.label)
        current = best.get(hit.benchmark_id)
        if current is None or rank > current[0]:
            best[hit.benchmark_id] = (rank, canonical)
    return {benchmark_id: label for benchmark_id, (_, label) in best.items()}


def _canonical_label(label: str) -> str:
    if label in {"contaminated", "exact", "near_duplicate"}:
        return "contaminated"
    if label in {"partial", "paraphrase_candidate"}:
        return "partial"
    return "clean"


def _normal_sf(z: float) -> float:
    """One-sided survival function of the standard normal (1 - CDF)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_proportion_z_test(
    correct_a: int, total_a: int, correct_b: int, total_b: int
) -> tuple[float, float]:
    """Return (z statistic, two-sided p-value) for H0: p_a == p_b.

    Group A is typically contaminated, group B clean. Uses the pooled-proportion
    z-test. Returns (0.0, 1.0) when either group is empty or has zero variance.
    """
    if total_a == 0 or total_b == 0:
        return 0.0, 1.0
    p_a = correct_a / total_a
    p_b = correct_b / total_b
    p_pool = (correct_a + correct_b) / (total_a + total_b)
    denom = p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b)
    if denom <= 0:
        return 0.0, 1.0
    z = (p_a - p_b) / math.sqrt(denom)
    try:  # exact p-value when scipy is present
        from scipy.stats import norm

        p_value = 2.0 * float(norm.sf(abs(z)))
    except ImportError:
        p_value = 2.0 * _normal_sf(abs(z))
    return z, p_value


@dataclass(slots=True)
class GroupStats:
    label: str
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(slots=True)
class AccuracyGapResult:
    scope: str  # e.g. "gsm8k::amd/Instella-3B" or "overall"
    by_label: dict[str, GroupStats] = field(default_factory=dict)
    z: float = 0.0
    p_value: float = 1.0
    gap: float = 0.0  # contaminated accuracy - clean accuracy

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "gap": round(self.gap, 6),
            "z": round(self.z, 6),
            "p_value": round(self.p_value, 6),
            "groups": {
                label: {
                    "total": stats.total,
                    "correct": stats.correct,
                    "accuracy": round(stats.accuracy, 6),
                }
                for label, stats in self.by_label.items()
            },
        }


def compute_accuracy_gap(
    scores: list[EvaluationRecord],
    contamination: list[ContaminationHit],
    per_benchmark: bool = True,
    per_model: bool = True,
) -> list[AccuracyGapResult]:
    """Compute contaminated-vs-clean accuracy gaps at the requested granularity.

    Every scored item is assigned a contamination label (clean when it has no hit),
    then grouped by (benchmark, model) as requested. For each group we report the
    per-label accuracy plus the contaminated-vs-clean z-test.
    """
    labels = contamination_labels_by_benchmark_id(contamination)

    grouped: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for record in scores:
        label = labels.get(record.benchmark_id, "clean")
        benchmark = str(record.metadata.get("benchmark", "unknown")) if per_benchmark else "all"
        model = record.model if per_model else "all"
        scope = f"{benchmark}::{model}"
        grouped[scope].append((label, record.correct))

    results: list[AccuracyGapResult] = []
    for scope, entries in sorted(grouped.items()):
        by_label: dict[str, GroupStats] = {}
        for label in ("contaminated", "partial", "clean"):
            subset = [correct for lab, correct in entries if lab == label]
            by_label[label] = GroupStats(label, len(subset), sum(subset))
        contaminated = by_label["contaminated"]
        clean = by_label["clean"]
        z, p_value = two_proportion_z_test(
            contaminated.correct, contaminated.total, clean.correct, clean.total
        )
        results.append(
            AccuracyGapResult(
                scope=scope,
                by_label=by_label,
                z=z,
                p_value=p_value,
                gap=contaminated.accuracy - clean.accuracy,
            )
        )
    return results
