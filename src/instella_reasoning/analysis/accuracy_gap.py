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
import random
from collections import defaultdict
from dataclasses import dataclass, field

from instella_reasoning.analysis.stats import benjamini_hochberg_qvalues
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


def _record_contamination_label(
    record: EvaluationRecord,
    labels: dict[str, str],
) -> str:
    """Resolve contamination for an original item or one of its variants."""
    return labels.get(record.benchmark_id, labels.get(record.parent_id, "clean"))


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
    q_value: float = 1.0  # BH-FDR adjusted p-value across all scopes
    gap: float = 0.0  # contaminated accuracy - clean accuracy

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "gap": round(self.gap, 6),
            "z": round(self.z, 6),
            "p_value": round(self.p_value, 6),
            "q_value": round(self.q_value, 6),
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
        label = _record_contamination_label(record, labels)
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
        # A gap is only meaningful when both groups are populated; otherwise report 0.
        gap = (
            contaminated.accuracy - clean.accuracy
            if contaminated.total and clean.total
            else 0.0
        )
        results.append(
            AccuracyGapResult(scope=scope, by_label=by_label, z=z, p_value=p_value, gap=gap)
        )

    # Benjamini-Hochberg FDR across every scope tested (reviewer concern M3: many
    # (benchmark x model) comparisons inflate false positives without correction).
    q_values = benjamini_hochberg_qvalues([r.p_value for r in results])
    for result, q in zip(results, q_values, strict=False):
        result.q_value = q
    return results


# -- difficulty-stratified, cluster-robust gap (reviewer concern M3) -------------


@dataclass(slots=True)
class StratumStats:
    difficulty_bin: int
    contaminated: GroupStats
    clean: GroupStats

    @property
    def gap(self) -> float:
        if not self.contaminated.total or not self.clean.total:
            return 0.0
        return self.contaminated.accuracy - self.clean.accuracy

    @property
    def mh_weight(self) -> float:
        """Mantel-Haenszel weight n1*n2/(n1+n2); zero when a cell is empty."""
        n1, n2 = self.contaminated.total, self.clean.total
        return (n1 * n2) / (n1 + n2) if (n1 + n2) else 0.0


@dataclass(slots=True)
class StratifiedGapResult:
    scope: str
    unadjusted_gap: float  # naive contaminated - clean, ignoring difficulty
    pooled_gap: float  # Mantel-Haenszel difficulty-adjusted gap
    ci_low: float
    ci_high: float
    n_clusters: int
    strata: list[StratumStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "unadjusted_gap": round(self.unadjusted_gap, 6),
            "pooled_gap_difficulty_adjusted": round(self.pooled_gap, 6),
            "cluster_bootstrap_ci": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "n_clusters": self.n_clusters,
            "strata": [
                {
                    "difficulty_bin": s.difficulty_bin,
                    "contaminated": {"n": s.contaminated.total, "acc": round(s.contaminated.accuracy, 6)},
                    "clean": {"n": s.clean.total, "acc": round(s.clean.accuracy, 6)},
                    "gap": round(s.gap, 6),
                    "weight": round(s.mh_weight, 6),
                }
                for s in self.strata
            ],
        }


def _mantel_haenszel_gap(rows: list[tuple[int, str, bool]]) -> float:
    """Difficulty-adjusted contaminated-vs-clean gap.

    ``rows`` are (difficulty_bin, label, correct). Within each bin we compute the
    contaminated-minus-clean accuracy difference, then pool across bins with MH weights
    n1*n2/(n1+n2). This removes a difficulty confound: if contaminated items are simply
    easier, the naive gap shrinks toward the within-difficulty gap.
    """
    by_bin: dict[int, list[tuple[str, bool]]] = defaultdict(list)
    for bin_id, label, correct in rows:
        by_bin[bin_id].append((label, correct))
    num = den = 0.0
    for entries in by_bin.values():
        c = [ok for lab, ok in entries if lab == "contaminated"]
        n = [ok for lab, ok in entries if lab == "clean"]
        if not c or not n:
            continue
        weight = (len(c) * len(n)) / (len(c) + len(n))
        gap = sum(c) / len(c) - sum(n) / len(n)
        num += weight * gap
        den += weight
    return num / den if den else 0.0


def compute_stratified_accuracy_gap(
    scores: list[EvaluationRecord],
    contamination: list[ContaminationHit],
    difficulty_bins: dict[str, int],
    per_benchmark: bool = True,
    per_model: bool = True,
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 6198,
) -> list[StratifiedGapResult]:
    """Difficulty-adjusted gap with a cluster-robust bootstrap CI.

    ``difficulty_bins`` maps benchmark_id -> equal-frequency difficulty bin (see
    :func:`instella_reasoning.difficulty.assign_difficulty_bins`). The bootstrap resamples
    **parent clusters** (not individual variants) so consistency-variant correlation does
    not shrink the interval artificially.
    """
    labels = contamination_labels_by_benchmark_id(contamination)

    # Group records by scope, carrying (parent_id, difficulty_bin, label, correct).
    grouped: dict[str, list[tuple[str, int, str, bool]]] = defaultdict(list)
    for record in scores:
        label = _record_contamination_label(record, labels)
        if label == "partial":
            continue  # gap contrasts contaminated vs clean; partial is excluded
        benchmark = str(record.metadata.get("benchmark", "unknown")) if per_benchmark else "all"
        model = record.model if per_model else "all"
        scope = f"{benchmark}::{model}"
        bin_id = difficulty_bins.get(record.benchmark_id, difficulty_bins.get(record.parent_id, 0))
        parent = record.parent_id or record.benchmark_id
        grouped[scope].append((parent, bin_id, label, record.correct))

    results: list[StratifiedGapResult] = []
    for scope, entries in sorted(grouped.items()):
        rows = [(bin_id, label, correct) for _, bin_id, label, correct in entries]
        pooled = _mantel_haenszel_gap(rows)

        # naive gap
        cont = [ok for _, _, lab, ok in entries if lab == "contaminated"]
        clean = [ok for _, _, lab, ok in entries if lab == "clean"]
        unadjusted = (sum(cont) / len(cont) - sum(clean) / len(clean)) if cont and clean else 0.0

        # per-stratum breakdown
        strata: list[StratumStats] = []
        bins = sorted({bin_id for bin_id, _, _ in rows})
        for bin_id in bins:
            c = [ok for b, lab, ok in rows if b == bin_id and lab == "contaminated"]
            n = [ok for b, lab, ok in rows if b == bin_id and lab == "clean"]
            strata.append(
                StratumStats(
                    difficulty_bin=bin_id,
                    contaminated=GroupStats("contaminated", len(c), sum(c)),
                    clean=GroupStats("clean", len(n), sum(n)),
                )
            )

        # cluster bootstrap: resample parent clusters with replacement
        clusters: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
        for parent, bin_id, label, correct in entries:
            clusters[parent].append((bin_id, label, correct))
        cluster_ids = list(clusters)
        rng = random.Random(f"{seed}:{scope}")
        estimates: list[float] = []
        for _ in range(n_bootstrap):
            sampled_rows: list[tuple[int, str, bool]] = []
            for _ in range(len(cluster_ids)):
                pick = cluster_ids[rng.randrange(len(cluster_ids))]
                sampled_rows.extend(clusters[pick])
            estimates.append(_mantel_haenszel_gap(sampled_rows))
        estimates.sort()
        ci_low, ci_high = _percentile_pair(estimates, level)

        results.append(
            StratifiedGapResult(
                scope=scope,
                unadjusted_gap=unadjusted,
                pooled_gap=pooled,
                ci_low=ci_low,
                ci_high=ci_high,
                n_clusters=len(cluster_ids),
                strata=strata,
            )
        )
    return results


def _percentile_pair(sorted_values: list[float], level: float) -> tuple[float, float]:
    if not sorted_values:
        return 0.0, 0.0
    alpha = 1.0 - level
    return (
        _percentile(sorted_values, 100 * alpha / 2),
        _percentile(sorted_values, 100 * (1 - alpha / 2)),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight
