"""Validation utilities that turn reviewer concerns into measurable numbers.

Two independent validations a rigorous submission needs:

1. **Contamination threshold calibration (M4).** Given a hand-labeled ground-truth set
   (benchmark_id -> is_contaminated), sweep the cosine threshold used by the embedding
   scanner and report precision / recall / F1, so the C/PC/N thresholds are *chosen*
   rather than asserted, and so "clean" can be reported as a detection lower bound with
   a known false-negative rate rather than as ground truth.

2. **Reliability-metric validation (M5).** The Reliability = accuracy x consistency
   metric claims to separate *genuine* reasoning from *fragile* pattern matching. This
   module scores synthetic clusters with known status and reports whether Reliability
   actually ranks genuine > fragile, giving a construct-validity number to cite.

Everything is dependency-free and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from instella_reasoning.metrics import summarize_reliability
from instella_reasoning.records import ContaminationHit, EvaluationRecord

# -- 1. contamination threshold calibration ------------------------------------


@dataclass(slots=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def best_cosine_by_benchmark_id(hits: list[ContaminationHit]) -> dict[str, float]:
    """Max cosine score seen for each benchmark item across all its hits."""
    best: dict[str, float] = {}
    for hit in hits:
        cosine = float(hit.metadata.get("cosine", hit.score))
        if hit.benchmark_id not in best or cosine > best[hit.benchmark_id]:
            best[hit.benchmark_id] = cosine
    return best


def calibrate_cosine_threshold(
    hits: list[ContaminationHit],
    ground_truth: dict[str, bool],
    thresholds: list[float] | None = None,
) -> list[ThresholdPoint]:
    """Sweep cosine thresholds against a labeled set; return a PR curve.

    ``ground_truth`` maps benchmark_id -> True (truly contaminated) / False (truly clean).
    An item is *predicted* contaminated at threshold t if its best hit cosine >= t.
    Items with no hit have best cosine 0.0 (predicted clean).
    """
    if thresholds is None:
        thresholds = [round(0.50 + 0.05 * i, 2) for i in range(11)]  # 0.50 .. 1.00
    best = best_cosine_by_benchmark_id(hits)
    points: list[ThresholdPoint] = []
    for t in thresholds:
        tp = fp = fn = 0
        for bid, truth in ground_truth.items():
            predicted = best.get(bid, 0.0) >= t
            if predicted and truth:
                tp += 1
            elif predicted and not truth:
                fp += 1
            elif not predicted and truth:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        points.append(ThresholdPoint(t, precision, recall, f1, tp, fp, fn))
    return points


def best_threshold(points: list[ThresholdPoint]) -> ThresholdPoint | None:
    """The threshold maximizing F1 (ties broken by higher precision)."""
    if not points:
        return None
    return max(points, key=lambda p: (p.f1, p.precision))


# -- 2. reliability-metric construct validation --------------------------------


@dataclass(slots=True)
class MetricValidation:
    genuine_reliability: float
    fragile_reliability: float
    separation: float  # genuine - fragile; > 0 means the metric orders them correctly
    genuine_acc_under_perturbation: float
    fragile_acc_under_perturbation: float
    separates: bool
    n_genuine: int = 0
    n_fragile: int = 0

    def to_dict(self) -> dict:
        return {
            "genuine_reliability": round(self.genuine_reliability, 6),
            "fragile_reliability": round(self.fragile_reliability, 6),
            "separation": round(self.separation, 6),
            "genuine_acc_under_perturbation": round(self.genuine_acc_under_perturbation, 6),
            "fragile_acc_under_perturbation": round(self.fragile_acc_under_perturbation, 6),
            "separates": self.separates,
            "n_genuine": self.n_genuine,
            "n_fragile": self.n_fragile,
        }


def validate_reliability_metric(
    scores: list[EvaluationRecord], labels: dict[str, str]
) -> MetricValidation:
    """Check that Reliability ranks *genuine* clusters above *fragile* ones.

    ``labels`` maps parent_id -> "genuine" | "fragile" (ground truth, e.g. from a synthetic
    probe set or manual annotation). Returns the mean Reliability and mean
    accuracy-under-perturbation for each group and whether the metric separates them.
    A positive ``separation`` is the construct-validity evidence a reviewer will ask for.
    """
    summaries = {s.parent_id: s for s in summarize_reliability(scores)}
    genuine = [summaries[p] for p, lab in labels.items() if lab == "genuine" and p in summaries]
    fragile = [summaries[p] for p, lab in labels.items() if lab == "fragile" and p in summaries]

    def _mean(values: list[float]) -> float:
        values = [v for v in values if v >= 0]
        return sum(values) / len(values) if values else 0.0

    g_rel = _mean([s.reliability for s in genuine])
    f_rel = _mean([s.reliability for s in fragile])
    return MetricValidation(
        genuine_reliability=g_rel,
        fragile_reliability=f_rel,
        separation=g_rel - f_rel,
        genuine_acc_under_perturbation=_mean([s.accuracy_under_perturbation for s in genuine]),
        fragile_acc_under_perturbation=_mean([s.accuracy_under_perturbation for s in fragile]),
        separates=g_rel > f_rel,
        n_genuine=len(genuine),
        n_fragile=len(fragile),
    )
