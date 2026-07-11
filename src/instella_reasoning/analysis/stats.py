"""Dependency-free statistics for the research analyses (Phase 6).

The implementation plan calls for: bootstrap confidence intervals on reliability
decomposition, Mann-Whitney U tests for distribution shifts, effect sizes
(Cohen's h for proportions, Cliff's delta for distributions), and Benjamini-Hochberg
FDR control across the many benchmark/sub-skill comparisons. All of it runs with the
standard library so the analysis layer has no hard scientific-Python dependency; when
``scipy`` is available the two-proportion p-value in
:mod:`instella_reasoning.analysis.accuracy_gap` uses its exact survival function.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    level: float

    def to_dict(self) -> dict:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "level": self.level,
        }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 6198,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for ``statistic`` over ``values``."""
    if not values:
        return ConfidenceInterval(0.0, 0.0, 0.0, level)
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(sample))
    estimates.sort()
    alpha = 1.0 - level
    low = _percentile(estimates, 100 * alpha / 2)
    high = _percentile(estimates, 100 * (1 - alpha / 2))
    return ConfidenceInterval(point=statistic(values), low=low, high=high, level=level)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion (better than normal at extremes)."""
    if total == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.95)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return ConfidenceInterval(point=p, low=max(0.0, center - margin), high=min(1.0, center + margin), level=0.95)


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for the difference between two proportions."""
    phi1 = 2 * math.asin(math.sqrt(_clamp01(p1)))
    phi2 = 2 * math.asin(math.sqrt(_clamp01(p2)))
    return phi1 - phi2


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]: P(a>b) - P(a<b)."""
    if not a or not b:
        return 0.0
    greater = 0
    less = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (len(a) * len(b))


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mann-Whitney U statistic and two-sided p-value (normal approximation, tie-corrected)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    combined = [(value, 0) for value in a] + [(value, 1) for value in b]
    combined.sort(key=lambda pair: pair[0])

    ranks = [0.0] * len(combined)
    tie_terms = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_terms += t**3 - t
        i = j + 1

    rank_sum_a = sum(rank for rank, (_, group) in zip(ranks, combined, strict=False) if group == 0)
    u1 = rank_sum_a - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    mu = n1 * n2 / 2
    sigma_sq = (n1 * n2 / 12) * ((n + 1) - tie_terms / (n * (n - 1))) if n > 1 else 0.0
    if sigma_sq <= 0:
        return u, 1.0
    z = (u - mu) / math.sqrt(sigma_sq)
    p_value = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return u, min(1.0, p_value)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return, for each p-value, whether it is rejected under BH FDR control at ``alpha``."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / m:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            rejected[idx] = True
    return rejected


def benjamini_hochberg_qvalues(p_values: Sequence[float]) -> list[float]:
    """BH-adjusted q-values, aligned with the input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        value = min(prev, p_values[idx] * m / rank)
        q[idx] = value
        prev = value
    return q


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
