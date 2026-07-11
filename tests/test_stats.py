
from instella_reasoning.analysis.stats import (
    benjamini_hochberg,
    benjamini_hochberg_qvalues,
    bootstrap_ci,
    cliffs_delta,
    cohens_h,
    mann_whitney_u,
    wilson_interval,
)


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 50
    ci = bootstrap_ci(values, n_resamples=500, seed=1)
    assert ci.low <= ci.point <= ci.high
    assert abs(ci.point - 0.5) < 1e-9
    assert 0.3 < ci.low < ci.high < 0.7


def test_wilson_interval_matches_known_value() -> None:
    ci = wilson_interval(50, 100)
    assert abs(ci.point - 0.5) < 1e-9
    assert ci.low < 0.5 < ci.high
    # Wilson interval for 50/100 is roughly [0.404, 0.596]
    assert abs(ci.low - 0.404) < 0.01
    assert abs(ci.high - 0.596) < 0.01


def test_cohens_h_zero_when_equal() -> None:
    assert abs(cohens_h(0.5, 0.5)) < 1e-12
    assert cohens_h(0.9, 0.5) > 0


def test_cliffs_delta_full_separation() -> None:
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    assert abs(cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-12


def test_mann_whitney_detects_shift() -> None:
    a = [10, 11, 12, 13, 14]
    b = [1, 2, 3, 4, 5]
    u, p = mann_whitney_u(a, b)
    assert p < 0.05
    # identical distributions -> not significant
    _, p_equal = mann_whitney_u([1, 2, 3, 4], [1, 2, 3, 4])
    assert p_equal > 0.5


def test_benjamini_hochberg_controls_fdr() -> None:
    p_values = [0.001, 0.01, 0.2, 0.5, 0.9]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected[0] is True
    assert rejected[-1] is False
    q = benjamini_hochberg_qvalues(p_values)
    assert all(0.0 <= value <= 1.0 for value in q)
    # q-values are monotone in the sorted p-value order
    assert q[0] <= q[-1]


def test_mann_whitney_empty_is_safe() -> None:
    u, p = mann_whitney_u([], [1, 2, 3])
    assert u == 0.0 and p == 1.0
