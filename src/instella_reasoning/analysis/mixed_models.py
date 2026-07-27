"""Item-level regression with cluster-robust inference — dependency-free.

arXiv:2605.28700 ("The Importance of Being Statistically Earnest") re-analysed 20
open-weight models on GSM-Symbolic and found that only about half of the reported
perturbation effects survive once per-question variation is modelled properly. Pooled
two-proportion tests treat every generation as independent, which they are not: variants
of the same item are strongly correlated (measured intra-cluster correlation on this
project's own data is ~0.48, so a naive test has roughly 2.4x too little variance and
correspondingly inflated significance).

This module provides the honest alternative without adding a hard dependency:

* :func:`fit_logistic` — logistic regression by IRLS;
* :func:`fit_logistic_cluster_robust` — the same fit with CR1 cluster-robust standard
  errors, clustering on ``parent_id``, which is the always-available primary analysis;
* :func:`fit_glmm` — a true logistic mixed model with per-item random intercepts via
  ``statsmodels`` when it is installed, reported alongside as confirmation.

Everything here is small-p (a handful of predictors), so plain Python linear algebra is
both fast enough and one less thing that can fail to install inside a Colab runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_MAX_ITER = 100
_TOL = 1e-9


# -- small dense linear algebra --------------------------------------------------


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n, k, m = len(a), len(b), len(b[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        arow = a[i]
        orow = out[i]
        for t in range(k):
            av = arow[t]
            if av == 0.0:
                continue
            brow = b[t]
            for j in range(m):
                orow[j] += av * brow[j]
    return out


def _invert(matrix: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inverse with partial pivoting; None if singular."""
    n = len(matrix)
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pval = aug[col][col]
        aug[col] = [v / pval for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            aug[r] = [v - factor * p for v, p in zip(aug[r], aug[col], strict=False)]
    return [row[n:] for row in aug]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# -- results ---------------------------------------------------------------------


@dataclass(slots=True)
class Coefficient:
    name: str
    estimate: float
    std_error: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def odds_ratio(self) -> float:
        return math.exp(self.estimate)

    def to_dict(self) -> dict:
        return {
            "term": self.name,
            "estimate": round(self.estimate, 6),
            "std_error": round(self.std_error, 6),
            "z": round(self.z, 4),
            "p_value": round(self.p_value, 6),
            "ci95": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "odds_ratio": round(self.odds_ratio, 6),
        }


@dataclass(slots=True)
class ModelFit:
    method: str
    n_obs: int
    n_clusters: int
    converged: bool
    coefficients: list[Coefficient] = field(default_factory=list)
    note: str = ""

    def term(self, name: str) -> Coefficient | None:
        return next((c for c in self.coefficients if c.name == name), None)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_obs": self.n_obs,
            "n_clusters": self.n_clusters,
            "converged": self.converged,
            "note": self.note,
            "coefficients": [c.to_dict() for c in self.coefficients],
        }


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _two_sided_p(z: float) -> float:
    return 2.0 * _normal_sf(abs(z))


# -- fitting ---------------------------------------------------------------------


def fit_logistic(
    design: list[list[float]], outcome: list[int], names: list[str]
) -> tuple[list[float], list[list[float]], list[float], bool]:
    """IRLS logistic fit. Returns (beta, bread = (X'WX)^-1, fitted probabilities, converged)."""
    n_obs = len(design)
    k = len(names)
    beta = [0.0] * k
    bread: list[list[float]] | None = None
    probs = [0.5] * n_obs
    converged = False

    for _ in range(_MAX_ITER):
        probs = [_sigmoid(sum(b * x for b, x in zip(beta, row, strict=False))) for row in design]
        # X'WX and X'(y - p)
        xtwx = [[0.0] * k for _ in range(k)]
        score = [0.0] * k
        for row, y, p in zip(design, outcome, probs, strict=False):
            w = max(p * (1.0 - p), 1e-10)
            resid = y - p
            for i in range(k):
                score[i] += row[i] * resid
                xi_w = row[i] * w
                for j in range(i, k):
                    xtwx[i][j] += xi_w * row[j]
        for i in range(k):
            for j in range(i):
                xtwx[i][j] = xtwx[j][i]

        inv = _invert(xtwx)
        if inv is None:
            return beta, [[0.0] * k for _ in range(k)], probs, False
        bread = inv
        step = [sum(inv[i][j] * score[j] for j in range(k)) for i in range(k)]
        # Damp the step: separation (a cell with 0% or 100% accuracy) otherwise sends a
        # coefficient to infinity and the fit never converges.
        norm = max(abs(s) for s in step) if step else 0.0
        if norm > 4.0:
            step = [s * (4.0 / norm) for s in step]
        beta = [b + s for b, s in zip(beta, step, strict=False)]
        if max(abs(s) for s in step) < _TOL:
            converged = True
            break

    probs = [_sigmoid(sum(b * x for b, x in zip(beta, row, strict=False))) for row in design]
    return beta, bread or [[0.0] * k for _ in range(k)], probs, converged


def fit_logistic_cluster_robust(
    design: list[list[float]],
    outcome: list[int],
    clusters: list[str],
    names: list[str],
) -> ModelFit:
    """Logistic regression with CR1 cluster-robust standard errors.

    Clustering on the benchmark item is the minimum correction the perturbation-evaluation
    literature now expects: variants of one problem are not independent observations, and
    treating them as such is what produces over-confident perturbation effects.
    """
    k = len(names)
    beta, bread, probs, converged = fit_logistic(design, outcome, names)

    # meat = sum over clusters of (X_g' u_g)(X_g' u_g)'
    per_cluster: dict[str, list[float]] = {}
    for row, y, p, g in zip(design, outcome, probs, clusters, strict=False):
        resid = y - p
        acc = per_cluster.setdefault(g, [0.0] * k)
        for i in range(k):
            acc[i] += row[i] * resid

    meat = [[0.0] * k for _ in range(k)]
    for vec in per_cluster.values():
        for i in range(k):
            if vec[i] == 0.0:
                continue
            for j in range(k):
                meat[i][j] += vec[i] * vec[j]

    n_obs, n_g = len(design), len(per_cluster)
    if n_g > 1 and n_obs > k:
        scale = (n_g / (n_g - 1)) * ((n_obs - 1) / (n_obs - k))
        meat = [[v * scale for v in row] for row in meat]

    cov = _matmul(_matmul(bread, meat), bread)

    coefficients = []
    for i, name in enumerate(names):
        var = cov[i][i]
        se = math.sqrt(var) if var > 0 else float("inf")
        z = beta[i] / se if se not in (0.0, float("inf")) else 0.0
        coefficients.append(
            Coefficient(
                name=name,
                estimate=beta[i],
                std_error=se,
                z=z,
                p_value=_two_sided_p(z) if se != float("inf") else 1.0,
                ci_low=beta[i] - 1.959964 * se if se != float("inf") else float("-inf"),
                ci_high=beta[i] + 1.959964 * se if se != float("inf") else float("inf"),
            )
        )
    return ModelFit(
        method="logistic_cluster_robust_CR1",
        n_obs=n_obs,
        n_clusters=n_g,
        converged=converged,
        coefficients=coefficients,
        note="SEs clustered on benchmark item (parent_id).",
    )


def fit_glmm(
    design: list[list[float]],
    outcome: list[int],
    clusters: list[str],
    names: list[str],
) -> ModelFit | None:
    """Logistic mixed model with per-item random intercepts, via statsmodels.

    Returns None when statsmodels/numpy are unavailable or the fit fails — the
    cluster-robust fit is the primary analysis and must never be blocked by this.
    """
    try:
        import numpy as np
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception:  # noqa: BLE001 - optional dependency, any failure is non-fatal
        return None

    try:
        exog = np.asarray(design, dtype=float)
        endog = np.asarray(outcome, dtype=float)
        uniq = {c: i for i, c in enumerate(sorted(set(clusters)))}
        exog_vc = np.zeros((len(clusters), len(uniq)))
        for row, c in enumerate(clusters):
            exog_vc[row, uniq[c]] = 1.0
        ident = np.zeros(len(uniq), dtype=int)
        model = BinomialBayesMixedGLM(
            endog, exog, exog_vc, ident, vcp_p=2.0, fe_p=2.0
        )
        res = model.fit_vb(verbose=False)
        coefficients = []
        for i, name in enumerate(names):
            est = float(res.fe_mean[i])
            se = float(res.fe_sd[i])
            z = est / se if se else 0.0
            coefficients.append(
                Coefficient(
                    name=name,
                    estimate=est,
                    std_error=se,
                    z=z,
                    p_value=_two_sided_p(z),
                    ci_low=est - 1.959964 * se,
                    ci_high=est + 1.959964 * se,
                )
            )
        return ModelFit(
            method="glmm_binomial_random_intercept_vb",
            n_obs=len(design),
            n_clusters=len(uniq),
            converged=True,
            coefficients=coefficients,
            note="Variational Bayes; per-item random intercepts (statsmodels).",
        )
    except Exception as exc:  # noqa: BLE001 - never let the optional path kill a run
        return ModelFit(
            method="glmm_binomial_random_intercept_vb",
            n_obs=len(design),
            n_clusters=len(set(clusters)),
            converged=False,
            coefficients=[],
            note=f"GLMM fit failed ({type(exc).__name__}: {exc}); rely on the cluster-robust fit.",
        )
