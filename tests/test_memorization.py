"""The memorisation DiD and its cluster-robust inference.

These are recovery tests: data is simulated from a known ground truth and the estimator
has to return it. A statistical routine that is only checked for "runs without error" is
not checked at all — and this one produces the paper's headline number.
"""

from __future__ import annotations

import math
import random

from instella_reasoning.analysis.memorization import (
    build_cells,
    build_trajectory,
    difference_in_differences,
    magnitude_matched,
    memorization_regression,
    trajectory_deltas,
)
from instella_reasoning.analysis.mixed_models import fit_logistic_cluster_robust
from instella_reasoning.records import EvaluationRecord


def _synthesise(
    accuracies: dict[tuple[str, str], float],
    n_items: int = 200,
    n_perturbed: int = 3,
    item_sd: float = 0.9,
    seed: int = 0,
    model: str = "amd/Instella-3B",
) -> list[EvaluationRecord]:
    """Records with a per-item random effect, so clustering is real rather than assumed."""
    rng = random.Random(seed)
    out: list[EvaluationRecord] = []
    for arm in ("seen", "unseen"):
        for i in range(n_items):
            parent = f"{arm}_{i:04d}"
            offset = rng.gauss(0, item_sd)

            # Bind the item effect as a default argument: a late-bound closure over the
            # loop variable would silently give every item the *last* item's effect and
            # destroy the very clustering these tests exist to exercise.
            def draw(p: float, offset: float = offset) -> bool:
                logit = math.log(p / (1 - p)) + offset
                return rng.random() < 1 / (1 + math.exp(-logit))

            for index in range(1 + n_perturbed):
                perturbed = index > 0
                base = accuracies[(arm, "perturbed" if perturbed else "original")]
                ratio = math.exp(rng.gauss(0, 0.3))
                out.append(
                    EvaluationRecord(
                        benchmark_id=f"{parent}_{index}",
                        parent_id=parent,
                        variant_type="gsm_symbolic" if perturbed else "original",
                        expected="1",
                        predicted="1",
                        normalized_expected="1",
                        normalized_predicted="1",
                        correct=draw(base),
                        model=model,
                        metadata={
                            "arm": arm,
                            "benchmark": "gsm8k",
                            **(
                                {
                                    "answer_changing": True,
                                    "magnitude_ratio": ratio,
                                    "log_magnitude_ratio": math.log(ratio),
                                }
                                if perturbed
                                else {}
                            ),
                        },
                    )
                )
    return out


def test_recovers_a_known_memorisation_effect() -> None:
    """Seen advantage exists on originals only => DiD should recover +0.25."""
    records = _synthesise(
        {
            ("seen", "original"): 0.80,
            ("unseen", "original"): 0.55,
            ("seen", "perturbed"): 0.50,
            ("unseen", "perturbed"): 0.50,
        },
        seed=1,
    )
    result = difference_in_differences(records, n_bootstrap=800)
    assert result.status == "measured"
    assert 0.15 < result.did < 0.35
    assert result.ci_low > 0, result.to_dict()
    assert result.significant
    assert "Memorisation" in result.to_dict()["interpretation"]


def test_recovers_a_true_null_without_crying_wolf() -> None:
    """No seen advantage anywhere => DiD ~ 0 and the interval must cover it."""
    records = _synthesise(
        {
            ("seen", "original"): 0.65,
            ("unseen", "original"): 0.65,
            ("seen", "perturbed"): 0.50,
            ("unseen", "perturbed"): 0.50,
        },
        seed=2,
    )
    result = difference_in_differences(records, n_bootstrap=800)
    assert abs(result.did) < 0.10
    assert result.ci_low <= 0 <= result.ci_high
    assert not result.significant
    assert "No evidence" in result.to_dict()["interpretation"]


def test_perturbation_drop_alone_does_not_read_as_memorisation() -> None:
    """Both arms drop equally under perturbation: that is fragility, not memorisation."""
    records = _synthesise(
        {
            ("seen", "original"): 0.70,
            ("unseen", "original"): 0.70,
            ("seen", "perturbed"): 0.40,
            ("unseen", "perturbed"): 0.40,
        },
        seed=3,
    )
    result = difference_in_differences(records, n_bootstrap=800)
    assert not result.significant, result.to_dict()


def test_empty_cell_is_reported_not_silently_zero() -> None:
    records = [
        r for r in _synthesise({k: 0.5 for k in
                                [("seen", "original"), ("unseen", "original"),
                                 ("seen", "perturbed"), ("unseen", "perturbed")]}, n_items=5)
        if r.metadata["arm"] == "seen"
    ]
    result = difference_in_differences(records, n_bootstrap=50)
    assert result.status == "insufficient_cells"
    assert result.did == 0.0


def test_regression_interaction_matches_the_did_sign() -> None:
    records = _synthesise(
        {
            ("seen", "original"): 0.80,
            ("unseen", "original"): 0.55,
            ("seen", "perturbed"): 0.50,
            ("unseen", "perturbed"): 0.50,
        },
        seed=4,
    )
    payload = memorization_regression(records)
    assert payload["status"] == "measured"
    assert "log_magnitude" in payload["formula"]
    term = payload["did_term"]
    # On the log-odds scale the DiD appears as a negative interaction: perturbation costs
    # the seen arm more than the unseen arm.
    assert term["estimate"] < 0
    assert term["p_value"] < 0.05


def test_magnitude_matching_keeps_originals_and_filters_perturbed() -> None:
    records = _synthesise(
        {k: 0.6 for k in [("seen", "original"), ("unseen", "original"),
                          ("seen", "perturbed"), ("unseen", "perturbed")]},
        n_items=40, seed=5,
    )
    matched = magnitude_matched(records, tolerance=0.20)
    originals = [r for r in matched if r.variant_type == "original"]
    perturbed = [r for r in matched if r.variant_type == "gsm_symbolic"]
    assert len(originals) == len([r for r in records if r.variant_type == "original"])
    assert perturbed, "expected some perturbed rows to survive"
    assert all(0.8 <= float(r.metadata["magnitude_ratio"]) <= 1.2 for r in perturbed)


def test_cells_partition_every_labelled_record() -> None:
    records = _synthesise({k: 0.5 for k in
                           [("seen", "original"), ("unseen", "original"),
                            ("seen", "perturbed"), ("unseen", "perturbed")]}, n_items=10)
    cells = build_cells(records)
    assert sum(c.n for c in cells.values()) == len(records)


def test_trajectory_is_ordered_by_training_step() -> None:
    payload = {
        "amd/Instella-3B-Instruct": _synthesise(
            {k: 0.6 for k in [("seen", "original"), ("unseen", "original"),
                              ("seen", "perturbed"), ("unseen", "perturbed")]},
            n_items=20, seed=6, model="amd/Instella-3B-Instruct"),
        "amd/Instella-3B-Stage1": _synthesise(
            {k: 0.3 for k in [("seen", "original"), ("unseen", "original"),
                              ("seen", "perturbed"), ("unseen", "perturbed")]},
            n_items=20, seed=7, model="amd/Instella-3B-Stage1"),
    }
    points = build_trajectory(payload, n_bootstrap=100)
    assert [p.tag for p in points] == ["stage1", "instruct"]
    deltas = trajectory_deltas(points)
    assert deltas[0]["transition"] == "stage1 -> instruct"
    assert deltas[0]["introduces_gsm8k_data"] is True


def test_cluster_robust_se_is_calibrated_on_an_independent_fit() -> None:
    """Intercept-only logistic: the reported SE must match the analytic 1/sqrt(n p (1-p))."""
    n, p = 500, 0.6
    rng = random.Random(11)
    outcome = [1 if rng.random() < p else 0 for _ in range(n)]
    design = [[1.0] for _ in range(n)]
    fit = fit_logistic_cluster_robust(design, outcome, [str(i) for i in range(n)], ["intercept"])
    analytic = 1 / math.sqrt(n * p * (1 - p))
    assert fit.converged
    assert abs(fit.term("intercept").std_error - analytic) < 0.02
    assert abs(fit.term("intercept").estimate - math.log(p / (1 - p))) < 0.25
