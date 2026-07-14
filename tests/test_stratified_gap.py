"""Difficulty-adjusted, cluster-robust, FDR-corrected accuracy gap (reviewer M3).

The key test constructs a pure difficulty *confound*: contaminated items are all easy and
correct; clean items are half easy-correct, half hard-wrong. The naive gap is large and
positive, but there is no contamination effect *within* a difficulty stratum — so the
Mantel-Haenszel-adjusted gap must collapse toward zero.
"""
from instella_reasoning.analysis.accuracy_gap import (
    compute_accuracy_gap,
    compute_stratified_accuracy_gap,
)
from instella_reasoning.records import ContaminationHit, EvaluationRecord


def _score(bid: str, correct: bool) -> EvaluationRecord:
    return EvaluationRecord(
        benchmark_id=bid,
        parent_id=bid,
        variant_type="original",
        expected="1",
        predicted="1" if correct else "0",
        normalized_expected="1",
        normalized_predicted="1" if correct else "0",
        correct=correct,
        model="instella-3b",
        metadata={"benchmark": "gsm8k"},
    )


def _hit(bid: str) -> ContaminationHit:
    return ContaminationHit(
        benchmark_id=bid, document_id="d", source="synthetic", label="contaminated",
        score=0.95, token_jaccard=0.9, char_ngram_jaccard=0.9, excerpt="", metadata={"cosine": 0.95},
    )


def _confounded_setup():
    scores, hits, bins = [], [], {}
    # contaminated: easy (bin 0), all correct
    for i in range(5):
        bid = f"c{i}"
        scores.append(_score(bid, True))
        hits.append(_hit(bid))
        bins[bid] = 0
    # clean easy (bin 0), all correct
    for i in range(5):
        bid = f"ne{i}"
        scores.append(_score(bid, True))
        bins[bid] = 0
    # clean hard (bin 1), all wrong
    for i in range(5):
        bid = f"nh{i}"
        scores.append(_score(bid, False))
        bins[bid] = 1
    return scores, hits, bins


def test_naive_gap_is_inflated_by_difficulty() -> None:
    scores, hits, _ = _confounded_setup()
    [result] = compute_accuracy_gap(scores, hits)
    # contaminated acc 1.0, clean acc 0.5 -> naive gap +0.5
    assert abs(result.gap - 0.5) < 1e-9


def test_difficulty_adjustment_collapses_the_confound() -> None:
    scores, hits, bins = _confounded_setup()
    [strat] = compute_stratified_accuracy_gap(scores, hits, bins, n_bootstrap=300)
    assert abs(strat.unadjusted_gap - 0.5) < 1e-9
    # Within difficulty, contaminated and clean are identical -> adjusted gap ~ 0.
    assert abs(strat.pooled_gap) < 1e-9
    assert strat.ci_low <= 0.0 <= strat.ci_high  # CI includes zero
    assert strat.n_clusters == 15


def test_fdr_qvalues_are_populated() -> None:
    scores, hits, _ = _confounded_setup()
    results = compute_accuracy_gap(scores, hits)
    assert all(0.0 <= r.q_value <= 1.0 for r in results)
    assert all(r.q_value >= r.p_value - 1e-9 for r in results)  # q >= p for BH
