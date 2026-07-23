from instella_reasoning.analysis.accuracy_gap import (
    compute_accuracy_gap,
    contamination_labels_by_benchmark_id,
    two_proportion_z_test,
)
from instella_reasoning.records import ContaminationHit, EvaluationRecord


def _hit(benchmark_id: str, label: str) -> ContaminationHit:
    return ContaminationHit(
        benchmark_id=benchmark_id,
        document_id="doc",
        source="src",
        label=label,
        score=0.9,
        token_jaccard=0.5,
        char_ngram_jaccard=0.5,
        excerpt="",
    )


def _eval(
    benchmark_id: str,
    correct: bool,
    benchmark: str = "gsm8k",
    model: str = "m",
    parent_id: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        benchmark_id=benchmark_id,
        parent_id=parent_id or benchmark_id,
        variant_type="rephrase" if parent_id else "original",
        expected="1",
        predicted="1" if correct else "0",
        normalized_expected="1",
        normalized_predicted="1" if correct else "0",
        correct=correct,
        model=model,
        metadata={"benchmark": benchmark},
    )


def test_strongest_label_wins() -> None:
    labels = contamination_labels_by_benchmark_id(
        [_hit("q1", "partial"), _hit("q1", "contaminated"), _hit("q2", "paraphrase_candidate")]
    )
    assert labels["q1"] == "contaminated"
    assert labels["q2"] == "partial"


def test_z_test_zero_variance_is_safe() -> None:
    z, p = two_proportion_z_test(0, 0, 5, 10)
    assert z == 0.0 and p == 1.0


def test_z_test_detects_large_gap() -> None:
    # 90/100 contaminated correct vs 40/100 clean correct -> highly significant.
    z, p = two_proportion_z_test(90, 100, 40, 100)
    assert z > 0
    assert p < 0.001


def test_gap_is_zero_when_a_group_is_empty() -> None:
    # No contaminated items -> gap must be 0, not -clean_accuracy.
    scores = [_eval("q1", True), _eval("q2", True)]  # both clean, both correct
    results = compute_accuracy_gap(scores, [])
    assert len(results) == 1
    assert results[0].gap == 0.0
    assert results[0].by_label["contaminated"].total == 0


def test_compute_accuracy_gap_splits_by_label() -> None:
    hits = [_hit("q1", "contaminated"), _hit("q2", "contaminated")]
    scores = [
        _eval("q1", True),
        _eval("q2", True),
        _eval("q3", False),  # clean (no hit)
        _eval("q4", False),  # clean
    ]
    results = compute_accuracy_gap(scores, hits)
    assert len(results) == 1
    result = results[0]
    assert result.by_label["contaminated"].accuracy == 1.0
    assert result.by_label["clean"].accuracy == 0.0
    assert result.gap == 1.0


def test_variant_inherits_parent_contamination_label() -> None:
    scores = [
        _eval("q1__rephrase", True, parent_id="q1"),
        _eval("q2__rephrase", False, parent_id="q2"),
    ]
    result = compute_accuracy_gap(scores, [_hit("q1", "contaminated")])[0]
    assert result.by_label["contaminated"].total == 1
    assert result.by_label["contaminated"].accuracy == 1.0
    assert result.by_label["clean"].total == 1
