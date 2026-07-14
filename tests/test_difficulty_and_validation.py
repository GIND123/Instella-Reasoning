"""Difficulty estimation, variant validation, contamination calibration, metric validation."""
from instella_reasoning.difficulty import assign_difficulty_bins, estimate_difficulty
from instella_reasoning.metrics import summarize_reliability
from instella_reasoning.perturbations import validate_variants
from instella_reasoning.records import BenchmarkItem, ContaminationHit, EvaluationRecord
from instella_reasoning.validation import (
    best_threshold,
    calibrate_cosine_threshold,
    validate_reliability_metric,
)

# -- difficulty ----------------------------------------------------------------


def test_difficulty_counts_gsm8k_steps() -> None:
    item = BenchmarkItem("g", "q", answer="9", metadata={
        "rationale": "16-3-4=<<16-3-4=9>>9. then 9*2=<<9*2=18>>18. #### 18"})
    d = estimate_difficulty(item)
    assert d.method == "gsm8k_steps"
    assert d.score == 2.0  # two <<>> annotations


def test_difficulty_uses_math_level() -> None:
    item = BenchmarkItem("m", "q", answer="1", metadata={"level": "Level 4"})
    d = estimate_difficulty(item)
    assert d.method == "math_level"
    assert d.score == 4.0


def test_equal_frequency_bins() -> None:
    items = [BenchmarkItem(f"i{n}", "q", metadata={"level": f"Level {n}"}) for n in range(1, 7)]
    bins = assign_difficulty_bins(items, n_bins=3)
    assert set(bins.values()) == {0, 1, 2}
    assert bins["i1"] == 0 and bins["i6"] == 2  # easiest/hardest at the extremes


# -- variant validation --------------------------------------------------------


def test_validate_variants_flags_answer_change() -> None:
    original = BenchmarkItem("p", "orig", answer="7", parent_id="p", variant_type="original")
    good = BenchmarkItem("p__rephrasing", "reworded", answer="7", parent_id="p", variant_type="rephrasing")
    bad = BenchmarkItem("p__entity", "renamed", answer="9", parent_id="p", variant_type="entity_substitution")
    numeric = BenchmarkItem(
        "p__gsm_symbolic_0", "new nums", answer="42", parent_id="p", variant_type="gsm_symbolic",
        metadata={"answer_changing": True},
    )
    report = validate_variants([original, good, bad, numeric])
    assert report.n_answer_preserving == 2
    assert report.n_preserved_ok == 1  # only `good`
    assert "p__entity" in report.violations
    assert report.n_answer_changing == 1  # the numeric variant, not checked against original


# -- contamination threshold calibration ---------------------------------------


def test_calibrate_cosine_threshold_pr_curve() -> None:
    # Two truly-contaminated (high cosine) and two truly-clean (low cosine) items.
    hits = [
        ContaminationHit("a", "d", "s", "contaminated", 0.95, 0.9, 0.9, "", {"cosine": 0.95}),
        ContaminationHit("b", "d", "s", "contaminated", 0.92, 0.9, 0.9, "", {"cosine": 0.92}),
        ContaminationHit("c", "d", "s", "partial", 0.60, 0.3, 0.3, "", {"cosine": 0.60}),
    ]
    truth = {"a": True, "b": True, "c": False, "d": False}
    points = calibrate_cosine_threshold(hits, truth, thresholds=[0.9, 0.5])
    at_90 = next(p for p in points if p.threshold == 0.9)
    assert at_90.tp == 2 and at_90.fp == 0  # 0.9 cleanly separates
    assert at_90.precision == 1.0 and at_90.recall == 1.0
    at_50 = next(p for p in points if p.threshold == 0.5)
    assert at_50.fp == 1  # item c (0.60) now a false positive
    assert best_threshold(points).threshold == 0.9


# -- reliability metric construct validation -----------------------------------


def _cluster(parent: str, corrects: list[bool]) -> list[EvaluationRecord]:
    rows = []
    for i, ok in enumerate(corrects):
        vt = "original" if i == 0 else "entity_substitution"
        rows.append(EvaluationRecord(
            benchmark_id=f"{parent}_{i}", parent_id=parent, variant_type=vt,
            expected="1", predicted="1" if ok else "0",
            normalized_expected="1", normalized_predicted="1" if ok else "0",
            correct=ok, model="m", metadata={"skill": "arithmetic"}))
    return rows


def test_reliability_separates_genuine_from_fragile() -> None:
    # genuine: correct & consistent across variants; fragile: original right, variants wrong.
    scores = _cluster("genuine", [True, True, True, True]) + _cluster("fragile", [True, False, False, False])
    labels = {"genuine": "genuine", "fragile": "fragile"}
    result = validate_reliability_metric(scores, labels)
    assert result.separates is True
    assert result.separation > 0.3
    assert result.n_genuine == 1 and result.n_fragile == 1


def test_type_aware_consistency_excludes_numeric_variants() -> None:
    # A cluster where the model is perfectly correct: 1 original + 2 numeric (different answers).
    rows = [
        EvaluationRecord("p_0", "p", "original", "5", "5", "5", "5", True, "m", {}),
        EvaluationRecord("p_1", "p", "gsm_symbolic", "8", "8", "8", "8", True, "m", {"answer_changing": True}),
        EvaluationRecord("p_2", "p", "gsm_symbolic", "3", "3", "3", "3", True, "m", {"answer_changing": True}),
    ]
    [summary] = summarize_reliability(rows)
    # Consistency must NOT be penalized for the differing numeric answers.
    assert summary.answer_consistency == 1.0
    assert summary.accuracy == 1.0
    assert summary.n_answer_changing == 2
    assert summary.accuracy_under_perturbation == 1.0
