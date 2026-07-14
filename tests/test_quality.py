"""Degeneracy quality gate — the guard that keeps looping/empty output out of the Atlas."""
from instella_reasoning.quality import (
    QualityThresholds,
    assess_completion,
    assess_generations,
)
from instella_reasoning.records import GenerationRecord


def test_coherent_answer_is_not_flagged() -> None:
    completion = (
        "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = 18 dollars "
        "every day at the market.\n#### 18"
    )
    quality = assess_completion("gsm8k_0", completion)
    assert quality.degenerate is False
    assert quality.reasons == []


def test_short_numeric_answer_is_not_flagged() -> None:
    # Short, high-distinct answers must never trip the repetition heuristics.
    assert assess_completion("gsm8k_1", "#### 42").degenerate is False


def test_looping_completion_is_flagged() -> None:
    # The exact failure mode from an Instruct model with no chat template.
    completion = "case? " * 40
    quality = assess_completion("gsm8k_2", completion)
    assert quality.degenerate is True
    assert "long_token_run" in quality.reasons or "repeated_trigram" in quality.reasons


def test_empty_completion_is_flagged() -> None:
    quality = assess_completion("gsm8k_3", "   ")
    assert quality.degenerate is True
    assert quality.reasons == ["empty"]


def test_assess_generations_reports_fraction() -> None:
    generations = [
        GenerationRecord("a", "The answer is 12 because 3 * 4 = 12.\n#### 12"),
        GenerationRecord("b", "word " * 50),  # degenerate
        GenerationRecord("c", ""),  # degenerate (empty)
    ]
    report = assess_generations(generations)
    assert report.n == 3
    assert report.n_degenerate == 2
    assert abs(report.degenerate_fraction - 2 / 3) < 1e-9
    assert {row.benchmark_id for row in report.flagged} == {"b", "c"}


def test_thresholds_are_tunable() -> None:
    text = "a b c d e f g h i j " * 3  # 30 tokens, distinct ratio 10/30 ~= 0.33
    strict = QualityThresholds(distinct_ratio_threshold=0.5)
    lax = QualityThresholds(distinct_ratio_threshold=0.1)
    assert assess_completion("x", text, strict).degenerate is True
    assert assess_completion("x", text, lax).degenerate is False
