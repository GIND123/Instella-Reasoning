from instella_reasoning.metrics import summarize_reliability
from instella_reasoning.records import EvaluationRecord


def test_summarize_reliability_groups_variants() -> None:
    records = [
        EvaluationRecord("a", "p", "original", "1", "1", "1", "1", True),
        EvaluationRecord("b", "p", "rename", "1", "1", "1", "1", True),
        EvaluationRecord("c", "p", "distractor", "1", "2", "1", "2", False),
    ]

    summary = summarize_reliability(records)

    assert len(summary) == 1
    assert summary[0].accuracy == 0.666667
    assert summary[0].answer_consistency == 0.666667
    assert summary[0].reliability == 0.444444
