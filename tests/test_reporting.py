"""Report provenance + quality-gate sections (bf16-vs-4bit labeling, degeneracy warning)."""
from instella_reasoning.quality import assess_generations
from instella_reasoning.records import EvaluationRecord, GenerationRecord
from instella_reasoning.reporting import precision_provenance, quality_gate_section


def _score(model: str, precision: str) -> EvaluationRecord:
    return EvaluationRecord(
        benchmark_id="b",
        parent_id="b",
        variant_type="original",
        expected="1",
        predicted="1",
        normalized_expected="1",
        normalized_predicted="1",
        correct=True,
        model=model,
        metadata={"precision": precision},
    )


def test_provenance_lists_precision_per_model() -> None:
    lines = precision_provenance([_score("instella", "bf16")])
    text = "\n".join(lines)
    assert "instella" in text and "bf16" in text
    assert "Caution" not in text  # no 4-bit -> no warning


def test_provenance_warns_on_4bit() -> None:
    lines = precision_provenance([_score("instella", "nf4-4bit")])
    text = "\n".join(lines)
    assert "Caution" in text
    assert "bf16" in text  # tells the reader what the headline should be


def test_quality_section_warns_when_degenerate() -> None:
    report = assess_generations(
        [
            GenerationRecord("a", "3 * 4 = 12, so the answer is 12.\n#### 12"),
            GenerationRecord("b", "loop " * 40),
        ]
    )
    text = "\n".join(quality_gate_section(report))
    assert "Warning" in text
    assert "`b`" in text  # the flagged item is shown


def test_quality_section_empty_when_no_generations() -> None:
    report = assess_generations([])
    assert quality_gate_section(report) == []
