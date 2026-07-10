from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instella_reasoning.records import ContaminationHit, EvaluationRecord, read_jsonl, write_jsonl


@dataclass(slots=True)
class AttributionRecord:
    benchmark_id: str
    document_id: str
    source: str
    attribution_method: str
    influence_proxy: float
    contamination_label: str
    correct: bool | None
    metadata: dict[str, object]


def build_retrieval_attribution(
    contamination_hits: list[ContaminationHit],
    evaluations: list[EvaluationRecord],
    output_path: str | Path,
) -> list[AttributionRecord]:
    """Join retrieval hits with correctness as a cheap attribution proxy.

    This is not a causal influence function. It is a triage layer that identifies
    examples worth running with TracIn/TRAK or gradient-based methods later.
    """

    eval_by_id = {record.benchmark_id: record for record in evaluations}
    rows: list[AttributionRecord] = []
    for hit in contamination_hits:
        evaluation = eval_by_id.get(hit.benchmark_id)
        correctness_bonus = 1.0 if evaluation and evaluation.correct else 0.0
        influence_proxy = (0.8 * hit.score) + (0.2 * correctness_bonus)
        rows.append(
            AttributionRecord(
                benchmark_id=hit.benchmark_id,
                document_id=hit.document_id,
                source=hit.source,
                attribution_method="retrieval_correctness_proxy",
                influence_proxy=round(influence_proxy, 6),
                contamination_label=hit.label,
                correct=None if evaluation is None else evaluation.correct,
                metadata={"hit_metadata": hit.metadata},
            )
        )
    write_jsonl(output_path, rows)
    return rows


def load_contamination_hits(path: str | Path) -> list[ContaminationHit]:
    return [ContaminationHit(**row) for row in read_jsonl(path)]


def load_evaluations(path: str | Path) -> list[EvaluationRecord]:
    return [EvaluationRecord(**row) for row in read_jsonl(path)]
