from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from instella_reasoning.records import EvaluationRecord


@dataclass(slots=True)
class ReliabilitySummary:
    parent_id: str
    n_variants: int
    accuracy: float
    answer_consistency: float
    reliability: float


def summarize_reliability(records: list[EvaluationRecord]) -> list[ReliabilitySummary]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.parent_id].append(record)

    summaries: list[ReliabilitySummary] = []
    for parent_id, group in sorted(grouped.items()):
        accuracy = sum(record.correct for record in group) / len(group)
        answer_counts = Counter(record.normalized_predicted for record in group)
        consistency = max(answer_counts.values()) / len(group)
        reliability = accuracy * consistency
        summaries.append(
            ReliabilitySummary(
                parent_id=parent_id,
                n_variants=len(group),
                accuracy=round(accuracy, 6),
                answer_consistency=round(consistency, 6),
                reliability=round(reliability, 6),
            )
        )
    return summaries


def aggregate_accuracy(records: list[EvaluationRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.correct for record in records) / len(records)
