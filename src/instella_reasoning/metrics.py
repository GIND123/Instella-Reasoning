from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from instella_reasoning.records import EvaluationRecord

# Perturbations that *change* the correct answer (numeric resampling). For these,
# answer agreement is meaningless (a good reasoner *should* answer differently), so
# they are scored by correctness, not agreement. Answer-preserving perturbations
# (entity/reorder/distractor/rephrase) keep the gold answer and drive the consistency
# term. This split fixes the incoherence flagged in reviewer concern M5.
ANSWER_CHANGING_VARIANTS = frozenset({"gsm_symbolic", "numeric_perturbation"})


def is_answer_changing(record: EvaluationRecord) -> bool:
    if record.variant_type in ANSWER_CHANGING_VARIANTS:
        return True
    return bool(record.metadata.get("answer_changing"))


@dataclass(slots=True)
class ReliabilitySummary:
    parent_id: str
    n_variants: int
    accuracy: float
    answer_consistency: float
    reliability: float
    # GSM-Symbolic-style accuracy on answer-*changing* (numeric) variants; NaN-free
    # sentinel -1.0 means "no answer-changing variants in this cluster".
    accuracy_under_perturbation: float = -1.0
    n_answer_changing: int = 0


def cluster_consistency(records: list[EvaluationRecord]) -> float:
    """Modal-answer agreement over the answer-*preserving* records of a cluster.

    Answer-changing (numeric) variants are excluded because their gold answers differ
    by construction. Falls back to the whole cluster when every record is answer-changing.
    """
    preserving = [r for r in records if not is_answer_changing(r)]
    pool = preserving or records
    if not pool:
        return 0.0
    answer_counts = Counter(r.normalized_predicted for r in pool)
    return max(answer_counts.values()) / len(pool)


def summarize_reliability(records: list[EvaluationRecord]) -> list[ReliabilitySummary]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.parent_id].append(record)

    summaries: list[ReliabilitySummary] = []
    for parent_id, group in sorted(grouped.items()):
        accuracy = sum(record.correct for record in group) / len(group)
        consistency = cluster_consistency(group)
        reliability = accuracy * consistency

        changing = [r for r in group if is_answer_changing(r)]
        acc_under_perturbation = (
            sum(r.correct for r in changing) / len(changing) if changing else -1.0
        )

        summaries.append(
            ReliabilitySummary(
                parent_id=parent_id,
                n_variants=len(group),
                accuracy=round(accuracy, 6),
                answer_consistency=round(consistency, 6),
                reliability=round(reliability, 6),
                accuracy_under_perturbation=(
                    round(acc_under_perturbation, 6) if changing else -1.0
                ),
                n_answer_changing=len(changing),
            )
        )
    return summaries


def aggregate_accuracy(records: list[EvaluationRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.correct for record in records) / len(records)
