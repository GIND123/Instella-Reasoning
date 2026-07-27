from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from instella_reasoning.records import EvaluationRecord

# Perturbations that *change* the correct answer (numeric resampling). For these,
# answer agreement is meaningless (a good reasoner *should* answer differently), so
# they are scored by correctness, not agreement. Answer-preserving perturbations
# (entity/reorder/distractor/rephrase) keep the gold answer and drive the consistency
# term. This split fixes the incoherence flagged in reviewer concern M5.
ANSWER_CHANGING_VARIANTS = frozenset(
    {
        "gsm_symbolic",  # auto-derived numeric resampling (instella_reasoning.gsm_symbolic)
        "gsm_symbolic_official",  # Apple's hand-written GSM-Symbolic instances
        "numeric_perturbation",
    }
)


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
    # Secondary metric: agreement of *correctness labels* with the original item, the
    # definition used in the project proposal. Reported alongside the primary
    # modal-answer share because the two can disagree and the difference is diagnostic
    # (see `label_agreement_consistency`).
    label_agreement: float = -1.0
    # Consistency restricted to `resample` copies: pure decoding noise, no perturbation.
    # This is the null against which the perturbation-driven consistency drop is judged.
    decoding_consistency: float = -1.0
    n_resample: int = 0


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


def label_agreement_consistency(records: list[EvaluationRecord]) -> float:
    """Fraction of answer-preserving variants whose *correctness label* matches the original.

    This is the definition stated in the project proposal, kept as a reported secondary so
    the paper's metric section can show both. It differs from
    :func:`cluster_consistency` in a way that matters: a model that is confidently and
    identically **wrong** across every variant scores 1.0 here but is also 1.0 on modal
    agreement, whereas a model that is wrong in a *different* way each time scores 1.0
    here and low on modal agreement. Modal agreement is therefore the stricter
    memorisation probe and stays primary; the divergence between the two is itself a
    diagnostic (high label-agreement with low modal-agreement = unstable guessing).
    """
    preserving = [r for r in records if not is_answer_changing(r)]
    originals = [r for r in preserving if r.variant_type == "original"]
    others = [r for r in preserving if r.variant_type != "original"]
    if not originals or not others:
        return -1.0
    reference = originals[0].correct
    return sum(1 for r in others if r.correct == reference) / len(others)


def decoding_noise_consistency(records: list[EvaluationRecord]) -> tuple[float, int]:
    """Modal-answer agreement over ``resample`` copies only — the sampling-noise null."""
    copies = [r for r in records if r.variant_type == "resample"]
    if len(copies) < 2:
        return -1.0, len(copies)
    counts = Counter(r.normalized_predicted for r in copies)
    return max(counts.values()) / len(copies), len(copies)


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
        decoding, n_resample = decoding_noise_consistency(group)

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
                label_agreement=round(label_agreement_consistency(group), 6),
                decoding_consistency=round(decoding, 6) if decoding >= 0 else -1.0,
                n_resample=n_resample,
            )
        )
    return summaries


def aggregate_accuracy(records: list[EvaluationRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.correct for record in records) / len(records)
