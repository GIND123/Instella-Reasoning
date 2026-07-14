"""The Reasoning Reliability Atlas — the study's central deliverable (Phase 3/6).

Given scored generations (with ``parent_id`` linking a cluster of variants and a
``metadata.skill`` tag) and contamination labels, this builds the per-sub-skill map
of ``accuracy``, ``consistency``, and ``reliability = accuracy x consistency``,
broken down by contamination level, and classifies each cell:

- ``genuine``  — reliability >= 0.50: the model understands the problem structure.
- ``partial``  — 0.30 <= reliability < 0.50.
- ``fragile``  — reliability < 0.30 *with* high accuracy (>= 0.60): recognises the
  original but fails semantically equivalent variants (the memorisation signature).
- ``gap``      — low accuracy and low reliability: the skill is largely absent.

The per-cluster reliability is computed exactly as in
:func:`instella_reasoning.metrics.summarize_reliability` (accuracy across a cluster
times the fraction of the cluster that shares the modal answer), then averaged
within each ``(skill, contamination_level)`` cell.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from instella_reasoning.analysis.accuracy_gap import contamination_labels_by_benchmark_id
from instella_reasoning.metrics import cluster_consistency
from instella_reasoning.records import ContaminationHit, EvaluationRecord

CONTAMINATION_LEVELS = ("all", "contaminated", "partial", "clean")

GENUINE_THRESHOLD = 0.50
PARTIAL_THRESHOLD = 0.30
FRAGILE_ACCURACY = 0.60


@dataclass(slots=True)
class ClusterReliability:
    parent_id: str
    skill: str
    contamination_level: str
    model: str
    n_variants: int
    accuracy: float
    consistency: float
    reliability: float


@dataclass(slots=True)
class AtlasCell:
    skill: str
    contamination_level: str
    n_clusters: int
    accuracy: float
    consistency: float
    reliability: float
    classification: str

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "contamination_level": self.contamination_level,
            "n_clusters": self.n_clusters,
            "accuracy": round(self.accuracy, 6),
            "consistency": round(self.consistency, 6),
            "reliability": round(self.reliability, 6),
            "classification": self.classification,
        }


@dataclass(slots=True)
class AtlasReport:
    cells: list[AtlasCell] = field(default_factory=list)
    clusters: list[ClusterReliability] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"cells": [cell.to_dict() for cell in self.cells]}

    def cell(self, skill: str, level: str) -> AtlasCell | None:
        for cell in self.cells:
            if cell.skill == skill and cell.contamination_level == level:
                return cell
        return None

    def to_markdown(self) -> str:
        return render_atlas_markdown(self)


def classify_cell(accuracy: float, reliability: float) -> str:
    if accuracy >= FRAGILE_ACCURACY and reliability < PARTIAL_THRESHOLD:
        return "fragile"
    if reliability >= GENUINE_THRESHOLD:
        return "genuine"
    if reliability >= PARTIAL_THRESHOLD:
        return "partial"
    return "gap"


def _cluster_reliabilities(
    scores: list[EvaluationRecord],
    labels: dict[str, str],
    skill_key: str,
) -> list[ClusterReliability]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in scores:
        grouped[record.parent_id].append(record)

    clusters: list[ClusterReliability] = []
    for parent_id, group in grouped.items():
        accuracy = sum(record.correct for record in group) / len(group)
        # Consistency ignores answer-*changing* numeric variants (see metrics.py / M5).
        consistency = cluster_consistency(group)
        reliability = accuracy * consistency
        skill = _modal_metadata(group, skill_key, default="unknown")
        model = group[0].model
        # A cluster inherits the contamination label of its parent/original item.
        level = labels.get(parent_id, "clean")
        clusters.append(
            ClusterReliability(
                parent_id=parent_id,
                skill=skill,
                contamination_level=level,
                model=model,
                n_variants=len(group),
                accuracy=accuracy,
                consistency=consistency,
                reliability=reliability,
            )
        )
    return clusters


def build_atlas(
    scores: list[EvaluationRecord],
    contamination: list[ContaminationHit],
    skill_key: str = "skill",
) -> AtlasReport:
    """Build the Reliability Atlas from scored generations and contamination hits."""
    labels = contamination_labels_by_benchmark_id(contamination)
    clusters = _cluster_reliabilities(scores, labels, skill_key)

    skills = sorted({cluster.skill for cluster in clusters})
    cells: list[AtlasCell] = []
    for skill in skills:
        for level in CONTAMINATION_LEVELS:
            subset = [
                cluster
                for cluster in clusters
                if cluster.skill == skill
                and (level == "all" or cluster.contamination_level == level)
            ]
            if not subset:
                continue
            cells.append(_aggregate_cell(skill, level, subset))
    return AtlasReport(cells=cells, clusters=clusters)


def _aggregate_cell(skill: str, level: str, subset: list[ClusterReliability]) -> AtlasCell:
    n = len(subset)
    accuracy = sum(c.accuracy for c in subset) / n
    consistency = sum(c.consistency for c in subset) / n
    reliability = sum(c.reliability for c in subset) / n
    return AtlasCell(
        skill=skill,
        contamination_level=level,
        n_clusters=n,
        accuracy=accuracy,
        consistency=consistency,
        reliability=reliability,
        classification=classify_cell(accuracy, reliability),
    )


def _modal_metadata(group: list[EvaluationRecord], key: str, default: str) -> str:
    values = [str(record.metadata.get(key)) for record in group if record.metadata.get(key)]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


_CLASSIFICATION_MARK = {
    "genuine": "GENUINE",
    "partial": "PARTIAL",
    "fragile": "FRAGILE",
    "gap": "GAP",
}


def render_atlas_markdown(report: AtlasReport) -> str:
    lines = [
        "## Reasoning Reliability Atlas",
        "",
        "Reliability = Accuracy x Consistency, per reasoning sub-skill and contamination level.",
        "",
        "| Sub-skill | Level | Clusters | Accuracy | Consistency | Reliability | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in report.cells:
        indent = "" if cell.contamination_level == "all" else "&nbsp;&nbsp;"
        lines.append(
            f"| {indent}{cell.skill} | {cell.contamination_level} | {cell.n_clusters} | "
            f"{cell.accuracy:.3f} | {cell.consistency:.3f} | {cell.reliability:.3f} | "
            f"{_CLASSIFICATION_MARK.get(cell.classification, cell.classification)} |"
        )
    lines.extend(
        [
            "",
            "**Legend** — GENUINE (R>=0.50), PARTIAL (0.30-0.50), "
            "FRAGILE (R<0.30 with high accuracy = memorisation signature), GAP (skill absent).",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
