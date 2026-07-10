from __future__ import annotations

from collections import Counter
from pathlib import Path

from instella_reasoning.metrics import aggregate_accuracy, summarize_reliability
from instella_reasoning.records import ContaminationHit, EvaluationRecord


def write_markdown_report(
    scores: list[EvaluationRecord],
    contamination: list[ContaminationHit],
    output_path: str | Path,
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    contamination_by_label = Counter(hit.label for hit in contamination)
    reliability = summarize_reliability(scores)
    mean_reliability = sum(row.reliability for row in reliability) / len(reliability) if reliability else 0.0

    lines = [
        "# Instella Reasoning Reliability Report",
        "",
        "## Summary",
        "",
        f"- Scored generations: {len(scores)}",
        f"- Accuracy: {aggregate_accuracy(scores):.3f}",
        f"- Benchmark groups: {len(reliability)}",
        f"- Mean reliability: {mean_reliability:.3f}",
        f"- Contamination hits: {len(contamination)}",
        "",
        "## Contamination Labels",
        "",
    ]
    if contamination_by_label:
        for label, count in sorted(contamination_by_label.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none: 0")

    lines.extend(["", "## Lowest Reliability Groups", ""])
    for row in sorted(reliability, key=lambda item: item.reliability)[:20]:
        lines.append(
            f"- `{row.parent_id}`: reliability={row.reliability:.3f}, "
            f"accuracy={row.accuracy:.3f}, consistency={row.answer_consistency:.3f}, "
            f"variants={row.n_variants}"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(
        "Retrieval contamination labels are diagnostic candidates. Treat them as triage signals, "
        "not proof of memorization, until validated with manual inspection or gradient-based attribution."
    )

    content = "\n".join(lines) + "\n"
    output.write_text(content, encoding="utf-8")
    return content
