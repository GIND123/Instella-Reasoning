from __future__ import annotations

from collections import Counter
from pathlib import Path

from instella_reasoning.metrics import aggregate_accuracy, summarize_reliability
from instella_reasoning.records import ContaminationHit, EvaluationRecord

_CONTAMINATION_CANON = {
    "contaminated": "contaminated",
    "exact": "contaminated",
    "near_duplicate": "contaminated",
    "partial": "partial",
    "paraphrase_candidate": "partial",
}


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


def precision_provenance(scores: list[EvaluationRecord]) -> list[str]:
    """Report which precision each model's numbers came from, warning on 4-bit.

    bf16 is the accurate headline precision; 4-bit NF4 is a fast, lower-fidelity pass.
    Mixing them silently would let quantization noise masquerade as a reasoning result,
    so the report states the provenance and flags any 4-bit numbers explicitly.
    """
    if not scores:
        return []
    by_model: dict[str, set[str]] = {}
    for record in scores:
        precision = str(record.metadata.get("precision", "unknown"))
        by_model.setdefault(record.model, set()).add(precision)

    lines = ["### Generation provenance", ""]
    any_4bit = False
    for model, precisions in sorted(by_model.items()):
        rendered = ", ".join(sorted(precisions))
        lines.append(f"- `{model}` — precision: {rendered}")
        any_4bit = any_4bit or any("4bit" in p for p in precisions)
    if any_4bit:
        lines.append("")
        lines.append(
            "> **Caution:** some numbers above come from a 4-bit (NF4) run. Treat these as a "
            "fast, lower-fidelity pass — report **bf16** results as the headline, since NF4 "
            "quantization measurably shifts accuracy."
        )
    lines.append("")
    return lines


def run_provenance_section(manifest: dict | None) -> list[str]:
    """Render the reproducibility manifest (git commit + key package versions)."""
    if not manifest:
        return []
    git = manifest.get("git", {}) or {}
    commit = git.get("commit")
    commit_short = commit[:10] if commit else "unknown"
    dirty = git.get("dirty")
    dirty_note = " (dirty tree)" if dirty else ""
    packages = manifest.get("packages", {}) or {}
    key = [f"{name} {packages[name]}" for name in ("transformers", "torch") if packages.get(name)]
    lines = [
        "### Reproducibility",
        "",
        f"- Git commit: `{commit_short}`{dirty_note}",
        f"- Generated: {manifest.get('generated_at_utc', 'unknown')}",
    ]
    if key:
        lines.append(f"- Key packages: {', '.join(key)}")
    lines.append("- Full manifest: `manifest.json`")
    lines.append("")
    return lines


def quality_gate_section(quality_report) -> list[str]:
    """Render the degeneracy quality gate (see :mod:`instella_reasoning.quality`)."""
    if quality_report is None or quality_report.n == 0:
        return []
    lines = [
        "### Quality gate (degenerate completions)",
        "",
        f"- Completions checked: **{quality_report.n}**",
        f"- Flagged degenerate: **{quality_report.n_degenerate}** "
        f"({quality_report.degenerate_fraction:.1%})",
    ]
    if quality_report.n_degenerate:
        lines.append(
            "> **Warning:** degenerate (looping/empty) completions detected — usually a "
            "prompt/format problem (e.g. an Instruct model without its chat template). Their "
            "scores are unreliable; fix generation before trusting the Atlas. Examples:"
        )
        for row in quality_report.flagged[:5]:
            lines.append(
                f">   - `{row.benchmark_id}`: {', '.join(row.reasons)} "
                f"(distinct-ratio={row.distinct_token_ratio:.2f}, tokens={row.n_tokens})"
            )
    lines.append("")
    return lines


def write_full_report(
    scores: list[EvaluationRecord],
    contamination: list[ContaminationHit],
    output_path: str | Path,
    atlas_report=None,
    gap_results=None,
    skill_attributions=None,
    emergence_report=None,
    figure_paths: dict[str, str] | None = None,
    quality_report=None,
    stratified_gap_results=None,
    manifest: dict | None = None,
) -> str:
    """Assemble the full reliability report: summary, atlas, accuracy gap, attribution.

    Every analysis section is optional so the report degrades gracefully when a stage
    was skipped (e.g. no model was run, so there are no scores/gap). Renders Markdown
    that embeds any generated figures by relative path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Instella Reasoning Reliability Report",
        "",
        "> When Instella answers a reasoning problem correctly, is it **reasoning** or "
        "**remembering**? This report combines contamination search, the Reliability "
        "metric (accuracy x consistency), and training-data attribution to tell them apart.",
        "",
        "## 1. Summary",
        "",
        f"- Scored generations: **{len(scores)}**",
        f"- Overall accuracy: **{aggregate_accuracy(scores):.3f}**",
        f"- Contamination hits: **{len(contamination)}**",
    ]
    label_counts = Counter(_CONTAMINATION_CANON.get(hit.label, "clean") for hit in contamination)
    if label_counts:
        breakdown = ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items()))
        lines.append(f"- Contamination breakdown: {breakdown}")
    lines.append("")

    lines.extend(precision_provenance(scores))
    lines.extend(quality_gate_section(quality_report))
    lines.extend(run_provenance_section(manifest))

    if atlas_report is not None and atlas_report.cells:
        lines.append(atlas_report.to_markdown())
        lines.append(_atlas_takeaway(atlas_report))

    if gap_results:
        lines.extend(_gap_section(gap_results))
    if stratified_gap_results:
        lines.extend(_stratified_gap_section(stratified_gap_results))

    if skill_attributions:
        lines.extend(_attribution_section(skill_attributions))

    if emergence_report is not None and (emergence_report.transitions or emergence_report.rl_effects):
        lines.extend(_emergence_section(emergence_report))

    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for name, path in figure_paths.items():
            lines.append(f"### {name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"![{name}]({path})")
            lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Contamination labels are diagnostic candidates from retrieval + n-gram overlap; "
            "confirm with manual inspection or gradient attribution before publication.",
            "- **\"Clean\" = not detected in the indexed sample**, a lower bound on contamination; "
            "do not treat it as a verified-uncontaminated control.",
            "- Prefer the **difficulty-adjusted** gap over the naive one — an unadjusted gap can be a "
            "difficulty confound (contaminated items may simply be easier).",
            "- A **fragile** cell (high accuracy, low reliability) is the memorisation signature: "
            "the model recognises the original but fails semantically equivalent variants.",
            "- A **concentrated** attribution signature (near-duplicate share >= 0.30) supports "
            "memorisation; a **diverse** signature supports generalised learning.",
            "",
        ]
    )
    content = "\n".join(lines) + "\n"
    output.write_text(content, encoding="utf-8")
    return content


def _atlas_takeaway(atlas_report) -> str:
    all_cells = [c for c in atlas_report.cells if c.contamination_level == "all"]
    fragile = [c.skill for c in all_cells if c.classification == "fragile"]
    genuine = [c.skill for c in all_cells if c.classification == "genuine"]
    parts = []
    if genuine:
        parts.append(f"Genuine reasoning: {', '.join(genuine)}.")
    if fragile:
        parts.append(f"Fragile / memorisation-like: {', '.join(fragile)}.")
    return ("**Atlas takeaway** — " + " ".join(parts) + "\n") if parts else ""


def _gap_section(gap_results) -> list[str]:
    lines = [
        "## 2. Contaminated vs Not-Detected Accuracy Gap",
        "",
        "> **\"Clean\" means \"not detected\", not \"provably absent\"** — contamination is a "
        "one-sided lower bound because only a sample of the web corpora is indexed. Read the "
        "gap as *at least this large*, and prefer the difficulty-adjusted estimate below.",
        "",
        "| Scope | Contaminated | Not-detected | Gap | z | p | q (FDR) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in gap_results:
        contaminated = result.by_label["contaminated"]
        clean = result.by_label["clean"]
        sig = " *" if result.q_value < 0.05 else ""
        lines.append(
            f"| {result.scope} | {contaminated.accuracy:.3f} (n={contaminated.total}) | "
            f"{clean.accuracy:.3f} (n={clean.total}) | {result.gap:+.3f} | "
            f"{result.z:.2f} | {result.p_value:.4f} | {result.q_value:.4f}{sig} |"
        )
    lines.extend(["", "`*` q < 0.05 (two-proportion z-test, Benjamini-Hochberg FDR across scopes).", ""])
    return lines


def _stratified_gap_section(stratified_results) -> list[str]:
    lines = [
        "### 2b. Difficulty-adjusted gap (confounder control)",
        "",
        "> Mantel-Haenszel pooling of the gap *within* equal-frequency difficulty strata, "
        "with a cluster-robust bootstrap CI (resampling parent clusters). If the adjusted gap "
        "collapses toward 0, the naive gap was largely a difficulty confound, not contamination.",
        "",
        "| Scope | Unadjusted gap | Difficulty-adjusted gap | 95% CI (cluster bootstrap) | Clusters |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in stratified_results:
        crosses_zero = r.ci_low <= 0.0 <= r.ci_high
        flag = "" if not crosses_zero else " (CI incl. 0)"
        lines.append(
            f"| {r.scope} | {r.unadjusted_gap:+.3f} | {r.pooled_gap:+.3f} | "
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]{flag} | {r.n_clusters} |"
        )
    lines.append("")
    return lines


def _attribution_section(skill_attributions) -> list[str]:
    lines = [
        "## 3. Training-Data Attribution (Tier 1: embedding)",
        "",
        "| Sub-skill | Items | Near-dup share | Gini | Concentrated | Top sources |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for agg in skill_attributions:
        top = ", ".join(
            f"{source} ({share:.0%})" for source, share in list(agg.source_shares.items())[:3]
        )
        lines.append(
            f"| {agg.skill} | {agg.n_items} | {agg.mean_near_duplicate_share:.3f} | "
            f"{agg.mean_gini:.3f} | {agg.concentrated_share:.0%} | {top} |"
        )
    lines.append("")
    return lines


def _emergence_section(emergence_report) -> list[str]:
    lines = ["## 4. Scale & Post-training Emergence", ""]
    if emergence_report.transitions:
        lines.extend(
            ["| Sub-skill | Small | Large | Delta | Class |", "|---|---:|---:|---:|---|"]
        )
        for t in emergence_report.transitions:
            lines.append(
                f"| {t.skill} | {t.small_reliability:.3f} | {t.large_reliability:.3f} | "
                f"{t.delta:+.3f} | {t.transition_class} |"
            )
        lines.append("")
    if emergence_report.rl_effects:
        lines.extend(
            [
                "**RL post-training effect** (consistency gain = genuine, accuracy-only = pattern matching):",
                "",
                "| Sub-skill | dAccuracy | dConsistency | dReliability | Verdict |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for e in emergence_report.rl_effects:
            lines.append(
                f"| {e.skill} | {e.delta_accuracy:+.3f} | {e.delta_consistency:+.3f} | "
                f"{e.delta_reliability:+.3f} | {e.verdict} |"
            )
        lines.append("")
    if emergence_report.rl_generalization:
        gen = emergence_report.rl_generalization
        lines.append(
            f"RL generalisation: math gain {gen.get('math_reliability_gain', 0):+.3f}, "
            f"non-math gain {gen.get('non_math_reliability_gain', 0):+.3f} "
            f"-> {gen.get('verdict', 'n/a')}."
        )
        lines.append("")
    return lines
