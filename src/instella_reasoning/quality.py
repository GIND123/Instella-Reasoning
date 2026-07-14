"""Generation quality gate — catch degenerate completions before they reach the Atlas.

A misconfigured prompt/format (e.g. an instruction model fed a raw string without its
chat template) produces *degenerate* text: endless repetition of a token or phrase.
Such output scores as "wrong" and silently drags down accuracy/reliability, so a
memorisation study can misread a formatting bug as a reasoning failure. This module
flags those completions with three cheap, dependency-free signals:

- **distinct-token ratio** — unique / total tokens; collapses toward 0 on loops.
- **top-trigram fraction** — share of the single most frequent word trigram.
- **longest token run** — longest streak of one token repeated back-to-back.

Everything here is a pure function of the text (no torch, no numpy), so it runs in CI
and in the pipeline. Thresholds are conservative: a short, correct answer like ``18``
is never flagged; only long, looping, or empty completions are.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from instella_reasoning.records import GenerationRecord, write_jsonl


@dataclass(slots=True)
class QualityThresholds:
    """Tunables for :func:`assess_completion`. Defaults chosen to avoid false positives."""

    min_tokens_for_repetition: int = 20  # ratio checks only apply to longer text
    distinct_ratio_threshold: float = 0.35
    trigram_fraction_threshold: float = 0.25
    run_threshold: int = 12


@dataclass(slots=True)
class CompletionQuality:
    benchmark_id: str
    n_tokens: int
    distinct_token_ratio: float
    top_trigram_fraction: float
    longest_token_run: int
    degenerate: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "n_tokens": self.n_tokens,
            "distinct_token_ratio": round(self.distinct_token_ratio, 6),
            "top_trigram_fraction": round(self.top_trigram_fraction, 6),
            "longest_token_run": self.longest_token_run,
            "degenerate": self.degenerate,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class QualityReport:
    n: int
    n_degenerate: int
    degenerate_fraction: float
    flagged: list[CompletionQuality] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_degenerate": self.n_degenerate,
            "degenerate_fraction": round(self.degenerate_fraction, 6),
            "flagged": [row.to_dict() for row in self.flagged],
        }


def _longest_token_run(tokens: list[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def _top_ngram_fraction(tokens: list[str], n: int = 3) -> float:
    if len(tokens) < n:
        return 0.0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    total = sum(grams.values())
    return max(grams.values()) / total if total else 0.0


def assess_completion(
    benchmark_id: str, completion: str, thresholds: QualityThresholds | None = None
) -> CompletionQuality:
    """Score one completion and decide whether it is degenerate."""
    thresholds = thresholds or QualityThresholds()
    tokens = completion.split()
    n = len(tokens)
    reasons: list[str] = []

    if n == 0:
        return CompletionQuality(benchmark_id, 0, 0.0, 0.0, 0, True, ["empty"])

    distinct_ratio = len(set(tokens)) / n
    top_trigram = _top_ngram_fraction(tokens, 3)
    longest_run = _longest_token_run(tokens)

    if n >= thresholds.min_tokens_for_repetition:
        if distinct_ratio < thresholds.distinct_ratio_threshold:
            reasons.append("low_distinct_token_ratio")
        if top_trigram > thresholds.trigram_fraction_threshold:
            reasons.append("repeated_trigram")
    if longest_run >= thresholds.run_threshold:
        reasons.append("long_token_run")

    return CompletionQuality(
        benchmark_id=benchmark_id,
        n_tokens=n,
        distinct_token_ratio=distinct_ratio,
        top_trigram_fraction=top_trigram,
        longest_token_run=longest_run,
        degenerate=bool(reasons),
        reasons=reasons,
    )


def assess_generations(
    generations: list[GenerationRecord], thresholds: QualityThresholds | None = None
) -> QualityReport:
    """Assess a batch of generations, returning a report with the flagged (degenerate) ones."""
    thresholds = thresholds or QualityThresholds()
    flagged: list[CompletionQuality] = []
    for generation in generations:
        quality = assess_completion(generation.benchmark_id, generation.completion, thresholds)
        if quality.degenerate:
            flagged.append(quality)
    n = len(generations)
    return QualityReport(
        n=n,
        n_degenerate=len(flagged),
        degenerate_fraction=len(flagged) / n if n else 0.0,
        flagged=flagged,
    )


def write_quality_report(
    generations: list[GenerationRecord],
    output_path: str | Path | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Assess generations and, when ``output_path`` is given, write the flagged rows to JSONL."""
    report = assess_generations(generations, thresholds)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, [row.to_dict() for row in report.flagged])
    return report
