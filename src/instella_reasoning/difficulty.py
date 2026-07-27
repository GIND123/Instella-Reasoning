"""Per-item difficulty estimation for confounder control (reviewer concern M3).

The headline contaminated-vs-clean accuracy gap is confounded if contaminated items
are systematically easier than clean ones (Ivanova et al.'s critique of GSM-Symbolic).
To control for this we need a difficulty score per benchmark item that does *not*
depend on the model under test. This module derives one from structural signals:

- **GSM8K**: number of ``<<a op b=c>>`` calculator annotations in the reference
  rationale = the number of reasoning steps (the strongest available proxy).
- **MATH**: the dataset's own ``Level 1..5`` label.
- **Fallback**: a length/quantity proxy (token count + count of numbers in the prompt),
  z-less and monotone so it only needs to *rank* items within a benchmark.

Downstream, :func:`assign_difficulty_bins` produces equal-frequency strata so the gap
can be pooled *within* difficulty (Mantel-Haenszel style), removing the confound.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

from instella_reasoning.records import BenchmarkItem

_CALC_ANNOTATION = re.compile(r"<<[^=<>]+=[^<>]+>>")
_MATH_LEVEL = re.compile(r"level\s*(\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(slots=True)
class Difficulty:
    benchmark_id: str
    score: float  # higher = harder; only meaningful for *ranking* within a benchmark
    method: str  # "gsm8k_steps" | "math_level" | "length_quantity"

    def to_dict(self) -> dict:
        return {"benchmark_id": self.benchmark_id, "score": round(self.score, 6), "method": self.method}


def estimate_difficulty(item: BenchmarkItem) -> Difficulty:
    """Estimate a model-independent difficulty score for one benchmark item."""
    rationale = str(item.metadata.get("rationale", ""))
    steps = len(_CALC_ANNOTATION.findall(rationale))
    if steps > 0:
        return Difficulty(item.id, float(steps), "gsm8k_steps")

    level = item.metadata.get("level")
    if level is not None:
        match = _MATH_LEVEL.search(str(level))
        if match:
            return Difficulty(item.id, float(match.group(1)), "math_level")
        if isinstance(level, (int, float)):
            return Difficulty(item.id, float(level), "math_level")

    # Fallback: longer prompts with more quantities are harder, on average.
    n_tokens = len(item.prompt.split())
    n_numbers = len(_NUMBER.findall(item.prompt))
    return Difficulty(item.id, float(n_tokens) + 5.0 * float(n_numbers), "length_quantity")


def estimate_all(items: list[BenchmarkItem]) -> dict[str, Difficulty]:
    return {item.id: estimate_difficulty(item) for item in items}


def assign_difficulty_bins(
    items: list[BenchmarkItem], n_bins: int = 3
) -> dict[str, int]:
    """Assign each item to an equal-frequency difficulty bin (0 = easiest).

    Equal-frequency (quantile) binning keeps strata balanced regardless of the raw
    score distribution, which matters because GSM8K step counts are discrete and
    right-skewed. Ties in score are kept in the same conceptual difficulty but may
    land in adjacent bins; that is acceptable for stratified pooling.
    """
    if not items:
        return {}
    # Ties must NOT be broken by input order. GSM8K step counts are discrete, so most
    # items tie; ranking by list position then hands the first-listed group the low bins
    # and the second group the high bins. When the two groups are the seen and unseen
    # arms, that manufactures a difficulty imbalance out of nothing and confounds the
    # headline contrast. Hashing the id breaks ties deterministically but independently of
    # the order items were passed in, so tied items spread evenly across bins.
    def _tiebreak(benchmark_id: str) -> int:
        return zlib.crc32(benchmark_id.encode("utf-8"))

    scored = sorted(
        estimate_all(items).values(),
        key=lambda d: (d.score, _tiebreak(d.benchmark_id)),
    )
    n = len(scored)
    n_bins = max(1, min(n_bins, n))
    bins: dict[str, int] = {}
    for rank, difficulty in enumerate(scored):
        # rank in [0, n) -> bin in [0, n_bins)
        bins[difficulty.benchmark_id] = min(n_bins - 1, rank * n_bins // n)
    return bins
