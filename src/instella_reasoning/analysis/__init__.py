"""Research analyses: contamination-aware accuracy, reliability, stats, plots."""

from instella_reasoning.analysis.accuracy_gap import (
    AccuracyGapResult,
    contamination_labels_by_benchmark_id,
    two_proportion_z_test,
)

__all__ = [
    "AccuracyGapResult",
    "contamination_labels_by_benchmark_id",
    "two_proportion_z_test",
]
