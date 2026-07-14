"""MATH full-subject loading — interleave all seven subjects with unique, balanced ids."""
from instella_reasoning.datasets.loaders import (
    BENCHMARK_LOADERS,
    MATH_SUBJECTS,
    _load_math_all_subjects,
)


def _fake_load_dataset_factory():
    """Return a load_dataset stub yielding 3 rows per subject, tagged by subject."""

    def _fake(path, subject, split=None):
        return [
            {
                "problem": f"{subject} problem {i}",
                "solution": f"...\\boxed{{{subject[:3]}{i}}}",
                "level": "Level 1",
                "type": subject,
            }
            for i in range(3)
        ]

    return _fake


def test_all_subjects_interleaved_with_unique_ids() -> None:
    spec = BENCHMARK_LOADERS["math"]
    rows = _load_math_all_subjects(_fake_load_dataset_factory(), spec, split=None, limit=None)

    assert len(rows) == 3 * len(MATH_SUBJECTS)
    ids = [row.id for row in rows]
    assert len(set(ids)) == len(ids)  # unique
    assert ids == [f"math_{i:05d}" for i in range(len(rows))]  # sequential
    assert all(row.parent_id == row.id for row in rows)
    # Interleaved: the first N rows are one-per-subject, not all of subject 0.
    first_types = [row.metadata["type"] for row in rows[: len(MATH_SUBJECTS)]]
    assert set(first_types) == set(MATH_SUBJECTS)


def test_limit_is_balanced_across_subjects() -> None:
    spec = BENCHMARK_LOADERS["math"]
    rows = _load_math_all_subjects(_fake_load_dataset_factory(), spec, split=None, limit=7)
    assert len(rows) == 7
    # Round-robin means the 7-item cap draws one from each subject before a second pass.
    assert set(row.metadata["type"] for row in rows) == set(MATH_SUBJECTS)


def test_failed_subject_is_skipped_not_fatal() -> None:
    spec = BENCHMARK_LOADERS["math"]

    def _flaky(path, subject, split=None):
        if subject == "geometry":
            raise RuntimeError("simulated 401")
        return [{"problem": "p", "solution": "\\boxed{1}", "level": "1", "type": subject}]

    rows = _load_math_all_subjects(_flaky, spec, split=None, limit=None)
    assert len(rows) == len(MATH_SUBJECTS) - 1  # geometry dropped, rest survive
    assert "geometry" not in {row.metadata["type"] for row in rows}
