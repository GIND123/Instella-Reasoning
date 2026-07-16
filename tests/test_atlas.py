from instella_reasoning.analysis.atlas import build_atlas, classify_cell
from instella_reasoning.records import ContaminationHit, EvaluationRecord


def _eval(parent: str, variant: str, correct: bool, predicted: str, skill: str) -> EvaluationRecord:
    return EvaluationRecord(
        benchmark_id=f"{parent}__{variant}",
        parent_id=parent,
        variant_type=variant,
        expected="16",
        predicted=predicted,
        normalized_expected="16",
        normalized_predicted=predicted,
        correct=correct,
        model="amd/Instella-3B",
        metadata={"skill": skill, "benchmark": "gsm8k"},
    )


def _hit(benchmark_id: str, label: str) -> ContaminationHit:
    return ContaminationHit(benchmark_id, "doc", "src", label, 0.95, 0.5, 0.5, "")


def test_classify_cell_boundaries() -> None:
    assert classify_cell(0.9, 0.6) == "genuine"
    assert classify_cell(0.9, 0.1) == "fragile"  # high accuracy, low reliability
    assert classify_cell(0.5, 0.35) == "partial"
    assert classify_cell(0.2, 0.1) == "gap"


def test_atlas_splits_by_skill_and_level() -> None:
    scores = [
        # contaminated cluster: correct but inconsistent (fragile signature)
        _eval("gsm_1", "original", True, "16", "arithmetic"),
        _eval("gsm_1", "entity_substitution", True, "16", "arithmetic"),
        _eval("gsm_1", "irrelevant_context", False, "42", "arithmetic"),
        # clean cluster: consistent
        _eval("gsm_2", "original", True, "20", "arithmetic"),
        _eval("gsm_2", "entity_substitution", True, "20", "arithmetic"),
    ]
    contamination = [_hit("gsm_1", "contaminated")]
    report = build_atlas(scores, contamination)

    all_cell = report.cell("arithmetic", "all")
    assert all_cell is not None
    assert all_cell.n_clusters == 2

    contaminated = report.cell("arithmetic", "contaminated")
    clean = report.cell("arithmetic", "clean")
    assert contaminated is not None and clean is not None
    assert contaminated.n_clusters == 1
    assert clean.n_clusters == 1
    # clean cluster is fully consistent -> higher reliability than the mixed one
    assert clean.reliability >= contaminated.reliability


def test_atlas_markdown_renders_table() -> None:
    scores = [_eval("gsm_1", "original", True, "16", "arithmetic")]
    report = build_atlas(scores, [])
    md = report.to_markdown()
    assert "Reasoning Reliability Atlas" in md
    assert "arithmetic" in md


def test_atlas_cell_carries_reliability_ci() -> None:
    scores = [
        _eval("gsm_1", "original", True, "16", "arithmetic"),
        _eval("gsm_2", "original", True, "16", "arithmetic"),
        _eval("gsm_3", "original", True, "16", "arithmetic"),
    ]
    report = build_atlas(scores, [])
    cell = report.cell("arithmetic", "all")
    assert cell is not None
    assert cell.reliability_ci_low <= cell.reliability <= cell.reliability_ci_high
    assert "reliability_ci" in cell.to_dict()


def test_verdict_withheld_below_min_clusters() -> None:
    # A single cluster cannot support a categorical verdict -> insufficient_data.
    one = build_atlas([_eval("gsm_1", "original", True, "16", "arithmetic")], [])
    assert one.cell("arithmetic", "all").classification == "insufficient_data"
    # With enough clusters and a lenient gate, a real verdict is emitted.
    scores = [_eval(f"gsm_{i}", "original", True, "16", "arithmetic") for i in range(3)]
    many = build_atlas(scores, [], min_clusters_for_verdict=3)
    assert many.cell("arithmetic", "all").classification == "genuine"
