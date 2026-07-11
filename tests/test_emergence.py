from instella_reasoning.analysis.atlas import AtlasCell, AtlasReport
from instella_reasoning.emergence import (
    analyze_rl_effect,
    capability_transitions,
    classify_divergence,
    classify_transition,
    rl_generalization_summary,
    schaeffer_artifact_test,
)


def _report(cells: dict[str, tuple[float, float, float]]) -> AtlasReport:
    """cells: skill -> (accuracy, consistency, reliability) at the 'all' level."""
    atlas_cells = [
        AtlasCell(skill, "all", 10, acc, cons, rel, "genuine")
        for skill, (acc, cons, rel) in cells.items()
    ]
    return AtlasReport(cells=atlas_cells)


def test_classify_transition_categories() -> None:
    assert classify_transition(0.6, 0.7) == "scale_independent"
    assert classify_transition(0.05, 0.6) == "emergent"
    assert classify_transition(0.1, 0.1) == "scale_resistant"
    assert classify_transition(0.2, 0.45) == "amplified"


def test_capability_transitions_across_models() -> None:
    small = _report({"arithmetic": (0.2, 0.3, 0.06), "logic": (0.5, 0.6, 0.30)})
    large = _report({"arithmetic": (0.7, 0.8, 0.56), "logic": (0.55, 0.62, 0.34)})
    transitions = capability_transitions(small, large, "olmo-1b", "instella-3b")
    by_skill = {t.skill: t for t in transitions}
    assert by_skill["arithmetic"].transition_class == "emergent"
    assert by_skill["arithmetic"].delta > 0


def test_schaeffer_flags_metric_artifact() -> None:
    # discontinuous jumps 0.1 -> 0.8; continuous rises smoothly
    result = schaeffer_artifact_test([0.05, 0.1, 0.8], [0.2, 0.35, 0.5])
    assert result["likely_metric_artifact"] is True

    genuine = schaeffer_artifact_test([0.1, 0.4, 0.8], [0.1, 0.45, 0.85])
    assert genuine["likely_metric_artifact"] is False


def test_rl_effect_distinguishes_genuine_from_accuracy_only() -> None:
    pre = _report({"arithmetic": (0.5, 0.5, 0.25), "logic": (0.4, 0.7, 0.28)})
    post = _report({"arithmetic": (0.7, 0.7, 0.49), "logic": (0.5, 0.7, 0.35)})
    effects = {e.skill: e for e in analyze_rl_effect(pre, post)}
    assert effects["arithmetic"].verdict == "genuine_improvement"
    # logic gained accuracy but not consistency
    assert effects["logic"].verdict == "accuracy_only"


def test_rl_generalization_summary_domain_specific() -> None:
    pre = _report({"arithmetic": (0.5, 0.5, 0.25), "logic": (0.4, 0.7, 0.28)})
    post = _report({"arithmetic": (0.8, 0.8, 0.64), "logic": (0.41, 0.7, 0.287)})
    effects = analyze_rl_effect(pre, post)
    summary = rl_generalization_summary(effects)
    assert summary["domain_specific"] is True


def test_cot_divergence_strategy_mismatch() -> None:
    result = classify_divergence(
        ["multiply 3 by 4", "done"], ["add 3 and 4", "subtract 2", "answer 5"], False, True
    )
    assert result["category"] == "strategy_mismatch"


def test_cot_divergence_premature_conclusion() -> None:
    result = classify_divergence(
        ["step one", "step two"],
        ["step one", "step two", "step three", "final step"],
        False,
        True,
    )
    assert result["category"] == "premature_conclusion"


def test_cot_no_divergence_when_both_correct() -> None:
    result = classify_divergence(["a"], ["a"], True, True)
    assert result["category"] == "no_divergence"
