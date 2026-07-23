"""Scale-controlled emergence analysis (proposal Phase 5).

CAVEAT (reviewer concern M6): AMD-OLMo-1B and Instella-3B share the OLMo *codebase* but
NOT the training *data* (different mixtures and token counts), so a 1B->3B comparison
entangles scale with data and does **not** cleanly isolate scale. Use
:func:`comparison_confound` to label an axis, and prefer Instella's own checkpoints
(Stage-1 vs Stage-2, SFT vs DPO) — which hold architecture and data lineage fixed — for
causal claims. This module provides:

- **Capability transitions** — classify each sub-skill's 1B->3B change as
  ``scale_independent`` / ``emergent`` / ``amplified`` / ``scale_resistant``.
- **Schaeffer metric-artifact test** — Schaeffer et al. (2023) showed apparent
  "emergence" can be an artifact of a discontinuous metric. We compare a discontinuous
  metric's scaling curve against a continuous one; a jump present only in the
  discontinuous metric is flagged as a likely artifact.
- **RL-effect analysis** — Instella-3B-Instruct (pre-RL) vs Instella-3B-Math (post-RL):
  does math RL raise *consistency* (genuine reasoning) or only *accuracy* (better
  pattern matching), and does it generalise beyond mathematics?
- **CoT-divergence taxonomy** — where a small model's chain of thought first diverges
  from a large model's on the same problem.

Inputs are the per-model :class:`instella_reasoning.analysis.atlas.AtlasReport`
objects, so the whole emergence layer is built on the same reliability decomposition
as the Atlas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from instella_reasoning.analysis.atlas import AtlasReport

HIGH_RELIABILITY = 0.50
LOW_RELIABILITY = 0.30
NEAR_ZERO = 0.10


# Comparison axes and whether they cleanly isolate a single factor. Only checkpoint
# axes within one lineage are "controlled"; cross-model-size axes confound scale + data.
_CONTROLLED_AXES = {
    "stage1_vs_stage2": "data stage within Instella-3B (architecture + lineage fixed)",
    "sft_vs_dpo": "post-training objective within Instella-3B (architecture fixed)",
    "instruct_vs_math": "post-training objective (SFT vs SFT+RL), same base",
}
_CONFOUNDED_AXES = {
    "olmo1b_vs_instella3b": "confounds scale AND data mixture AND token count",
}


def comparison_confound(axis: str) -> dict:
    """Label whether an emergence/scale comparison axis is controlled or confounded."""
    if axis in _CONTROLLED_AXES:
        return {"axis": axis, "controlled": True, "note": _CONTROLLED_AXES[axis]}
    if axis in _CONFOUNDED_AXES:
        return {"axis": axis, "controlled": False, "note": _CONFOUNDED_AXES[axis]}
    return {"axis": axis, "controlled": False, "note": "unknown axis; assume confounded until verified"}


# -- capability transitions ----------------------------------------------------


@dataclass(slots=True)
class Transition:
    skill: str
    small_model: str
    large_model: str
    small_reliability: float
    large_reliability: float
    delta: float
    transition_class: str

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "small_model": self.small_model,
            "large_model": self.large_model,
            "small_reliability": round(self.small_reliability, 6),
            "large_reliability": round(self.large_reliability, 6),
            "delta": round(self.delta, 6),
            "transition_class": self.transition_class,
        }


def classify_transition(
    small: float,
    large: float,
    high: float = HIGH_RELIABILITY,
    low: float = LOW_RELIABILITY,
    near_zero: float = NEAR_ZERO,
) -> str:
    if small >= high and large >= high:
        return "scale_independent"
    if large >= high and small < near_zero:
        return "emergent"
    if large < low and small < low:
        return "scale_resistant"
    return "amplified"


def reliability_by_skill(report: AtlasReport, level: str = "all") -> dict[str, float]:
    """Map skill -> reliability at the requested contamination level for one model."""
    return {
        cell.skill: cell.reliability
        for cell in report.cells
        if cell.contamination_level == level
    }


def capability_transitions(
    small: AtlasReport,
    large: AtlasReport,
    small_model: str = "small",
    large_model: str = "large",
    level: str = "all",
) -> list[Transition]:
    """Classify the small->large transition for every shared sub-skill."""
    small_map = reliability_by_skill(small, level)
    large_map = reliability_by_skill(large, level)
    transitions: list[Transition] = []
    for skill in sorted(set(small_map) | set(large_map)):
        small_r = small_map.get(skill, 0.0)
        large_r = large_map.get(skill, 0.0)
        transitions.append(
            Transition(
                skill=skill,
                small_model=small_model,
                large_model=large_model,
                small_reliability=small_r,
                large_reliability=large_r,
                delta=large_r - small_r,
                transition_class=classify_transition(small_r, large_r),
            )
        )
    return transitions


# -- Schaeffer metric-artifact test --------------------------------------------


def schaeffer_artifact_test(
    discontinuous: list[float],
    continuous: list[float],
    jump_threshold: float = 0.30,
) -> dict:
    """Flag apparent emergence that only exists under a discontinuous metric.

    ``discontinuous`` and ``continuous`` are two metric curves over the *same* models
    in increasing-size order (e.g. strict cluster accuracy vs mean per-variant
    accuracy). If the discontinuous curve has a large single-step jump but the
    continuous one is smooth, the "emergence" is likely a metric artifact.
    """
    disc_jump = _max_step(discontinuous)
    cont_jump = _max_step(continuous)
    likely_artifact = disc_jump >= jump_threshold and cont_jump < jump_threshold
    return {
        "discontinuous_max_jump": round(disc_jump, 6),
        "continuous_max_jump": round(cont_jump, 6),
        "likely_metric_artifact": likely_artifact,
        "verdict": "metric_artifact" if likely_artifact else "genuine_or_smooth",
    }


def _max_step(curve: list[float]) -> float:
    if len(curve) < 2:
        return 0.0
    return max(curve[i + 1] - curve[i] for i in range(len(curve) - 1))


# -- RL-effect analysis --------------------------------------------------------

_MATH_SKILLS = {"arithmetic", "mathematical"}


@dataclass(slots=True)
class RLEffect:
    skill: str
    delta_accuracy: float
    delta_consistency: float
    delta_reliability: float
    verdict: str

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "delta_accuracy": round(self.delta_accuracy, 6),
            "delta_consistency": round(self.delta_consistency, 6),
            "delta_reliability": round(self.delta_reliability, 6),
            "verdict": self.verdict,
        }


def _rl_verdict(delta_accuracy: float, delta_consistency: float, tol: float = 0.02) -> str:
    if delta_consistency > tol and delta_accuracy > tol:
        return "genuine_improvement"
    if delta_accuracy > tol and delta_consistency <= tol:
        return "accuracy_only"
    if delta_accuracy < -tol or delta_consistency < -tol:
        return "regression"
    return "no_change"


def analyze_rl_effect(pre_rl: AtlasReport, post_rl: AtlasReport, level: str = "all") -> list[RLEffect]:
    """Per-skill effect of RL post-training (consistency vs accuracy gain)."""
    pre = {c.skill: c for c in pre_rl.cells if c.contamination_level == level}
    post = {c.skill: c for c in post_rl.cells if c.contamination_level == level}
    effects: list[RLEffect] = []
    for skill in sorted(set(pre) & set(post)):
        d_acc = post[skill].accuracy - pre[skill].accuracy
        d_cons = post[skill].consistency - pre[skill].consistency
        d_rel = post[skill].reliability - pre[skill].reliability
        effects.append(RLEffect(skill, d_acc, d_cons, d_rel, _rl_verdict(d_acc, d_cons)))
    return effects


def rl_generalization_summary(effects: list[RLEffect]) -> dict:
    """Does RL improvement generalise beyond math, or is it domain-specific?"""
    math_effects = [e for e in effects if e.skill in _MATH_SKILLS]
    other_effects = [e for e in effects if e.skill not in _MATH_SKILLS]
    math_gain = _mean([e.delta_reliability for e in math_effects])
    other_gain = _mean([e.delta_reliability for e in other_effects])
    domain_specific = math_gain > 0.05 and other_gain <= 0.05
    return {
        "math_reliability_gain": round(math_gain, 6),
        "non_math_reliability_gain": round(other_gain, 6),
        "domain_specific": domain_specific,
        "verdict": "domain_specific_reasoning" if domain_specific else "general_or_none",
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# -- CoT-divergence taxonomy ---------------------------------------------------

DIVERGENCE_CATEGORIES = (
    "premature_conclusion",
    "error_propagation",
    "capacity_failure",
    "strategy_mismatch",
    "no_divergence",
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_step(step: str) -> str:
    return " ".join(step.lower().split())


def classify_divergence(
    small_steps: list[str],
    large_steps: list[str],
    small_correct: bool,
    large_correct: bool,
) -> dict:
    """Categorise where a small model's chain of thought first diverges from a large one.

    Heuristic, following the proposal's taxonomy:
    - ``strategy_mismatch``   — the very first step already differs.
    - ``error_propagation``   — steps agree then diverge at an intermediate step whose
      numbers differ (a local slip that then cascades).
    - ``premature_conclusion``— the small chain is materially shorter (stopped early).
    - ``capacity_failure``    — long, many-variable chains that diverge late.
    """
    if large_correct and small_correct:
        return {"category": "no_divergence", "divergence_step": None}

    first_diff = _first_divergence_index(small_steps, large_steps)
    if first_diff is None:
        # chains agree as far as they go; the shorter one stopped early
        if len(small_steps) < len(large_steps):
            return {"category": "premature_conclusion", "divergence_step": len(small_steps)}
        return {"category": "no_divergence", "divergence_step": None}

    if first_diff == 0:
        return {"category": "strategy_mismatch", "divergence_step": 0}

    if len(small_steps) < len(large_steps) - 1 and first_diff >= len(small_steps) - 1:
        return {"category": "premature_conclusion", "divergence_step": first_diff}

    # divergence in the middle with differing numbers -> a local error that propagates,
    # unless the chains are long and variable-heavy (capacity failure)
    variable_heavy = _max_numbers_per_step(large_steps) >= 3 and len(large_steps) >= 5
    if variable_heavy and first_diff >= len(large_steps) // 2:
        return {"category": "capacity_failure", "divergence_step": first_diff}
    return {"category": "error_propagation", "divergence_step": first_diff}


def _first_divergence_index(small_steps: list[str], large_steps: list[str]) -> int | None:
    for i in range(min(len(small_steps), len(large_steps))):
        if _normalize_step(small_steps[i]) != _normalize_step(large_steps[i]):
            return i
    return None


def _max_numbers_per_step(steps: list[str]) -> int:
    return max((len(_NUMBER.findall(step)) for step in steps), default=0)


def atlas_consistency_probed(*reports: AtlasReport) -> bool:
    """True iff every given atlas has at least one cell that probed consistency.

    When this is False the atlases were built from base benchmarks only (no answer-
    preserving variants), so every 'reliability' equals plain accuracy and the
    transition/RL verdicts below describe an *accuracy* curve, not a reasoning one.
    """
    return bool(reports) and all(
        any(cell.consistency_probed for cell in report.cells) for report in reports
    )


@dataclass(slots=True)
class EmergenceReport:
    transitions: list[Transition] = field(default_factory=list)
    rl_effects: list[RLEffect] = field(default_factory=list)
    rl_generalization: dict = field(default_factory=dict)
    # False when the underlying atlases have no answer-preserving variants: every
    # reliability collapses to accuracy, so these verdicts are accuracy-only (see
    # :func:`atlas_consistency_probed`). Surfaced so no reader mistakes this for a
    # measured reasoning-emergence result.
    consistency_probed: bool = True

    def to_dict(self) -> dict:
        return {
            "consistency_probed": self.consistency_probed,
            "reliability_note": (
                None
                if self.consistency_probed
                else "consistency was NOT probed (no answer-preserving variants); every "
                "reliability below equals accuracy and these transition/RL verdicts are "
                "accuracy-only, not measured reasoning emergence"
            ),
            "transitions": [t.to_dict() for t in self.transitions],
            "rl_effects": [e.to_dict() for e in self.rl_effects],
            "rl_generalization": self.rl_generalization,
        }
