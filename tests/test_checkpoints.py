"""The Instella trajectory registry — the model axis, and the facts it encodes.

The registry is the single source of truth for token budgets, few-shot policy, and which
checkpoints have seen GSM8K-derived data. Each of those has already caused a wrong result
once when it lived as a magic constant in a shell script instead.
"""

from __future__ import annotations

import pytest

from instella_reasoning.checkpoints import (
    INSTELLA_TRAJECTORY,
    contamination_intervention,
    resolve,
    tier,
    transitions,
    try_resolve,
)


def test_trajectory_steps_are_unique_and_ordered() -> None:
    steps = [c.step for c in INSTELLA_TRAJECTORY]
    assert steps == sorted(steps)
    assert len(set(steps)) == len(steps)


def test_stage1_is_the_only_checkpoint_without_gsm8k_data() -> None:
    """Stage 1 is the control arm; everything downstream of stage 2 has seen the data."""
    clean = [c for c in INSTELLA_TRAJECTORY if not c.saw_gsm8k_derived_data]
    assert [c.tag for c in clean] == ["stage1"]


def test_contamination_intervention_is_the_stage1_to_stage2_step() -> None:
    before, after = contamination_intervention()
    assert before.tag == "stage1"
    assert after.tag == "stage2"
    assert not before.saw_gsm8k_derived_data
    assert after.saw_gsm8k_derived_data


def test_long_cot_checkpoints_get_a_larger_token_budget() -> None:
    """The Math checkpoints emitted their answer marker on 2% of items at 512 tokens."""
    for tag in ("math", "math_sft"):
        assert resolve(tag).max_new_tokens >= 1536


def test_base_checkpoints_request_few_shot_prompting() -> None:
    """Zero-shot CoT on a base model loops; that measures instruction-following, not skill."""
    for ckpt in INSTELLA_TRAJECTORY:
        assert (ckpt.n_shot > 0) == ckpt.is_base


def test_resolution_accepts_tags_aliases_hf_ids_and_bare_names() -> None:
    for key in ("stage2", "base", "amd/Instella-3B", "Instella-3B"):
        assert resolve(key).tag == "stage2"
    assert try_resolve("not-a-model") is None
    with pytest.raises(KeyError):
        resolve("not-a-model")


def test_tier_one_is_the_core_run_and_excludes_off_trajectory_models() -> None:
    core = tier(1)
    assert [c.tag for c in core] == ["stage1", "stage2", "instruct", "math"]
    assert all(c.step >= 0 for c in core)
    assert "olmo1b" not in {c.tag for c in tier(3)}
    assert "olmo1b" in {c.tag for c in tier(3, include_off_trajectory=True)}


def test_transitions_are_consecutive_pairs() -> None:
    pairs = transitions(tier(1))
    assert [(a.tag, b.tag) for a, b in pairs] == [
        ("stage1", "stage2"),
        ("stage2", "instruct"),
        ("instruct", "math"),
    ]


def test_load_path_prefers_the_assembled_local_copy() -> None:
    """Repos shipping vLLM-only remote code cannot be loaded from the Hub directly."""
    assert resolve("math").load_path == "models/Instella-3B-Math-hf"
    assert resolve("stage2").load_path == "amd/Instella-3B"
