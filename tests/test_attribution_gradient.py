import math

import pytest

from instella_reasoning.attribution_gradient import (
    ConceptAttributor,
    GradientAttributionConfig,
    TracInAttributor,
    concept_similarity_scores,
    influence_concentration,
    tracin_scores_from_gradients,
)


def test_tracin_ranks_aligned_gradients_first() -> None:
    test = [[1.0, 0.0, 0.0]]  # one checkpoint
    candidates = {
        "aligned": [[2.0, 0.0, 0.0]],      # same direction -> high positive dot
        "orthogonal": [[0.0, 1.0, 0.0]],   # zero dot
        "opposed": [[-1.0, 0.0, 0.0]],     # negative dot
    }
    scores = tracin_scores_from_gradients(test, candidates)
    assert scores[0].train_id == "aligned"
    assert scores[-1].train_id == "opposed"
    assert scores[0].score > 0 > scores[-1].score


def test_tracin_sums_over_checkpoints_with_lr() -> None:
    test = [[1.0], [1.0]]
    candidates = {"c": [[3.0], [5.0]]}
    scores = tracin_scores_from_gradients(test, candidates, learning_rates=[1.0, 2.0])
    # 1*(1*3) + 2*(1*5) = 3 + 10 = 13
    assert math.isclose(scores[0].score, 13.0)


def test_tracin_rejects_mismatched_checkpoints() -> None:
    with pytest.raises(ValueError):
        tracin_scores_from_gradients([[1.0]], {"c": [[1.0], [1.0]]})


def test_concept_similarity_orders_by_cosine() -> None:
    test = [1.0, 1.0]
    candidates = {"same": [2.0, 2.0], "orth": [1.0, -1.0]}
    scores = concept_similarity_scores(test, candidates)
    assert scores[0].train_id == "same"
    assert math.isclose(scores[0].score, 1.0, abs_tol=1e-9)
    assert math.isclose(scores[1].score, 0.0, abs_tol=1e-9)


def test_influence_concentration_flags_dominant_source() -> None:
    scores = tracin_scores_from_gradients([[1.0]], {"a": [[10.0]], "b": [[0.1]], "c": [[0.1]]})
    summary = influence_concentration(scores, top_n=1)
    assert summary["top1_share"] > 0.9
    assert summary["n"] == 3


def test_torch_attributors_raise_without_extra() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; graceful-failure path not exercised")
    config = GradientAttributionConfig(model_name_or_path="amd/Instella-3B")
    with pytest.raises(RuntimeError, match="train extra"):
        TracInAttributor(config)
    with pytest.raises(RuntimeError, match="train extra"):
        ConceptAttributor(config)
