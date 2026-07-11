"""Tier-2/3 gradient- and representation-based attribution (proposal Section 6.4).

Two methods refine the embedding attribution for the most informative items:

- **TracIn-CP** (Pruthi et al., 2020): influence of a training example on a test
  example is the sum, over saved checkpoints, of the dot product of their loss
  gradients (scaled by the checkpoint learning rate). Following Yeh et al. (2022,
  "First is Better Than Last"), gradients are taken over early-to-middle layer
  parameters selected by a name filter, which is both cheaper and often more faithful
  than last-layer gradients.

- **Concept Influence** (Kowal et al., 2026, cheap variant): a forward-only,
  representation-level signal — mean-pooled hidden states at a chosen layer, compared
  by cosine similarity. It needs no backward pass, so it is the T4 fallback when
  gradients for the 3B model do not fit.

Design note: the *numeric cores* (``tracin_scores_from_gradients``,
``concept_similarity_scores``, ranking/normalisation) are pure Python and fully
tested. The Torch machinery that produces the gradient/representation vectors is
imported lazily, so this module imports and its cores run with no heavy dependencies.
Install the ``train`` extra (``pip install -e .[train]``) to run the real attributors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from instella_reasoning.records import write_jsonl

Vector = Sequence[float]

# Default parameter name substrings for "early-to-middle" layers of a 36-layer model.
DEFAULT_LAYER_FILTER = tuple(f".layers.{i}." for i in range(4, 18))


# -- numeric cores (pure Python, no torch) -------------------------------------


@dataclass(slots=True)
class InfluenceScore:
    train_id: str
    score: float
    method: str
    per_checkpoint: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "train_id": self.train_id,
            "score": round(self.score, 8),
            "method": self.method,
            "per_checkpoint": [round(value, 8) for value in self.per_checkpoint],
        }


def _dot(a: Vector, b: Vector) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=False))


def _cosine(a: Vector, b: Vector) -> float:
    na = math.sqrt(_dot(a, a))
    nb = math.sqrt(_dot(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def tracin_scores_from_gradients(
    test_gradients: Sequence[Vector],
    candidate_gradients: dict[str, Sequence[Vector]],
    learning_rates: Sequence[float] | None = None,
) -> list[InfluenceScore]:
    """Aggregate TracIn scores from precomputed per-checkpoint gradient vectors.

    ``test_gradients`` is one flattened gradient per checkpoint for the test example;
    ``candidate_gradients[train_id]`` is the aligned list for a candidate. The score is
    ``sum_c lr_c * <g_test^c, g_train^c>``. Kept separate from Torch so it is testable.
    """
    n_checkpoints = len(test_gradients)
    lrs = list(learning_rates) if learning_rates is not None else [1.0] * n_checkpoints
    if len(lrs) != n_checkpoints:
        raise ValueError("learning_rates must have one entry per checkpoint")

    scores: list[InfluenceScore] = []
    for train_id, grads in candidate_gradients.items():
        if len(grads) != n_checkpoints:
            raise ValueError(f"candidate {train_id} has {len(grads)} checkpoints, expected {n_checkpoints}")
        per_checkpoint = [lrs[c] * _dot(test_gradients[c], grads[c]) for c in range(n_checkpoints)]
        scores.append(
            InfluenceScore(
                train_id=train_id,
                score=math.fsum(per_checkpoint),
                method="tracin_cp",
                per_checkpoint=per_checkpoint,
            )
        )
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def concept_similarity_scores(
    test_repr: Vector,
    candidate_reprs: dict[str, Vector],
) -> list[InfluenceScore]:
    """Rank candidates by cosine similarity of their concept-level representations."""
    scores = [
        InfluenceScore(train_id=train_id, score=_cosine(test_repr, repr_vec), method="concept_influence")
        for train_id, repr_vec in candidate_reprs.items()
    ]
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def influence_concentration(scores: Sequence[InfluenceScore], top_n: int = 10) -> dict:
    """Summarise how concentrated influence is (memorisation vs generalisation signal)."""
    if not scores:
        return {"top1_share": 0.0, "top_n_share": 0.0, "n": 0}
    positive = [max(0.0, s.score) for s in scores]
    total = math.fsum(positive) or 1.0
    ordered = sorted(positive, reverse=True)
    return {
        "top1_share": ordered[0] / total,
        "top_n_share": math.fsum(ordered[:top_n]) / total,
        "n": len(scores),
    }


# -- Torch-backed attributors (lazy import) ------------------------------------


@dataclass(slots=True)
class GradientAttributionConfig:
    model_name_or_path: str
    layer_filter: tuple[str, ...] = DEFAULT_LAYER_FILTER
    concept_layer: int = 8
    max_length: int = 1024
    device: str | None = None
    trust_remote_code: bool = True


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "Gradient/concept attribution needs the train extra: `pip install -e .[train]`."
        ) from exc


class TracInAttributor:
    """Compute TracIn-CP attributions for a test item over candidate training docs.

    Requires the ``train`` extra. Pre-filter candidates with
    :class:`instella_reasoning.attribution_embedding.EmbeddingAttributor` (FAISS top-N)
    before calling this — TracIn is only feasible on a small candidate set per item.
    """

    def __init__(self, config: GradientAttributionConfig) -> None:  # pragma: no cover - needs torch
        _require_torch()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.model.eval()
        self._selected = [
            (name, param)
            for name, param in self.model.named_parameters()
            if any(token in name for token in config.layer_filter)
        ]
        if not self._selected:  # fall back to all params if the filter matched nothing
            self._selected = list(self.model.named_parameters())

    def _loss_gradient(self, prompt: str, completion: str):  # pragma: no cover - needs torch
        import torch

        text = prompt + completion
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        prompt_len = len(
            self.tokenizer(prompt, truncation=True, max_length=self.config.max_length)["input_ids"]
        )
        input_ids = enc["input_ids"].to(self.device)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100  # only score completion tokens
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_ids=input_ids, labels=labels)
        grads = torch.autograd.grad(outputs.loss, [p for _, p in self._selected], allow_unused=True)
        flat = [g.detach().reshape(-1) for g in grads if g is not None]
        return torch.cat(flat).cpu().tolist() if flat else [0.0]

    def attribute(
        self,
        test_prompt: str,
        test_completion: str,
        candidates: dict[str, tuple[str, str]],
    ) -> list[InfluenceScore]:  # pragma: no cover - needs torch
        test_grad = self._loss_gradient(test_prompt, test_completion)
        candidate_grads = {
            train_id: [self._loss_gradient(prompt, completion)]
            for train_id, (prompt, completion) in candidates.items()
        }
        return tracin_scores_from_gradients([test_grad], candidate_grads)


class ConceptAttributor:
    """Forward-only concept-level attribution via mean-pooled hidden states."""

    def __init__(self, config: GradientAttributionConfig) -> None:  # pragma: no cover - needs torch
        _require_torch()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            output_hidden_states=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

    def _represent(self, text: str):  # pragma: no cover - needs torch
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model(**enc)
        hidden = outputs.hidden_states[self.config.concept_layer][0]
        return hidden.mean(dim=0).cpu().tolist()

    def attribute(
        self, test_text: str, candidates: dict[str, str]
    ) -> list[InfluenceScore]:  # pragma: no cover - needs torch
        test_repr = self._represent(test_text)
        candidate_reprs = {train_id: self._represent(text) for train_id, text in candidates.items()}
        return concept_similarity_scores(test_repr, candidate_reprs)


def write_influence_scores(
    output_path: str | Path, benchmark_id: str, scores: Sequence[InfluenceScore]
) -> None:
    """Write ranked influence scores for one benchmark item to JSONL."""
    rows = [{"benchmark_id": benchmark_id, **score.to_dict()} for score in scores]
    write_jsonl(output_path, rows)
