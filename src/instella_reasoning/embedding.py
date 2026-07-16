"""Sentence embedding backends for contamination search and attribution.

This module is portable: on a Colab T4 it uses the GPU automatically; on a
CPU-only Windows box it runs on CPU. When ``sentence-transformers`` is not
installed, a dependency-free hashing embedder is provided so the rest of the
pipeline (FAISS indexing, contamination classification, tests) can still run at
smoke scale.

Two backends implement the same interface:

- ``SentenceTransformerEmbedder`` — the real semantic embedder (MiniLM / GTE).
  Requires the ``retrieval`` extra (``pip install -e .[retrieval]``).
- ``HashingEmbedder`` — a deterministic bag-of-character-n-grams embedder with no
  third-party dependencies. Lower quality, but lets the pipeline run anywhere and
  makes tests hermetic.

Vectors are always L2-normalized so that inner product equals cosine similarity,
which is what the FAISS index and contamination thresholds assume.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from instella_reasoning.text import char_ngrams, normalize_text

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HASHING_DIM = 256


class Embedder(Protocol):
    """Minimal embedder interface used across the pipeline."""

    dimension: int
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input text."""
        ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


@dataclass(slots=True)
class HashingEmbedder:
    """Dependency-free deterministic embedder over character n-grams.

    Not competitive with a trained sentence encoder, but it is stable, requires
    no downloads, and gives the FAISS + contamination code a real vector space to
    operate on for smoke tests and CI. Use ``SentenceTransformerEmbedder`` for any
    quantitative contamination claim.
    """

    dimension: int = _HASHING_DIM
    ngram_size: int = 4
    name: str = "hashing-char4"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        grams = char_ngrams(text, self.ngram_size)
        if not grams:
            # Fall back to whole normalized string so empty-ngram inputs still map
            # somewhere deterministic rather than to the zero vector.
            grams = {normalize_text(text)} if normalize_text(text) else set()
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)


class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (MiniLM, GTE, ...).

    Chooses CUDA automatically when available, else CPU, so the same code runs on
    a Colab T4 and a Windows laptop. Vectors are L2-normalized on return.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "Install the retrieval extra to use SentenceTransformerEmbedder: "
                "`pip install -e .[retrieval]`. For a dependency-free fallback use "
                "HashingEmbedder or pass backend='hashing'."
            ) from exc

        if device is None:
            device = _autodetect_device()
        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.batch_size = batch_size
        # sentence-transformers >=3.1 renamed get_sentence_embedding_dimension ->
        # get_embedding_dimension (the old name warns). Prefer the new name and fall
        # back so we support both installed versions without a deprecation warning.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def _autodetect_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is part of hf/train extras
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_embedder(
    backend: str = "auto",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    batch_size: int = 64,
) -> Embedder:
    """Construct an embedder.

    ``backend``:
    - ``"auto"`` — use sentence-transformers if importable, else the hashing fallback.
    - ``"sentence-transformers"`` — force the real model (raises if the extra is missing).
    - ``"hashing"`` — force the dependency-free fallback.
    """

    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name, device=device, batch_size=batch_size)
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return HashingEmbedder()
        return SentenceTransformerEmbedder(model_name, device=device, batch_size=batch_size)
    raise ValueError(f"Unknown embedder backend: {backend!r}")
