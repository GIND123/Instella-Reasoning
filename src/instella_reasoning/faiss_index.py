"""Vector index for near-duplicate retrieval over Instella training data.

Provides a single ``VectorIndex`` abstraction with two backends:

- FAISS (``flat``, ``ivfpq``, ``hnsw``) when ``faiss`` is installed — the
  production path for 10M+ passage scans described in the proposal.
- A dependency-free brute-force NumPy/pure-Python backend otherwise, so the
  contamination pipeline and tests run on any machine at smoke scale.

Vectors are assumed L2-normalized (see :mod:`instella_reasoning.embedding`), so
inner-product search returns cosine similarity directly.

The index stores a parallel list of string document ids alongside the vectors so
callers get back ``(document_id, cosine_similarity)`` pairs. A small JSON sidecar
records metadata (backend, dimension, model name) so an index can be rebuilt or
audited later.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SearchResult:
    document_id: str
    score: float  # cosine similarity in [-1, 1]


def _try_import_numpy():
    try:
        import numpy as np

        return np
    except ImportError:  # pragma: no cover - numpy ships with retrieval/hf extras
        return None


def _try_import_faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        return None


class VectorIndex:
    """Cosine-similarity nearest-neighbor index with graceful fallback.

    Parameters
    ----------
    dimension:
        Embedding dimension.
    index_type:
        ``"flat"`` (exact), ``"ivfpq"`` (approximate, memory-light), or ``"hnsw"``
        (graph-based, best recall). Ignored by the brute-force fallback.
    backend:
        ``"auto"`` picks FAISS when available, else brute force. Force with
        ``"faiss"`` or ``"bruteforce"``.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        backend: str = "auto",
        nlist: int = 4096,
        pq_m: int = 48,
        hnsw_m: int = 32,
        embedding_model: str = "unknown",
    ) -> None:
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.pq_m = pq_m
        self.hnsw_m = hnsw_m
        self.embedding_model = embedding_model
        self.document_ids: list[str] = []

        faiss = _try_import_faiss() if backend in {"auto", "faiss"} else None
        if backend == "faiss" and faiss is None:
            raise RuntimeError(
                "backend='faiss' requested but faiss is not installed. "
                "Install with `pip install -e .[retrieval]` or use backend='bruteforce'."
            )
        self._faiss = faiss
        self.backend = "faiss" if faiss is not None else "bruteforce"

        if self._faiss is not None:
            self._index = self._build_faiss_index()
            self._vectors = None
        else:
            self._index = None
            self._np = _try_import_numpy()
            # Brute-force store: numpy matrix if available, else list of lists.
            self._vectors: list[list[float]] | object = [] if self._np is None else None
            self._np_matrix = None

    # -- construction ------------------------------------------------------

    def _build_faiss_index(self):
        faiss = self._faiss
        if self.index_type == "flat":
            return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            return faiss.IndexIDMap2(index)
        if self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, self.pq_m, 8, faiss.METRIC_INNER_PRODUCT
            )
            return faiss.IndexIDMap2(index)
        raise ValueError(f"Unknown index_type: {self.index_type!r}")

    # -- population --------------------------------------------------------

    def add(self, document_ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if len(document_ids) != len(vectors):
            raise ValueError("document_ids and vectors must have equal length")
        if not document_ids:
            return

        start = len(self.document_ids)
        self.document_ids.extend(str(doc_id) for doc_id in document_ids)

        if self._faiss is not None:
            np = _try_import_numpy()
            matrix = np.asarray(vectors, dtype="float32")
            ids = np.arange(start, start + len(document_ids), dtype="int64")
            if self.index_type == "ivfpq" and not self._index.index.is_trained:
                self._index.index.train(matrix)
            self._index.add_with_ids(matrix, ids)
            return

        # brute-force
        if self._np is not None:
            import numpy as np

            block = np.asarray(vectors, dtype="float32")
            self._np_matrix = block if self._np_matrix is None else np.vstack([self._np_matrix, block])
        else:
            self._vectors.extend([list(map(float, vector)) for vector in vectors])

    # -- query -------------------------------------------------------------

    def search(self, query: Sequence[float], top_k: int = 20) -> list[SearchResult]:
        if not self.document_ids:
            return []
        top_k = min(top_k, len(self.document_ids))

        if self._faiss is not None:
            import numpy as np

            q = np.asarray([query], dtype="float32")
            scores, ids = self._index.search(q, top_k)
            results: list[SearchResult] = []
            for score, idx in zip(scores[0], ids[0], strict=False):
                if idx < 0:
                    continue
                results.append(SearchResult(self.document_ids[int(idx)], float(score)))
            return results

        if self._np is not None:
            import numpy as np

            q = np.asarray(query, dtype="float32")
            sims = self._np_matrix @ q
            order = np.argsort(-sims)[:top_k]
            return [SearchResult(self.document_ids[int(i)], float(sims[int(i)])) for i in order]

        # pure-python brute force
        scored = [
            (doc_id, _dot(query, vector))
            for doc_id, vector in zip(self.document_ids, self._vectors, strict=False)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [SearchResult(doc_id, score) for doc_id, score in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.document_ids)

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": self.backend,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "count": len(self.document_ids),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (path / "document_ids.json").write_text(
            json.dumps(self.document_ids), encoding="utf-8"
        )
        if self._faiss is not None:
            self._faiss.write_index(self._index, str(path / "index.faiss"))
        elif self._np is not None and self._np_matrix is not None:
            import numpy as np

            np.save(path / "vectors.npy", self._np_matrix)
        else:
            (path / "vectors.json").write_text(json.dumps(self._vectors), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, backend: str = "auto") -> VectorIndex:
        path = Path(directory)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        index = cls(
            dimension=int(meta["dimension"]),
            index_type=str(meta.get("index_type", "flat")),
            backend=backend,
            embedding_model=str(meta.get("embedding_model", "unknown")),
        )
        index.document_ids = json.loads((path / "document_ids.json").read_text(encoding="utf-8"))
        if index._faiss is not None and (path / "index.faiss").exists():
            index._index = index._faiss.read_index(str(path / "index.faiss"))
        elif (path / "vectors.npy").exists() and index._np is not None:
            import numpy as np

            index._np_matrix = np.load(path / "vectors.npy")
        elif (path / "vectors.json").exists():
            index._vectors = json.loads((path / "vectors.json").read_text(encoding="utf-8"))
        return index


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=False))
