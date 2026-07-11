from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instella_reasoning.embedding import Embedder, build_embedder
from instella_reasoning.faiss_index import VectorIndex
from instella_reasoning.records import BenchmarkItem, ContaminationHit, CorpusDocument, write_jsonl
from instella_reasoning.text import (
    char_ngrams,
    compact_excerpt,
    jaccard,
    normalize_text,
    token_set,
    weighted_token_overlap,
    word_ngram_overlap,
)


@dataclass(slots=True)
class ContaminationThresholds:
    exact: float = 0.98
    near_duplicate: float = 0.72
    paraphrase_candidate: float = 0.45


@dataclass(slots=True)
class EmbeddingContaminationThresholds:
    """Cosine + n-gram thresholds for the C / PC / N scheme (proposal Section 6.1).

    Following the proposal:

    - Contaminated (C):        cosine > 0.90  OR  13-gram overlap > 0.60
    - Partially contaminated:  0.75 < cosine <= 0.90  OR  0.30 < 13-gram <= 0.60
    - Clean (N):               cosine <= 0.75  AND  13-gram <= 0.30
    """

    contaminated_cosine: float = 0.90
    partial_cosine: float = 0.75
    contaminated_ngram: float = 0.60
    partial_ngram: float = 0.30
    ngram_n: int = 13


@dataclass(slots=True)
class IndexedDocument:
    document: CorpusDocument
    normalized: str
    tokens: set[str]
    ngrams: set[str]


class LexicalSearchIndex:
    """Small dependency-free retrieval index for smoke tests and sharded scans.

    For production-scale Instella scans, use this class as the first-pass
    reference behavior and replace the candidate generator with FAISS or another
    distributed embedding index (see :class:`EmbeddingContaminationScanner`).
    """

    def __init__(self, documents: list[CorpusDocument], ngram_size: int = 5) -> None:
        self.ngram_size = ngram_size
        self.documents = [
            IndexedDocument(
                document=doc,
                normalized=normalize_text(doc.text),
                tokens=token_set(doc.text),
                ngrams=char_ngrams(doc.text, ngram_size),
            )
            for doc in documents
        ]

    def search(
        self,
        item: BenchmarkItem,
        top_k: int = 5,
        thresholds: ContaminationThresholds | None = None,
    ) -> list[ContaminationHit]:
        thresholds = thresholds or ContaminationThresholds()
        query_normalized = normalize_text(item.prompt)
        query_tokens = token_set(item.prompt)
        query_ngrams = char_ngrams(item.prompt, self.ngram_size)

        hits: list[ContaminationHit] = []
        for indexed in self.documents:
            token_score = jaccard(query_tokens, indexed.tokens)
            ngram_score = jaccard(query_ngrams, indexed.ngrams)
            overlap_score = weighted_token_overlap(item.prompt, indexed.document.text)
            score = max(token_score, ngram_score, overlap_score)
            label = classify_match(query_normalized, indexed.normalized, score, thresholds)
            if label == "none":
                continue
            hits.append(
                ContaminationHit(
                    benchmark_id=item.id,
                    document_id=indexed.document.id,
                    source=indexed.document.source,
                    label=label,
                    score=round(score, 6),
                    token_jaccard=round(token_score, 6),
                    char_ngram_jaccard=round(ngram_score, 6),
                    excerpt=compact_excerpt(indexed.document.text),
                    metadata={
                        "parent_id": item.parent_id,
                        "variant_type": item.variant_type,
                    },
                )
            )

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def classify_match(
    query_normalized: str,
    document_normalized: str,
    score: float,
    thresholds: ContaminationThresholds,
) -> str:
    if not query_normalized or not document_normalized:
        return "none"
    if query_normalized == document_normalized or query_normalized in document_normalized:
        return "exact"
    if score >= thresholds.exact:
        return "exact"
    if score >= thresholds.near_duplicate:
        return "near_duplicate"
    if score >= thresholds.paraphrase_candidate:
        return "paraphrase_candidate"
    return "none"


def classify_contamination(
    cosine: float,
    ngram_overlap: float,
    thresholds: EmbeddingContaminationThresholds,
) -> str:
    """Return the C / PC / N label for a single (cosine, n-gram) candidate."""

    if cosine > thresholds.contaminated_cosine or ngram_overlap > thresholds.contaminated_ngram:
        return "contaminated"
    if cosine > thresholds.partial_cosine or ngram_overlap > thresholds.partial_ngram:
        return "partial"
    return "clean"


class EmbeddingContaminationScanner:
    """Embedding + n-gram contamination scan implementing the proposal's C/PC/N scheme.

    Uses a :class:`VectorIndex` (FAISS when available) to generate the top-K nearest
    training passages per benchmark item, then computes exact cosine and word-level
    13-gram overlap for each candidate and applies the C/PC/N thresholds. This is the
    production replacement for :class:`LexicalSearchIndex`; both share the same
    ``ContaminationHit`` output contract so downstream code is unchanged.
    """

    def __init__(
        self,
        documents: list[CorpusDocument],
        embedder: Embedder | None = None,
        index_type: str = "flat",
        index_backend: str = "auto",
        thresholds: EmbeddingContaminationThresholds | None = None,
    ) -> None:
        self.documents = {doc.id: doc for doc in documents}
        self.embedder = embedder or build_embedder(backend="auto")
        self.thresholds = thresholds or EmbeddingContaminationThresholds()

        doc_ids = [doc.id for doc in documents]
        doc_vectors = self.embedder.encode([doc.text for doc in documents]) if documents else []
        self.index = VectorIndex(
            dimension=self.embedder.dimension,
            index_type=index_type,
            backend=index_backend,
            embedding_model=self.embedder.name,
        )
        self.index.add(doc_ids, doc_vectors)

    def search(self, item: BenchmarkItem, top_k: int = 20) -> list[ContaminationHit]:
        if not self.documents:
            return []
        query_vector = self.embedder.encode([item.prompt])[0]
        candidates = self.index.search(query_vector, top_k=top_k)

        hits: list[ContaminationHit] = []
        for candidate in candidates:
            document = self.documents.get(candidate.document_id)
            if document is None:
                continue
            cosine = max(-1.0, min(1.0, candidate.score))
            ngram = word_ngram_overlap(item.prompt, document.text, n=self.thresholds.ngram_n)
            label = classify_contamination(cosine, ngram, self.thresholds)
            if label == "clean":
                continue
            hits.append(
                ContaminationHit(
                    benchmark_id=item.id,
                    document_id=document.id,
                    source=document.source,
                    label=label,
                    score=round(cosine, 6),
                    token_jaccard=round(jaccard(token_set(item.prompt), token_set(document.text)), 6),
                    char_ngram_jaccard=round(ngram, 6),
                    excerpt=compact_excerpt(document.text),
                    metadata={
                        "parent_id": item.parent_id,
                        "variant_type": item.variant_type,
                        "cosine": round(cosine, 6),
                        "word_ngram_overlap": round(ngram, 6),
                        "ngram_n": self.thresholds.ngram_n,
                        "embedding_model": self.embedder.name,
                    },
                )
            )
        return hits


def run_contamination_scan(
    benchmark: list[BenchmarkItem],
    corpus: list[CorpusDocument],
    output_path: str | Path,
    top_k: int = 5,
    thresholds: ContaminationThresholds | None = None,
) -> list[ContaminationHit]:
    """Lexical smoke-scale scan (backward compatible)."""

    index = LexicalSearchIndex(corpus)
    rows: list[ContaminationHit] = []
    for item in benchmark:
        rows.extend(index.search(item, top_k=top_k, thresholds=thresholds))
    write_jsonl(output_path, rows)
    return rows


def run_embedding_contamination_scan(
    benchmark: list[BenchmarkItem],
    corpus: list[CorpusDocument],
    output_path: str | Path,
    top_k: int = 20,
    embedder_backend: str = "auto",
    embedding_model: str | None = None,
    index_type: str = "flat",
    index_backend: str = "auto",
    thresholds: EmbeddingContaminationThresholds | None = None,
) -> list[ContaminationHit]:
    """Embedding + 13-gram scan producing C / PC / N labels (proposal Phase 1)."""

    embedder = build_embedder(
        backend=embedder_backend,
        model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
    )
    scanner = EmbeddingContaminationScanner(
        documents=corpus,
        embedder=embedder,
        index_type=index_type,
        index_backend=index_backend,
        thresholds=thresholds,
    )
    rows: list[ContaminationHit] = []
    for item in benchmark:
        rows.extend(scanner.search(item, top_k=top_k))
    write_jsonl(output_path, rows)
    return rows
