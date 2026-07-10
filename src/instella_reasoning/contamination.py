from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from instella_reasoning.records import BenchmarkItem, ContaminationHit, CorpusDocument, write_jsonl
from instella_reasoning.text import (
    char_ngrams,
    compact_excerpt,
    jaccard,
    normalize_text,
    token_set,
    weighted_token_overlap,
)


@dataclass(slots=True)
class ContaminationThresholds:
    exact: float = 0.98
    near_duplicate: float = 0.72
    paraphrase_candidate: float = 0.45


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
    distributed embedding index.
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


def run_contamination_scan(
    benchmark: list[BenchmarkItem],
    corpus: list[CorpusDocument],
    output_path: str | Path,
    top_k: int = 5,
    thresholds: ContaminationThresholds | None = None,
) -> list[ContaminationHit]:
    index = LexicalSearchIndex(corpus)
    rows: list[ContaminationHit] = []
    for item in benchmark:
        rows.extend(index.search(item, top_k=top_k, thresholds=thresholds))
    write_jsonl(output_path, rows)
    return rows
