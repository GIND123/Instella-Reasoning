"""Tier-1 embedding-based training-data attribution (proposal Section 6.4).

For each benchmark item, retrieve its nearest training passages and characterise
them: is the influence *concentrated* on a few near-duplicates (a memorisation
signature) or *diverse* across many thematically related documents (a generalisation
signature)? This is the cheap, always-available attribution tier; TracIn-CP and
Concept Influence (:mod:`instella_reasoning.attribution_gradient`) refine the most
interesting items later.

Signatures computed per item:

- ``near_duplicate_share`` — fraction of the top-k neighbours with cosine >= the
  near-duplicate threshold. The plan flags ``>= 0.30`` as a memorisation signal.
- ``gini`` — Gini coefficient of the neighbour similarity mass (0 = uniform / diverse,
  1 = all mass on one document / concentrated).
- ``source_shares`` — similarity-weighted share of each training data source, so an
  item's reasoning can be traced to OpenWebMath vs GSM8K-synthetic vs code, etc.
- ``concentration`` — ``"concentrated"`` or ``"diverse"`` verdict.

Runs on the same portable stack as contamination search: real MiniLM/GTE + FAISS
with the ``retrieval`` extra, or the dependency-free hashing embedder + brute-force
index otherwise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from instella_reasoning.embedding import Embedder, build_embedder
from instella_reasoning.faiss_index import VectorIndex
from instella_reasoning.records import BenchmarkItem, CorpusDocument, write_jsonl

NEAR_DUPLICATE_COSINE = 0.85
CONCENTRATION_SHARE = 0.30


@dataclass(slots=True)
class Neighbor:
    document_id: str
    source: str
    cosine: float


@dataclass(slots=True)
class AttributionProfile:
    benchmark_id: str
    skill: str
    n_neighbors: int
    mean_cosine: float
    top1_cosine: float
    near_duplicate_share: float
    gini: float
    concentration: str
    top_source: str
    source_shares: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "skill": self.skill,
            "n_neighbors": self.n_neighbors,
            "mean_cosine": round(self.mean_cosine, 6),
            "top1_cosine": round(self.top1_cosine, 6),
            "near_duplicate_share": round(self.near_duplicate_share, 6),
            "gini": round(self.gini, 6),
            "concentration": self.concentration,
            "top_source": self.top_source,
            "source_shares": {k: round(v, 6) for k, v in self.source_shares.items()},
        }


@dataclass(slots=True)
class SkillAttribution:
    skill: str
    n_items: int
    mean_near_duplicate_share: float
    mean_gini: float
    concentrated_share: float
    source_shares: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "n_items": self.n_items,
            "mean_near_duplicate_share": round(self.mean_near_duplicate_share, 6),
            "mean_gini": round(self.mean_gini, 6),
            "concentrated_share": round(self.concentrated_share, 6),
            "source_shares": {k: round(v, 6) for k, v in self.source_shares.items()},
        }


def gini(values: list[float]) -> float:
    """Gini coefficient of non-negative values (0 = perfectly uniform, ->1 = concentrated)."""
    clean = [max(0.0, value) for value in values]
    total = sum(clean)
    n = len(clean)
    if n == 0 or total == 0:
        return 0.0
    ordered = sorted(clean)
    cumulative = 0.0
    for i, value in enumerate(ordered, start=1):
        cumulative += i * value
    return (2 * cumulative) / (n * total) - (n + 1) / n


def characterize_neighbors(
    benchmark_id: str,
    skill: str,
    neighbors: list[Neighbor],
    near_duplicate_cosine: float = NEAR_DUPLICATE_COSINE,
    concentration_share: float = CONCENTRATION_SHARE,
) -> AttributionProfile:
    if not neighbors:
        return AttributionProfile(
            benchmark_id, skill, 0, 0.0, 0.0, 0.0, 0.0, "diverse", "none", {}
        )
    cosines = [n.cosine for n in neighbors]
    near_dupes = sum(1 for c in cosines if c >= near_duplicate_cosine)
    near_duplicate_share = near_dupes / len(neighbors)

    weights: dict[str, float] = defaultdict(float)
    total_weight = 0.0
    for neighbor in neighbors:
        weight = max(0.0, neighbor.cosine)
        weights[neighbor.source] += weight
        total_weight += weight
    if total_weight > 0:
        source_shares = {source: value / total_weight for source, value in weights.items()}
    else:
        # Degenerate (all-zero similarity): fall back to uniform counts by source.
        counts: dict[str, float] = defaultdict(float)
        for neighbor in neighbors:
            counts[neighbor.source] += 1
        source_shares = {source: value / len(neighbors) for source, value in counts.items()}
    top_source = max(source_shares, key=source_shares.get)

    concentration = (
        "concentrated"
        if near_duplicate_share >= concentration_share
        else "diverse"
    )
    return AttributionProfile(
        benchmark_id=benchmark_id,
        skill=skill,
        n_neighbors=len(neighbors),
        mean_cosine=sum(cosines) / len(cosines),
        top1_cosine=max(cosines),
        near_duplicate_share=near_duplicate_share,
        gini=gini(cosines),
        concentration=concentration,
        top_source=top_source,
        source_shares=dict(sorted(source_shares.items(), key=lambda kv: kv[1], reverse=True)),
    )


class EmbeddingAttributor:
    """Retrieve and characterise the nearest training neighbours of each benchmark item."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        embedder: Embedder | None = None,
        index_type: str = "flat",
        index_backend: str = "auto",
        near_duplicate_cosine: float = NEAR_DUPLICATE_COSINE,
        concentration_share: float = CONCENTRATION_SHARE,
    ) -> None:
        self.documents = {doc.id: doc for doc in documents}
        self.embedder = embedder or build_embedder(backend="auto")
        self.near_duplicate_cosine = near_duplicate_cosine
        self.concentration_share = concentration_share
        doc_ids = [doc.id for doc in documents]
        vectors = self.embedder.encode([doc.text for doc in documents]) if documents else []
        self.index = VectorIndex(
            dimension=self.embedder.dimension,
            index_type=index_type,
            backend=index_backend,
            embedding_model=self.embedder.name,
        )
        self.index.add(doc_ids, vectors)

    def neighbors(self, item: BenchmarkItem, top_k: int = 20) -> list[Neighbor]:
        if not self.documents:
            return []
        query = self.embedder.encode([item.prompt])[0]
        results = self.index.search(query, top_k=top_k)
        neighbors: list[Neighbor] = []
        for result in results:
            document = self.documents.get(result.document_id)
            if document is None:
                continue
            neighbors.append(
                Neighbor(document.id, document.source, max(-1.0, min(1.0, result.score)))
            )
        return neighbors

    def profile(self, item: BenchmarkItem, top_k: int = 20) -> AttributionProfile:
        skill = str(item.metadata.get("skill", "unknown"))
        return characterize_neighbors(
            item.id,
            skill,
            self.neighbors(item, top_k=top_k),
            near_duplicate_cosine=self.near_duplicate_cosine,
            concentration_share=self.concentration_share,
        )


def run_embedding_attribution(
    benchmark: list[BenchmarkItem],
    corpus: list[CorpusDocument],
    output_path: str | Path,
    top_k: int = 20,
    embedder_backend: str = "auto",
    embedding_model: str | None = None,
    index_type: str = "flat",
    index_backend: str = "auto",
    near_duplicate_cosine: float = NEAR_DUPLICATE_COSINE,
) -> list[AttributionProfile]:
    """Build per-item attribution profiles and write them to JSONL."""
    embedder = build_embedder(
        backend=embedder_backend,
        model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
    )
    attributor = EmbeddingAttributor(
        documents=corpus,
        embedder=embedder,
        index_type=index_type,
        index_backend=index_backend,
        near_duplicate_cosine=near_duplicate_cosine,
    )
    profiles = [attributor.profile(item, top_k=top_k) for item in benchmark]
    write_jsonl(output_path, [profile.to_dict() for profile in profiles])
    return profiles


def aggregate_by_skill(profiles: list[AttributionProfile]) -> list[SkillAttribution]:
    """Aggregate per-item profiles into a per-sub-skill attribution signature."""
    grouped: dict[str, list[AttributionProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.skill].append(profile)

    aggregates: list[SkillAttribution] = []
    for skill, group in sorted(grouped.items()):
        n = len(group)
        source_weights: dict[str, float] = defaultdict(float)
        for profile in group:
            for source, share in profile.source_shares.items():
                source_weights[source] += share
        total = sum(source_weights.values()) or 1.0
        source_shares = {
            source: weight / total
            for source, weight in sorted(source_weights.items(), key=lambda kv: kv[1], reverse=True)
        }
        aggregates.append(
            SkillAttribution(
                skill=skill,
                n_items=n,
                mean_near_duplicate_share=sum(p.near_duplicate_share for p in group) / n,
                mean_gini=sum(p.gini for p in group) / n,
                concentrated_share=sum(1 for p in group if p.concentration == "concentrated") / n,
                source_shares=source_shares,
            )
        )
    return aggregates
