from instella_reasoning.attribution_embedding import (
    EmbeddingAttributor,
    Neighbor,
    aggregate_by_skill,
    characterize_neighbors,
    gini,
    run_embedding_attribution,
)
from instella_reasoning.embedding import HashingEmbedder
from instella_reasoning.records import BenchmarkItem, CorpusDocument, read_jsonl


def test_gini_bounds() -> None:
    assert gini([1, 1, 1, 1]) == 0.0  # uniform -> 0
    assert gini([]) == 0.0
    assert gini([0, 0, 0, 10]) > 0.5  # concentrated -> high


def test_characterize_concentrated_vs_diverse() -> None:
    concentrated = characterize_neighbors(
        "q1",
        "arithmetic",
        [Neighbor("d1", "gsm8k_synth", 0.95), Neighbor("d2", "gsm8k_synth", 0.92)],
    )
    assert concentrated.concentration == "concentrated"
    assert concentrated.near_duplicate_share == 1.0
    assert concentrated.top_source == "gsm8k_synth"

    diverse = characterize_neighbors(
        "q2",
        "arithmetic",
        [Neighbor("d1", "openwebmath", 0.4), Neighbor("d2", "code", 0.35), Neighbor("d3", "web", 0.3)],
    )
    assert diverse.concentration == "diverse"
    assert diverse.near_duplicate_share == 0.0


def test_characterize_empty_neighbors() -> None:
    profile = characterize_neighbors("q", "arithmetic", [])
    assert profile.n_neighbors == 0
    assert profile.concentration == "diverse"


def test_attributor_finds_near_duplicate_source() -> None:
    corpus = [
        CorpusDocument(
            "doc_syn",
            "A store has 12 pencils. It sells 5 pencils and then receives 9 more. "
            "How many pencils are in the store now? The answer is 16.",
            "instella_gsm8k_synthetic",
        ),
        CorpusDocument("doc_web", "Tokenizer vocabulary and GPU memory planning.", "web"),
    ]
    attributor = EmbeddingAttributor(corpus, embedder=HashingEmbedder(), near_duplicate_cosine=0.2)
    item = BenchmarkItem(
        "gsm_1",
        "A store has 12 pencils. It sells 5 pencils and then receives 9 more. "
        "How many pencils are in the store now?",
        metadata={"skill": "arithmetic"},
    )
    profile = attributor.profile(item, top_k=2)
    assert profile.top_source == "instella_gsm8k_synthetic"
    assert profile.n_neighbors == 2


def test_run_and_aggregate(tmp_path) -> None:
    corpus = [
        CorpusDocument("d1", "A store has 12 pencils it sells 5 receives 9", "gsm8k_synth"),
        CorpusDocument("d2", "unrelated content about weather", "web"),
    ]
    benchmark = [
        BenchmarkItem("q1", "A store has 12 pencils it sells 5 receives 9", metadata={"skill": "arithmetic"}),
    ]
    out = tmp_path / "attr.jsonl"
    profiles = run_embedding_attribution(
        benchmark, corpus, out, top_k=2, embedder_backend="hashing", near_duplicate_cosine=0.2
    )
    assert len(list(read_jsonl(out))) == 1
    skill_agg = aggregate_by_skill(profiles)
    assert skill_agg[0].skill == "arithmetic"
    assert skill_agg[0].n_items == 1
