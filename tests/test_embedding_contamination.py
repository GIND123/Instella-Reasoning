from instella_reasoning.contamination import (
    EmbeddingContaminationScanner,
    EmbeddingContaminationThresholds,
    classify_contamination,
)
from instella_reasoning.embedding import HashingEmbedder, build_embedder
from instella_reasoning.faiss_index import VectorIndex
from instella_reasoning.records import BenchmarkItem, CorpusDocument
from instella_reasoning.text import word_ngram_overlap, word_ngrams


def test_hashing_embedder_is_normalized_and_deterministic() -> None:
    embedder = HashingEmbedder()
    a = embedder.encode(["a store has 12 pencils"])[0]
    b = embedder.encode(["a store has 12 pencils"])[0]
    assert a == b
    norm = sum(value * value for value in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_build_embedder_hashing_backend() -> None:
    embedder = build_embedder(backend="hashing")
    assert embedder.dimension > 0
    assert len(embedder.encode(["x", "y"])) == 2


def test_vector_index_bruteforce_ranks_nearest_first() -> None:
    embedder = HashingEmbedder()
    docs = ["a store has 12 pencils", "the mitochondria is the powerhouse"]
    vectors = embedder.encode(docs)
    index = VectorIndex(dimension=embedder.dimension, backend="bruteforce")
    index.add(["d0", "d1"], vectors)

    query = embedder.encode(["a store has 12 pencils"])[0]
    results = index.search(query, top_k=2)
    assert results[0].document_id == "d0"
    assert results[0].score >= results[1].score


def test_vector_index_save_load_roundtrip(tmp_path) -> None:
    embedder = HashingEmbedder()
    vectors = embedder.encode(["alpha beta", "gamma delta"])
    index = VectorIndex(dimension=embedder.dimension, backend="bruteforce")
    index.add(["a", "b"], vectors)
    index.save(tmp_path / "idx")

    reloaded = VectorIndex.load(tmp_path / "idx", backend="bruteforce")
    assert reloaded.document_ids == ["a", "b"]
    query = embedder.encode(["alpha beta"])[0]
    assert reloaded.search(query, top_k=1)[0].document_id == "a"


def test_word_ngram_overlap_detects_verbatim_substring() -> None:
    prompt = "a store has 12 pencils it sells 5 and receives 9 how many are there"
    document = "intro text " + prompt + " the answer is 16"
    assert word_ngram_overlap(prompt, document, n=5) == 1.0
    assert word_ngrams("one two", n=13)  # short input falls back to whole sequence


def test_classify_contamination_thresholds() -> None:
    t = EmbeddingContaminationThresholds()
    assert classify_contamination(0.95, 0.0, t) == "contaminated"
    assert classify_contamination(0.0, 0.7, t) == "contaminated"
    assert classify_contamination(0.80, 0.0, t) == "partial"
    assert classify_contamination(0.0, 0.4, t) == "partial"
    assert classify_contamination(0.5, 0.1, t) == "clean"


def test_embedding_scanner_labels_near_duplicate() -> None:
    corpus = [
        CorpusDocument(
            "doc_syn",
            "A store has 12 pencils. It sells 5 pencils and then receives 9 more. "
            "How many pencils are in the store now? The answer is 16.",
            "instella_gsm8k_synthetic",
        ),
        CorpusDocument("doc_unrel", "Tokenizer vocabulary and GPU memory planning.", "web"),
    ]
    scanner = EmbeddingContaminationScanner(
        documents=corpus,
        embedder=HashingEmbedder(),
        # Hashing embedder cosine is lower than a real encoder, so rely on the
        # directional 13-gram overlap to flag the verbatim training document.
        thresholds=EmbeddingContaminationThresholds(partial_cosine=0.99, contaminated_cosine=0.999),
    )
    item = BenchmarkItem(
        "gsm_1",
        "A store has 12 pencils. It sells 5 pencils and then receives 9 more. "
        "How many pencils are in the store now?",
    )
    hits = scanner.search(item, top_k=5)
    assert hits
    assert hits[0].document_id == "doc_syn"
    assert hits[0].label in {"contaminated", "partial"}
