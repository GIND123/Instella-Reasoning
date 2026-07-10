from instella_reasoning.contamination import LexicalSearchIndex
from instella_reasoning.records import BenchmarkItem, CorpusDocument


def test_lexical_index_finds_exact_match() -> None:
    index = LexicalSearchIndex(
        [
            CorpusDocument("doc1", "A store has 12 pencils and sells 5.", "test"),
            CorpusDocument("doc2", "Unrelated text.", "test"),
        ]
    )

    hits = index.search(BenchmarkItem("q1", "A store has 12 pencils and sells 5."))

    assert hits
    assert hits[0].document_id == "doc1"
    assert hits[0].label == "exact"
