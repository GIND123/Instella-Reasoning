import json

from instella_reasoning.datasets.loaders import (
    BENCHMARK_LOADERS,
    _coerce_corpus_text,
    _load_humaneval,
    _load_logiqa,
    _load_reclor,
)


def test_load_logiqa_nested_json_text():
    inner = {
        "text": "One seminar had 18 participants.",
        "question": "Which follows?",
        "options": ["A opt", "B opt", "C opt", "D opt"],
        "answer": 3,
    }
    rows = [{"text": json.dumps(inner)}]
    items = list(_load_logiqa(rows, "logical_deduction"))
    assert len(items) == 1
    it = items[0]
    assert it.answer == "D"
    assert "One seminar had 18 participants." in it.prompt
    assert "Which follows?" in it.prompt
    assert "A. A opt" in it.prompt


def test_load_logiqa_flat_columns():
    rows = [{"context": "ctx", "question": "q", "options": ["x", "y"], "answer": 0}]
    items = list(_load_logiqa(rows, "logical_deduction"))
    assert items[0].answer == "A"
    assert "ctx" in items[0].prompt


def test_reclor_and_humaneval_registered():
    assert "reclor" in BENCHMARK_LOADERS
    assert "humaneval" in BENCHMARK_LOADERS


def test_load_reclor_multiple_choice():
    rows = [
        {
            "context": "All cats are mammals.",
            "question": "Which follows?",
            "answers": ["Cats are mammals", "Cats are birds", "No cats exist", "Mammals are cats"],
            "label": 0,
            "id_string": "x",
        }
    ]
    items = list(_load_reclor(rows, "logical_reading"))
    assert len(items) == 1
    it = items[0]
    assert it.answer == "A"
    assert it.metadata["choices"] == ["A", "B", "C", "D"]
    assert "All cats are mammals." in it.prompt
    assert "A. Cats are mammals" in it.prompt


def test_load_reclor_hidden_label_is_none():
    rows = [{"context": "c", "question": "q", "answers": ["a", "b"], "label": -1}]
    items = list(_load_reclor(rows, "logical_reading"))
    assert items[0].answer is None


def test_load_humaneval_has_no_answer():
    rows = [{"prompt": "def add(a, b):\n    ", "entry_point": "add", "canonical_solution": "x"}]
    items = list(_load_humaneval(rows, "code"))
    assert items[0].answer is None
    assert items[0].metadata["scoring"] == "execution"
    assert items[0].metadata["entry_point"] == "add"


def test_coerce_plain_string():
    assert _coerce_corpus_text("hello world") == "hello world"


def test_coerce_none_is_empty():
    assert _coerce_corpus_text(None) == ""


def test_coerce_chat_messages():
    messages = [
        {"role": "user", "content": "Natalia sold clips to 36 friends."},
        {"role": "assistant", "content": "She sold 54 clips total."},
    ]
    text = _coerce_corpus_text(messages)
    assert "Natalia sold clips to 36 friends." in text
    assert "She sold 54 clips total." in text


def test_coerce_list_of_strings():
    assert _coerce_corpus_text(["a", "b"]) == "a b"


def test_coerce_skips_empty_content():
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "Q"}]
    assert _coerce_corpus_text(messages) == "Q"
