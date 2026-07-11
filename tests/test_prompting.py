from instella_reasoning.prompting import (
    build_prompt,
    extract_answer,
    extract_boxed,
    extract_multiple_choice,
    extract_numeric,
    extract_yes_no,
    extractor_for,
)
from instella_reasoning.records import BenchmarkItem


def _item(**meta) -> BenchmarkItem:
    return BenchmarkItem("id", "prompt text", answer=meta.pop("answer", "1"), metadata=meta)


def test_extract_numeric_prefers_hash_answer() -> None:
    completion = "First 12-5=7, then 7+9=16.\n#### 16"
    assert extract_numeric(completion) == "16"


def test_extract_numeric_strips_commas_and_dollar() -> None:
    assert extract_numeric("The answer is $1,234.") == "1234"


def test_extract_numeric_falls_back_to_last_number() -> None:
    assert extract_numeric("we get 3 apples and 5 oranges") == "5"


def test_extract_boxed() -> None:
    assert extract_boxed("Thus the value is \\boxed{42}.") == "42"
    assert extract_boxed("no box here #### 7") == "7"


def test_extract_multiple_choice_letter() -> None:
    assert extract_multiple_choice("I think it is option C.\n#### C") == "C"
    assert extract_multiple_choice("The answer is B because ...") == "B"


def test_extract_yes_no() -> None:
    assert extract_yes_no("Reasoning...\n#### Yes") == "yes"
    assert extract_yes_no("No, that cannot be true.") == "no"


def test_extractor_selection_by_metadata() -> None:
    assert extractor_for(_item(benchmark="gsm8k", skill="arithmetic")).name == "numeric"
    assert extractor_for(_item(benchmark="math", skill="mathematical")).name == "boxed"
    assert extractor_for(_item(benchmark="arc_challenge", choices=["A", "B"])).name == "multiple_choice"


def test_build_prompt_adds_cot_suffix() -> None:
    item = _item(benchmark="gsm8k", skill="arithmetic")
    prompt = build_prompt(item)
    assert "####" in prompt
    assert prompt.startswith("prompt text")


def test_extract_answer_end_to_end() -> None:
    item = _item(benchmark="gsm8k", skill="arithmetic")
    assert extract_answer(item, "steps...\n#### 16") == "16"
