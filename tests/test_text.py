from instella_reasoning.text import extract_final_answer, normalize_answer


def test_extract_final_answer_prefers_gsm_marker() -> None:
    assert extract_final_answer("Reasoning here. #### 16") == "16"


def test_normalize_answer_handles_articles_and_punctuation() -> None:
    assert normalize_answer("The answer is: Yes.") == "yes"


def test_normalize_answer_handles_leading_yes_no() -> None:
    assert normalize_answer("Yes. If dax implies wug, the answer is yes.") == "yes"
