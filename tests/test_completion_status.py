from instella_reasoning.evaluation import _finalize_completion


def test_eos_marks_completion_finished() -> None:
    text, finished = _finalize_completion("The answer is 7.", emitted_eos=True, n_shot=0)
    assert text == "The answer is 7."
    assert finished


def test_final_answer_marker_is_a_semantic_stop() -> None:
    text, finished = _finalize_completion("work\n#### 7", emitted_eos=False, n_shot=0)
    assert text == "work\n#### 7"
    assert finished


def test_next_few_shot_exemplar_is_trimmed_and_finished() -> None:
    text, finished = _finalize_completion(
        "work\n#### 7\nQuestion: another problem", emitted_eos=False, n_shot=4
    )
    assert text == "work\n#### 7"
    assert finished


def test_token_cap_without_semantic_stop_is_unfinished() -> None:
    text, finished = _finalize_completion("still reasoning", emitted_eos=False, n_shot=0)
    assert text == "still reasoning"
    assert not finished
