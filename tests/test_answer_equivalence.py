from __future__ import annotations

import pytest

from instella_reasoning.answer_equivalence import (
    answers_equivalent,
    numeric_equal,
    numeric_value,
    resolve_answer_kind,
    symbolic_equal,
    sympy_available,
)
from instella_reasoning.records import BenchmarkItem


def test_normalized_string_match_is_kind_independent() -> None:
    assert answers_equivalent("Yes", "yes", "yes_no")
    assert answers_equivalent("B", "b", "multiple_choice")
    assert not answers_equivalent("A", "B", "multiple_choice")


def test_numeric_value_parses_fractions_decimals_currency_percent() -> None:
    assert numeric_value("1/2") == pytest.approx(0.5)
    assert numeric_value("$1,200") == pytest.approx(1200.0)
    assert numeric_value("50%") == pytest.approx(0.5)
    assert numeric_value(r"\frac{3}{4}") == pytest.approx(0.75)
    assert numeric_value("not a number") is None


def test_numeric_equal_recovers_format_differences() -> None:
    assert numeric_equal("0.5", "1/2")
    assert numeric_equal("18", "18.0")
    assert not numeric_equal("18", "19")


def test_answers_equivalent_numeric_kind_uses_numeric_path() -> None:
    # These differ as strings but are the same number; numeric kind must accept them.
    assert answers_equivalent("1/2", "0.5", "numeric")
    # A text kind does not invoke the numeric path, so unequal strings stay unequal.
    assert not answers_equivalent("1/2", "0.5", "text")


def test_answers_equivalent_is_conservative_on_missing() -> None:
    assert not answers_equivalent(None, "1", "numeric")
    assert not answers_equivalent("1", None, "numeric")


def test_resolve_answer_kind_from_metadata() -> None:
    math_item = BenchmarkItem(id="m", prompt="p", answer="2", metadata={"benchmark": "math"})
    gsm_item = BenchmarkItem(id="g", prompt="p", answer="2", metadata={"benchmark": "gsm8k"})
    mc_item = BenchmarkItem(id="a", prompt="p", answer="B", metadata={"benchmark": "arc_challenge"})
    yn_item = BenchmarkItem(id="y", prompt="p", answer="yes", metadata={"benchmark": "bbh"})
    assert resolve_answer_kind(math_item) == "math"
    assert resolve_answer_kind(gsm_item) == "numeric"
    assert resolve_answer_kind(mc_item) == "multiple_choice"
    assert resolve_answer_kind(yn_item) == "yes_no"


@pytest.mark.skipif(not sympy_available(), reason="sympy not installed")
def test_symbolic_equal_handles_latex_math() -> None:
    assert symbolic_equal(r"\frac{1}{2}", "0.5")
    assert symbolic_equal(r"\sqrt{4}", "2")
    assert symbolic_equal(r"\dfrac{2}{4}", r"\frac{1}{2}")
    assert not symbolic_equal(r"\frac{1}{2}", r"\frac{1}{3}")


@pytest.mark.skipif(not sympy_available(), reason="sympy not installed")
def test_answers_equivalent_math_kind_uses_symbolic_path() -> None:
    assert answers_equivalent(r"\frac{1}{2}", "0.5", "math")
    assert answers_equivalent("2", r"\sqrt{4}", "math")


def test_symbolic_equal_survives_unparseable_garbage():
    # Degenerate model output with stray backslashes made sympy's tokenizer raise
    # tokenize.TokenError (not in the old except-list), crashing a whole scoring run.
    from instella_reasoning.answer_equivalence import answers_equivalent, symbolic_equal

    garbage = "\\text{the answer is } 42 \\\\ \\x"
    assert symbolic_equal("1", garbage) is False
    assert answers_equivalent("1", garbage, kind="math") is False
