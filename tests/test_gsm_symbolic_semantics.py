"""Semantic correctness of auto-derived numeric variants.

The existing ``test_gsm_symbolic.py`` asserts *structural* properties — leaves extracted,
counts, ``None`` on malformed input. All of it passed while the generator was emitting
variants with **wrong gold answers** on 15% of items, because nothing checked that a
variant's stated answer is the answer a human would compute from the variant's own text.

Every test here re-derives the answer independently of the generator and compares. That is
the only kind of test that can catch the failure class these bugs belong to.

The three defects, each with a regression test:

1. a value that plays two roles (question quantity vs structural constant) in one step —
   ``"2 bolts of blue fiber and half that much white"`` with chain ``<<2/2=1>>``;
2. a decimal split into two integers by a ``\\d+`` tokeniser — ``10*1.2`` yielding leaves
   ``{10, 1, 2}``, which rewrote "1.2 times" as "5.2 times";
3. leaves resampled from ``[v//2, 2v+2]`` (expectation 1.25 v), producing variants whose
   median gold answer was 2.61x the original — the magnitude confound of arXiv:2605.28700.
"""

from __future__ import annotations

import random
import re

import pytest

from instella_reasoning.gsm_symbolic import (
    MAGNITUDE_RATIO_BAND,
    build_template,
    magnitude_report,
    make_numeric_variants,
)
from instella_reasoning.records import BenchmarkItem


def _item(prompt: str, answer: str, rationale: str, item_id: str = "probe") -> BenchmarkItem:
    return BenchmarkItem(
        id=item_id,
        prompt=prompt,
        answer=answer,
        parent_id=item_id,
        metadata={"benchmark": "gsm8k", "skill": "arithmetic", "rationale": rationale},
    )


# -- 1. role conflation ----------------------------------------------------------


def test_structural_constant_sharing_a_leaf_value_is_rejected() -> None:
    """'half that much' is a constant 2, not the quantity 2. Value-based substitution
    cannot tell them apart, so the template must be refused rather than corrupted."""
    robe = _item(
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "3",
        "It takes 2/2=<<2/2=1>>1 bolt of white fiber\n"
        "So the total amount of fabric is 2+1=<<2+1=3>>3 bolts\n#### 3",
    )
    assert build_template(robe) is None
    assert make_numeric_variants(robe, k=3, rng=random.Random(0)).n_variants == 0


def test_self_multiplied_value_is_rejected() -> None:
    """``4*4`` is 'cartons x price', where only the price is in the question."""
    cynthia = _item(
        "Cynthia eats one serving of ice cream every night. She buys cartons with 15 "
        "servings at a cost of $4.00 per carton. After 60 days, how much will she spend?",
        "16",
        "She needs 60/15 = 4 containers\n"
        "If each carton costs $4.00 and she needs 4 then it costs 4*4 = $<<4*4=16>>16\n#### 16",
    )
    assert build_template(cynthia) is None


def test_rejection_reason_is_recorded_for_review() -> None:
    robe = _item(
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts?",
        "3",
        "It takes 2/2=<<2/2=1>>1 bolt\nTotal is 2+1=<<2+1=3>>3\n#### 3",
    )
    rejections: list = []
    build_template(robe, rejections)
    assert [r.reason for r in rejections] == ["value_repeated_within_step"]


# -- 2. decimal tokenisation -----------------------------------------------------


def test_decimal_rate_is_held_fixed_not_split() -> None:
    """``1.2`` must survive into the variant unchanged; splitting it rewrites the rate."""
    eliza = _item(
        "Eliza's rate per hour for the first 40 hours she works each week is $10. She also "
        "receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for "
        "45 hours this week, how much are her earnings for this week?",
        "460",
        "Eliza is entitled to 45 -40 = <<45-40=5>>5 hours overtime.\n"
        "Her overtime rate is $10 x 1.2 = $<<10*1.2=12>>12.\n"
        "So she receives $12 x 5 = $<<12*5=60>>60 overtime.\n"
        "Her regular earning is $10 x 40 = $<<10*40=400>>400.\n"
        "Total $400 + $60 = $<<400+60=460>>460\n#### 460",
    )
    template = build_template(eliza)
    assert template is not None
    assert 1.2 in template.constants
    assert 1 not in template.leaf_values and 2 not in template.leaf_values

    for variant in make_numeric_variants(eliza, k=4, rng=random.Random(1)).items:
        assert "1.2 times" in variant.prompt


def test_decimal_variants_have_independently_correct_answers() -> None:
    """Re-solve each variant from its own text and compare with the stated answer."""
    eliza = _item(
        "Eliza's rate per hour for the first 40 hours she works each week is $10. She also "
        "receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for "
        "45 hours this week, how much are her earnings for this week?",
        "460",
        "Eliza is entitled to 45 -40 = <<45-40=5>>5 hours overtime.\n"
        "Her overtime rate is $10 x 1.2 = $<<10*1.2=12>>12.\n"
        "So she receives $12 x 5 = $<<12*5=60>>60 overtime.\n"
        "Her regular earning is $10 x 40 = $<<10*40=400>>400.\n"
        "Total $400 + $60 = $<<400+60=460>>460\n#### 460",
    )
    variants = make_numeric_variants(eliza, k=5, rng=random.Random(2)).items
    assert variants, "expected at least one variant"
    pattern = re.compile(
        r"first (\d+) hours .*? is \$(\d+)\..*?worked for (\d+) hours", re.DOTALL
    )
    for variant in variants:
        match = pattern.search(variant.prompt)
        assert match, variant.prompt
        threshold, rate, worked = (int(g) for g in match.groups())
        expected = rate * threshold + (rate * 1.2) * (worked - threshold)
        assert int(variant.answer) == pytest.approx(expected), variant.prompt


# -- 3. magnitude neutrality -----------------------------------------------------


def _corpus() -> list[BenchmarkItem]:
    return [
        _item(
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins "
            "with 4. She sells the rest at 2 dollars per egg. How much does she make daily?",
            "18",
            "16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n9 * 2 = <<9*2=18>>18 dollars.\n#### 18",
            "a",
        ),
        _item(
            "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace "
            "weighs. What are their combined weights in pounds?",
            "623",
            "Alex weighs 125*4-2 = <<125*4-2=498>>498.\n"
            "Combined 125+498 = <<125+498=623>>623 pounds.\n#### 623",
            "b",
        ),
        _item(
            "A store sells 30 shirts at 12 dollars each on Monday. How much did it take in?",
            "360",
            "30 * 12 = <<30*12=360>>360 dollars.\n#### 360",
            "c",
        ),
    ]


def test_resampling_is_magnitude_neutral() -> None:
    """The median variant/original answer ratio must sit near 1.

    The previous generator produced 2.61x with 85% of variants larger, which is exactly
    the confound that removed the significance of half the models re-analysed in
    arXiv:2605.28700. This is a hard requirement, not a preference.
    """
    produced: list[BenchmarkItem] = []
    for item in _corpus():
        produced.extend(
            make_numeric_variants(item, k=8, rng=random.Random(f"mag:{item.id}")).items
        )
    report = magnitude_report(produced)
    assert report.n >= 10
    low, high = MAGNITUDE_RATIO_BAND
    assert low <= report.median_ratio <= high, report.to_dict()
    assert 0.3 <= report.share_larger <= 0.7, report.to_dict()


def test_magnitude_report_flags_an_inflated_set() -> None:
    inflated = [
        BenchmarkItem(
            id=f"v{i}", prompt="q", answer="1", parent_id="p",
            variant_type="gsm_symbolic",
            metadata={"magnitude_ratio": 2.6, "answer_changing": True},
        )
        for i in range(20)
    ]
    assert magnitude_report(inflated).in_band is False


# -- substitution integrity ------------------------------------------------------


def test_substitution_never_cascades() -> None:
    """Mapping 5->12 then 12->7 must not rewrite the 12 the first substitution produced."""
    item = _item(
        "A box holds 5 pens and a crate holds 12 boxes. How many pens are in a crate?",
        "60",
        "5 * 12 = <<5*12=60>>60 pens.\n#### 60",
    )
    for variant in make_numeric_variants(item, k=8, rng=random.Random(7)).items:
        leaves = variant.metadata["numeric_leaves"]
        numbers = [int(n) for n in re.findall(r"(?<!\d)\d+(?!\d)", variant.prompt)]
        assert numbers == leaves, (variant.prompt, leaves)
        assert int(variant.answer) == leaves[0] * leaves[1]


def test_value_used_more_in_question_than_chain_is_rejected() -> None:
    """'C bakes 4. D has 4 more.' — the chain consumes one 4, the question holds two."""
    item = _item(
        "A has 16. B eats 3. C bakes 4. D has 4 more. Sells at 2. How much?",
        "18",
        "16 - 3 - 4 = <<16-3-4=9>>9.\n9 * 2 = <<9*2=18>>18.\n#### 18",
    )
    rejections: list = []
    assert build_template(item, rejections) is None
    assert rejections[0].reason == "value_ambiguous_in_question"


def test_sentence_final_period_is_not_a_decimal_point() -> None:
    """Boundary regression: ``bakes 4.`` must count as an occurrence of 4."""
    from instella_reasoning.gsm_symbolic import _int_occurrences

    assert _int_occurrences(4, "C bakes 4. D has 4 more.") == 2
    assert _int_occurrences(4, "the price is 1.4 dollars") == 0
    assert _int_occurrences(4, "he has 14 apples") == 0
    assert _int_occurrences(2, "a rate of 1.2 times") == 0


def test_leaves_are_never_resampled_to_one_for_real_quantities() -> None:
    """'1 times' / '1 pounds' reads as broken text and confounds fluency with reasoning."""
    item = _item(
        "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace "
        "weighs. What are their combined weights in pounds?",
        "623",
        "Alex weighs 125*4-2 = <<125*4-2=498>>498.\n"
        "Combined 125+498 = <<125+498=623>>623 pounds.\n#### 623",
    )
    for variant in make_numeric_variants(item, k=10, rng=random.Random(5)).items:
        # 125 and 4 are >2, so neither may collapse to 1.
        leaves = dict(zip(variant.metadata["original_leaves"], variant.metadata["numeric_leaves"], strict=False))
        assert leaves[125] >= 2
        assert leaves[4] >= 2
