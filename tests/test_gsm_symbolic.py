"""GSM-Symbolic-lite templater: validated numeric variants with recomputed answers."""
import random

from instella_reasoning.gsm_symbolic import build_template, make_numeric_variants
from instella_reasoning.records import BenchmarkItem


def _janet() -> BenchmarkItem:
    return BenchmarkItem(
        id="gsm8k_0",
        prompt=(
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins "
            "with 4. She sells the rest at 2 dollars per egg. How much does she make daily?"
        ),
        answer="18",
        parent_id="gsm8k_0",
        metadata={
            "benchmark": "gsm8k",
            "skill": "arithmetic",
            "rationale": "16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n9 * 2 = <<9*2=18>>18 dollars.\n#### 18",
        },
    )


def test_build_template_extracts_leaves_and_results() -> None:
    t = build_template(_janet())
    assert t is not None
    assert t.leaf_values == [16, 3, 4, 2]
    assert t.step_results == [9, 18]
    assert t.original_answer == 18


def test_no_annotations_yields_no_template() -> None:
    item = _janet()
    item.metadata["rationale"] = "She makes 18 dollars. #### 18"  # no <<>> annotations
    assert build_template(item) is None


def test_broken_annotation_fails_round_trip() -> None:
    item = _janet()
    item.metadata["rationale"] = "16 - 3 - 4 = <<16-3-4=10>>10.\n#### 10"  # 16-3-4 != 10
    assert build_template(item) is None


def test_numeric_variants_have_correct_recomputed_answers() -> None:
    item = _janet()
    result = make_numeric_variants(item, k=5, rng=random.Random(0))
    assert result.n_templated == 1
    assert result.n_variants >= 1
    for v in result.items:
        assert v.variant_type == "gsm_symbolic"
        assert v.parent_id == "gsm8k_0"
        assert v.metadata["answer_changing"] is True
        # Independently recompute from the substituted numbers in the new prompt.
        leaves = v.metadata["numeric_leaves"]  # [a, b, c, d] for 16,3,4,2 slots
        a, b, c, d = leaves
        assert int(v.answer) == (a - b - c) * d
        assert int(v.answer) > 0


def test_ambiguous_leaf_appearing_twice_is_skipped() -> None:
    # The value 4 appears twice in the question -> substitution is ambiguous -> no template.
    item = _janet()
    item.prompt = "A has 16. B eats 3. C bakes 4. D has 4 more. Sells at 2. How much?"
    item.metadata["rationale"] = "16 - 3 - 4 = <<16-3-4=9>>9.\n9 * 2 = <<9*2=18>>18.\n#### 18"
    assert build_template(item) is None
