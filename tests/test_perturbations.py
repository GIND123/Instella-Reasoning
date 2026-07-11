import random

from instella_reasoning.perturbations import (
    PerturbationConfig,
    entity_substitution,
    expand_benchmark_file,
    irrelevant_context,
    make_variants,
    numeric_perturbation,
    premise_reordering,
    rephrasing,
)
from instella_reasoning.records import BenchmarkItem, read_benchmark


def _gsm_item() -> BenchmarkItem:
    return BenchmarkItem(
        id="gsm_1",
        prompt="Alice has 12 pencils. She sells 5 pencils and then receives 9 more. "
        "How many pencils are there now?",
        answer="16",
        parent_id="gsm_1",
        metadata={"benchmark": "gsm8k", "skill": "arithmetic"},
    )


def test_entity_substitution_preserves_answer_and_changes_prompt() -> None:
    item = _gsm_item()
    variant = entity_substitution(item, random.Random(0))
    assert variant is not None
    assert variant.answer == "16"
    assert variant.parent_id == "gsm_1"
    assert variant.variant_type == "entity_substitution"
    assert "Alice" not in variant.prompt or "pencils" not in variant.prompt
    # numbers must be untouched so the answer stays valid
    assert "12" in variant.prompt and "5" in variant.prompt and "9" in variant.prompt


def test_premise_reordering_keeps_question_last_and_answer() -> None:
    item = _gsm_item()
    variant = premise_reordering(item, random.Random(1))
    assert variant is not None
    assert variant.prompt.strip().endswith("?")
    assert variant.answer == "16"


def test_irrelevant_context_adds_a_distractor() -> None:
    item = _gsm_item()
    variant = irrelevant_context(item, random.Random(2))
    assert variant is not None
    assert len(variant.prompt) > len(item.prompt)
    assert variant.answer == "16"


def test_rephrasing_changes_surface_form_only() -> None:
    item = _gsm_item()
    variant = rephrasing(item, random.Random(3))
    assert variant is not None
    assert variant.prompt != item.prompt
    assert variant.answer == "16"


def test_numeric_perturbation_recomputes_answer() -> None:
    item = BenchmarkItem(
        id="tmpl_1",
        prompt="A store has 12 pencils. It sells 5. How many are left?",
        answer="7",
        parent_id="tmpl_1",
        metadata={
            "numeric_template": {
                "template": "A store has {a} pencils. It sells {b}. How many are left?",
                "slots": {"a": {"low": 10, "high": 20}, "b": {"choices": [1, 2, 3, 4, 5]}},
                "answer_expr": "a - b",
            }
        },
    )
    variant = numeric_perturbation(item, random.Random(7))
    assert variant is not None
    assert variant.variant_type == "numeric_perturbation"
    # recomputed answer must equal a - b for the sampled values
    values = variant.metadata["numeric_values"]
    assert int(variant.answer) == int(values["a"] - values["b"])


def test_numeric_perturbation_skips_without_template() -> None:
    assert numeric_perturbation(_gsm_item(), random.Random(0)) is None


def test_make_variants_is_deterministic() -> None:
    item = _gsm_item()
    a = make_variants(item, PerturbationConfig(seed=42))
    b = make_variants(item, PerturbationConfig(seed=42))
    assert [v.prompt for v in a] == [v.prompt for v in b]
    assert all(v.parent_id == "gsm_1" for v in a)


def test_expand_benchmark_file_roundtrip(tmp_path) -> None:
    from instella_reasoning.records import write_jsonl

    src = tmp_path / "bench.jsonl"
    write_jsonl(src, [_gsm_item()])
    out = tmp_path / "suite.jsonl"
    count = expand_benchmark_file(src, out, PerturbationConfig(seed=1))
    suite = read_benchmark(out)
    assert count == len(suite)
    # original + at least one variant, all sharing the parent
    assert count >= 2
    assert {v.parent_id for v in suite} == {"gsm_1"}
    assert any(v.variant_type == "original" for v in suite)
