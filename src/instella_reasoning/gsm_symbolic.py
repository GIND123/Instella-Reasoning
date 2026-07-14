"""GSM-Symbolic-lite: auto-derive validated numeric variants of GSM8K items.

GSM-Symbolic (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) shows the decisive
robustness signal is changing *numerical values* (models are largely robust to name
changes), with correctness guaranteed by symbolic templates. Their templates are hand
annotated. This module derives them automatically from GSM8K's built-in
``<<a op b=c>>`` calculator annotations, then resamples the input numbers and
*recomputes* the ground-truth answer — so each variant carries a correct new label
rather than a guessed one.

Precision over recall, by design. We emit a variant only when:

1. the annotated calc chain parses into steps whose results we can recompute;
2. re-running the chain on the *original* leaf numbers reproduces every annotated
   intermediate result and the final ``#### answer`` (round-trip validation — the
   analogue of GSM-Symbolic's "verify original values satisfy the conditions");
3. every leaf constant appears **exactly once** in the question (unambiguous
   substitution) and no leaf value collides with an intermediate/result value.

Items failing any check are skipped, not corrupted. Resampling keeps all
intermediates and the final answer positive integers (the GSM8K invariant).
"""

from __future__ import annotations

import ast
import operator
import random
import re
from dataclasses import dataclass, field

from instella_reasoning.records import BenchmarkItem

_STEP = re.compile(r"<<\s*([^=<>]+?)\s*=\s*([^<>]+?)\s*>>")
_INT_TOKEN = re.compile(r"\d+")
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@dataclass(slots=True)
class SymbolicTemplate:
    benchmark_id: str
    question: str
    original_answer: int
    leaf_values: list[int]  # distinct input constants, in first-appearance order
    step_exprs: list[str]  # raw calc expressions, e.g. "16-3-4"
    step_results: list[int]  # annotated result of each step

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "leaf_values": self.leaf_values,
            "step_exprs": self.step_exprs,
            "step_results": self.step_results,
            "original_answer": self.original_answer,
        }


def _eval_arith(expr: str) -> float | None:
    """Evaluate a pure-number arithmetic expression with +,-,*,/ only."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_node(node.operand)
        return None if operand is None else -operand
    return None


def _int_or_none(value: str) -> int | None:
    try:
        f = float(value.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None
    return int(f) if f.is_integer() else None


def build_template(item: BenchmarkItem) -> SymbolicTemplate | None:
    """Derive a validated symbolic template from a GSM8K item, or None if not safe."""
    rationale = str(item.metadata.get("rationale", ""))
    raw_steps = _STEP.findall(rationale)
    if not raw_steps or len(raw_steps) > 8:
        return None

    step_exprs: list[str] = []
    step_results: list[int] = []
    for expr, result in raw_steps:
        r = _int_or_none(result)
        if r is None:
            return None
        step_exprs.append(expr.strip())
        step_results.append(r)

    # Round-trip: recompute each expression as pure arithmetic and require it matches
    # the annotated result. This confirms the annotations form a faithful integer chain.
    for expr, expected in zip(step_exprs, step_results, strict=False):
        value = _eval_arith(expr)
        if value is None or not float(value).is_integer() or int(value) != expected:
            return None

    # Final answer must equal the last step result.
    answer = _int_or_none(str(item.answer or ""))
    if answer is None or answer != step_results[-1]:
        return None

    # Leaf constants: integers appearing in expressions that are not a prior step result.
    prior_results: set[int] = set()
    leaves: list[int] = []
    seen_leaves: set[int] = set()
    for expr, res in zip(step_exprs, step_results, strict=False):
        for tok in _INT_TOKEN.findall(expr):
            value = int(tok)
            if value in prior_results:
                continue
            if value not in seen_leaves:
                seen_leaves.add(value)
                leaves.append(value)
        prior_results.add(res)

    if not leaves or len(leaves) > 6:
        return None
    # No leaf may collide with an intermediate/result value, and results must be distinct,
    # so a number token in an expression maps unambiguously to either a leaf or a prior
    # step result (recomputation substitutes by value).
    if seen_leaves & set(step_results):
        return None
    if len(set(step_results)) != len(step_results):
        return None
    # Each leaf must appear exactly once as a standalone integer in the question.
    for value in leaves:
        if len(re.findall(rf"(?<!\d){value}(?!\d)", item.prompt)) != 1:
            return None

    return SymbolicTemplate(
        benchmark_id=item.id,
        question=item.prompt,
        original_answer=answer,
        leaf_values=leaves,
        step_exprs=step_exprs,
        step_results=step_results,
    )


def _recompute(
    step_exprs: list[str], step_results: list[int], mapping: dict[int, int]
) -> tuple[list[int], int] | None:
    """Re-evaluate the chain, substituting leaf values via ``mapping`` and prior results.

    A number token in an expression that equals a *previous step's original result* is a
    prior-result reference and is replaced with that step's *recomputed* value; otherwise
    it is a leaf, replaced via ``mapping``. (Leaves are disjoint from results and results
    are distinct — enforced in :func:`build_template` — so this mapping is unambiguous.)
    Returns (results, final) with all values positive integers, or None if a step is invalid.
    """
    old_to_new: dict[int, int] = {}
    results: list[int] = []
    for expr, old_result in zip(step_exprs, step_results, strict=False):
        def _sub(match: re.Match) -> str:
            value = int(match.group(0))
            if value in old_to_new:  # references a prior step's result -> use recomputed value
                return str(old_to_new[value])
            return str(mapping.get(value, value))  # leaf constant

        new_expr = _INT_TOKEN.sub(_sub, expr)
        value = _eval_arith(new_expr)
        if value is None or not float(value).is_integer():
            return None
        ivalue = int(value)
        if ivalue <= 0:
            return None
        old_to_new[old_result] = ivalue
        results.append(ivalue)
    return results, results[-1]


@dataclass(slots=True)
class NumericVariantResult:
    items: list[BenchmarkItem] = field(default_factory=list)
    n_templated: int = 0  # 1 if the item yielded a template, else 0
    n_variants: int = 0


def make_numeric_variants(
    item: BenchmarkItem, k: int = 3, rng: random.Random | None = None, max_tries: int = 40
) -> NumericVariantResult:
    """Produce up to ``k`` validated numeric variants of a GSM8K item.

    Each variant resamples the leaf constants, recomputes the answer, and substitutes
    the new numbers into the question. Variants carry ``variant_type='gsm_symbolic'`` and
    the same ``parent_id`` so the reliability metric treats them as an answer-*changing*
    cluster (scored by correctness, not answer agreement).
    """
    rng = rng or random.Random(f"gsm_symbolic:{item.id}")
    template = build_template(item)
    result = NumericVariantResult()
    if template is None:
        return result
    result.n_templated = 1

    seen: set[tuple[int, ...]] = {tuple(template.leaf_values)}
    for _ in range(max_tries):
        if len(result.items) >= k:
            break
        mapping: dict[int, int] = {}
        for value in template.leaf_values:
            low = max(1, value // 2)
            high = value * 2 + 2
            mapping[value] = rng.randint(low, high)
        new_leaves = tuple(mapping[v] for v in template.leaf_values)
        if new_leaves in seen:
            continue
        # distinct new leaf values, no collision with any recomputed result
        if len(set(new_leaves)) != len(new_leaves):
            continue
        recomputed = _recompute(template.step_exprs, template.step_results, mapping)
        if recomputed is None:
            continue
        results, final = recomputed
        if set(new_leaves) & set(results):
            continue  # ambiguity between a leaf value and an intermediate
        seen.add(new_leaves)

        new_question = template.question
        ok = True
        for old_value in template.leaf_values:
            pattern = rf"(?<!\d){old_value}(?!\d)"
            new_question, n = re.subn(pattern, str(mapping[old_value]), new_question, count=1)
            if n != 1:
                ok = False
                break
        if not ok:
            continue

        idx = len(result.items)
        parent = item.parent_id or item.id
        result.items.append(
            BenchmarkItem(
                id=f"{parent}__gsm_symbolic_{idx}",
                prompt=new_question,
                answer=str(final),
                parent_id=parent,
                variant_type="gsm_symbolic",
                metadata={
                    **item.metadata,
                    "original_id": parent,
                    "numeric_leaves": list(new_leaves),
                    "answer_changing": True,
                },
            )
        )
    result.n_variants = len(result.items)
    return result
