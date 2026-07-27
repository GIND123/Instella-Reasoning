"""GSM-Symbolic-lite: auto-derive validated numeric variants of GSM8K items.

GSM-Symbolic (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) shows the decisive
robustness signal is changing *numerical values* (models are largely robust to name
changes), with correctness guaranteed by symbolic templates. Their templates are hand
annotated over 100 GSM8K **test** items. This module derives templates automatically
from GSM8K's built-in ``<<a op b=c>>`` calculator annotations, which is what lets the
same perturbation be applied to GSM8K **train** items — required for the seen/unseen
memorisation contrast, where the official release gives no coverage.

Precision over recall, by design. We emit a variant only when:

1. the annotated calc chain parses into steps whose results we can recompute;
2. re-running the chain on the *original* leaf numbers reproduces every annotated
   intermediate result and the final ``#### answer`` (round-trip validation);
3. every substitutable value passes the role-disambiguation rules in
   :func:`build_template` (see "Why value-based substitution is dangerous" below).

Items failing any check are skipped, not corrupted, and the reason is recorded in
:class:`TemplateRejection` so the hand-verification harness can surface near-misses.

Why value-based substitution is dangerous
-----------------------------------------
A number in the calc chain can play two different roles that look identical:

    "A robe takes 2 bolts of blue fiber and half that much white fiber."
    chain: <<2/2=1>>, <<2+1=3>>

The first ``2`` is a *question quantity* (bolts of blue fiber). The second is a
*structural constant* (the "half"). Substituting the value everywhere yields
``5/5=1, 5+1=6`` for a question whose true answer is ``5 + 5/2 = 7.5``. An earlier
version of this module shipped that bug and mislabelled 15% of its numeric variants.

The disambiguation rules, in order:

* a **decimal** token (``1.2``) is always a structural constant — never substituted
  (previously ``\\d+`` split it into ``1`` and ``2``, which rewrote "1.2 times" as
  "5.2 times");
* a value appearing **twice inside a single step expression** (``2/2``, ``4*4``) is
  the self-referential signature of a structural constant — reject the template;
* a value **absent from the question** is a structural constant — hold it fixed;
* otherwise the value is a leaf: substitute *every* occurrence, in both the question
  and the chain, consistently.

Magnitude neutrality
--------------------
Resampling must not systematically enlarge the numbers. arXiv:2605.28700 shows that a
shifted integer distribution — a K-S statistic of only 0.12 in the original
GSM-Symbolic release — accounted for the statistical significance of roughly half the
models they re-analysed. An earlier version here sampled leaves from ``[v//2, 2v+2]``
(expectation ≈ 1.25 v), producing variants whose median gold answer was **2.61x** the
original: the measured "reasoning fragility" was substantially a bigger-arithmetic
effect. :func:`_resample_leaf` is now log-symmetric (median ratio 1.0) and
:func:`magnitude_report` gives the run a hard check on the realised distribution.
"""

from __future__ import annotations

import ast
import math
import operator
import random
import re
from dataclasses import dataclass, field

from instella_reasoning.records import BenchmarkItem

_STEP = re.compile(r"<<\s*([^=<>]+?)\s*=\s*([^<>]+?)\s*>>")
# Atomic numeric token: decimals stay whole so "1.2" is never split into "1" and "2".
_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

MAX_STEPS = 8
MAX_LEAVES = 6
#: Realised median (variant answer / original answer). A run outside this band is
#: reporting a magnitude effect, not a reasoning effect — see module docstring.
MAGNITUDE_RATIO_BAND = (0.80, 1.25)


# Numeric-token boundaries that distinguish a decimal point from sentence punctuation.
# A naive ``(?<![\d.])N(?![\d.])`` silently fails to match "bakes 4." — the trailing period
# ends the sentence but the lookahead reads it as the start of a decimal. That under-count
# made an ambiguity guard here pass items it should have rejected.
_NUM_START = r"(?<!\d)(?<!\d\.)"  # not inside a longer integer, not the tail of "1.4"
_NUM_END = r"(?!\d)(?!\.\d)"  # not followed by more digits, not followed by ".5"


def _int_occurrences(value: int, text: str) -> int:
    """Count standalone occurrences of ``value`` not embedded in a longer number."""
    return len(re.findall(rf"{_NUM_START}{value}{_NUM_END}", text))


@dataclass(slots=True)
class SymbolicTemplate:
    benchmark_id: str
    question: str
    original_answer: int
    leaf_values: list[int]  # substitutable question quantities, first-appearance order
    constants: list[float]  # structural constants held fixed (divisors, rates, ...)
    step_exprs: list[str]
    step_results: list[int]

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "leaf_values": self.leaf_values,
            "constants": self.constants,
            "step_exprs": self.step_exprs,
            "step_results": self.step_results,
            "original_answer": self.original_answer,
        }


@dataclass(slots=True)
class TemplateRejection:
    """Why an item did not yield a template. Feeds the hand-verification harness."""

    benchmark_id: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"benchmark_id": self.benchmark_id, "reason": self.reason, "detail": self.detail}


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


def build_template(
    item: BenchmarkItem, rejections: list[TemplateRejection] | None = None
) -> SymbolicTemplate | None:
    """Derive a validated symbolic template from a GSM8K item, or None if not safe.

    ``rejections``, if given, receives a :class:`TemplateRejection` explaining any
    failure — the hand-verification harness uses these to surface near-misses that a
    human can approve manually.
    """

    def _reject(reason: str, detail: str = "") -> None:
        if rejections is not None:
            rejections.append(TemplateRejection(item.id, reason, detail))

    rationale = str(item.metadata.get("rationale", ""))
    raw_steps = _STEP.findall(rationale)
    if not raw_steps:
        _reject("no_calc_annotations")
        return None
    if len(raw_steps) > MAX_STEPS:
        _reject("chain_too_long", f"{len(raw_steps)} steps")
        return None

    step_exprs: list[str] = []
    step_results: list[int] = []
    for expr, result in raw_steps:
        r = _int_or_none(result)
        if r is None:
            _reject("non_integer_step_result", result)
            return None
        step_exprs.append(expr.strip())
        step_results.append(r)

    # Round-trip: recompute each expression and require it matches the annotation. This
    # confirms the annotations form a faithful integer chain before we perturb anything.
    for expr, expected in zip(step_exprs, step_results, strict=False):
        value = _eval_arith(expr)
        if value is None or not float(value).is_integer() or int(value) != expected:
            _reject("step_does_not_reproduce_result", f"{expr} != {expected}")
            return None

    answer = _int_or_none(str(item.answer or ""))
    if answer is None or answer != step_results[-1]:
        _reject("final_answer_mismatch", f"answer={item.answer} last_step={step_results[-1]}")
        return None

    if len(set(step_results)) != len(step_results):
        _reject("duplicate_step_results")
        return None

    # --- classify every chain token as leaf / structural constant / prior result -----
    prior_results: set[int] = set()
    leaves: list[int] = []
    seen_leaves: set[int] = set()
    constants: list[float] = []
    # How often each integer occurs across the whole chain. Compared against its count in
    # the question below: a question occurrence the chain cannot account for means the
    # number denotes two different quantities ("C bakes 4. D has 4 more."), and blanket
    # substitution would silently rewrite the one the chain never touched.
    chain_counts: dict[int, int] = {}
    for expr in step_exprs:
        for tok in _NUM_TOKEN.findall(expr):
            if "." not in tok:
                chain_counts[int(tok)] = chain_counts.get(int(tok), 0) + 1

    for expr, result in zip(step_exprs, step_results, strict=False):
        tokens = _NUM_TOKEN.findall(expr)
        counts_in_step: dict[str, int] = {}
        for tok in tokens:
            counts_in_step[tok] = counts_in_step.get(tok, 0) + 1

        for tok in tokens:
            if "." in tok:  # decimal -> always a structural constant (rate, "1.2 times")
                value_f = float(tok)
                if value_f not in constants:
                    constants.append(value_f)
                continue
            value = int(tok)
            if value in prior_results:  # reference to an earlier step's output
                continue
            if value in seen_leaves:
                continue
            # Self-referential within one step ("2/2", "4*4") is the structural-constant
            # signature: one occurrence is a question quantity, the other is the operator's
            # own constant. They are indistinguishable by value, so refuse the template.
            if counts_in_step[tok] > 1:
                _reject("value_repeated_within_step", f"{value} in '{expr}'")
                return None
            in_question = _int_occurrences(value, item.prompt)
            if in_question == 0:
                # Not in the question at all -> a structural constant. Hold it fixed.
                if float(value) not in constants:
                    constants.append(float(value))
                continue
            if in_question > chain_counts.get(value, 0):
                # The question uses this number more often than the calc chain does, so at
                # least one occurrence denotes something the chain never consumes. We
                # cannot tell which, so refuse rather than corrupt the item.
                _reject(
                    "value_ambiguous_in_question",
                    f"{value} appears {in_question}x in question but {chain_counts.get(value, 0)}x in chain",
                )
                return None
            seen_leaves.add(value)
            leaves.append(value)
        prior_results.add(result)

    if not leaves:
        _reject("no_substitutable_leaves")
        return None
    if len(leaves) > MAX_LEAVES:
        _reject("too_many_leaves", str(len(leaves)))
        return None
    if seen_leaves & set(step_results):
        _reject("leaf_collides_with_step_result")
        return None

    return SymbolicTemplate(
        benchmark_id=item.id,
        question=item.prompt,
        original_answer=answer,
        leaf_values=leaves,
        constants=constants,
        step_exprs=step_exprs,
        step_results=step_results,
    )


def _recompute(
    step_exprs: list[str], step_results: list[int], mapping: dict[int, int]
) -> tuple[list[int], int] | None:
    """Re-evaluate the chain with leaves substituted via ``mapping``.

    A token is (a) a decimal -> left alone; (b) a previous step's original result ->
    replaced with that step's *recomputed* value; (c) a leaf -> replaced via ``mapping``;
    (d) anything else -> a structural constant, left alone. Returns
    ``(results, final)`` with all values positive integers, or None if a step is invalid.
    """
    old_to_new: dict[int, int] = {}
    results: list[int] = []
    for expr, old_result in zip(step_exprs, step_results, strict=False):

        def _sub(match: re.Match) -> str:
            tok = match.group(0)
            if "." in tok:
                return tok
            value = int(tok)
            if value in old_to_new:
                return str(old_to_new[value])
            return str(mapping.get(value, value))

        new_expr = _NUM_TOKEN.sub(_sub, expr)
        value = _eval_arith(new_expr)
        if value is None or not float(value).is_integer():
            return None
        ivalue = int(value)
        if ivalue <= 0:
            return None
        old_to_new[old_result] = ivalue
        results.append(ivalue)
    return results, results[-1]


def _resample_leaf(value: int, rng: random.Random) -> int:
    """Draw a replacement for ``value`` with a **median ratio of 1.0**.

    Small values use additive jitter (a multiplicative draw on 1 or 2 cannot go down
    without clamping to 1, which biases the whole run upward). Larger values use a
    log-symmetric multiplicative draw: scale up by f or down by the same f with equal
    probability, so the median of new/old is exactly 1.
    """
    # Never resample a real quantity down to 1: it produces "1 times", "1 pounds",
    # "1 friends" — ungrammatical text that makes the variant look broken to a reader
    # and confounds the perturbation with a fluency drop. GSM-Symbolic hand-wrote its
    # sampling ranges for exactly this reason.
    floor = 1 if value <= 2 else 2
    if value <= 4:
        return rng.randint(max(floor, value - 3), value + 3)
    factor = rng.uniform(1.15, 2.0)
    if rng.random() < 0.5:
        return max(floor, int(round(value * factor)))
    return max(floor, int(round(value / factor)))


@dataclass(slots=True)
class NumericVariantResult:
    items: list[BenchmarkItem] = field(default_factory=list)
    n_templated: int = 0
    n_variants: int = 0
    rejection: TemplateRejection | None = None


def make_numeric_variants(
    item: BenchmarkItem,
    k: int = 3,
    rng: random.Random | None = None,
    max_tries: int = 60,
    rejections: list[TemplateRejection] | None = None,
) -> NumericVariantResult:
    """Produce up to ``k`` validated numeric variants of a GSM8K item.

    Each variant resamples the leaf constants, recomputes the answer from the item's own
    calc chain, and substitutes the new numbers into the question. Variants carry
    ``variant_type='gsm_symbolic'`` and the parent's id so the reliability metric treats
    them as an answer-*changing* cluster (scored by correctness, not answer agreement).
    """
    rng = rng or random.Random(f"gsm_symbolic:{item.id}")
    local_rejections: list[TemplateRejection] = []
    template = build_template(item, local_rejections)
    result = NumericVariantResult()
    if template is None:
        result.rejection = local_rejections[0] if local_rejections else None
        if rejections is not None:
            rejections.extend(local_rejections)
        return result
    result.n_templated = 1

    protected = {int(c) for c in template.constants if float(c).is_integer()}
    seen: set[tuple[int, ...]] = {tuple(template.leaf_values)}
    # Balancing the *answer* direction per slot, not just the leaf draws. Median-neutral
    # leaves are not enough: the answer is a sum/product of several leaves, so Jensen's
    # inequality still skews it upward (measured median ratio 1.19 with neutral leaves).
    # Alternating the target direction pins the realised median ratio at ~1.0, which is
    # what the magnitude confound actually requires.
    relaxed_after = max_tries // 2
    for attempt in range(max_tries):
        if len(result.items) >= k:
            break
        mapping = {v: _resample_leaf(v, rng) for v in template.leaf_values}
        new_leaves = tuple(mapping[v] for v in template.leaf_values)
        if new_leaves in seen:
            continue
        if len(set(new_leaves)) != len(new_leaves):
            continue
        # A new leaf must not collide with a held-fixed structural constant, or the
        # substituted question would read two different quantities as the same number.
        if set(new_leaves) & protected:
            continue
        recomputed = _recompute(template.step_exprs, template.step_results, mapping)
        if recomputed is None:
            continue
        results, final = recomputed
        if set(new_leaves) & set(results):
            continue  # ambiguity between a leaf and an intermediate
        if attempt < relaxed_after:
            want_smaller = len(result.items) % 2 == 0
            if (final < template.original_answer) is not want_smaller:
                continue
        seen.add(new_leaves)

        # Substitute in the question. Counts were validated at template time, so every
        # occurrence of a leaf refers to the same quantity and all are replaced. Done via
        # a single simultaneous pass so a new value can never be re-substituted by a later
        # leaf ("5 -> 12" followed by "12 -> 7" must not touch the freshly written 12).
        new_question = _substitute_simultaneous(template.question, mapping)
        if new_question is None:
            continue

        idx = len(result.items)
        parent = item.parent_id or item.id
        ratio = final / template.original_answer if template.original_answer else 1.0
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
                    "original_answer": template.original_answer,
                    "numeric_leaves": list(new_leaves),
                    "original_leaves": list(template.leaf_values),
                    "answer_changing": True,
                    # Recorded per item so the analysis can include magnitude as a GLMM
                    # covariate and run the magnitude-matched sensitivity check.
                    "magnitude_ratio": round(ratio, 6),
                    "log_magnitude_ratio": round(math.log(ratio), 6) if ratio > 0 else 0.0,
                },
            )
        )
    result.n_variants = len(result.items)
    if rejections is not None:
        rejections.extend(local_rejections)
    return result


def _substitute_simultaneous(question: str, mapping: dict[int, int]) -> str | None:
    """Replace every leaf occurrence in one pass, so substitutions cannot cascade.

    Sequential ``re.sub`` calls are unsafe: mapping 5->12 then 12->7 would rewrite the
    ``12`` that the first substitution just produced. One combined pattern with a single
    dispatching callback makes every replacement read the *original* text.
    """
    if not mapping:
        return None
    alternation = "|".join(str(v) for v in sorted(mapping, key=lambda v: -v))
    pattern = re.compile(rf"{_NUM_START}({alternation}){_NUM_END}")
    replaced = 0

    def _sub(match: re.Match) -> str:
        nonlocal replaced
        replaced += 1
        return str(mapping[int(match.group(1))])

    new_question = pattern.sub(_sub, question)
    if replaced == 0:
        return None
    return new_question


# -- run-level magnitude audit ---------------------------------------------------


@dataclass(slots=True)
class MagnitudeReport:
    """Realised magnitude shift of a numeric variant set — a hard gate for the run."""

    n: int
    median_ratio: float
    mean_log_ratio: float
    share_larger: float
    in_band: bool
    band: tuple[float, float] = MAGNITUDE_RATIO_BAND

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "median_answer_magnitude_ratio": round(self.median_ratio, 4),
            "mean_log_ratio": round(self.mean_log_ratio, 4),
            "share_variant_larger_than_original": round(self.share_larger, 4),
            "band": list(self.band),
            "in_band": self.in_band,
        }


def magnitude_report(items: list[BenchmarkItem]) -> MagnitudeReport:
    """Audit the answer-magnitude distribution of a numeric variant set.

    The original generator produced a median ratio of 2.61 with 85% of variants larger
    than their original — a textbook instance of the confound arXiv:2605.28700 identifies.
    The suite calls this and refuses to spend GPU hours when the result is out of band.
    """
    ratios = [
        float(i.metadata["magnitude_ratio"])
        for i in items
        if i.variant_type == "gsm_symbolic" and "magnitude_ratio" in i.metadata
    ]
    if not ratios:
        return MagnitudeReport(0, 1.0, 0.0, 0.0, True)
    ordered = sorted(ratios)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    logs = [math.log(r) for r in ratios if r > 0]
    low, high = MAGNITUDE_RATIO_BAND
    return MagnitudeReport(
        n=len(ratios),
        median_ratio=median,
        mean_log_ratio=sum(logs) / len(logs) if logs else 0.0,
        share_larger=sum(1 for r in ratios if r > 1.0) / len(ratios),
        in_band=low <= median <= high,
    )
