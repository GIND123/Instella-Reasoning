"""Semantics-preserving perturbation generators for consistency clusters (Phase 3).

The Reliability metric (``accuracy x consistency``) needs, for each benchmark item,
a *cluster* of semantically equivalent problem variants. This module builds those
clusters deterministically and dependency-free, following the perturbation taxonomy
in the proposal (GSM-Symbolic / GSM-Plus style):

- ``entity_substitution``   — rename people/objects consistently (answer unchanged).
- ``premise_reordering``    — shuffle the non-question sentences (answer unchanged).
- ``irrelevant_context``    — inject a distractor clause that does not change the answer.
- ``rephrasing``            — light lexical rewrites (answer unchanged).
- ``numeric_perturbation``  — change the numbers *and recompute the answer* from a
  symbolic template carried in ``metadata.numeric_template`` (answer changes, but is
  known exactly). Items without a template skip this type rather than emit a wrong
  answer.

Every generator is a pure function of ``(item, seed)`` so a run is fully
reproducible. Variants inherit the original's ``parent_id`` and carry a
``variant_type`` tag, so :func:`instella_reasoning.metrics.summarize_reliability`
groups a cluster automatically.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from instella_reasoning.records import BenchmarkItem, read_benchmark, write_jsonl

# Curated pools kept small and unambiguous so substitutions never change arithmetic.
_NAME_POOL = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Liam", "Maya", "Noah", "Olivia", "Peter",
]
_OBJECT_POOL = [
    "pencils", "marbles", "apples", "coins", "stickers", "books", "oranges",
    "cards", "candies", "balloons", "cookies", "erasers",
]
_DISTRACTORS = [
    "The weather that day was pleasant.",
    "The items were stored in a blue box.",
    "This happened on a Tuesday afternoon.",
    "A friend watched the whole time.",
    "The shop had recently been repainted.",
    "Everyone was in a cheerful mood.",
]
# Light, answer-preserving rephrase rules applied in order.
_REPHRASE_RULES: list[tuple[str, str]] = [
    (r"\bHow many\b", "In total, how many"),
    (r"\bare there\b", "are present"),
    (r"\bare in\b", "can be found in"),
    (r"\bthen\b", "after that"),
    (r"\bIf\b", "Suppose that"),
    (r"\breceives\b", "gets"),
    (r"\bsells\b", "gives away"),
]

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

DEFAULT_PERTURBATIONS = (
    "entity_substitution",
    "premise_reordering",
    "irrelevant_context",
    "rephrasing",
)


@dataclass(slots=True)
class PerturbationConfig:
    """Which perturbations to apply and how many variants to keep per item.

    ``numeric_variants`` adds that many GSM-Symbolic-style *answer-changing* variants per
    templatable GSM8K item (see :mod:`instella_reasoning.gsm_symbolic`). Per GSM-Symbolic
    (arXiv:2410.05229) these numeric perturbations are the decisive robustness probe, so a
    rigorous run should set ``numeric_variants > 0`` rather than rely on surface changes.
    """

    types: tuple[str, ...] = DEFAULT_PERTURBATIONS
    seed: int = 6198
    include_original: bool = True
    max_variants: int | None = None
    numeric_variants: int = 0


# -- sentence utilities --------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split into sentences while preserving their trailing punctuation/whitespace."""
    parts = re.findall(r".*?(?:[.!?](?:\s+|$)|$)", text, flags=re.DOTALL)
    return [part for part in parts if part.strip()]


def _looks_like_question(sentence: str) -> bool:
    stripped = sentence.strip()
    return stripped.endswith("?") or bool(
        re.match(r"^(how|what|which|who|when|where|is|are|does|do|can)\b", stripped, re.I)
    )


# -- individual perturbations --------------------------------------------------


def entity_substitution(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Rename any people/objects consistently. Answer is preserved by construction."""
    prompt = item.prompt
    mapping: dict[str, str] = {}

    present_names = [name for name in _NAME_POOL if re.search(rf"\b{name}\b", prompt)]
    if present_names:
        replacements = _distinct_choices(_NAME_POOL, len(present_names), present_names, rng)
        mapping.update(dict(zip(present_names, replacements, strict=False)))

    present_objects = [obj for obj in _OBJECT_POOL if re.search(rf"\b{obj}\b", prompt)]
    if present_objects:
        replacements = _distinct_choices(_OBJECT_POOL, len(present_objects), present_objects, rng)
        mapping.update(dict(zip(present_objects, replacements, strict=False)))

    if not mapping:
        return None

    new_prompt = prompt
    for source, target in mapping.items():
        new_prompt = re.sub(rf"\b{re.escape(source)}\b", target, new_prompt)
    return _variant(item, new_prompt, "entity_substitution", answer=item.answer)


def premise_reordering(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Shuffle the non-question sentences. Answer is preserved."""
    sentences = _split_sentences(item.prompt)
    if len(sentences) < 3:
        return None
    question_idx = next(
        (i for i in range(len(sentences) - 1, -1, -1) if _looks_like_question(sentences[i])),
        len(sentences) - 1,
    )
    body = sentences[:question_idx] + sentences[question_idx + 1 :]
    if len(body) < 2:
        return None
    shuffled = body[:]
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != body:
            break
    else:
        return None
    rebuilt = shuffled[:]
    rebuilt.insert(min(question_idx, len(rebuilt)), sentences[question_idx])
    return _variant(item, "".join(rebuilt).strip(), "premise_reordering", answer=item.answer)


def irrelevant_context(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Insert a distractor sentence before the question. Answer is preserved."""
    sentences = _split_sentences(item.prompt)
    distractor = rng.choice(_DISTRACTORS)
    if not sentences:
        return None
    insert_at = next(
        (i for i in range(len(sentences)) if _looks_like_question(sentences[i])),
        len(sentences),
    )
    padded = distractor if distractor.endswith(" ") else distractor + " "
    rebuilt = sentences[:insert_at] + [padded] + sentences[insert_at:]
    return _variant(item, "".join(rebuilt).strip(), "irrelevant_context", answer=item.answer)


def rephrasing(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Apply light lexical rewrites. Answer is preserved."""
    new_prompt = item.prompt
    applied = 0
    for pattern, replacement in _REPHRASE_RULES:
        rewritten, n = re.subn(pattern, replacement, new_prompt, count=1)
        if n:
            new_prompt = rewritten
            applied += 1
    if applied == 0 or new_prompt == item.prompt:
        return None
    return _variant(item, new_prompt, "rephrasing", answer=item.answer)


def numeric_perturbation(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Resample the numbers and recompute the answer from a symbolic template.

    Requires ``item.metadata['numeric_template']`` shaped like::

        {"template": "A store has {a} pencils. It sells {b}. How many are left?",
         "slots": {"a": {"low": 5, "high": 50}, "b": {"low": 1, "high": 5}},
         "answer_expr": "a - b"}

    ``slots`` values may be an explicit ``choices`` list or a ``low/high[/step]``
    range. The answer is recomputed with a restricted arithmetic evaluator, so the
    variant carries a *correct* new ground truth rather than a guessed one.
    """
    template_spec = item.metadata.get("numeric_template")
    if not isinstance(template_spec, dict):
        return None
    template = template_spec.get("template")
    slots = template_spec.get("slots", {})
    answer_expr = template_spec.get("answer_expr")
    if not template or not answer_expr or not isinstance(slots, dict):
        return None

    values: dict[str, float] = {}
    for name, spec in slots.items():
        values[name] = _sample_slot(spec, rng)

    try:
        new_prompt = template.format(**{k: _fmt_number(v) for k, v in values.items()})
        answer_value = _safe_eval(answer_expr, values)
    except (KeyError, ValueError, ZeroDivisionError):
        return None

    return _variant(
        item,
        new_prompt,
        "numeric_perturbation",
        answer=_fmt_number(answer_value),
        extra_meta={"numeric_values": values},
    )


_PERTURBATION_FUNCS = {
    "entity_substitution": entity_substitution,
    "premise_reordering": premise_reordering,
    "irrelevant_context": irrelevant_context,
    "rephrasing": rephrasing,
    "numeric_perturbation": numeric_perturbation,
}


# -- cluster construction ------------------------------------------------------


def make_variants(item: BenchmarkItem, config: PerturbationConfig | None = None) -> list[BenchmarkItem]:
    """Return the perturbation variants for one item (originals excluded)."""
    config = config or PerturbationConfig()
    variants: list[BenchmarkItem] = []
    for perturbation in config.types:
        func = _PERTURBATION_FUNCS.get(perturbation)
        if func is None:
            raise ValueError(f"Unknown perturbation {perturbation!r}. Known: {sorted(_PERTURBATION_FUNCS)}")
        # Per-(item,type) seed so a variant is stable regardless of call order.
        rng = random.Random(f"{config.seed}:{item.id}:{perturbation}")
        variant = func(item, rng)
        if variant is not None:
            variants.append(variant)
        if config.max_variants is not None and len(variants) >= config.max_variants:
            break
    return variants


def make_variant_suite(
    items: Iterable[BenchmarkItem], config: PerturbationConfig | None = None
) -> list[BenchmarkItem]:
    """Expand every item into its cluster (original + variants) in a flat list."""
    config = config or PerturbationConfig()
    suite: list[BenchmarkItem] = []
    for item in items:
        original = _ensure_parent(item)
        if config.include_original:
            suite.append(original)
        suite.extend(make_variants(original, config))
        if config.numeric_variants > 0:
            from instella_reasoning.gsm_symbolic import make_numeric_variants

            rng = random.Random(f"{config.seed}:{original.id}:gsm_symbolic")
            suite.extend(make_numeric_variants(original, k=config.numeric_variants, rng=rng).items)
    return suite


def make_resample_cluster(item: BenchmarkItem, n: int) -> list[BenchmarkItem]:
    """Duplicate an item ``n`` times as the **decoding-noise control**.

    Every copy is textually identical and shares the parent's gold answer, so under
    temperature sampling the cluster's consistency measures *run-to-run variance alone* —
    with no perturbation applied. That number is the baseline the perturbation-consistency
    drop has to beat.

    Without it the study cannot answer the first question a reviewer asks: at temperature
    0 there is no sampling variance at all, so an observed inconsistency across variants
    has no null to be compared against. The copies are answer-*preserving*, which puts
    them in the consistency term exactly like a surface variant.
    """
    parent = item.parent_id or item.id
    return [
        BenchmarkItem(
            id=f"{parent}__resample_{i:02d}",
            prompt=item.prompt,
            answer=item.answer,
            parent_id=parent,
            variant_type="resample",
            metadata={**item.metadata, "sample_index": i, "decoding_noise_control": True},
        )
        for i in range(n)
    ]


def make_resample_suite(items: Iterable[BenchmarkItem], n: int) -> list[BenchmarkItem]:
    """Flat original+copies suite for the decoding-noise control."""
    suite: list[BenchmarkItem] = []
    for item in items:
        original = _ensure_parent(item)
        suite.append(original)
        suite.extend(make_resample_cluster(original, n))
    return suite


# -- variant validation (reviewer concern M2) ----------------------------------


@dataclass(slots=True)
class ValidationReport:
    """Answer-preservation / well-formedness audit for a perturbation suite."""

    n_variants: int
    n_answer_preserving: int
    n_preserved_ok: int
    n_answer_changing: int
    n_degenerate_text: int
    violations: list[str] = field(default_factory=list)  # variant ids that changed the answer

    @property
    def answer_preservation_rate(self) -> float:
        return self.n_preserved_ok / self.n_answer_preserving if self.n_answer_preserving else 1.0

    def to_dict(self) -> dict:
        return {
            "n_variants": self.n_variants,
            "n_answer_preserving": self.n_answer_preserving,
            "n_preserved_ok": self.n_preserved_ok,
            "answer_preservation_rate": round(self.answer_preservation_rate, 6),
            "n_answer_changing": self.n_answer_changing,
            "n_degenerate_text": self.n_degenerate_text,
            "violations": self.violations[:50],
        }


def validate_variants(items: list[BenchmarkItem]) -> ValidationReport:
    """Audit a perturbation suite for answer preservation and text well-formedness.

    Answer-*preserving* variants (entity/reorder/distractor/rephrase) must carry the same
    gold answer as their original; a mismatch is a bug in the generator or an unintended
    difficulty change. Answer-*changing* variants (``gsm_symbolic``) are recomputed and
    self-validated, so they are counted separately, not checked against the original.
    """
    from instella_reasoning.gsm_symbolic import _int_or_none  # local: keep import graph light

    originals = {
        item.parent_id or item.id: item
        for item in items
        if item.variant_type == "original"
    }
    n_variants = n_preserving = n_preserved_ok = n_changing = n_degenerate = 0
    violations: list[str] = []
    changing_types = {"gsm_symbolic", "numeric_perturbation"}

    for item in items:
        if item.variant_type == "original":
            continue
        n_variants += 1
        # crude well-formedness: non-empty, changed from the original, no doubled blanks
        parent = originals.get(item.parent_id or "")
        if not item.prompt.strip() or "  " in item.prompt.strip():
            n_degenerate += 1
        if item.variant_type in changing_types or item.metadata.get("answer_changing"):
            n_changing += 1
            continue
        n_preserving += 1
        if parent is None:
            continue
        # Compare answers numerically when possible, else as normalized strings.
        want, got = str(parent.answer), str(item.answer)
        equal = (
            _int_or_none(want) == _int_or_none(got)
            if _int_or_none(want) is not None and _int_or_none(got) is not None
            else want.strip() == got.strip()
        )
        if equal:
            n_preserved_ok += 1
        else:
            violations.append(item.id)

    return ValidationReport(
        n_variants=n_variants,
        n_answer_preserving=n_preserving,
        n_preserved_ok=n_preserved_ok,
        n_answer_changing=n_changing,
        n_degenerate_text=n_degenerate,
        violations=violations,
    )


def expand_benchmark_file(
    input_path: str | Path,
    output_path: str | Path,
    config: PerturbationConfig | None = None,
) -> int:
    """Read a benchmark JSONL, write the expanded consistency suite, return its size."""
    items = read_benchmark(input_path)
    suite = make_variant_suite(items, config)
    write_jsonl(output_path, suite)
    return len(suite)


# -- helpers -------------------------------------------------------------------


def _variant(
    item: BenchmarkItem,
    prompt: str,
    variant_type: str,
    answer: str | None,
    extra_meta: dict | None = None,
) -> BenchmarkItem:
    metadata = dict(item.metadata)
    if extra_meta:
        metadata.update(extra_meta)
    metadata["original_id"] = item.parent_id or item.id
    return BenchmarkItem(
        id=f"{item.parent_id or item.id}__{variant_type}",
        prompt=prompt,
        answer=answer,
        parent_id=item.parent_id or item.id,
        variant_type=variant_type,
        metadata=metadata,
    )


def _ensure_parent(item: BenchmarkItem) -> BenchmarkItem:
    if item.parent_id is not None:
        return item
    return BenchmarkItem(
        id=item.id,
        prompt=item.prompt,
        answer=item.answer,
        parent_id=item.id,
        variant_type=item.variant_type,
        metadata=item.metadata,
    )


def _distinct_choices(
    pool: list[str], count: int, avoid: Iterable[str], rng: random.Random
) -> list[str]:
    available = [value for value in pool if value not in set(avoid)]
    rng.shuffle(available)
    return available[:count]


def _sample_slot(spec, rng: random.Random) -> float:
    if isinstance(spec, dict) and "choices" in spec:
        return float(rng.choice(spec["choices"]))
    if isinstance(spec, dict) and "low" in spec and "high" in spec:
        low, high = spec["low"], spec["high"]
        step = spec.get("step", 1)
        n_steps = int((high - low) / step)
        return float(low + rng.randint(0, max(0, n_steps)) * step)
    if isinstance(spec, (list, tuple)):
        return float(rng.choice(list(spec)))
    raise ValueError(f"Unsupported slot spec: {spec!r}")


def _fmt_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def _safe_eval(expr: str, names: dict[str, float]) -> float:
    """Evaluate an arithmetic expression over ``names`` with no attribute/call access."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body, names)


def _eval_node(node: ast.AST, names: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in names:
            return float(names[node.id])
        raise KeyError(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_eval_node(node.operand, names))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")
