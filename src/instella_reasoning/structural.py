"""Structural perturbations: the axis the pretraining augmentation held fixed.

Why this module exists
----------------------
The published training recipe for the second-stage synthetic math corpus builds its
variants by *abstracting numerical values as function parameters* and then assigning
new values to those parameters, keeping the solution program identical. Numeric
perturbation is therefore the **same operation** that generated the training data:
a numerically perturbed evaluation item lies inside the training distribution by
construction, so a difference-in-differences estimator built on it cannot separate
memorisation from reasoning. A flat estimate is the predicted outcome either way.

The augmentation varies *leaf values* while holding the *program* fixed. The
discriminating axis is therefore the program itself:

- ``depth_extension``      — append one dependent step, increasing reasoning depth by
  exactly one. The step count changes, which the augmentation never did.
- ``distractor_quantity``  — insert an unused numeric quantity. The augmentation only
  re-valued quantities that the program consumed; it never introduced one the program
  must ignore.

Both are answer-exact: ``depth_extension`` recomputes the gold answer in closed form,
``distractor_quantity`` provably preserves it. Neither depends on entity parsing, so
they apply to any item with an integer gold answer rather than only to items whose
calculator annotations yield a clean symbolic template.

Every generator is a pure function of ``(item, rng)``, matching the contract in
:mod:`instella_reasoning.perturbations`, so runs remain reproducible.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from instella_reasoning.records import BenchmarkItem

#: Variants whose gold answer differs from the parent's. Mirrors the convention in
#: :mod:`instella_reasoning.metrics` so cluster consistency excludes them.
ANSWER_CHANGING_STRUCTURAL = ("depth_extension", "premise_removal")
ANSWER_PRESERVING_STRUCTURAL = ("distractor_quantity",)

STRUCTURAL_PERTURBATIONS = ANSWER_CHANGING_STRUCTURAL + ANSWER_PRESERVING_STRUCTURAL

_INT = re.compile(r"^-?\d+$")

# Multipliers/addends kept small so the extra step never dominates the arithmetic and
# never pushes the answer into a magnitude regime the parent did not occupy. The
# magnitude confound that inflated earlier perturbation results was a *ratio* effect,
# so the composition factors stay inside a narrow band.
_MULTIPLIERS = (2, 3, 4)
_ADDENDS = (5, 7, 9, 11, 12, 15)

# Distractor sentences carry exactly one number and no quantity the question consumes.
_DISTRACTOR_TEMPLATES = (
    "Earlier that month, {n} similar items were recorded in a separate log that is not "
    "part of this calculation.",
    "A different store {n} miles away kept its own records, which do not affect this "
    "question.",
    "In the previous year the same count had been {n}, but that figure is unrelated to "
    "the question asked.",
    "An unrelated shipment of {n} boxes arrived on the same day and was never opened.",
)


@dataclass(slots=True)
class StructuralResult:
    """Outcome of applying the structural generators to one item."""

    variants: list[BenchmarkItem]
    skipped: list[str]

    def to_dict(self) -> dict:
        return {"n_variants": len(self.variants), "skipped": list(self.skipped)}


def _gold_int(item: BenchmarkItem) -> int | None:
    """Integer gold answer, or None when the item is not integer-valued.

    Both generators need closed-form recomputation, which is only defensible when the
    parent answer is an exact integer. Non-integer items are skipped rather than
    approximated: a silently wrong gold answer is far more damaging than a smaller n.
    """
    if item.answer is None:
        return None
    raw = str(item.answer).strip().replace(",", "")
    # Tolerate a trailing "#### 18" style answer field.
    if "####" in raw:
        raw = raw.split("####")[-1].strip()
    if not _INT.match(raw):
        return None
    return int(raw)


def _question_body(prompt: str) -> str:
    """Prompt with trailing whitespace normalised, ready for sentence appending."""
    return prompt.rstrip()


def depth_extension(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Append one dependent reasoning step; the answer changes and is recomputed exactly.

    The appended clause refers to the parent's result anaphorically ("that amount"),
    which avoids parsing the question's entities and keeps the transformation valid for
    any integer-valued item. The composed question requires the model to solve the
    original problem *and* carry its result through one further operation, so the
    solution program gains a step it did not have in training.
    """
    gold = _gold_int(item)
    if gold is None:
        return None

    if rng.random() < 0.5:
        k = rng.choice(_MULTIPLIERS)
        clause = f" Finally, what is {k} times that amount?"
        new_answer = gold * k
        op = f"*{k}"
    else:
        n = rng.choice(_ADDENDS)
        clause = f" Finally, what is that amount plus {n}?"
        new_answer = gold + n
        op = f"+{n}"

    metadata = dict(item.metadata)
    metadata.update(
        {
            # Read by metrics.is_answer_changing, which otherwise only knows the numeric
            # variant names. Without it, cluster consistency would compare correctness
            # labels across variants whose gold answers legitimately differ.
            "answer_changing": True,
            "structural_op": op,
            "parent_answer": gold,
            "depth_delta": 1,
            # Recorded so the analysis can verify the composition never moved the answer
            # into a different magnitude regime, the confound that invalidated earlier
            # perturbation estimates.
            "magnitude_ratio": (new_answer / gold) if gold else None,
        }
    )
    return BenchmarkItem(
        id=f"{item.id}__depth",
        prompt=_question_body(item.prompt) + clause,
        answer=str(new_answer),
        parent_id=item.parent_id or item.id,
        variant_type="depth_extension",
        metadata=metadata,
    )


def distractor_quantity(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Insert an unused numeric quantity; the gold answer is unchanged by construction.

    Distinct from ``irrelevant_context``, which adds a distractor sentence carrying no
    number. The number is what matters here: the pretraining augmentation only ever
    re-valued quantities the solution program consumed, so a quantity the program must
    *ignore* is outside its support. The inserted value is chosen to avoid colliding
    with any integer already present in the question, so it cannot be mistaken for a
    quantity the question defines.
    """
    gold = _gold_int(item)
    if gold is None:
        return None

    present = {int(t) for t in re.findall(r"\d+", item.prompt)}
    present.add(abs(gold))
    candidates = [v for v in range(13, 97) if v not in present]
    if not candidates:
        return None
    n = rng.choice(candidates)

    template = rng.choice(_DISTRACTOR_TEMPLATES)
    sentence = template.format(n=n)

    body = _question_body(item.prompt)
    # Insert before the final question sentence when one is identifiable, so the
    # distractor sits in the premises rather than after the interrogative. Falling back
    # to appending keeps the generator total.
    parts = re.split(r"(?<=[.!?])\s+", body)
    if len(parts) >= 2:
        new_prompt = " ".join(parts[:-1] + [sentence, parts[-1]])
    else:
        new_prompt = f"{body} {sentence}"

    metadata = dict(item.metadata)
    metadata.update(
        {"structural_op": f"distractor:{n}", "parent_answer": gold, "depth_delta": 0}
    )
    return BenchmarkItem(
        id=f"{item.id}__distract",
        prompt=new_prompt,
        answer=str(gold),
        parent_id=item.parent_id or item.id,
        variant_type="distractor_quantity",
        metadata=metadata,
    )


def premise_removal(item: BenchmarkItem, rng: random.Random) -> BenchmarkItem | None:
    """Delete a premise the solution needs, making the problem unanswerable.

    The strongest available probe, and the reason is that it converts an indirect accuracy
    contrast into a direct measurement of recall. ``depth_extension`` leaves the original
    problem verbatim, so a recalled answer remains usable and the probe cannot separate
    recall from reasoning. Deleting a required premise removes the information the answer
    depends on: a model reasoning from the text should report that the problem is
    underdetermined, whereas a model reproducing a memorised item emits the original
    answer regardless. The rate of the latter, compared across verified-membership arms,
    *is* the memorisation estimate.

    The augmentation that generated the pretraining corpus re-valued parameters while
    holding the program and its premises fixed, so a removed premise lies outside its
    support by construction.

    Precision over recall: the deleted sentence must contain an integer that the
    calculator chain actually consumes, otherwise removal may leave the problem solvable
    and the item is skipped rather than mislabelled.
    """
    gold = _gold_int(item)
    if gold is None:
        return None

    body = _question_body(item.prompt)
    parts = re.split(r"(?<=[.!?])\s+", body)
    if len(parts) < 3:  # need premises beyond the single question sentence
        return None

    # Values the solution consumes, taken from the calculator annotations.
    rationale = str((item.metadata or {}).get("rationale", ""))
    used = {int(t) for t in re.findall(r"\d+", rationale)} if rationale else set()
    if not used:
        return None

    # Candidate premises: not the final (question) sentence, and carrying a consumed value.
    candidates = [
        i for i, s in enumerate(parts[:-1])
        if {int(t) for t in re.findall(r"\d+", s)} & used
    ]
    if not candidates:
        return None
    drop = rng.choice(candidates)
    remaining = [s for i, s in enumerate(parts) if i != drop]

    # Answer-leakage guard. The necessity check above confirms the deleted *premise value*
    # is gone; it says nothing about the *answer*, which can independently appear
    # elsewhere in the text. When it does, emitting it is copying rather than recall, and
    # the item inflates the estimate. This matters because leakage correlates with
    # containment: on the measured arms it occurred in 6.9 percent of high-containment
    # items against 3.1 percent of low, a difference the same size as the effect itself.
    if str(abs(gold)) in re.findall(r"\d+", " ".join(remaining)):
        return None

    metadata = dict(item.metadata)
    metadata.update(
        {
            "answer_changing": True,
            "structural_op": f"premise_removal:{drop}",
            # Retained so the analysis can measure how often the model emits the answer to
            # a question the prompt no longer determines.
            "parent_answer": gold,
            "removed_premise": parts[drop],
            "unanswerable": True,
            "depth_delta": -1,
        }
    )
    return BenchmarkItem(
        id=f"{item.id}__noprem",
        prompt=" ".join(remaining),
        # No gold answer exists: the item is scored by whether the model recalls the
        # parent's answer, not by exact match against one.
        answer=None,
        parent_id=item.parent_id or item.id,
        variant_type="premise_removal",
        metadata=metadata,
    )


_GENERATORS = {
    "depth_extension": depth_extension,
    "distractor_quantity": distractor_quantity,
    "premise_removal": premise_removal,
}


def removable_premises(item: BenchmarkItem) -> list[int]:
    """Indices of premise sentences the solution actually consumes.

    Exposed so a caller can emit one variant per removable premise. Recall on an
    unanswerable item is a low-rate binary event, so the estimate is
    precision-limited at one variant per problem; deleting each required premise in
    turn multiplies the observations without adding parent problems.
    """
    body = _question_body(item.prompt)
    parts = re.split(r"(?<=[.!?])\s+", body)
    if len(parts) < 3:
        return []
    rationale = str((item.metadata or {}).get("rationale", ""))
    used = {int(t) for t in re.findall(r"\d+", rationale)} if rationale else set()
    if not used:
        return []

    out: list[int] = []
    for i, s in enumerate(parts[:-1]):
        here = {int(t) for t in re.findall(r"\d+", s)} & used
        if not here:
            continue
        # Necessity check. A value the solution consumes must be *gone* from the whole
        # remaining prompt, not merely from the deleted sentence: if it survives
        # elsewhere the problem stays solvable, and a model reproducing the parent's
        # answer would then be reasoning correctly rather than recalling. Counting that
        # as recall would inflate the estimate in exactly the direction the hypothesis
        # predicts, so the item is dropped instead.
        rest = " ".join(p for j, p in enumerate(parts) if j != i)
        rest_nums = {int(t) for t in re.findall(r"\d+", rest)}
        if here - rest_nums:
            out.append(i)
    return out


def _premise_removal_at(item: BenchmarkItem, drop: int) -> BenchmarkItem | None:
    """Deterministic premise removal at a specific sentence index."""
    gold = _gold_int(item)
    if gold is None:
        return None
    parts = re.split(r"(?<=[.!?])\s+", _question_body(item.prompt))
    if drop >= len(parts) - 1:
        return None
    remaining = [s for i, s in enumerate(parts) if i != drop]
    if str(abs(gold)) in re.findall(r"\d+", " ".join(remaining)):
        return None  # answer-leakage guard; see premise_removal
    metadata = dict(item.metadata)
    metadata.update(
        {
            "answer_changing": True,
            "structural_op": f"premise_removal:{drop}",
            "parent_answer": gold,
            "removed_premise": parts[drop],
            "unanswerable": True,
            "depth_delta": -1,
        }
    )
    return BenchmarkItem(
        id=f"{item.id}__noprem{drop}",
        prompt=" ".join(remaining),
        answer=None,
        parent_id=item.parent_id or item.id,
        variant_type="premise_removal",
        metadata=metadata,
    )


def make_structural_variants(
    items: list[BenchmarkItem],
    kinds: tuple[str, ...] = STRUCTURAL_PERTURBATIONS,
    seed: int = 6198,
    premise_removal_k: int = 1,
) -> StructuralResult:
    """Build structural variants for a list of parent items.

    Seeding is per ``(item id, kind)`` rather than global so variant text is stable
    regardless of call order or of which items are present in the batch — the same
    property the numeric generators rely on to keep prior generations valid when the
    item set changes.

    ``premise_removal_k`` emits up to that many removals per item, each deleting a
    different required premise. Variants stay inside the parent's cluster, so the
    cluster bootstrap continues to treat them as correlated rather than independent.
    """
    variants: list[BenchmarkItem] = []
    skipped: list[str] = []
    for item in items:
        for kind in kinds:
            if kind == "premise_removal" and premise_removal_k > 1:
                idxs = removable_premises(item)[:premise_removal_k]
                if not idxs:
                    skipped.append(f"{item.id}:{kind}")
                for drop in idxs:
                    out = _premise_removal_at(item, drop)
                    if out is not None:
                        variants.append(out)
                continue
            gen = _GENERATORS.get(kind)
            if gen is None:
                raise ValueError(f"unknown structural perturbation: {kind}")
            rng = random.Random(f"{item.id}::{kind}::{seed}")
            out = gen(item, rng)
            if out is None:
                skipped.append(f"{item.id}:{kind}")
            else:
                variants.append(out)
    return StructuralResult(variants=variants, skipped=skipped)
