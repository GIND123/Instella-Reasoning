"""Chain-of-thought prompt templates and per-benchmark answer extractors.

The evaluation stage (Phase 2 of the proposal) runs each Instella variant with
chain-of-thought prompting, then extracts a final answer with benchmark-specific
parsing before exact-match scoring. Keeping prompting and extraction here — keyed by
the ``metadata.benchmark`` / ``metadata.skill`` tags the loaders attach — lets the
generation and scoring code stay generic.

Extractors are deliberately conservative: they look for the strongest final-answer
signal (``#### N``, ``\\boxed{}``, "the answer is X", a trailing multiple-choice
letter) and fall back to the last number or the raw text. Each returns a *normalized*
string so scoring is a plain equality check.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from instella_reasoning.records import BenchmarkItem

# -- prompt templates ----------------------------------------------------------

COT_SUFFIX = "\nLet's think step by step, then give the final answer after '####'."
MC_SUFFIX = (
    "\nThink step by step, then answer with the single letter of the correct option "
    "after '####'."
)
YESNO_SUFFIX = "\nThink step by step, then answer 'yes' or 'no' after '####'."


# -- few-shot exemplars --------------------------------------------------------
#
# Base (non-instruction-tuned) checkpoints cannot follow a zero-shot CoT instruction:
# measured on this repo's own run, AMD-OLMo-1B produced degenerate loops on 408/556 items
# and emitted the '####' marker on 2%. Comparing a base checkpoint to an instruction-tuned
# one under zero-shot prompting measures instruction-following, not reasoning, which
# silently confounds every scale and post-training claim.
#
# Exemplars are hand-written in the style of Wei et al. (2022) rather than sampled from
# GSM8K, deliberately: this study's treatment variable is whether a *benchmark item* was
# in training, so putting real GSM8K items in the prompt would contaminate the prompt with
# the very thing being measured.

_ARITHMETIC_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "A baker has 5 trays with 8 muffins on each tray. He sells 17 muffins. "
        "How many muffins are left?",
        "The baker starts with 5 * 8 = 40 muffins.\n"
        "After selling 17, he has 40 - 17 = 23 muffins left.\n#### 23",
    ),
    (
        "Maria reads 12 pages every night. Her book has 96 pages. "
        "How many nights will it take her to finish?",
        "Each night Maria reads 12 pages.\n"
        "To finish 96 pages she needs 96 / 12 = 8 nights.\n#### 8",
    ),
    (
        "A shirt costs $24. It is on sale for a quarter off. How much does it cost now?",
        "A quarter of $24 is 24 / 4 = $6.\n"
        "So the sale price is 24 - 6 = $18.\n#### 18",
    ),
    (
        "Tom has 3 boxes of pencils with 15 pencils each, and he buys 20 more pencils. "
        "How many pencils does he have?",
        "The boxes hold 3 * 15 = 45 pencils.\n"
        "Adding the 20 he bought gives 45 + 20 = 65 pencils.\n#### 65",
    ),
)

_MC_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "Which object is the best conductor of electricity?\n"
        "A. a rubber band\nB. a copper wire\nC. a glass rod\nD. a wooden spoon",
        "Metals conduct electricity well, and copper is a metal.\n"
        "Rubber, glass, and wood are insulators.\n#### B",
    ),
    (
        "All birds have feathers. A robin is a bird. What follows?\n"
        "A. A robin has feathers.\nB. A robin can swim.\n"
        "C. All feathered animals are robins.\nD. Nothing follows.",
        "Every bird has feathers, and a robin is a bird.\n"
        "So a robin must have feathers.\n#### A",
    ),
)

_YESNO_SHOTS: tuple[tuple[str, str], ...] = (
    ("Is the expression ( True and False ) or True true or false?",
     "( True and False ) is False.\nFalse or True is True.\n#### yes"),
    ("Is the expression not ( True or False ) true or false?",
     "( True or False ) is True.\nnot True is False.\n#### no"),
)


def _render_shots(shots: tuple[tuple[str, str], ...], n_shot: int) -> str:
    used = shots[: max(0, n_shot)]
    if not used:
        return ""
    blocks = [f"Question: {q}\nAnswer: {a}" for q, a in used]
    return "\n\n".join(blocks) + "\n\n"


def _shots_for(item: BenchmarkItem) -> tuple[tuple[str, str], ...]:
    benchmark = str(item.metadata.get("benchmark", ""))
    if benchmark in {"arc_challenge", "logiqa2", "reclor"} or "choices" in item.metadata:
        return _MC_SHOTS
    if benchmark == "bbh" and _looks_like_yes_no(item):
        return _YESNO_SHOTS
    return _ARITHMETIC_SHOTS


def build_prompt(item: BenchmarkItem, n_shot: int = 0) -> str:
    """Wrap a benchmark prompt with the chain-of-thought instruction for its type.

    ``n_shot`` > 0 prepends worked exemplars in a ``Question:``/``Answer:`` format and
    frames the item the same way, which is what makes a base checkpoint emit a parseable
    final answer instead of looping. Instruction-tuned checkpoints should stay at
    ``n_shot=0`` and use their chat template.
    """
    benchmark = str(item.metadata.get("benchmark", ""))
    skill = str(item.metadata.get("skill", ""))
    if benchmark in {"arc_challenge", "logiqa2", "reclor"} or "choices" in item.metadata:
        suffix = MC_SUFFIX
    elif benchmark == "bbh" and _looks_like_yes_no(item):
        suffix = YESNO_SUFFIX
    elif skill in {"arithmetic", "mathematical"} or benchmark in {"gsm8k", "math"}:
        suffix = COT_SUFFIX
    else:
        suffix = COT_SUFFIX

    if n_shot > 0:
        preamble = _render_shots(_shots_for(item), n_shot)
        # The trailing "Answer:" is load-bearing: it puts the model in completion mode so
        # it continues the pattern rather than restating the question.
        return f"{preamble}Question: {item.prompt}{suffix}\nAnswer:"
    return item.prompt + suffix


def _looks_like_yes_no(item: BenchmarkItem) -> bool:
    answer = (item.answer or "").strip().lower()
    return answer in {"yes", "no", "true", "false"}


# -- answer extraction ---------------------------------------------------------

_HASH_ANSWER = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_IS = re.compile(
    r"(?:final\s+answer|answer\s+is|answer\s*[:=])\s*\$?\\?\.?\s*([^\n\.]+)", re.IGNORECASE
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_MC_LETTER = re.compile(r"\b([A-E])\b")


def _clean_number(text: str) -> str:
    return text.replace(",", "").replace("$", "").strip().rstrip(".")


def _after_hash(completion: str) -> str | None:
    matches = _HASH_ANSWER.findall(completion)
    return matches[-1].strip() if matches else None


def extract_numeric(completion: str) -> str:
    hashed = _after_hash(completion)
    if hashed:
        numbers = _NUMBER.findall(hashed)
        if numbers:
            return _clean_number(numbers[-1])
        return _clean_number(hashed)
    match = _ANSWER_IS.search(completion)
    if match:
        numbers = _NUMBER.findall(match.group(1))
        if numbers:
            return _clean_number(numbers[-1])
    numbers = _NUMBER.findall(completion)
    return _clean_number(numbers[-1]) if numbers else completion.strip()


def extract_boxed(completion: str) -> str:
    boxed = _BOXED.findall(completion)
    if boxed:
        return boxed[-1].strip()
    hashed = _after_hash(completion)
    if hashed:
        return hashed.strip()
    return extract_numeric(completion)


def extract_multiple_choice(completion: str) -> str:
    hashed = _after_hash(completion)
    search_space = hashed if hashed else completion
    match = _ANSWER_IS.search(search_space) if not hashed else None
    if match:
        search_space = match.group(1)
    letters = _MC_LETTER.findall(search_space.upper())
    if letters:
        return letters[-1]
    # Numeric option labels (1-5) as a fallback.
    numbers = _NUMBER.findall(search_space)
    return numbers[-1] if numbers else search_space.strip()


def extract_yes_no(completion: str) -> str:
    hashed = _after_hash(completion) or completion
    match = re.search(r"\b(yes|no|true|false)\b", hashed, re.IGNORECASE)
    return match.group(1).lower() if match else hashed.strip().lower()


@dataclass(slots=True)
class Extractor:
    name: str
    fn: Callable[[str], str]


_EXTRACTORS: dict[str, Extractor] = {
    "numeric": Extractor("numeric", extract_numeric),
    "boxed": Extractor("boxed", extract_boxed),
    "multiple_choice": Extractor("multiple_choice", extract_multiple_choice),
    "yes_no": Extractor("yes_no", extract_yes_no),
}


def extractor_for(item: BenchmarkItem) -> Extractor:
    """Choose the answer extractor for an item from its benchmark/skill tags."""
    benchmark = str(item.metadata.get("benchmark", ""))
    skill = str(item.metadata.get("skill", ""))
    if benchmark == "math" or skill == "mathematical":
        return _EXTRACTORS["boxed"]
    if benchmark in {"arc_challenge", "logiqa2", "reclor"} or "choices" in item.metadata:
        return _EXTRACTORS["multiple_choice"]
    if benchmark == "gsm8k" or skill == "arithmetic":
        return _EXTRACTORS["numeric"]
    if _looks_like_yes_no(item):
        return _EXTRACTORS["yes_no"]
    return _EXTRACTORS["numeric"]


def extract_answer(item: BenchmarkItem, completion: str) -> str:
    """Extract the final answer from a completion using the item's extractor."""
    return extractor_for(item).fn(completion)
