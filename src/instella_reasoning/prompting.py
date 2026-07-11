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


def build_prompt(item: BenchmarkItem) -> str:
    """Wrap a benchmark prompt with the chain-of-thought instruction for its type."""
    benchmark = str(item.metadata.get("benchmark", ""))
    skill = str(item.metadata.get("skill", ""))
    if benchmark in {"arc_challenge", "logiqa2", "reclor"} or "choices" in item.metadata:
        return item.prompt + MC_SUFFIX
    if benchmark == "bbh" and _looks_like_yes_no(item):
        return item.prompt + YESNO_SUFFIX
    if skill in {"arithmetic", "mathematical"} or benchmark in {"gsm8k", "math"}:
        return item.prompt + COT_SUFFIX
    return item.prompt + COT_SUFFIX


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
