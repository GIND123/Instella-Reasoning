from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable

_FINAL_ANSWER_PATTERNS = [
    re.compile(r"####\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"(?:final\s+answer|answer\s+is|therefore)\s*[:=]?\s*([^\n\.]+)", re.IGNORECASE),
]


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    lowered = value.lower().strip()
    translated = lowered.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(translated.split())


def normalize_answer(value: str | None) -> str:
    text = normalize_text(extract_final_answer(value or ""))
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return " ".join(text.split())


def extract_final_answer(completion: str) -> str:
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(completion)
        if match:
            return match.group(1).strip()

    yes_no = re.match(r"^\s*(yes|no)\b", completion, re.IGNORECASE)
    if yes_no:
        return yes_no.group(1).strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", completion)
    if numbers:
        return numbers[-1]
    return completion.strip()


def token_set(value: str) -> set[str]:
    return set(normalize_text(value).split())


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def char_ngrams(value: str, n: int = 5) -> set[str]:
    normalized = normalize_text(value)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def word_ngrams(value: str, n: int = 13) -> set[str]:
    """Return the set of word-level n-grams (Brown et al., 2020 contamination check)."""
    tokens = normalize_text(value).split()
    if len(tokens) < n:
        # Shorter than a full n-gram: treat the whole token sequence as one gram so
        # short prompts can still match a containing document.
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def word_ngram_overlap(query: str, document: str, n: int = 13) -> float:
    """Fraction of the query's word n-grams that also appear in the document.

    This is directional (query-normalized) so a short benchmark prompt embedded
    verbatim inside a long training document scores 1.0, matching the intent of
    contamination detection: "was this problem seen during training?".
    """
    query_grams = word_ngrams(query, n)
    if not query_grams:
        return 0.0
    doc_grams = word_ngrams(document, n)
    if not doc_grams:
        return 0.0
    return len(query_grams & doc_grams) / len(query_grams)


def weighted_token_overlap(query: str, document: str) -> float:
    query_counts = Counter(normalize_text(query).split())
    doc_counts = Counter(normalize_text(document).split())
    if not query_counts or not doc_counts:
        return 0.0
    overlap = sum(min(count, doc_counts[token]) for token, count in query_counts.items())
    return overlap / sum(query_counts.values())


def compact_excerpt(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
