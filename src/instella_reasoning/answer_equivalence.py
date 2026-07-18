"""Answer-equivalence checking for scoring — especially MATH (proposal Phase 2).

Exact string match badly *under-counts* correct MATH answers: ``\\frac{1}{2}``,
``0.5``, ``1/2`` and ``\\dfrac{1}{2}`` are the same number but differ as strings, so a
naive comparison reports a correct model as wrong and biases every accuracy number (the
headline contaminated-vs-clean gap included). This module compares answers at three
increasingly permissive levels, each *conservative* (a parse failure returns ``False``,
never a spurious match, so accuracy is never inflated):

1. **Normalized string equality** — cheap, matches the historical behavior.
2. **Numeric equality** — parse both sides as numbers (fractions ``a/b``, decimals,
   ``$``/``,``/``%``) and compare with a relative tolerance.
3. **Symbolic equality** — for MATH, LaTeX-normalize both sides and check
   ``simplify(lhs - rhs) == 0`` with SymPy. SymPy is optional: without it (or on any
   parse error) this level is skipped and we fall back to levels 1-2.

The ``kind`` argument selects how hard to try; :func:`resolve_answer_kind` derives it
from an item's ``benchmark``/``skill`` tags, mirroring
:func:`instella_reasoning.prompting.extractor_for`.
"""

from __future__ import annotations

import math
import re
from typing import Any

# -- answer kind resolution ----------------------------------------------------


def resolve_answer_kind(item: Any) -> str:
    """Return the comparison kind for an item: math / numeric / multiple_choice / yes_no / text."""
    metadata = getattr(item, "metadata", {}) or {}
    benchmark = str(metadata.get("benchmark", ""))
    skill = str(metadata.get("skill", ""))
    answer = (getattr(item, "answer", None) or "").strip().lower()
    if benchmark == "math" or skill == "mathematical":
        return "math"
    if benchmark in {"arc_challenge", "logiqa2", "reclor"} or "choices" in metadata:
        return "multiple_choice"
    if benchmark == "gsm8k" or skill == "arithmetic":
        return "numeric"
    if answer in {"yes", "no", "true", "false"}:
        return "yes_no"
    return "text"


# -- level 1: normalized string ------------------------------------------------

_ARTICLES = {"the", "a", "an"}


def normalize_answer_text(value: str) -> str:
    """Lowercase, drop articles/currency/spacing noise; the cheap first-pass comparison."""
    text = value.strip().lower()
    text = text.replace("$", "").replace("\\$", "")
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    text = " ".join(word for word in text.split() if word not in _ARTICLES)
    return text


# -- level 2: numeric ----------------------------------------------------------

_FRAC_LATEX = re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def numeric_value(value: str) -> float | None:
    """Best-effort parse of a scalar answer to a float, or None if it is not numeric.

    Handles ``\\frac{a}{b}``, ``a/b``, decimals, thousands separators, ``$`` and a
    trailing ``%`` (interpreted as a fraction). Returns None on anything non-scalar
    (e.g. an interval or a symbolic expression) so the caller can try the symbolic path.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    percent = text.endswith("%")
    text = text.rstrip("%").strip()
    text = text.replace("$", "").replace("\\$", "").replace(",", "").replace(" ", "")

    frac = _FRAC_LATEX.search(text)
    if frac:
        num, den = _plain_float(frac.group(1)), _plain_float(frac.group(2))
        result = _safe_div(num, den)
    elif "/" in text:
        parts = text.split("/")
        if len(parts) != 2:
            return None
        result = _safe_div(_plain_float(parts[0]), _plain_float(parts[1]))
    else:
        result = _plain_float(text)

    if result is None:
        return None
    return result / 100.0 if percent else result


def _plain_float(text: str) -> float | None:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def numeric_equal(a: str, b: str, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    va, vb = numeric_value(a), numeric_value(b)
    if va is None or vb is None:
        return False
    return math.isclose(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)


# -- level 3: symbolic (SymPy, optional) ---------------------------------------


def sympy_available() -> bool:
    try:
        import sympy  # noqa: F401

        return True
    except ImportError:
        return False


def _latex_to_expr(value: str) -> str:
    """Turn a light subset of LaTeX math into a SymPy-parseable expression string."""
    text = value.strip()
    for token in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\ ", r"\displaystyle", "$"):
        text = text.replace(token, "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # \frac{A}{B} -> ((A)/(B)); iterate to catch a second (non-nested) fraction.
    for _ in range(4):
        new = _FRAC_LATEX.sub(r"((\1)/(\2))", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]+)\}", r"((\2)**(1/(\1)))", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt((\1))", text)
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\pi", "pi").replace(r"\%", "").replace("%", "")
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    return text.strip()


def symbolic_equal(a: str, b: str) -> bool:
    """True when a and b denote the same value symbolically (SymPy). Conservative on error."""
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return False

    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        expr_a = parse_expr(_latex_to_expr(a), transformations=transformations, evaluate=True)
        expr_b = parse_expr(_latex_to_expr(b), transformations=transformations, evaluate=True)
        diff = sympy.simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        # Inputs are arbitrary (often degenerate) model output; any parse/simplify failure
        # means "not provably equal", never a crash. An enumerated except-list already
        # missed tokenize.TokenError (stray backslash from unhandled LaTeX), which killed
        # a whole scoring run — hence the deliberate blanket catch.
        return False


# -- public entry point --------------------------------------------------------


def answers_equivalent(expected: str | None, predicted: str | None, kind: str = "text") -> bool:
    """Return True if ``predicted`` matches ``expected`` for the given answer ``kind``.

    Levels escalate by kind: every kind gets normalized string match; ``numeric``/``math``
    also get numeric comparison; ``math`` additionally gets SymPy symbolic comparison.
    A missing side, or any parse failure, yields False — the check never *invents* a match.
    """
    if expected is None or predicted is None:
        return False
    if normalize_answer_text(expected) == normalize_answer_text(predicted):
        return True
    if kind in {"numeric", "math"} and numeric_equal(expected, predicted):
        return True
    if kind == "math" and symbolic_equal(expected, predicted):
        return True
    return False
