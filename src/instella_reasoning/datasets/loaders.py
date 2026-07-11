"""Load public reasoning benchmarks and Instella corpus shards into the JSONL contract.

Each benchmark loader converts a HuggingFace dataset into ``BenchmarkItem`` rows with
a consistent ``metadata.skill`` tag (the reasoning sub-skill) and ``metadata.benchmark``
name, so the contamination, evaluation, and reliability stages treat every benchmark
uniformly.

The ``datasets`` library is only imported when a loader actually runs, and a clear
install message is raised otherwise. This keeps the core pipeline importable on a
machine that has not installed the ``hf`` extra.

Answer normalization is benchmark-specific (GSM8K ``#### N``, MATH ``\\boxed{}``,
multiple-choice letter, etc.) and lives here so downstream scoring stays generic.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from instella_reasoning.records import BenchmarkItem, CorpusDocument, write_jsonl


@dataclass(slots=True)
class BenchmarkSpec:
    name: str
    skill: str
    hf_path: str
    hf_name: str | None
    split: str
    loader: Callable[[object, str], Iterator[BenchmarkItem]]


def _require_datasets():
    try:
        from datasets import load_dataset

        return load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "Benchmark/corpus loading needs the `datasets` library. "
            "Install with `pip install -e .[hf]`."
        ) from exc


# -- answer extraction helpers --------------------------------------------------

_GSM8K_ANSWER = re.compile(r"####\s*(.+)")
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def _gsm8k_answer(raw: str) -> str:
    match = _GSM8K_ANSWER.search(raw)
    value = match.group(1) if match else raw
    return value.replace(",", "").strip()


def _math_answer(solution: str) -> str:
    matches = _BOXED.findall(solution)
    return matches[-1].strip() if matches else solution.strip()


def _letter(index: int) -> str:
    return chr(ord("A") + index)


# -- per-benchmark row generators ----------------------------------------------


def _load_gsm8k(dataset, skill: str) -> Iterator[BenchmarkItem]:
    for i, row in enumerate(dataset):
        yield BenchmarkItem(
            id=f"gsm8k_{i:05d}",
            prompt=str(row["question"]).strip(),
            answer=_gsm8k_answer(str(row["answer"])),
            parent_id=f"gsm8k_{i:05d}",
            metadata={"benchmark": "gsm8k", "skill": skill, "rationale": str(row["answer"])},
        )


def _load_math(dataset, skill: str) -> Iterator[BenchmarkItem]:
    for i, row in enumerate(dataset):
        level = row.get("level", "")
        yield BenchmarkItem(
            id=f"math_{i:05d}",
            prompt=str(row["problem"]).strip(),
            answer=_math_answer(str(row.get("solution", ""))),
            parent_id=f"math_{i:05d}",
            metadata={
                "benchmark": "math",
                "skill": skill,
                "level": level,
                "type": row.get("type", ""),
            },
        )


def _load_arc(dataset, skill: str) -> Iterator[BenchmarkItem]:
    for i, row in enumerate(dataset):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=False))
        prompt = f"{str(row['question']).strip()}\n{options}"
        yield BenchmarkItem(
            id=f"arc_{i:05d}",
            prompt=prompt,
            answer=str(row["answerKey"]).strip(),
            parent_id=f"arc_{i:05d}",
            metadata={"benchmark": "arc_challenge", "skill": skill, "choices": list(labels)},
        )


def _load_logiqa(dataset, skill: str) -> Iterator[BenchmarkItem]:
    for i, row in enumerate(dataset):
        options = row.get("options") or row.get("choices") or []
        rendered = "\n".join(f"{_letter(j)}. {opt}" for j, opt in enumerate(options))
        context = str(row.get("context", "")).strip()
        query = str(row.get("query") or row.get("question", "")).strip()
        prompt = "\n".join(part for part in [context, query, rendered] if part)
        correct = row.get("correct_option", row.get("answer"))
        answer = _letter(int(correct)) if isinstance(correct, int) else str(correct)
        yield BenchmarkItem(
            id=f"logiqa_{i:05d}",
            prompt=prompt,
            answer=answer,
            parent_id=f"logiqa_{i:05d}",
            metadata={"benchmark": "logiqa2", "skill": skill},
        )


def _load_bbh(dataset, skill: str) -> Iterator[BenchmarkItem]:
    for i, row in enumerate(dataset):
        yield BenchmarkItem(
            id=f"bbh_{i:05d}",
            prompt=str(row["input"]).strip(),
            answer=str(row["target"]).strip(),
            parent_id=f"bbh_{i:05d}",
            metadata={"benchmark": "bbh", "skill": skill},
        )


def _load_generic_qa(dataset, skill: str) -> Iterator[BenchmarkItem]:
    """Fallback loader for question/answer-style datasets (best effort)."""
    for i, row in enumerate(dataset):
        prompt = row.get("question") or row.get("problem") or row.get("input")
        answer = row.get("answer") or row.get("target") or row.get("label")
        if prompt is None:
            continue
        yield BenchmarkItem(
            id=f"{skill}_{i:05d}",
            prompt=str(prompt).strip(),
            answer=None if answer is None else str(answer).strip(),
            parent_id=f"{skill}_{i:05d}",
            metadata={"skill": skill},
        )


BENCHMARK_LOADERS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec("gsm8k", "arithmetic", "openai/gsm8k", "main", "test", _load_gsm8k),
    "math": BenchmarkSpec(
        "math", "mathematical", "EleutherAI/hendrycks_math", "algebra", "test", _load_math
    ),
    "arc_challenge": BenchmarkSpec(
        "arc_challenge",
        "commonsense_science",
        "allenai/ai2_arc",
        "ARC-Challenge",
        "test",
        _load_arc,
    ),
    "logiqa2": BenchmarkSpec(
        "logiqa2", "logical_deduction", "datatune/LogiQA2.0", None, "test", _load_logiqa
    ),
    "bbh": BenchmarkSpec(
        "bbh", "multi_step", "lukaemon/bbh", "boolean_expressions", "test", _load_bbh
    ),
}


def load_benchmark_to_jsonl(
    benchmark: str,
    output_path: str | Path,
    split: str | None = None,
    hf_name: str | None = None,
    limit: int | None = None,
) -> int:
    """Download a benchmark from HuggingFace and write it as a normalized JSONL.

    Returns the number of items written. ``hf_name`` overrides the default config
    (e.g. a different MATH subject or BBH subtask); ``limit`` caps the item count
    for quick local runs.
    """

    spec = BENCHMARK_LOADERS.get(benchmark)
    if spec is None:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. Known: {sorted(BENCHMARK_LOADERS)}"
        )
    load_dataset = _require_datasets()
    config = hf_name if hf_name is not None else spec.hf_name
    dataset = load_dataset(spec.hf_path, config, split=split or spec.split)

    rows: list[BenchmarkItem] = []
    for item in spec.loader(dataset, spec.skill):
        rows.append(item)
        if limit is not None and len(rows) >= limit:
            break
    write_jsonl(output_path, rows)
    return len(rows)


def load_corpus_dataset_to_jsonl(
    hf_path: str,
    output_path: str | Path,
    text_field: str = "text",
    source: str | None = None,
    hf_name: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> int:
    """Stream a HuggingFace corpus/dataset into ``CorpusDocument`` JSONL.

    For very large corpora (DCLM, FineWeb-Edu) pass a ``limit`` to sample; the
    proposal indexes small high-risk datasets (Instella-GSM8K-synthetic,
    OpenMathInstruct-2, dm_math) fully and samples the web corpora.
    """

    load_dataset = _require_datasets()
    streaming = limit is not None
    dataset = load_dataset(hf_path, hf_name, split=split, streaming=streaming)
    src = source or hf_path

    rows: list[CorpusDocument] = []
    for i, row in enumerate(dataset):
        text = row.get(text_field)
        if text is None:
            # For instruction datasets, join problem/solution style fields.
            text = " ".join(
                str(row[key]) for key in ("problem", "question", "solution", "answer") if key in row
            )
        if not text:
            continue
        rows.append(
            CorpusDocument(
                id=f"{src.replace('/', '_')}_{i:07d}",
                text=str(text),
                source=src,
                metadata={"hf_path": hf_path, "hf_name": hf_name, "split": split},
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    write_jsonl(output_path, rows)
    return len(rows)
