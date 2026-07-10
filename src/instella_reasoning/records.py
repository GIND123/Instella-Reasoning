from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


@dataclass(slots=True)
class BenchmarkItem:
    id: str
    prompt: str
    answer: str | None = None
    parent_id: str | None = None
    variant_type: str = "original"
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: JsonDict) -> BenchmarkItem:
        prompt = row.get("prompt") or row.get("question") or row.get("input")
        if prompt is None:
            raise ValueError(f"Benchmark row {row.get('id', '<missing id>')} has no prompt/question/input field")
        return cls(
            id=str(row.get("id") or row.get("uid")),
            prompt=str(prompt),
            answer=None if row.get("answer") is None else str(row["answer"]),
            parent_id=None if row.get("parent_id") is None else str(row["parent_id"]),
            variant_type=str(row.get("variant_type", "original")),
            metadata=dict(row.get("metadata", {})),
        )


@dataclass(slots=True)
class CorpusDocument:
    id: str
    text: str
    source: str = "unknown"
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: JsonDict) -> CorpusDocument:
        text = row.get("text") or row.get("content") or row.get("document")
        if text is None:
            raise ValueError(f"Corpus row {row.get('id', '<missing id>')} has no text/content/document field")
        return cls(
            id=str(row.get("id") or row.get("doc_id")),
            text=str(text),
            source=str(row.get("source", "unknown")),
            metadata=dict(row.get("metadata", {})),
        )


@dataclass(slots=True)
class ContaminationHit:
    benchmark_id: str
    document_id: str
    source: str
    label: str
    score: float
    token_jaccard: float
    char_ngram_jaccard: float
    excerpt: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRecord:
    benchmark_id: str
    completion: str
    model: str = "unknown"
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: JsonDict) -> GenerationRecord:
        completion = row.get("completion") or row.get("prediction") or row.get("response")
        if completion is None:
            raise ValueError(f"Generation row {row.get('benchmark_id', '<missing id>')} has no completion")
        return cls(
            benchmark_id=str(row.get("benchmark_id") or row.get("id")),
            completion=str(completion),
            model=str(row.get("model", "unknown")),
            metadata=dict(row.get("metadata", {})),
        )


@dataclass(slots=True)
class EvaluationRecord:
    benchmark_id: str
    parent_id: str
    variant_type: str
    expected: str | None
    predicted: str
    normalized_expected: str | None
    normalized_predicted: str
    correct: bool
    model: str = "unknown"
    metadata: JsonDict = field(default_factory=dict)


def read_jsonl(path: str | Path) -> Iterator[JsonDict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if is_dataclass(row):
                payload = asdict(row)
            else:
                payload = row
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def read_benchmark(path: str | Path) -> list[BenchmarkItem]:
    return [BenchmarkItem.from_dict(row) for row in read_jsonl(path)]


def read_corpus(path: str | Path) -> list[CorpusDocument]:
    return [CorpusDocument.from_dict(row) for row in read_jsonl(path)]
