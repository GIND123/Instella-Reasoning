from __future__ import annotations

from pathlib import Path

from instella_reasoning.records import (
    BenchmarkItem,
    EvaluationRecord,
    GenerationRecord,
    write_jsonl,
)
from instella_reasoning.text import normalize_answer


def score_generations(
    benchmark: list[BenchmarkItem],
    generations: list[GenerationRecord],
    output_path: str | Path,
) -> list[EvaluationRecord]:
    by_id = {item.id: item for item in benchmark}
    rows: list[EvaluationRecord] = []

    for generation in generations:
        item = by_id.get(generation.benchmark_id)
        if item is None:
            raise ValueError(f"Generation references unknown benchmark id: {generation.benchmark_id}")
        normalized_expected = normalize_answer(item.answer) if item.answer is not None else None
        normalized_predicted = normalize_answer(generation.completion)
        rows.append(
            EvaluationRecord(
                benchmark_id=item.id,
                parent_id=item.parent_id or item.id,
                variant_type=item.variant_type,
                expected=item.answer,
                predicted=generation.completion,
                normalized_expected=normalized_expected,
                normalized_predicted=normalized_predicted,
                correct=normalized_expected is not None and normalized_expected == normalized_predicted,
                model=generation.model,
                metadata=generation.metadata,
            )
        )

    write_jsonl(output_path, rows)
    return rows


def generate_with_transformers(
    benchmark: list[BenchmarkItem],
    model_name_or_path: str,
    output_path: str | Path,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    trust_remote_code: bool = True,
) -> list[GenerationRecord]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install Hugging Face dependencies with `pip install -e .[hf,train]` to generate completions."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    rows: list[GenerationRecord] = []
    do_sample = temperature > 0
    for item in benchmark:
        inputs = tokenizer(item.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(tokens[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        rows.append(
            GenerationRecord(
                benchmark_id=item.id,
                completion=completion,
                model=model_name_or_path,
                metadata={"max_new_tokens": max_new_tokens, "temperature": temperature},
            )
        )

    write_jsonl(output_path, rows)
    return rows
