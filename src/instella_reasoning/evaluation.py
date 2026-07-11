from __future__ import annotations

from pathlib import Path

from instella_reasoning.prompting import build_prompt, extract_answer
from instella_reasoning.records import (
    BenchmarkItem,
    EvaluationRecord,
    GenerationRecord,
    write_jsonl,
)
from instella_reasoning.text import normalize_text


def _normalize_final(value: str | None) -> str | None:
    """Normalize an already-extracted final answer for exact-match comparison."""
    if value is None:
        return None
    text = normalize_text(value)
    text = " ".join(word for word in text.split() if word not in {"the", "a", "an"})
    return text


def score_generations(
    benchmark: list[BenchmarkItem],
    generations: list[GenerationRecord],
    output_path: str | Path,
    use_extractors: bool = True,
) -> list[EvaluationRecord]:
    """Score generations against benchmark answers.

    When ``use_extractors`` is True (default), each completion is parsed with the
    benchmark-specific extractor (GSM8K numeric, MATH boxed, multiple-choice letter,
    yes/no) before normalization — far more reliable than the generic last-number
    heuristic. Set it False to reproduce the original lenient behavior.
    """

    by_id = {item.id: item for item in benchmark}
    rows: list[EvaluationRecord] = []

    for generation in generations:
        item = by_id.get(generation.benchmark_id)
        if item is None:
            raise ValueError(f"Generation references unknown benchmark id: {generation.benchmark_id}")

        if use_extractors:
            predicted_final = extract_answer(item, generation.completion)
            normalized_predicted = _normalize_final(predicted_final)
            normalized_expected = _normalize_final(item.answer) if item.answer is not None else None
        else:
            from instella_reasoning.text import normalize_answer

            normalized_predicted = normalize_answer(generation.completion)
            normalized_expected = normalize_answer(item.answer) if item.answer is not None else None

        rows.append(
            EvaluationRecord(
                benchmark_id=item.id,
                parent_id=item.parent_id or item.id,
                variant_type=item.variant_type,
                expected=item.answer,
                predicted=generation.completion,
                normalized_expected=normalized_expected,
                normalized_predicted=normalized_predicted or "",
                correct=normalized_expected is not None
                and normalized_expected == normalized_predicted,
                model=generation.model,
                metadata={**generation.metadata, **item.metadata},
            )
        )

    write_jsonl(output_path, rows)
    return rows


def _load_model(
    model_name_or_path: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError as exc:  # pragma: no cover - needs train extra
            raise RuntimeError(
                "4-bit quantization needs bitsandbytes. Install with `pip install bitsandbytes`, "
                "or pass load_in_4bit=False to run in bf16/fp32."
            ) from exc
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return tokenizer, model


def generate_with_transformers(
    benchmark: list[BenchmarkItem],
    model_name_or_path: str,
    output_path: str | Path,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    batch_size: int = 1,
    use_cot_prompt: bool = True,
) -> list[GenerationRecord]:
    """Generate model completions with chain-of-thought prompting.

    Portable across a Colab T4 (CUDA, optionally 4-bit for the 3B models) and a
    CPU-only machine (bf16/fp32). ``use_cot_prompt`` wraps each item with the
    benchmark-appropriate CoT instruction so answers are parseable by the extractors.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install Hugging Face dependencies with `pip install -e .[hf,train]` to generate completions."
        ) from exc

    tokenizer, model = _load_model(model_name_or_path, load_in_4bit, trust_remote_code)
    do_sample = temperature > 0
    prompts = [build_prompt(item) if use_cot_prompt else item.prompt for item in benchmark]

    rows: list[GenerationRecord] = []
    for start in range(0, len(benchmark), batch_size):
        batch_items = benchmark[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        for item, sequence in zip(batch_items, tokens, strict=False):
            completion = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
            rows.append(
                GenerationRecord(
                    benchmark_id=item.id,
                    completion=completion,
                    model=model_name_or_path,
                    metadata={
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "load_in_4bit": load_in_4bit,
                        "cot_prompt": use_cot_prompt,
                    },
                )
            )

    write_jsonl(output_path, rows)
    return rows
