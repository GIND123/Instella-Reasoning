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


def describe_device() -> dict:
    """Report the compute device the loader will actually use.

    Surfaces the two things that silently sank the first Colab run: a CPU-only
    torch build (``+cpu``, so ``--load-in-4bit`` is a no-op) and the absence of any
    visible CUDA GPU. Import-safe so a preflight can call it before any download.
    """
    import torch

    cuda = torch.cuda.is_available()
    info: dict = {
        "torch_version": torch.__version__,
        "cpu_only_build": "+cpu" in torch.__version__,
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
    }
    return info


def _load_model(
    model_name_or_path: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = describe_device()
    cuda = info["cuda_available"]
    print(
        f"[load] torch={info['torch_version']} device={info['device']}"
        + (f" gpu={info['gpu_name']} (x{info['gpu_count']})" if cuda else "")
    )

    # Instella's tokenizer needs left padding for correct batched decoder-only generation.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit and not cuda:
        print(
            "[load] WARNING: --load-in-4bit requested but no CUDA GPU is visible "
            "(torch is a CPU-only build if the version ends in '+cpu'). Falling back "
            "to full precision on CPU. Switch the Colab runtime to a GPU for 4-bit."
        )

    if load_in_4bit and cuda:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        except ImportError as exc:  # pragma: no cover - needs train extra
            raise RuntimeError(
                "4-bit quantization needs bitsandbytes. Install with `pip install bitsandbytes`, "
                "or pass load_in_4bit=False to run in bf16/fp32."
            ) from exc
    elif cuda:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    else:
        # CPU: keep the whole model in RAM. Never use device_map="auto" here — with no
        # GPU it silently spills weights to disk (the "offloaded to the disk" message),
        # which makes generation effectively hang. bf16 halves the footprint (~6 GB for
        # a 3B model) so it fits in a stock Colab CPU runtime for a load/smoke check.
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if "device_map" not in kwargs:
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


def _format_prompts(tokenizer, prompts: list[str], use_chat_template: bool):
    """Render user prompts for the model, applying its chat template when it has one.

    Instruction-tuned checkpoints (``Instella-3B-Instruct``/``-Math``) are trained with
    a chat format; feeding them a raw string produces degenerate, repetitive output.
    When the tokenizer exposes a ``chat_template`` we wrap each prompt as a single user
    turn with ``add_generation_prompt=True``; the template already injects the special
    tokens, so the caller must tokenize with ``add_special_tokens=False`` to avoid a
    duplicate BOS. Base checkpoints (no template) fall back to the raw prompt and normal
    special-token handling. Returns ``(texts, add_special_tokens, used_chat_template)``.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if use_chat_template and chat_template:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        return texts, False, True
    return prompts, True, False


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
    use_chat_template: bool = True,
) -> list[GenerationRecord]:
    """Generate model completions with chain-of-thought prompting.

    Portable across a Colab T4 (CUDA, optionally 4-bit for the 3B models) and a
    CPU-only machine (bf16/fp32). ``use_cot_prompt`` wraps each item with the
    benchmark-appropriate CoT instruction so answers are parseable by the extractors;
    ``use_chat_template`` then formats it for instruction-tuned models (see
    :func:`_format_prompts`). Disable the latter only for a base (non-Instruct) model.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install Hugging Face dependencies with `pip install -e .[hf,train]` to generate completions."
        ) from exc

    tokenizer, model = _load_model(model_name_or_path, load_in_4bit, trust_remote_code)
    quantized = load_in_4bit and torch.cuda.is_available()
    # Precision provenance so downstream analysis can separate the accurate bf16 headline
    # run from a fast, lower-fidelity 4-bit pass (NF4 measurably shifts accuracy).
    precision = "nf4-4bit" if quantized else "bf16"
    do_sample = temperature > 0
    prompts = [build_prompt(item) if use_cot_prompt else item.prompt for item in benchmark]

    rows: list[GenerationRecord] = []
    templated_any = False
    for start in range(0, len(benchmark), batch_size):
        batch_items = benchmark[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        texts, add_special, used_template = _format_prompts(
            tokenizer, batch_prompts, use_chat_template
        )
        templated_any = templated_any or used_template
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=add_special,
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
                        "load_in_4bit": quantized,  # effective, not just requested
                        "precision": precision,  # "bf16" (headline) or "nf4-4bit" (fast pass)
                        "cot_prompt": use_cot_prompt,
                        "chat_template": used_template,
                    },
                )
            )

    if use_chat_template and not templated_any:
        print(
            f"[generate] NOTE: '{model_name_or_path}' has no chat template; used raw prompts. "
            "If this is an -Instruct/-Math checkpoint, output may be degenerate — verify the model id."
        )

    write_jsonl(output_path, rows)
    return rows
