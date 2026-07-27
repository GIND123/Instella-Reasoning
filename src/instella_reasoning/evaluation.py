from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm

from instella_reasoning.answer_equivalence import answers_equivalent, resolve_answer_kind
from instella_reasoning.prompting import build_prompt, extract_answer
from instella_reasoning.records import (
    BenchmarkItem,
    EvaluationRecord,
    GenerationRecord,
    read_jsonl,
    write_jsonl,
)
from instella_reasoning.text import normalize_text

#: A generation run must clear this semantic-completion rate, or its accuracy is an artifact
#: of the token budget rather than a property of the model.
MIN_TERMINATION_RATE = 0.85

# In few-shot mode the model continues the Question/Answer pattern past its own answer.
# Everything after the next exemplar header belongs to a hallucinated follow-up question,
# not to this item's response, and would otherwise poison last-number answer extraction.
_NEXT_EXEMPLAR = re.compile(r"\n\s*(?:Question|Q)\s*:", re.IGNORECASE)


def _truncate_at_next_exemplar(completion: str) -> str:
    match = _NEXT_EXEMPLAR.search(completion)
    return completion[: match.start()].rstrip() if match else completion


def _finalize_completion(completion: str, emitted_eos: bool, n_shot: int) -> tuple[str, bool]:
    """Return scoreable text and whether generation reached a semantic stopping point.

    Base checkpoints often finish an answer and continue into the next few-shot exemplar
    instead of emitting EOS. GSM-style models may likewise emit the final-answer marker
    before their EOS token. Both are completed measurements, not token-budget truncations.
    """
    scoreable = _truncate_at_next_exemplar(completion) if n_shot > 0 else completion
    reached_next_exemplar = scoreable != completion
    reached_answer_marker = "####" in scoreable
    return scoreable, emitted_eos or reached_next_exemplar or reached_answer_marker


@dataclass(slots=True)
class TerminationReport:
    """Whether a generation set reached a semantic stop or was cut off at the token cap."""

    model: str
    n: int
    n_finished: int
    n_with_answer_marker: int
    median_chars: int
    termination_rate: float
    marker_rate: float
    passes: bool
    threshold: float = MIN_TERMINATION_RATE

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n": self.n,
            "n_finished": self.n_finished,
            "termination_rate": round(self.termination_rate, 4),
            "n_with_answer_marker": self.n_with_answer_marker,
            "marker_rate": round(self.marker_rate, 4),
            "median_chars": self.median_chars,
            "threshold": self.threshold,
            "passes": self.passes,
        }


def termination_report(
    generations: list[GenerationRecord], threshold: float = MIN_TERMINATION_RATE
) -> TerminationReport:
    """Audit a generation set for truncation before its accuracy is believed.

    ``finished`` records EOS, a final-answer marker, or a few-shot exemplar boundary.
    ``marker_rate`` remains separate because a semantically complete response can still be
    hard to parse. A model that reaches neither signal is being scored on truncated text,
    which is exactly how a long-CoT checkpoint gets mistaken for a weak one.
    """
    if not generations:
        return TerminationReport("unknown", 0, 0, 0, 0, 0.0, 0.0, False, threshold)
    model = generations[0].model
    n = len(generations)
    n_finished = sum(1 for g in generations if g.metadata.get("finished", True))
    n_marker = sum(1 for g in generations if "####" in g.completion)
    lengths = sorted(len(g.completion) for g in generations)
    rate = n_finished / n
    return TerminationReport(
        model=model,
        n=n,
        n_finished=n_finished,
        n_with_answer_marker=n_marker,
        median_chars=lengths[len(lengths) // 2],
        termination_rate=rate,
        marker_rate=n_marker / n,
        passes=rate >= threshold,
        threshold=threshold,
    )


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
            # Correctness uses the *raw* extracted answers through a kind-aware equivalence
            # check (string -> numeric -> symbolic), so e.g. `\frac{1}{2}` == `0.5` on MATH.
            # This recovers correct answers that exact string match drops; it is conservative
            # (a parse failure never invents a match), so accuracy is not inflated.
            correct = item.answer is not None and answers_equivalent(
                item.answer, predicted_final, resolve_answer_kind(item)
            )
        else:
            from instella_reasoning.text import normalize_answer

            normalized_predicted = normalize_answer(generation.completion)
            normalized_expected = normalize_answer(item.answer) if item.answer is not None else None
            correct = (
                normalized_expected is not None and normalized_expected == normalized_predicted
            )

        rows.append(
            EvaluationRecord(
                benchmark_id=item.id,
                parent_id=item.parent_id or item.id,
                variant_type=item.variant_type,
                expected=item.answer,
                predicted=generation.completion,
                normalized_expected=normalized_expected,
                normalized_predicted=normalized_predicted or "",
                correct=correct,
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


def _resolve_dtype(dtype: str, cuda: bool):
    """Map a dtype name to a torch dtype. ``auto`` picks bf16 on GPU, fp32 on CPU.

    On a Turing card (T4, sm_75) bf16 has no native tensor-core support — it is
    numerically fine but slow. ``fp16`` is the faster Turing choice for a headline run;
    it is exposed explicitly rather than auto-selected because fp16 can overflow on some
    activations, so bf16 stays the safe default when the caller does not choose.
    """
    import torch

    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype != "auto":
        if dtype not in table:
            raise ValueError(f"Unknown dtype {dtype!r}; choose from auto/bf16/fp16/fp32.")
        return table[dtype]
    # auto: bf16 everywhere (matches how Instella was trained; halves the CPU footprint
    # so a 3B model still fits a stock CPU runtime for a load/smoke check).
    return torch.bfloat16


def _load_model(
    model_name_or_path: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
    dtype: str = "auto",
    revision: str | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    info = describe_device()
    cuda = info["cuda_available"]
    resolved_dtype = _resolve_dtype(dtype, cuda)
    print(
        f"[load] torch={info['torch_version']} device={info['device']} dtype={resolved_dtype}"
        + (f" revision={revision}" if revision else "")
        + (f" gpu={info['gpu_name']} (x{info['gpu_count']})" if cuda else "")
    )

    # Instella's tokenizer needs left padding for correct batched decoder-only generation.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
        revision=revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "revision": revision,
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
                bnb_4bit_compute_dtype=resolved_dtype,
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
        kwargs["torch_dtype"] = resolved_dtype
        kwargs["device_map"] = "auto"
    else:
        # CPU: keep the whole model in RAM. Never use device_map="auto" here — with no
        # GPU it silently spills weights to disk (the "offloaded to the disk" message),
        # which makes generation effectively hang. bf16 halves the footprint (~6 GB for
        # a 3B model) so it fits in a stock Colab CPU runtime for a load/smoke check.
        kwargs["torch_dtype"] = resolved_dtype

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
    dtype: str = "auto",
    revision: str | None = None,
    n_shot: int = 0,
) -> list[GenerationRecord]:
    """Generate model completions with chain-of-thought prompting.

    Portable across a Colab T4 (CUDA, optionally 4-bit for the 3B models) and a
    CPU-only machine (bf16/fp32). ``use_cot_prompt`` wraps each item with the
    benchmark-appropriate CoT instruction so answers are parseable by the extractors;
    ``use_chat_template`` then formats it for instruction-tuned models (see
    :func:`_format_prompts`). Disable the latter only for a base (non-Instruct) model.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: a batch is flushed to disk as soon as it is generated (below), so an
    # interrupted run leaves valid partial output. Re-running skips items already present,
    # continuing mid-benchmark instead of restarting it — critical on preemptible GPUs.
    # This runs before importing torch so a fully-complete run is recognized as done even
    # on a machine without the heavy ML deps (and CI can test the fast-forward path).
    existing: list[GenerationRecord] = []
    done_ids: set[str] = set()
    if out.exists():
        for row in read_jsonl(out):
            try:
                rec = GenerationRecord.from_dict(row)
            except ValueError:
                continue
            existing.append(rec)
            done_ids.add(rec.benchmark_id)
    pending = [item for item in benchmark if item.id not in done_ids]
    if done_ids:
        print(f"[generate] resume: {len(done_ids)} done, {len(pending)} remaining -> {out}")
    if not pending:
        return existing

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install Hugging Face dependencies with `pip install -e .[hf,train]` to generate completions."
        ) from exc

    def _append(records: list[GenerationRecord]) -> None:
        with out.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    tokenizer, model = _load_model(
        model_name_or_path, load_in_4bit, trust_remote_code, dtype=dtype, revision=revision
    )
    quantized = load_in_4bit and torch.cuda.is_available()
    # Precision provenance so downstream analysis can separate the accurate bf16 headline
    # run from a fast, lower-fidelity 4-bit pass (NF4 measurably shifts accuracy).
    base_dtype = "bf16" if dtype == "auto" else dtype
    precision = "nf4-4bit" if quantized else base_dtype
    do_sample = temperature > 0
    prompts = [
        build_prompt(item, n_shot=n_shot) if use_cot_prompt else item.prompt for item in pending
    ]
    eos_id = getattr(tokenizer, "eos_token_id", None)

    rows: list[GenerationRecord] = list(existing)
    templated_any = False
    model_label = Path(model_name_or_path).name
    with tqdm(
        total=len(benchmark),
        initial=len(benchmark) - len(pending),
        desc=f"Generate {model_label}",
        unit="item",
        dynamic_ncols=True,
        leave=True,
    ) as progress:
        for start in range(0, len(pending), batch_size):
            batch_items = pending[start : start + batch_size]
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
            batch_rows: list[GenerationRecord] = []
            for item, sequence in zip(batch_items, tokens, strict=False):
                generated = sequence[prompt_len:]
                # Did the model reach a semantic stop, or run out of budget mid-sentence?
                # This distinction is not cosmetic: at 512 tokens the Math checkpoint hit the
                # cap on 81% of items, emitted its '####' marker on 2%, and 26% of its
                # responses contained the gold answer yet scored wrong. Recording it per row
                # lets `termination_report` gate the run before the numbers are believed.
                if eos_id is not None:
                    emitted_eos = bool((generated == eos_id).any().item())
                else:  # pragma: no cover - tokenizer without an EOS id
                    emitted_eos = int(generated.shape[-1]) < max_new_tokens
                raw_completion = tokenizer.decode(generated, skip_special_tokens=True)
                completion, finished = _finalize_completion(
                    raw_completion, emitted_eos=emitted_eos, n_shot=n_shot
                )
                batch_rows.append(
                    GenerationRecord(
                        benchmark_id=item.id,
                        completion=completion,
                        model=model_name_or_path,
                        metadata={
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "load_in_4bit": quantized,  # effective, not just requested
                            "precision": precision,  # "bf16" (headline) / "fp16" / "nf4-4bit"
                            "revision": revision,  # model commit pinned for reproducibility
                            "cot_prompt": use_cot_prompt,
                            "chat_template": used_template,
                            "n_shot": n_shot,
                            "emitted_eos": emitted_eos,
                            "finished": finished,
                        },
                    )
                )
            _append(batch_rows)  # flush so an interrupt leaves valid output
            rows.extend(batch_rows)
            progress.update(len(batch_rows))

    if use_chat_template and not templated_any:
        print(
            f"[generate] NOTE: '{model_name_or_path}' has no chat template; used raw prompts. "
            "If this is an -Instruct/-Math checkpoint, output may be degenerate — verify the model id."
        )

    return rows
