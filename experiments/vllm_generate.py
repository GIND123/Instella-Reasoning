#!/usr/bin/env python
"""Generate completions with vLLM, emitting the repo's own GenerationRecord schema.

Why a second engine at all: the transformers path pads a fixed-size batch and runs it to
the token cap, so a deletion probe at a 2048-token budget on a base checkpoint costs
~8 s/item. vLLM's continuous batching retires each sequence at its own stop point, which
is worth roughly an order of magnitude on exactly this workload — most completions are
far shorter than the cap, and under static batching every one of them waits for the
longest sequence in its batch.

What must NOT change is the measurement. Three pieces are imported from the transformers
path rather than reimplemented, because a difference in any of them would enter the
checkpoint contrast as though it were a model difference:

* ``build_prompt``          — identical prompt text, including few-shot exemplars
* ``_format_prompts``       — identical chat-template handling and BOS accounting
* ``_finalize_completion``  — identical truncation-at-next-exemplar and ``finished`` flag

Tokenisation is done here rather than handed to the engine as a string, so
``add_special_tokens`` follows ``_format_prompts``' contract exactly: a chat template
already injects BOS, and letting vLLM re-tokenise the rendered string would add a second
one. That single token would shift greedy decoding.

The engine is recorded per row. Rows produced by different engines must never be pooled
into one contrast, and the metadata is what makes that checkable after the fact.

  python experiments/vllm_generate.py --benchmark X.jsonl --model amd/Instella-3B \
      --output gen.jsonl --max-new-tokens 2048 --n-shot 4 --no-chat-template
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from instella_reasoning.evaluation import _finalize_completion, _format_prompts
from instella_reasoning.prompting import build_prompt
from instella_reasoning.records import GenerationRecord, read_benchmark, read_jsonl

_ARCH_NAME = "InstellaForCausalLM"


def _register_arch() -> None:
    """Register the out-of-tree model class so the engine accepts the 3B checkpoints.

    Passed as a string, not a class object: the engine imports the model inside a separate
    process where a class captured here would not survive. The module must therefore be
    importable from that subprocess's environment (PYTHONPATH), which
    lambda_vllm_bootstrap.sh arranges via ~/.instella_vllm_env.
    """
    import importlib.util

    from vllm import ModelRegistry

    if importlib.util.find_spec("vllm_arch") is None:
        raise RuntimeError(
            "vllm_arch not importable. Run experiments/lambda_vllm_bootstrap.sh and "
            "source ~/.instella_vllm_env (it puts the arch dir on PYTHONPATH)."
        )
    ModelRegistry.register_model(_ARCH_NAME, "vllm_arch:InstellaForCausalLM")
    print("[arch] custom architecture registered", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None, help="default: 1.0 at T=0, else 0.95")
    ap.add_argument("--n-shot", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="draws per item. >1 only makes sense at T>0; a rate needs k draws.")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-cot-prompt", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    args = ap.parse_args()

    if args.n_samples > 1 and args.temperature == 0:
        raise SystemExit(
            "--n-samples>1 at temperature 0 draws the same completion k times. "
            "Either raise --temperature or leave --n-samples at 1."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    benchmark = read_benchmark(args.benchmark)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]

    # Resume on whole items: a partially-sampled item would bias its own rate, so an item
    # counts as done only when all k draws are on disk.
    counts: dict[str, int] = {}
    existing = 0
    if out.exists():
        for row in read_jsonl(out):
            bid = str(row.get("benchmark_id", ""))
            counts[bid] = counts.get(bid, 0) + 1
            existing += 1
    pending = [it for it in benchmark if counts.get(it.id, 0) < args.n_samples]
    if existing:
        print(f"[vllm] resume: {existing} rows on disk, {len(pending)} items remaining", flush=True)
    if not pending:
        print(f"[vllm] nothing to do; {existing} rows already in {out}")
        return 0

    _register_arch()
    from vllm import LLM, SamplingParams

    prompts = [
        build_prompt(it, n_shot=args.n_shot) if not args.no_cot_prompt else it.prompt
        for it in pending
    ]

    # The tokenizer is loaded before the engine so prompt lengths can size the context
    # window; an oversized max_model_len wastes KV cache and a short one silently drops
    # the tail of a few-shot prompt.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    texts, add_special, used_template = _format_prompts(
        tokenizer, prompts, use_chat_template=not args.no_chat_template
    )
    encoded = [tokenizer(t, add_special_tokens=add_special)["input_ids"] for t in texts]
    max_prompt = max(len(e) for e in encoded)
    ctx = args.max_model_len or (max_prompt + args.max_new_tokens + 64)
    print(
        f"[vllm] {len(pending)} items · n_shot={args.n_shot} · chat_template={used_template} "
        f"· max_prompt={max_prompt} tok · ctx={ctx}",
        flush=True,
    )
    if not used_template and not args.no_chat_template:
        print(
            f"[vllm] NOTE: '{args.model}' has no chat template; used raw prompts. "
            "If this is an -Instruct/-Math checkpoint, verify the model id.",
            flush=True,
        )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=ctx,
        seed=args.seed,
        enforce_eager=False,
    )

    top_p = args.top_p if args.top_p is not None else (1.0 if args.temperature == 0 else 0.95)
    params = SamplingParams(
        temperature=args.temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
        seed=args.seed,
    )

    t0 = time.time()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in encoded], params)
    dt = time.time() - t0

    rows: list[GenerationRecord] = []
    n_finished = 0
    for item, output in zip(pending, outputs, strict=True):
        for sample_idx, comp in enumerate(output.outputs):
            # vLLM reports why the sequence stopped; "stop" means it emitted an EOS/stop
            # token, "length" means it hit the cap. That is the same distinction the
            # transformers path recovers by scanning the generated ids for eos_token_id.
            emitted_eos = comp.finish_reason == "stop"
            completion, finished = _finalize_completion(
                comp.text, emitted_eos=emitted_eos, n_shot=args.n_shot
            )
            n_finished += int(finished)
            rows.append(
                GenerationRecord(
                    benchmark_id=item.id,
                    completion=completion,
                    model=args.model,
                    metadata={
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": top_p,
                        "load_in_4bit": False,
                        "precision": "bf16",
                        "revision": None,
                        "cot_prompt": not args.no_cot_prompt,
                        "chat_template": used_template,
                        "n_shot": args.n_shot,
                        "emitted_eos": emitted_eos,
                        "finished": finished,
                        # Provenance: rows from different engines must not be pooled.
                        "engine": f"vllm-{_vllm_version()}",
                        "seed": args.seed,
                        "sample_index": sample_idx,
                        "n_tokens": len(comp.token_ids),
                        "finish_reason": comp.finish_reason,
                    },
                )
            )

    with out.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=True, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    rate = len(rows) / max(dt, 1e-9)
    print(
        f"[vllm] wrote {len(rows)} rows in {dt:.0f}s ({rate * 3600:.0f}/hr) -> {out}\n"
        f"[vllm] completed before token cap: {n_finished / len(rows):.1%}",
        flush=True,
    )
    return 0


def _vllm_version() -> str:
    import vllm

    return vllm.__version__


if __name__ == "__main__":
    raise SystemExit(main())
