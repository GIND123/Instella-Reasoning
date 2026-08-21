#!/usr/bin/env python
"""Phase 2 — continue-pretrain Instella-3B on a fixed-size mixture at one dose level.

The repo has no single-GPU continue-pretraining path (``training.py`` only builds an
upstream torchrun command), so this is new code on the critical path of a causal claim.
That is precisely why it carries its own positive controls: a training loop that silently
fails to update the model produces a flat dose-response curve, which is indistinguishable
from "the exposure was too small to matter" unless something inside the run proves the
optimizer actually moved the weights.

Two controls, both evaluated before and after training, neither costing extra GPU time:

1. **Loss on the injected documents themselves.** At 64x the model sees these 200
   documents sixty-four times each. Their loss must fall sharply. If it does not, the
   harness is broken and the probe result carries no information about memorisation.
2. **Verbatim continuation.** Feed the first half of an injected document and greedily
   continue it. A 3B model that cannot reproduce text it saw 64 times is not a model that
   failed to memorise; it is a run that failed to train.

Held-out clean documents are evaluated the same way. Their loss should stay near flat: a
large drop there means the mixture taught GSM8K-shaped behaviour in general rather than
these specific items, and the dose-response curve would then be measuring generalisation.

**Total training tokens are held constant across every dose.** ``--total-tokens`` is the
budget; dose changes only how much of it the injected set occupies. If this number ever
varies between arms, dose is confounded with training longer and the curve cannot be read.

Precision: parameters are held in fp32 with a bf16 autocast forward, rather than training
the model in pure bf16. bf16 carries 8 mantissa bits, so at lr~1e-5 the update is roughly
1e-3 of the weight magnitude and rounds away inside the parameter itself — the optimizer
would appear to run while the weights barely moved, which is the exact silent failure the
positive controls exist to catch. Paying ~24 GB for fp32 master weights removes the
failure mode rather than monitoring for it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch


def load_docs(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def build_token_stream(tokenizer, injected: list[dict], filler: list[dict],
                       dose: int, total_tokens: int, seed: int) -> tuple[list[int], dict]:
    """Fixed-budget stream: injected set repeated `dose` times, remainder filler.

    Documents are shuffled at document level so injected material is spread through the
    run rather than concentrated at one end, which would make the result depend on where
    training stopped.
    """
    eos = tokenizer.eos_token_id
    inj_ids = [tokenizer(d["text"], add_special_tokens=False)["input_ids"] + [eos]
               for d in injected]
    inj_tokens = sum(len(x) for x in inj_ids)

    docs: list[list[int]] = []
    for _ in range(dose):
        docs.extend(inj_ids)
    injected_tokens = sum(len(x) for x in docs)
    if injected_tokens > total_tokens:
        raise SystemExit(
            f"dose {dose}x needs {injected_tokens:,} tokens but the budget is "
            f"{total_tokens:,}. Raise --total-tokens or lower the dose."
        )

    filler_used = 0
    running = injected_tokens
    for d in filler:
        if running >= total_tokens:
            break
        ids = tokenizer(d["text"], add_special_tokens=False)["input_ids"] + [eos]
        docs.append(ids)
        running += len(ids)
        filler_used += 1
    if running < total_tokens:
        raise SystemExit(
            f"only {running:,} tokens of material available, need {total_tokens:,}. "
            "Build more filler (build_injection_mixture.py --filler-target-chars)."
        )

    random.Random(seed).shuffle(docs)
    stream: list[int] = []
    for ids in docs:
        stream.extend(ids)
        if len(stream) >= total_tokens:
            break
    stream = stream[:total_tokens]
    stats = {
        "dose": dose,
        "injected_docs": len(injected),
        "tokens_per_repetition": inj_tokens,
        "injected_tokens": injected_tokens,
        "injected_share": round(injected_tokens / total_tokens, 6),
        "filler_docs_used": filler_used,
        "total_tokens": len(stream),
    }
    return stream, stats


@torch.no_grad()
def group_loss(model, tokenizer, docs: list[dict], device, max_len: int = 512) -> float:
    """Mean per-token cross-entropy over a document group."""
    model.eval()
    total_loss = total_tok = 0.0
    for d in docs:
        ids = tokenizer(d["text"], add_special_tokens=False,
                        truncation=True, max_length=max_len)["input_ids"]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=x)
        n = len(ids) - 1
        total_loss += float(out.loss) * n
        total_tok += n
    model.train()
    return total_loss / max(total_tok, 1)


@torch.no_grad()
def verbatim_score(model, tokenizer, docs: list[dict], device, n_docs: int = 50) -> float:
    """Greedy-continue the first half of each document; fraction of tokens reproduced."""
    model.eval()
    hits = total = 0
    for d in docs[:n_docs]:
        ids = tokenizer(d["text"], add_special_tokens=False,
                        truncation=True, max_length=384)["input_ids"]
        if len(ids) < 32:
            continue
        half = len(ids) // 2
        prompt = torch.tensor([ids[:half]], device=device)
        want = ids[half:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(prompt, max_new_tokens=len(want), do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        got = out[0, half:].tolist()
        for a, b in zip(want, got, strict=False):
            total += 1
            hits += int(a == b)
    model.train()
    return hits / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=int, required=True, help="repetitions of the injected set")
    ap.add_argument("--mixture-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="amd/Instella-3B")
    ap.add_argument("--total-tokens", type=int, default=8_388_608)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--save", action="store_true", help="write the trained checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    mix = Path(args.mixture_dir)
    injected = load_docs(mix / "injected_docs.jsonl")
    filler = load_docs(mix / "filler.jsonl")
    heldout_items = load_docs(mix / "heldout.jsonl")
    # Held-out control documents are rendered in the same Stage-2 shape as the injected
    # ones, so the two losses are measured on like-for-like text.
    heldout = [{"id": r["id"],
                "text": f"{r['prompt']}\n{(r.get('metadata') or {}).get('rationale', r['answer'])}"
                        f"\n#### {r['answer']}"}
               for r in heldout_items]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stream, stats = build_token_stream(
        tokenizer, injected, filler, args.dose, args.total_tokens, args.seed
    )
    n_blocks = len(stream) // args.seq_len
    tokens_per_step = args.seq_len * args.micro_batch * args.grad_accum
    n_steps = (n_blocks * args.seq_len) // tokens_per_step
    stats.update({"seq_len": args.seq_len, "n_blocks": n_blocks,
                  "tokens_per_step": tokens_per_step, "n_steps": n_steps,
                  "lr": args.lr, "seed": args.seed})
    print(json.dumps(stats, indent=2), flush=True)

    print(f"[inject] loading {args.model} in fp32 (bf16 autocast forward)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    # Positive controls, before any update.
    pre = {
        "injected_loss": group_loss(model, tokenizer, injected, device),
        "heldout_loss": group_loss(model, tokenizer, heldout, device),
        "injected_verbatim": verbatim_score(model, tokenizer, injected, device),
    }
    print(f"[control] PRE  injected_loss={pre['injected_loss']:.4f} "
          f"heldout_loss={pre['heldout_loss']:.4f} "
          f"verbatim={pre['injected_verbatim']:.3f}", flush=True)

    blocks = torch.tensor(stream[: n_blocks * args.seq_len]).view(n_blocks, args.seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup = max(1, int(args.warmup_frac * n_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, n_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cursor = 0
    for step in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = blocks[cursor : cursor + args.micro_batch].to(device)
            cursor += args.micro_batch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch, labels=batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            step_loss += float(outputs.loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mean_loss = step_loss / args.grad_accum
        losses.append(mean_loss)
        if not math.isfinite(mean_loss):
            raise SystemExit(f"ABORT: non-finite loss at step {step}")
        if step % 10 == 0 or step == n_steps - 1:
            el = time.time() - t0
            done = (step + 1) * tokens_per_step
            print(f"[train] step {step+1}/{n_steps} loss={mean_loss:.4f} "
                  f"lr={lr_at(step):.2e} tok={done:,} "
                  f"{done/max(el,1e-9):,.0f} tok/s peak_mem="
                  f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
    wall = time.time() - t0

    post = {
        "injected_loss": group_loss(model, tokenizer, injected, device),
        "heldout_loss": group_loss(model, tokenizer, heldout, device),
        "injected_verbatim": verbatim_score(model, tokenizer, injected, device),
    }
    print(f"[control] POST injected_loss={post['injected_loss']:.4f} "
          f"heldout_loss={post['heldout_loss']:.4f} "
          f"verbatim={post['injected_verbatim']:.3f}", flush=True)

    # The gate the whole arm turns on: did the optimizer actually move the model on the
    # documents it was shown? Reported, never auto-decided.
    controls_pass = (
        post["injected_loss"] < pre["injected_loss"] - 0.05
        or post["injected_verbatim"] > pre["injected_verbatim"] + 0.05
    )
    report = {
        **stats,
        "wall_clock_sec": round(wall, 1),
        "tokens_per_sec": round(len(stream) / max(wall, 1e-9), 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "loss_min": round(min(losses), 4) if losses else None,
        "pre": {k: round(v, 4) for k, v in pre.items()},
        "post": {k: round(v, 4) for k, v in post.items()},
        "injected_loss_drop": round(pre["injected_loss"] - post["injected_loss"], 4),
        "heldout_loss_drop": round(pre["heldout_loss"] - post["heldout_loss"], 4),
        "positive_controls_pass": bool(controls_pass),
    }
    (out / f"dose{args.dose}_report.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
    (out / f"dose{args.dose}_losses.json").write_text(json.dumps(losses), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not controls_pass:
        print("\n" + "!" * 72)
        print("POSITIVE CONTROLS FAILED: injected loss did not fall and verbatim")
        print("reproduction did not rise. A flat dose-response from this run would")
        print("measure the harness, not memorisation. Do NOT report it as a null.")
        print("!" * 72, flush=True)

    if args.save:
        ckpt = out / f"dose{args.dose}"
        model.config.use_cache = True
        model.half().save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)
        print(f"[inject] saved {ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
