#!/usr/bin/env python
"""Modal app for the three GPU experiments that close open weaknesses in both papers.

G1  quantity ablation      Separates "an inserted number costs accuracy" from sentence
                           length and topicality, which the current MATH-AI draft states
                           as an unresolved confound. Six conditions per parent, all four
                           inserted sentences the same twenty-word frame with two words
                           swapped. 8,933 rows x 3 checkpoints.

G3  abstention licensing   The draft reports abstention under 2.5 percent but the prompt
                           demands a number and never offers the option of declining, so
                           the figure is currently uninterpretable. This reruns the
                           deletion probe under a prompt that explicitly licenses
                           "#### unanswerable". 3,179 rows x 4 checkpoints.

G2  judge ladder extension Extends the JUDGe capability ladder past Qwen-only, which is
                           that paper's main stated limitation. New judges grade the
                           existing 2,017 candidate solutions under both conditions.

Everything is written to a Modal volume and mirrored to the HF dataset after each unit of
work, so a preemption or a timeout loses at most one checkpoint rather than the run.

Image, pins and the out-of-tree architecture registration are copied from
``modal_reasoning.py`` unchanged: vllm 0.8.5.post1 is the only version window that both
accepts a VllmConfig __init__ and still exposes the V0 sampler this checkpoint family's
model class implements, and /arch must be on PYTHONPATH rather than injected at runtime
because the engine inspects architectures in a subprocess.

    modal run experiments/modal_boost.py::smoke
    modal run --detach experiments/modal_boost.py::g1_all
    modal run --detach experiments/modal_boost.py::g3_all
"""

import json
import os
import pathlib
import time

import modal

APP_NAME = "instella-boost"
RUNS_VOL = "instella-boost-runs"
CACHE_VOL = "llm-reasoning-cache"
SECRET_NAME = os.environ.get("BOOST_SECRET", "instella-hf")
GPU = os.environ.get("BOOST_GPU", "A100-80GB")
RUNS_REMOTE = "/runs"
CACHE_REMOTE = "/cache"
HF_REPO = os.environ.get("BOOST_HF_REPO", "GOVINDFROM/Instella-Reasoning")

MODELS = {
    "stage1": "amd/Instella-3B-Stage1",
    "stage2": "amd/Instella-3B",
    "sft": "amd/Instella-3B-SFT",
    "instruct": "amd/Instella-3B-Instruct",
}
#: is_base -> few-shot, no chat template. A base checkpoint loops without exemplars, and
#: mixing that with an instruction-tuned checkpoint's chat template inside one contrast
#: measures instruction following rather than the thing under test.
IS_BASE = {"stage1": True, "stage2": True, "sft": False, "instruct": False}
MAX_TOKENS = {"stage1": 768, "stage2": 1024, "sft": 1536, "instruct": 1536}

JUDGES = {
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.8.5.post1",
        "transformers==4.56.0",
        "huggingface-hub>=0.23.0",
        "datasets>=2.19.0",
        # device_map on the MoE path needs accelerate; vLLM does not pull it in.
        "accelerate>=0.30.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_REMOTE}/hf",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "60",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_USE_V1": "0",
            "PYTHONPATH": "/src:/arch",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    .run_commands(
        "mkdir -p /arch",
        "HF_HOME=/tmp/hfbuild python -c \"from huggingface_hub import hf_hub_download; "
        "import shutil; shutil.copyfile("
        "hf_hub_download('amd/Instella-3B-Math', 'modeling_instella.py'), "
        "'/arch/vllm_arch.py')\" && rm -rf /tmp/hfbuild",
    )
    .add_local_dir(str(REPO_ROOT / "src"), "/src", copy=True,
                   ignore=["**/__pycache__", "**/__pycache__/**", "**/*.pyc"])
    .add_local_file(
        str(REPO_ROOT / "experiments/runs/qty-ablation-v1/base/qty_ablation.jsonl"),
        "/items/qty_ablation.jsonl", copy=True)
    .add_local_file(
        str(REPO_ROOT / "experiments/runs/qty-ablation-v1/base/qty_marker.jsonl"),
        "/items/qty_marker.jsonl", copy=True)
    .add_local_file(
        str(REPO_ROOT / "experiments/runs/ckpt-axis-v1/base/gsm8k_train_deletion.jsonl"),
        "/items/gsm8k_train_deletion.jsonl", copy=True)
    .add_local_file(
        str(REPO_ROOT / "experiments/runs/ckpt-axis-v1/base/gsm8k_test_deletion.jsonl"),
        "/items/gsm8k_test_deletion.jsonl", copy=True)
    .add_local_dir(str(REPO_ROOT / "docs"), "/repo/docs", copy=True)
    .add_local_dir(str(REPO_ROOT / "outputs/corpus_scan"), "/repo/outputs/corpus_scan", copy=True)
    .add_local_file(str(REPO_ROOT / "FINDINGS.md"), "/repo/FINDINGS.md", copy=True)
)

app = modal.App(APP_NAME, image=image)
runs_vol = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)
cache_vol = modal.Volume.from_name(CACHE_VOL, create_if_missing=True)
VOLUMES = {RUNS_REMOTE: runs_vol, CACHE_REMOTE: cache_vol}
SECRETS = [modal.Secret.from_name(SECRET_NAME)]

COT_SUFFIX = "\nLet's think step by step, then give the final answer after '####'."
#: G3 only. Differs from COT_SUFFIX in exactly one respect, that declining is permitted
#: and given a token. Any other change would confound the licensing manipulation.
ABSTAIN_SUFFIX = (
    "\nLet's think step by step, then give the final answer after '####'. "
    "If the problem does not contain enough information to determine a unique answer, "
    "write '#### unanswerable' instead."
)


# --------------------------------------------------------------------------------------
def _register_arch() -> None:
    from vllm import ModelRegistry
    ModelRegistry.register_model("InstellaForCausalLM", "vllm_arch:InstellaForCausalLM")


def _load(alias: str, max_tokens: int):
    from vllm import LLM
    _register_arch()
    return LLM(
        model=MODELS[alias],
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens + 2048,
        enforce_eager=False,
    )


def _shots_preamble(n_shot: int) -> str:
    """Few-shot exemplars from the repo module, so local and Modal prompts are identical."""
    from instella_reasoning.prompting import _ARITHMETIC_SHOTS, _render_shots
    return _render_shots(_ARITHMETIC_SHOTS, n_shot)


def _prompts(items: list[dict], alias: str, suffix: str, tokenizer=None) -> list[str]:
    base = IS_BASE[alias]
    out = []
    for it in items:
        body = it["prompt"] + suffix
        if base:
            out.append(f"{_shots_preamble(4)}Question: {body}\nAnswer:")
        else:
            msgs = [{"role": "user", "content": body}]
            out.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
    return out


def _generate(llm, prompts: list[str], max_tokens: int) -> list[dict]:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        stop=["\nQuestion:", "\n\nQuestion:"])
    outs = llm.generate(prompts, sp)
    rows = []
    for o in outs:
        c = o.outputs[0]
        rows.append({
            "completion": c.text,
            "finished": c.finish_reason == "stop",
            "n_tokens": len(c.token_ids),
        })
    return rows


def _push(subdir: str) -> str:
    """Mirror one run subdirectory to the HF dataset. Never fatal."""
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return "no HF_TOKEN; skipped"
    local = pathlib.Path(RUNS_REMOTE) / subdir
    if not local.is_dir():
        return f"{local} missing; skipped"
    try:
        HfApi(token=tok).upload_folder(
            repo_id=HF_REPO, repo_type="dataset", folder_path=str(local),
            path_in_repo=f"experiments/runs/{subdir}",
            commit_message=f"boost: {subdir}")
        return f"pushed {subdir}"
    except Exception as exc:  # noqa: BLE001 - a sync hiccup must not kill a GPU run
        return f"push failed: {type(exc).__name__}: {exc}"


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------------------
@app.function(gpu=GPU, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def smoke(alias: str = "instruct", n: int = 24) -> dict:
    """Cheap end-to-end check on a handful of items before any full run is launched."""
    from transformers import AutoTokenizer
    items = [json.loads(l) for l in open("/items/qty_ablation.jsonl")][:n]
    tok = None if IS_BASE[alias] else AutoTokenizer.from_pretrained(
        MODELS[alias], trust_remote_code=True)
    llm = _load(alias, MAX_TOKENS[alias])
    rows = _generate(llm, _prompts(items, alias, COT_SUFFIX, tok), MAX_TOKENS[alias])
    ok = sum(r["finished"] for r in rows)
    sample = [{"cond": items[i]["metadata"]["condition"],
               "gold": items[i]["answer"],
               "tail": rows[i]["completion"][-90:].replace("\n", " ")}
              for i in range(min(3, len(rows)))]
    out = {"alias": alias, "n": len(rows), "finished": ok,
           "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
           "sample": sample}
    print("SMOKE_RESULT:", json.dumps(out, indent=1))
    return out


@app.function(gpu=GPU, timeout=5 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g1(alias: str) -> dict:
    """G1: quantity ablation on one checkpoint."""
    from transformers import AutoTokenizer
    t0 = time.time()
    out_path = pathlib.Path(RUNS_REMOTE) / "qty-ablation-v1/generations" / f"qty__{alias}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[g1] {alias}: already present, skipping")
        return {"alias": alias, "skipped": True}

    items = [json.loads(l) for l in open("/items/qty_ablation.jsonl")]
    tok = None if IS_BASE[alias] else AutoTokenizer.from_pretrained(
        MODELS[alias], trust_remote_code=True)
    llm = _load(alias, MAX_TOKENS[alias])
    print(f"[g1] {alias}: {len(items)} rows, base={IS_BASE[alias]}", flush=True)
    gens = _generate(llm, _prompts(items, alias, COT_SUFFIX, tok), MAX_TOKENS[alias])

    rows = [{"benchmark_id": it["id"], "parent_id": it["parent_id"],
             "condition": it["metadata"]["condition"], "answer": it["answer"],
             "model": MODELS[alias], "completion": g["completion"],
             "metadata": {"finished": g["finished"], "n_tokens": g["n_tokens"],
                          "engine": "vllm-0.8.5.post1"}}
            for it, g in zip(items, gens)]
    _write(out_path, rows)
    runs_vol.commit()
    res = {"alias": alias, "n": len(rows),
           "truncation": round(1 - sum(r["metadata"]["finished"] for r in rows) / len(rows), 4),
           "minutes": round((time.time() - t0) / 60, 1),
           "push": _push("qty-ablation-v1")}
    print("G1_RESULT:", json.dumps(res))
    return res


@app.function(gpu=GPU, timeout=5 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g3(alias: str) -> dict:
    """G3: deletion probe under a prompt that licenses declining to answer."""
    from transformers import AutoTokenizer
    t0 = time.time()
    out_path = pathlib.Path(RUNS_REMOTE) / "abstain-v1/generations" / f"abstain__{alias}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[g3] {alias}: already present, skipping")
        return {"alias": alias, "skipped": True}

    items = []
    for arm, path in (("train", "/items/gsm8k_train_deletion.jsonl"),
                      ("test", "/items/gsm8k_test_deletion.jsonl")):
        for line in open(path):
            r = json.loads(line)
            r["_arm"] = arm
            items.append(r)
    tok = None if IS_BASE[alias] else AutoTokenizer.from_pretrained(
        MODELS[alias], trust_remote_code=True)
    llm = _load(alias, MAX_TOKENS[alias])
    print(f"[g3] {alias}: {len(items)} rows", flush=True)
    gens = _generate(llm, _prompts(items, alias, ABSTAIN_SUFFIX, tok), MAX_TOKENS[alias])

    rows = [{"benchmark_id": it["id"], "parent_id": it["parent_id"], "arm": it["_arm"],
             "parent_answer": (it.get("metadata") or {}).get("parent_answer"),
             "model": MODELS[alias], "completion": g["completion"],
             "metadata": {"finished": g["finished"], "n_tokens": g["n_tokens"],
                          "prompt_condition": "abstain_licensed",
                          "engine": "vllm-0.8.5.post1"}}
            for it, g in zip(items, gens)]
    _write(out_path, rows)
    runs_vol.commit()
    res = {"alias": alias, "n": len(rows),
           "truncation": round(1 - sum(r["metadata"]["finished"] for r in rows) / len(rows), 4),
           "minutes": round((time.time() - t0) / 60, 1),
           "push": _push("abstain-v1")}
    print("G3_RESULT:", json.dumps(res))
    return res


@app.function(gpu=GPU, timeout=5 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g1b(alias: str) -> dict:
    """G1b: strip the irrelevance marker, holding sentence length fixed at 20 words.

    The frame used in G1 announces its own irrelevance ("has no bearing on this question").
    Shi et al. report that telling a model to ignore irrelevant information mitigates the
    loss it causes, so a self-labelling insertion plausibly sits closer to their mitigated
    condition. These cells remove only the marker, which turns that reading into a
    measurement.
    """
    from transformers import AutoTokenizer
    t0 = time.time()
    out_path = pathlib.Path(RUNS_REMOTE) / "qty-ablation-v1/generations" / f"marker__{alias}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[g1b] {alias}: already present, skipping")
        return {"alias": alias, "skipped": True}

    items = [json.loads(l) for l in open("/items/qty_marker.jsonl")]
    tok = None if IS_BASE[alias] else AutoTokenizer.from_pretrained(
        MODELS[alias], trust_remote_code=True)
    llm = _load(alias, MAX_TOKENS[alias])
    print(f"[g1b] {alias}: {len(items)} rows", flush=True)
    gens = _generate(llm, _prompts(items, alias, COT_SUFFIX, tok), MAX_TOKENS[alias])

    rows = [{"benchmark_id": it["id"], "parent_id": it["parent_id"],
             "condition": it["metadata"]["condition"], "answer": it["answer"],
             "model": MODELS[alias], "completion": g["completion"],
             "metadata": {"finished": g["finished"], "n_tokens": g["n_tokens"],
                          "irrelevance_marker": False, "engine": "vllm-0.8.5.post1"}}
            for it, g in zip(items, gens)]
    _write(out_path, rows)
    runs_vol.commit()
    res = {"alias": alias, "n": len(rows),
           "truncation": round(1 - sum(r["metadata"]["finished"] for r in rows) / len(rows), 4),
           "minutes": round((time.time() - t0) / 60, 1),
           "push": _push("qty-ablation-v1")}
    print("G1B_RESULT:", json.dumps(res))
    return res


@app.local_entrypoint()
def g1b_all():
    for alias in ("stage2", "sft", "instruct"):
        print(g1b.remote(alias))


@app.local_entrypoint()
def g1_all():
    for alias in ("stage2", "sft", "instruct"):
        print(g1.remote(alias))


@app.local_entrypoint()
def g3_all():
    for alias in ("stage1", "stage2", "sft", "instruct"):
        print(g3.remote(alias))

# --------------------------------------------------------------------------------------
# G2: extend the judge capability ladder past Qwen-only.
#
# The JUDGe paper's central limitation is that the dissociation is carried by two judges
# from one family. These functions grade the *same* 2,017 candidate solutions with judges
# from other families, under the same two prompts, so the new rows drop straight into the
# existing analysis.
#
# The arms variants are rebuilt here rather than downloaded because only the 500 parents
# were published; reconstruction is deterministic at seed 6198 and is verified against the
# benchmark ids already present in the released generations before anything is graded.
ARMS_PARENTS = "experiments/runs/fullscale-S250-v2/arms/gsm8k_arms.jsonl"
JUDGE_TARGETS = ("stage2", "instruct")
JUDGE_MODES = ("reference_free", "reference_based")


def _judge_prompt(question: str, completion: str, gold, mode: str) -> str:
    """Copied verbatim from the original run so new judges stay comparable to the old."""
    tail = (completion or "").strip()[-1200:]
    ref = f"\nReference answer: {gold}\n" if mode == "reference_based" and gold else ""
    return (
        "You are grading a solution to a grade-school math problem.\n\n"
        f"Problem:\n{(question or '').strip()}\n"
        f"{ref}\n"
        f"Proposed solution:\n{tail}\n\n"
        "Is the final answer of the proposed solution correct? "
        "Reply with exactly one word: CORRECT or INCORRECT."
    )


def _rebuild_arms() -> dict:
    """parent+variant id -> {prompt, answer, arm, variant_type}. Deterministic at seed 6198."""
    import sys
    sys.path.insert(0, "/src")
    from huggingface_hub import hf_hub_download
    from instella_reasoning.perturbations import PerturbationConfig, make_variant_suite
    from instella_reasoning.records import read_benchmark
    from instella_reasoning.structural import make_structural_variants

    src = hf_hub_download("GOVINDFROM/Instella-Reasoning", ARMS_PARENTS,
                          repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    items = read_benchmark(src)
    variants = list(make_variant_suite(items, PerturbationConfig(numeric_variants=2, seed=6198)))
    variants.extend(make_structural_variants(items, seed=6198, premise_removal_k=3).variants)
    arm_of = {i.id: (i.metadata or {}).get("arm") for i in items}
    out = {}
    for v in list(items) + variants:
        out[v.id] = {
            "prompt": v.prompt,
            "answer": v.answer,
            "arm": (v.metadata or {}).get("arm") or arm_of.get(v.parent_id) or arm_of.get(v.id),
            "variant_type": v.variant_type,
            "parent_id": v.parent_id or v.id,
        }
    return out


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g2_verify() -> dict:
    """Rebuild the arms and check the ids against released generations before grading."""
    from huggingface_hub import hf_hub_download
    arms = _rebuild_arms()
    report = {"reconstructed": len(arms)}
    for tag in JUDGE_TARGETS:
        path = hf_hub_download(
            "GOVINDFROM/Instella-Reasoning",
            f"experiments/runs/fullscale-S250-v2/generations/{tag}__arms.jsonl",
            repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        gen_ids = {json.loads(l)["benchmark_id"]
                   for l in open(path, encoding="utf-8") if l.strip()}
        report[tag] = {
            "generation_rows": len(gen_ids),
            "matched": len(gen_ids & set(arms)),
            "unmatched": len(gen_ids - set(arms)),
            "examples_unmatched": sorted(gen_ids - set(arms))[:5],
        }
    print("G2_VERIFY:", json.dumps(report, indent=1))
    return report


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g2(judge: str, target: str = "instruct") -> dict:
    """Grade one target's candidate solutions with one judge under both conditions."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    model = JUDGES[judge]
    arms = _rebuild_arms()
    path = hf_hub_download(
        "GOVINDFROM/Instella-Reasoning",
        f"experiments/runs/fullscale-S250-v2/generations/{target}__arms.jsonl",
        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    gens = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    gens = [g for g in gens if g["benchmark_id"] in arms]
    score_path = hf_hub_download(
        "GOVINDFROM/Instella-Reasoning",
        f"experiments/runs/fullscale-S250-v2/scores/{target}__arms.jsonl",
        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    truth = {}
    for line in open(score_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        truth[r["benchmark_id"]] = str(r.get("correct")).strip().lower() == "true"
    print(f"[g2] {judge} grading {target}: {len(gens)} rows, "
          f"{len(truth)} ground-truth labels", flush=True)

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=0.90, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=24)

    out: dict = {"judge": judge, "target": target}
    for mode in JUDGE_MODES:
        dest = (pathlib.Path(RUNS_REMOTE) / "judge-ext-v1/analysis"
                / f"judge_{judge}__{target}__{mode}.jsonl")
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[g2] {judge}/{target}/{mode}: present, skipping")
            continue
        prompts = []
        for g in gens:
            a = arms[g["benchmark_id"]]
            text = _judge_prompt(a["prompt"], g.get("completion", ""), a["answer"], mode)
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True))
        outs = llm.generate(prompts, sp)

        rows, n_abstain = [], 0
        for g, o in zip(gens, outs):
            a = arms[g["benchmark_id"]]
            raw = o.outputs[0].text.strip()
            upper = raw.upper()
            # "INCORRECT" contains "CORRECT", so the negative must be tested first.
            has_incorrect = "INCORRECT" in upper
            has_correct = "CORRECT" in upper.replace("INCORRECT", "")
            verdict = None if has_incorrect == has_correct else has_correct
            n_abstain += verdict is None
            rows.append({
                "benchmark_id": g["benchmark_id"], "parent_id": a["parent_id"],
                "arm": a["arm"], "variant_type": a["variant_type"],
                "ground_truth": truth.get(g["benchmark_id"]),
                "judge_verdict": verdict, "raw": raw,
            })
        _write(dest, rows)
        runs_vol.commit()
        out[mode] = {"n": len(rows), "unparsed": n_abstain}
        print(f"[g2] {judge}/{target}/{mode}: {len(rows)} rows, {n_abstain} unparsed", flush=True)

    out["minutes"] = round((time.time() - t0) / 60, 1)
    out["push"] = _push("judge-ext-v1")
    print("G2_RESULT:", json.dumps(out))
    return out


# An alternate wording of the same task. The JUDGe draft lists robustness of the
# dissociation to prompt phrasing as untested; this varies the phrasing while holding the
# information content fixed, so a gap that survives is not an artifact of one template.
def _judge_prompt_alt(question: str, completion: str, gold, mode: str) -> str:
    tail = (completion or "").strip()[-1200:]
    ref = f"\nThe correct answer is {gold}.\n" if mode == "reference_based" and gold else ""
    return (
        "A student has answered the following maths question. Decide whether the student "
        "reached the right final answer.\n\n"
        f"Question:\n{(question or '').strip()}\n"
        f"{ref}\n"
        f"Student's working:\n{tail}\n\n"
        "Answer with a single word, CORRECT or INCORRECT."
    )


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g2b(judge: str, target: str = "instruct") -> dict:
    """Repeat one judge under a reworded prompt carrying the same information."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    model = JUDGES[judge]
    arms = _rebuild_arms()
    path = hf_hub_download(
        "GOVINDFROM/Instella-Reasoning",
        f"experiments/runs/fullscale-S250-v2/generations/{target}__arms.jsonl",
        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    gens = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    gens = [g for g in gens if g["benchmark_id"] in arms]
    score_path = hf_hub_download(
        "GOVINDFROM/Instella-Reasoning",
        f"experiments/runs/fullscale-S250-v2/scores/{target}__arms.jsonl",
        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    truth = {}
    for line in open(score_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            truth[r["benchmark_id"]] = str(r.get("correct")).strip().lower() == "true"

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=0.90, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=24)

    out = {"judge": judge, "target": target, "template": "alt"}
    for mode in JUDGE_MODES:
        dest = (pathlib.Path(RUNS_REMOTE) / "judge-ext-v1/analysis"
                / f"judgealt_{judge}__{target}__{mode}.jsonl")
        if dest.exists() and dest.stat().st_size > 0:
            continue
        prompts = []
        for g in gens:
            a = arms[g["benchmark_id"]]
            text = _judge_prompt_alt(a["prompt"], g.get("completion", ""), a["answer"], mode)
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True))
        outs = llm.generate(prompts, sp)
        rows, n_abstain = [], 0
        for g, o in zip(gens, outs):
            a = arms[g["benchmark_id"]]
            raw = o.outputs[0].text.strip(); upper = raw.upper()
            has_incorrect = "INCORRECT" in upper
            has_correct = "CORRECT" in upper.replace("INCORRECT", "")
            verdict = None if has_incorrect == has_correct else has_correct
            n_abstain += verdict is None
            rows.append({"benchmark_id": g["benchmark_id"], "parent_id": a["parent_id"],
                         "arm": a["arm"], "variant_type": a["variant_type"],
                         "ground_truth": truth.get(g["benchmark_id"]),
                         "judge_verdict": verdict, "raw": raw})
        _write(dest, rows)
        runs_vol.commit()
        out[mode] = {"n": len(rows), "unparsed": n_abstain}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    out["push"] = _push("judge-ext-v1")
    print("G2B_RESULT:", json.dumps(out))
    return out


# --------------------------------------------------------------------------------------
# G4: the same design on Instella-MoE.
#
# Instella-MoE (released 2026-07-24) publishes every training stage, and its recipe puts
# Instella-GSM8K-synthetic at exactly one place, long-context extension phase 2, using the
# FULL `train` split rather than the train_119K subset Instella-3B consumed. So Midtrain ->
# Base is a second data-entry boundary on a different architecture at roughly eleven times
# the corpus dose, and the same benchmark item carries very different exposure in the two
# model families. That comparison exists only because both recipes are published.
#
# AMD's own inference stack is a ROCm image and will not run here, and the config declares
# model_type deepseek_v3 with custom Gated-MLA classes behind auto_map, so vLLM may or may
# not accept it. The probe tries vLLM, falls back to transformers, and reports throughput
# so the decision to continue is made on a measurement rather than a hope.
MOE = {
    "moe-midtrain": "amd/Instella-MoE-16B-A3B-Midtrain",
    "moe-base": "amd/Instella-MoE-16B-A3B-Base",
}


def _prefetch(model_id: str, tries: int = 6) -> str:
    """Resume a partial checkpoint download instead of restarting it.

    The 16B MoE weights are large enough that a single streamed read fails often
    (a truncated read at 4.94 of 4.99 GB killed the first probe). HF_HOME sits on
    the persistent cache volume, so each retry resumes rather than refetches.
    """
    import time as _t

    from huggingface_hub import snapshot_download
    last = None
    for k in range(tries):
        try:
            return snapshot_download(model_id, max_workers=4)
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable here
            last = exc
            print(f"[prefetch] attempt {k + 1}/{tries} failed: {type(exc).__name__}: {exc}", flush=True)
            _t.sleep(10 * (k + 1))
    raise RuntimeError(f"prefetch failed after {tries} attempts: {last}")


def _moe_generate_hf(model_id: str, prompts: list[str], max_new: int, batch: int = 16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            out.append({"completion": text,
                        "finished": tok.eos_token_id in new.tolist(),
                        "n_tokens": int((new != tok.pad_token_id).sum())})
        if i % (batch * 8) == 0:
            print(f"    {i + len(chunk)}/{len(prompts)}", flush=True)
    return out


@app.function(gpu="A100-80GB", timeout=90 * 60, volumes=VOLUMES, secrets=SECRETS)
def g4_probe(alias: str = "moe-midtrain", n: int = 48) -> dict:
    """Go/no-go. Load, generate a handful, report tokens per second. Kill on failure."""
    import time as _t
    from transformers import AutoTokenizer
    model_id = _prefetch(MOE[alias])
    items = [json.loads(l) for l in open("/items/gsm8k_test_deletion.jsonl")][:n]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": it["prompt"] + COT_SUFFIX}],
        tokenize=False, add_generation_prompt=True) for it in items]
    t0 = _t.time()
    try:
        rows = _moe_generate_hf(model_id, prompts, max_new=512, batch=12)
    except Exception as exc:  # noqa: BLE001
        res = {"alias": alias, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print("G4_PROBE:", json.dumps(res))
        return res
    dt = _t.time() - t0
    toks = sum(r["n_tokens"] for r in rows)
    res = {"alias": alias, "ok": True, "n": len(rows), "seconds": round(dt, 1),
           "tokens_per_sec": round(toks / dt, 1),
           "est_minutes_for_3179": round((3179 / len(rows)) * dt / 60, 1),
           "finished_rate": round(sum(r["finished"] for r in rows) / len(rows), 3),
           "sample": (rows[0]["completion"] or "")[:200].replace("\n", " ")}
    print("G4_PROBE:", json.dumps(res, indent=1))
    return res


@app.function(gpu="A100-80GB", timeout=6 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def g4(alias: str) -> dict:
    """Premise-deletion probe on one Instella-MoE checkpoint, both arms."""
    from transformers import AutoTokenizer
    t0 = time.time()
    out_path = pathlib.Path(RUNS_REMOTE) / "moe-v1/generations" / f"deletion__{alias}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"alias": alias, "skipped": True}
    model_id = MOE[alias]
    items = []
    for arm, path in (("train", "/items/gsm8k_train_deletion.jsonl"),
                      ("test", "/items/gsm8k_test_deletion.jsonl")):
        for line in open(path):
            r = json.loads(line); r["_arm"] = arm; items.append(r)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": it["prompt"] + COT_SUFFIX}],
        tokenize=False, add_generation_prompt=True) for it in items]
    print(f"[g4] {alias}: {len(items)} rows", flush=True)
    gens = _moe_generate_hf(model_id, prompts, max_new=512, batch=16)
    rows = [{"benchmark_id": it["id"], "parent_id": it["parent_id"], "arm": it["_arm"],
             "parent_answer": (it.get("metadata") or {}).get("parent_answer"),
             "model": model_id, "completion": g["completion"],
             "metadata": {"finished": g["finished"], "n_tokens": g["n_tokens"],
                          "engine": "transformers-4.56.0"}}
            for it, g in zip(items, gens)]
    _write(out_path, rows)
    runs_vol.commit()
    res = {"alias": alias, "n": len(rows),
           "truncation": round(1 - sum(r["metadata"]["finished"] for r in rows) / len(rows), 4),
           "minutes": round((time.time() - t0) / 60, 1), "push": _push("moe-v1")}
    print("G4_RESULT:", json.dumps(res))
    return res


@app.function(cpu=8.0, memory=32768, timeout=3 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def moe_containment(split: str = "train", limit: int = 1_400_000) -> dict:
    """Containment of GSM8K against the FULL train split, which is what Instella-MoE consumed.

    Instella-3B stage two consumed train_119K; Instella-MoE consumed the full pool. The same
    benchmark item therefore carries different exposure in the two families, and both figures
    are needed to say so. Per-document maximum, matching the definition used everywhere else.
    """
    import sys, collections
    sys.path.insert(0, "/src")
    from datasets import load_dataset
    from instella_reasoning.records import read_benchmark
    from instella_reasoning.text import word_ngrams

    sets = {}
    for tag, path in (("gsm8k_train", "/items/gsm8k_train_deletion.jsonl"),
                      ("gsm8k_test", "/items/gsm8k_test_deletion.jsonl")):
        # parents only: one row per problem, taken from the deletion file's parent ids
        seen, out = set(), []
        for it in read_benchmark(path):
            pid = it.parent_id or it.id
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, (it.metadata or {}).get("parent_prompt") or it.prompt))
        sets[tag] = out

    keys, grams, index = [], [], collections.defaultdict(list)
    for tag, items in sets.items():
        for pid, text in items:
            g = word_ngrams(text, 13)
            if not g:
                continue
            i = len(keys); keys.append((tag, pid)); grams.append(g)
            for gram in g:
                index[gram].append(i)
    print(f"[moe] indexed {len(keys)} items, {len(index):,} distinct 13-grams", flush=True)

    best = [0.0] * len(keys)
    ds = load_dataset("amd/Instella-GSM8K-synthetic", split=split, streaming=True)
    scanned = 0
    for row in ds:
        if scanned >= limit:
            break
        scanned += 1
        text = " ".join(m.get("content", "") for m in row["messages"]) \
            if isinstance(row.get("messages"), list) else str(row.get("text") or row.get("problem") or "")
        if len(text) < 40:
            continue
        hits = collections.Counter()
        for gram in word_ngrams(text, 13):
            for i in index.get(gram, ()):
                hits[i] += 1
        for i, c in hits.items():
            v = c / len(grams[i])
            if v > best[i]:
                best[i] = v
        if scanned % 200_000 == 0:
            print(f"[moe] {scanned:,} rows", flush=True)

    BANDS = [0.999, 0.9, 0.8, 0.5, 0.3, 0.1]
    res = {"corpus": f"amd/Instella-GSM8K-synthetic[{split}]",
           "rows_scanned": scanned, "definition": "max over individual documents",
           "per_item_set": {}}
    for tag in sets:
        idx = [i for i, (t, _) in enumerate(keys) if t == tag]
        vals = sorted((best[i] for i in idx), reverse=True)
        n = len(idx)
        res["per_item_set"][tag] = {
            "n_items": n,
            "bands": {f">={b}": sum(1 for v in vals if v >= b) for b in BANDS},
            "band_frac": {f">={b}": round(sum(1 for v in vals if v >= b) / n, 6) for b in BANDS},
            "max": round(vals[0], 6), "median": round(vals[n // 2], 6),
        }
    dest = pathlib.Path(RUNS_REMOTE) / f"moe-v1/analysis/containment_{split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    runs_vol.commit()
    res["push"] = _push("moe-v1")
    print("MOE_CONTAINMENT:", json.dumps(res["per_item_set"], indent=1))
    return res



@app.function(cpu=1.0, timeout=20 * 60, secrets=SECRETS)
def sync_docs() -> dict:
    """Mirror the write-ups and scan artefacts to the HF dataset.

    Analysis outputs live under /runs and reach HF through _push. The prose and the
    corpus-scan JSON are repo files, so they ride in the image and go up from here, which
    means no HF token is needed on the laptop.
    """
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return {"ok": False, "error": "no HF_TOKEN in secret"}
    api, sent = HfApi(token=tok), []
    for local, remote in (("/repo/docs", "docs"),
                          ("/repo/outputs/corpus_scan", "outputs/corpus_scan")):
        if pathlib.Path(local).is_dir():
            api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=local,
                              path_in_repo=remote, commit_message="sync docs and scans")
            sent.append(remote)
    if pathlib.Path("/repo/FINDINGS.md").is_file():
        api.upload_file(path_or_fileobj="/repo/FINDINGS.md", path_in_repo="FINDINGS.md",
                        repo_id=HF_REPO, repo_type="dataset", commit_message="sync findings")
        sent.append("FINDINGS.md")
    res = {"ok": True, "repo": HF_REPO, "uploaded": sent}
    print("SYNC_DOCS:", json.dumps(res))
    return res

@app.local_entrypoint()
def g4_all():
    for alias in ("moe-midtrain", "moe-base"):
        print(g4.remote(alias))


@app.local_entrypoint()
def g2b_all():
    for judge in ("llama-3.1-8b-instruct", "qwen2.5-32b-instruct"):
        print(g2b.remote(judge, "instruct"))


@app.local_entrypoint()
def g2_all():
    for judge in ("llama-3.1-8b-instruct", "qwen2.5-32b-instruct"):
        for target in JUDGE_TARGETS:
            print(g2.remote(judge, target))
