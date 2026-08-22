"""vLLM reasoning-reliability suite (anonymised Modal app).

Naming: this app runs in a shared Modal workspace, so the app name, volume names, and
every log line use neutral aliases. Model identifiers are resolved from aliases at call
time and are never printed. Set ``LLMR_SECRET`` if the HF secret is named differently.

Why vLLM rather than transformers
---------------------------------
The reasoning-specialised checkpoint (alias ``M3-math``) is a long chain-of-thought model:
its post-training raised context 4K -> 32K and ran policy optimisation with 8K-16K output
tokens, and it emits its final answer inside ``\\boxed{}``. Generating it under a 512-3072
token cap truncates it mid-derivation, which is what produced the earlier collapse: 26% of
its responses contained the gold answer yet scored wrong, and completion never exceeded
75%. Its upstream repo ships modeling code targeting the vLLM engine, so vLLM is both the
faster and the *intended* path. Per-model budgets live in ``PROFILES`` below.

Entry points, in the order they should be run:

    modal run experiments/modal_reasoning.py::smoke
    modal run experiments/modal_reasoning.py::validate
    modal run experiments/modal_reasoning.py::generate --alias M3-instruct --suite atlas

``smoke`` proves every checkpoint loads and decodes. ``validate`` reproduces published
accuracy and is a hard gate: nothing downstream is trustworthy until it passes.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time

import modal

# --- naming ------------------------------------------------------------------------------
APP_NAME = "llm-reasoning"
RUNS_VOL = "llm-reasoning-runs"
CACHE_VOL = "llm-reasoning-cache"
SECRET_NAME = os.environ.get("LLMR_SECRET", "llm-reasoning-hf")

GPU = os.environ.get("LLMR_GPU", "A100-80GB")
RUNS_REMOTE = "/runs"
CACHE_REMOTE = "/cache"

#: Public alias -> upstream repo. Aliases are the only thing that appears in logs.
MODELS: dict[str, str] = {
    "M1": "amd/AMD-OLMo-1B",
    "M3-stage1": "amd/Instella-3B-Stage1",
    "M3-base": "amd/Instella-3B",
    "M3-sft": "amd/Instella-3B-SFT",
    "M3-instruct": "amd/Instella-3B-Instruct",
    "M3-math": "amd/Instella-3B-Math",
}

#: Per-model decoding profile. ``answer_style`` selects the prompt suffix and the
#: extraction order, which must match how the checkpoint was trained to answer.
#: ``max_model_len`` overrides the default (max_tokens + 2048) where a checkpoint's own
#: position embeddings are smaller; exceeding them is a hard error in the engine.
PROFILES: dict[str, dict] = {
    "M1": {"max_tokens": 640, "answer_style": "hash", "chat": False,
           "max_model_len": 2048},
    "M3-stage1": {"max_tokens": 768, "answer_style": "hash", "chat": False},
    "M3-base": {"max_tokens": 1024, "answer_style": "hash", "chat": False},
    "M3-sft": {"max_tokens": 1536, "answer_style": "hash", "chat": True},
    "M3-instruct": {"max_tokens": 1536, "answer_style": "hash", "chat": True},
    # Long-CoT. 16384 (its final rollout length) still truncated 25/200 GSM8K test items
    # and cost ~7 accuracy points against the published figure. Headroom matters more than
    # throughput here: the structural variants compose an extra step onto the problem, so
    # they run longer, and truncation that scales with condition difficulty would imitate
    # exactly the effect this study is testing for. Its SFT context is 32K.
    "M3-math": {"max_tokens": 24576, "answer_style": "boxed", "chat": True},
}

#: Published reference accuracy for the validation gate (GSM8K). The gate compares against
#: these; a large shortfall means the harness is wrong, not the checkpoint.
PUBLISHED_GSM8K: dict[str, float] = {
    "M3-stage1": 10.8,
    "M3-base": 59.8,
    "M3-sft": 71.7,
    "M3-instruct": 73.9,
    "M3-math": 92.48,
}

HASH_SUFFIX = "\nLet's think step by step, then give the final answer after '####'."
BOXED_SUFFIX = "\nLet's think step by step and output the final answer within \\boxed{}."

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # 0.6.3 predates this checkpoint family: its engine rejects the architecture
        # outright. The upstream repo ships a *native vLLM* model class whose __init__
        # takes a VllmConfig and which imports vllm.model_executor.layers.sampler.Sampler.
        # The VllmConfig signature arrived after 0.6.3; Sampler is removed once the V1
        # engine becomes mandatory. 0.8.5 is inside the window that has both.
        "vllm==0.8.5.post1",
        "transformers==4.56.0",
        "huggingface-hub>=0.23.0",
        "datasets>=2.19.0",
        "sympy>=1.12",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_REMOTE}/hf",
            "HF_HUB_DISABLE_XET": "1",
            # Keeps upstream repo paths out of the shared workspace's log stream.
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            # The registered model class uses the V0 sampler interface. V1 is default from
            # 0.8 onward and would import a sampler path the class does not implement.
            "VLLM_USE_V1": "0",
            # /arch must be on PYTHONPATH rather than injected via sys.path at runtime:
            # the engine inspects model architectures in a subprocess
            # (python -m vllm.model_executor.models.registry), which inherits the
            # environment but not the parent's sys.path edits.
            "PYTHONPATH": "/src:/arch",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    # Bake the out-of-tree model class into the image, pinned at build time. Downloading
    # it at call time would leave the inspection subprocess unable to import it.
    .run_commands(
        "mkdir -p /arch",
        # HF_HOME is redirected for this step only, then removed: the image-level HF_HOME
        # lives under /cache, and leaving a build artifact there makes the mount point
        # non-empty, which blocks the cache volume from mounting at run time.
        "HF_HOME=/tmp/hfbuild python -c \"from huggingface_hub import hf_hub_download; "
        "import shutil; shutil.copyfile("
        "hf_hub_download('amd/Instella-3B-Math', 'modeling_instella.py'), "
        "'/arch/vllm_arch.py')\" && rm -rf /tmp/hfbuild",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent.parent / "src"),
        "/src",
        copy=True,
        ignore=["**/__pycache__", "**/__pycache__/**", "**/*.pyc"],
    )
)

app = modal.App(APP_NAME, image=image)
runs_vol = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)
cache_vol = modal.Volume.from_name(CACHE_VOL, create_if_missing=True)
VOLUMES = {RUNS_REMOTE: runs_vol, CACHE_REMOTE: cache_vol}
SECRETS = [modal.Secret.from_name(SECRET_NAME)]

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_BOXED_TOKEN = "\\boxed{"


def boxed_balanced(text: str | None) -> str | None:
    r"""Last ``\boxed{...}`` extracted by brace matching.

    The character-class form cannot match a nested group, so any answer containing
    ``\frac``, ``\sqrt`` or ``\text`` fails to extract and falls through to a weaker
    rule. On MATH this silently mis-scored both sides of the comparison: 54 of 200 gold
    answers were left as the entire solution text, and predictions degraded to the last
    number in the completion. Recovering them moves measured accuracy from 0.235 to 0.300.
    """
    if not text:
        return None
    out, i = None, 0
    while True:
        j = text.find(_BOXED_TOKEN, i)
        if j < 0:
            break
        k = j + len(_BOXED_TOKEN)
        depth, buf = 1, []
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            k += 1
        if depth == 0:
            out = "".join(buf).strip()
        i = k + 1
    return out
_HASH = re.compile(r"####\s*([^\n]+)")
_LAST_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


# --- extraction --------------------------------------------------------------------------
def extract_answer(text: str, style: str) -> tuple[str | None, str]:
    """Return ``(answer, tier)``. ``tier`` records which rule fired, for the audit.

    Order matters and is style-dependent: a long-CoT model answers in ``\\boxed{}`` and
    only incidentally emits ``####``, so scoring it with the ``####`` rule first discards
    correct answers. Both rules are always tried; only their precedence changes.
    """
    rules = (
        [("boxed", _BOXED), ("hash", _HASH)]
        if style == "boxed"
        else [("hash", _HASH), ("boxed", _BOXED)]
    )
    for tier, pattern in rules:
        if tier == "boxed":
            bal = boxed_balanced(text)
            if bal is not None:
                return bal, tier
            continue
        found = pattern.findall(text)
        if found:
            return found[-1].strip(), tier
    nums = _LAST_NUM.findall(text)
    if nums:
        return nums[-1].strip(), "last_number"
    return None, "none"


def normalise(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().replace(",", "").replace("$", "").rstrip(".")
    v = re.sub(r"^\\text\{(.*)\}$", r"\1", v).strip()
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return v or None


def _suffix(style: str) -> str:
    return BOXED_SUFFIX if style == "boxed" else HASH_SUFFIX


# --- engine ------------------------------------------------------------------------------
#: The 3B checkpoints share one architecture, and the reasoning-specialised repo is the
#: only one that ships a *vLLM-native* implementation of it (the others ship a
#: transformers-style module the engine cannot consume). Registering that single class
#: therefore serves every 3B checkpoint, base through reasoning-specialised.
_ARCH_SOURCE = MODELS["M3-math"]
_ARCH_NAME = "InstellaForCausalLM"
_registered = False


def _register_custom_arch() -> None:
    """Register the out-of-tree model class so the engine accepts the 3B checkpoints.

    Downloaded under a distinct module name: the upstream file and the transformers-style
    file shipped by the other repos share a basename, and the dynamic-module loader would
    otherwise resolve whichever landed in ``sys.modules`` first.
    """
    global _registered
    if _registered:
        return
    from vllm import ModelRegistry

    # The module itself is baked into /arch at image build time and reachable via
    # PYTHONPATH, so the engine's inspection subprocess can import it too.
    if not pathlib.Path("/arch/vllm_arch.py").exists():
        raise RuntimeError("architecture module missing from image; rebuild required")

    # String form rather than the class object: the engine imports the model inside a
    # separate process, where a class captured here would not survive.
    ModelRegistry.register_model(_ARCH_NAME, "vllm_arch:InstellaForCausalLM")
    _registered = True
    print("[arch] custom architecture registered", flush=True)


def _load(alias: str):
    """Build a vLLM engine for ``alias``. Raises with a neutral message on failure."""
    from vllm import LLM

    repo = MODELS[alias]
    profile = PROFILES[alias]
    if alias.startswith("M3"):
        _register_custom_arch()
    ctx = profile.get("max_model_len") or (profile["max_tokens"] + 2048)
    print(f"[{alias}] loading engine (ctx={ctx})", flush=True)
    llm = LLM(
        model=repo,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=ctx,
        enforce_eager=False,
    )
    return llm


def _render(llm, alias: str, prompts: list[str]) -> list[str]:
    """Apply the checkpoint's own chat template when it has one."""
    profile = PROFILES[alias]
    suffix = _suffix(profile["answer_style"])
    tok = llm.get_tokenizer()
    out = []
    for p in prompts:
        text = p + suffix
        if profile["chat"] and getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        out.append(text)
    return out


def _generate(llm, alias: str, prompts: list[str], temperature: float = 0.0) -> list[dict]:
    from vllm import SamplingParams

    profile = PROFILES[alias]
    params = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0 else 0.95,
        max_tokens=profile["max_tokens"],
    )
    rendered = _render(llm, alias, prompts)
    t0 = time.time()
    outs = llm.generate(rendered, params)
    dt = time.time() - t0
    rows = []
    truncated = 0
    for o in outs:
        comp = o.outputs[0]
        finished = comp.finish_reason == "stop"
        truncated += 0 if finished else 1
        ans, tier = extract_answer(comp.text, profile["answer_style"])
        rows.append(
            {
                "completion": comp.text,
                "n_tokens": len(comp.token_ids),
                "finished": finished,
                "extracted": ans,
                "extraction_tier": tier,
            }
        )
    print(
        f"[{alias}] {len(prompts)} prompts in {dt:.0f}s "
        f"({len(prompts) / max(dt, 1e-9):.2f}/s) truncated={truncated}",
        flush=True,
    )
    return rows


def _gsm8k(split: str, n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for r in ds.select(range(min(n, len(ds)))):
        gold = r["answer"].split("####")[-1].strip()
        rows.append({"prompt": r["question"], "gold": gold})
    return rows


# --- entry points ------------------------------------------------------------------------
@app.function(gpu=GPU, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def smoke(aliases: str = "M3-instruct,M3-math") -> dict:
    """Prove each checkpoint loads under vLLM and decodes a trivial prompt.

    Runs before anything expensive: a checkpoint whose remote code will not load under
    the engine should cost one minute to discover, not an hour.
    """
    report = {}
    for alias in [a.strip() for a in aliases.split(",") if a.strip()]:
        try:
            llm = _load(alias)
            rows = _generate(llm, alias, ["What is 2 + 3?"])
            report[alias] = {
                "loaded": True,
                "extracted": rows[0]["extracted"],
                "tier": rows[0]["extraction_tier"],
                "n_tokens": rows[0]["n_tokens"],
                "finished": rows[0]["finished"],
            }
            print(f"[{alias}] OK -> {rows[0]['extracted']!r} ({rows[0]['extraction_tier']})")
            del llm
        except Exception as exc:  # surfaced per-alias so one failure does not hide the rest
            report[alias] = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"[{alias}] FAILED {type(exc).__name__}", flush=True)
    print(json.dumps(report, indent=2))
    return report


@app.function(gpu=GPU, timeout=3 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def validate(aliases: str = "M3-base,M3-instruct,M3-math", n: int = 200) -> dict:
    """Hard gate: reproduce published GSM8K accuracy before spending the run budget.

    Uses the GSM8K *test* split, zero-shot with each checkpoint's native answer style.
    Published figures use few-shot prompting for the base checkpoints, so a base model
    landing a few points low is expected; a checkpoint landing tens of points low means
    the harness is wrong. The gate reports the delta and lets the caller judge.
    """
    items = _gsm8k("test", n)
    golds = [normalise(r["gold"]) for r in items]
    prompts = [r["prompt"] for r in items]

    report = {}
    for alias in [a.strip() for a in aliases.split(",") if a.strip()]:
        llm = _load(alias)
        rows = _generate(llm, alias, prompts)
        correct = sum(1 for r, g in zip(rows, golds, strict=False) if normalise(r["extracted"]) == g)
        acc = 100.0 * correct / len(items)
        tiers: dict[str, int] = {}
        for r in rows:
            tiers[r["extraction_tier"]] = tiers.get(r["extraction_tier"], 0) + 1
        published = PUBLISHED_GSM8K.get(alias)
        report[alias] = {
            "n": len(items),
            "accuracy": round(acc, 2),
            "published": published,
            "delta": None if published is None else round(acc - published, 2),
            "truncated": sum(1 for r in rows if not r["finished"]),
            "median_tokens": sorted(r["n_tokens"] for r in rows)[len(rows) // 2],
            "extraction_tiers": tiers,
        }
        print(f"[{alias}] acc={acc:.2f} published={published} "
              f"delta={report[alias]['delta']} tiers={tiers}", flush=True)
        del llm

    out = pathlib.Path(RUNS_REMOTE) / "validation" / "gsm8k_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    runs_vol.commit()
    print(json.dumps(report, indent=2))
    return report


# --- suite definition --------------------------------------------------------------------
#: benchmark -> (reasoning subskill, applies structural/numeric perturbation).
#: Structural and numeric variants need an exact recomputable gold answer, which the
#: multiple-choice benchmarks do not provide; those carry surface variants only, and the
#: atlas reports that difference rather than hiding it.
SUITE: dict[str, tuple[str, bool]] = {
    "gsm8k": ("arithmetic", True),
    "math": ("mathematical", True),
    "arc_challenge": ("commonsense_science", False),
    "bbh": ("multi_step", False),
    "logiqa2": ("logical_deduction", False),
    "reclor": ("logical_reading", False),
    # Apple's hand-written GSM-Symbolic release. Correct by construction, so it validates
    # our auto-derived numeric variants against the published gold standard; and because
    # p1/p2 add one and two extra clauses respectively, the trio is an externally authored
    # *structural* difficulty ladder we did not design and cannot have tuned.
    "gsm_symbolic": ("arithmetic_symbolic", False),
    "gsm_symbolic_p1": ("arithmetic_symbolic", False),
    "gsm_symbolic_p2": ("arithmetic_symbolic", False),
}

#: Benchmarks passed through unperturbed: the release already ships the cluster structure
#: (one template head plus its instances), so re-perturbing would confound our variants
#: with theirs.
RAW_BENCHMARKS = {"gsm_symbolic", "gsm_symbolic_p1", "gsm_symbolic_p2"}

#: The verified-membership arms, carried as a seventh suite entry. Distinct from the
#: ``gsm8k`` benchmark above: those are test-split items with no membership label, whereas
#: these 250 seen / 250 unseen items were assigned by exact 13-gram containment against the
#: published pretraining corpus and difficulty-matched across three bins.
ARMS = "arms"
ARMS_SOURCE = "experiments/runs/fullscale-S250-v2/arms/gsm8k_arms.jsonl"
SKILL_OF = {**{k: v[0] for k, v in SUITE.items()}, ARMS: "arithmetic",
            "dose": "arithmetic_dose"}

RUN = os.environ.get("LLMR_RUN", "atlas-v1")


def _run_dir() -> pathlib.Path:
    return pathlib.Path(RUNS_REMOTE) / RUN


@app.function(cpu=4.0, timeout=2 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def build(limit: int = 200, numeric_k: int = 2, seed: int = 6198) -> dict:
    """Download benchmarks and expand each item into a variant cluster.

    Each cluster carries the original, answer-preserving surface rewrites, and — where the
    benchmark admits an exact gold answer — numeric and structural variants. The structural
    pair is the load-bearing addition: the pretraining augmentation for this family was
    itself built by re-valuing GSM8K parameters while holding the solution program fixed,
    so numeric perturbation lands inside the training distribution and cannot separate
    memorisation from reasoning. Changing the program is the axis that can.
    """
    import sys

    sys.path.insert(0, "/src")
    from instella_reasoning.datasets.loaders import load_benchmark_to_jsonl
    from instella_reasoning.perturbations import PerturbationConfig, make_variant_suite
    from instella_reasoning.records import read_benchmark, write_jsonl
    from instella_reasoning.structural import make_structural_variants

    out = _run_dir()
    (out / "base").mkdir(parents=True, exist_ok=True)
    (out / "variants").mkdir(parents=True, exist_ok=True)

    report: dict = {}
    for bench, (skill, exact) in SUITE.items():
        base_path = out / "base" / f"{bench}.jsonl"
        raw = bench in RAW_BENCHMARKS
        if not base_path.exists():
            # max_per_parent trims 50 instances/template to the handful the clustered
            # design actually gains from; limit still caps total rows.
            load_benchmark_to_jsonl(bench, base_path, limit=600 if raw else limit,
                                    max_per_parent=3 if raw else None)
        items = read_benchmark(base_path)

        cfg = (PerturbationConfig(types=(), numeric_variants=0, seed=seed) if raw
               else PerturbationConfig(numeric_variants=numeric_k if exact else 0, seed=seed))
        variants = list(make_variant_suite(items, cfg))
        n_structural = 0
        if exact and not raw:
            structural = make_structural_variants(items, seed=seed)
            variants.extend(structural.variants)
            n_structural = len(structural.variants)

        write_jsonl(out / "variants" / f"{bench}.jsonl", variants)
        counts: dict[str, int] = {}
        for v in variants:
            counts[v.variant_type] = counts.get(v.variant_type, 0) + 1
        report[bench] = {
            "skill": skill,
            "base_items": len(items),
            "rows": len(variants),
            "structural": n_structural,
            "by_type": counts,
        }
        print(f"[build] {bench:14s} skill={skill:20s} base={len(items):4d} "
              f"rows={len(variants):5d} structural={n_structural}", flush=True)

    (out / "variants" / "manifest.json").write_text(json.dumps(report, indent=2), "utf-8")
    runs_vol.commit()
    return report


@app.function(cpu=4.0, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def build_arms(numeric_k: int = 2, seed: int = 6198,
               premise_removal_k: int = 3) -> dict:
    """Expand the verified-membership arms into perturbation clusters.

    This is the experiment the subskill atlas cannot run. Structural perturbation makes a
    problem harder for any model, so an accuracy drop under it means nothing on its own.
    The claim requires the drop to be *larger on items verified to be in the training
    corpus* than on matched items that are not, which differences out the added
    difficulty:

        DiD = (seen - unseen | original) - (seen - unseen | depth_extension)

    The arms are reused rather than re-derived: their containment scan was audited at a
    0.0% false-positive rate against 1000 items that cannot be contaminated, and they are
    already matched on a model-independent difficulty bin.
    """
    import sys

    sys.path.insert(0, "/src")
    from huggingface_hub import hf_hub_download

    from instella_reasoning.perturbations import PerturbationConfig, make_variant_suite
    from instella_reasoning.records import read_benchmark, write_jsonl
    from instella_reasoning.structural import make_structural_variants

    out = _run_dir()
    (out / "variants").mkdir(parents=True, exist_ok=True)
    src = hf_hub_download("GOVINDFROM/Instella-Reasoning", ARMS_SOURCE, repo_type="dataset")
    items = read_benchmark(src)

    variants = list(make_variant_suite(items, PerturbationConfig(
        numeric_variants=numeric_k, seed=seed)))
    variants.extend(make_structural_variants(
        items, seed=seed, premise_removal_k=premise_removal_k).variants)

    # Variants inherit the parent's arm label: perturbation changes the question, not
    # whether the underlying problem was in the training corpus.
    arm_of = {i.id: (i.metadata or {}).get("arm") for i in items}
    for v in variants:
        v.metadata = dict(v.metadata or {})
        v.metadata["arm"] = arm_of.get(v.parent_id) or arm_of.get(v.id)

    write_jsonl(out / "variants" / f"{ARMS}.jsonl", variants)
    counts: dict[str, int] = {}
    for v in variants:
        key = f"{v.metadata.get('arm')}|{v.variant_type}"
        counts[key] = counts.get(key, 0) + 1
    print(f"[arms] parents={len(items)} rows={len(variants)}", flush=True)
    for k in sorted(counts):
        print(f"  {k:34s} {counts[k]:4d}", flush=True)
    runs_vol.commit()
    return {"parents": len(items), "rows": len(variants), "cells": counts}


DOSE = "dose"


@app.function(cpu=4.0, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def build_dose(numeric_k: int = 2, premise_removal_k: int = 3, seed: int = 6198) -> dict:
    """Within-split contrast: high- versus low-containment items, both from GSM8K *train*.

    The arms used elsewhere compare train items (in the corpus) against test items (not in
    it), which leaves one objection unanswered: the contrast is train-versus-test, so a
    difference could be a generalisation gap rather than a membership effect. That
    objection is fatal to a memorisation claim and cannot be argued away.

    It can, however, be designed away. Containment is continuous, and within the train
    split alone it spans the full range: 303 items sit at or above 0.80 and 163 at or
    below 0.10. Both arms are then the same split, written by the same annotators to the
    same specification, and differ only in how heavily the augmentation reproduced them.
    Any residual difference cannot be a train/test artifact because there is no test item
    in the design.

    Arms are matched on reasoning-step count, the model-independent difficulty proxy, so
    the harder-item confound is also closed.
    """
    import sys
    from collections import defaultdict

    sys.path.insert(0, "/src")
    from huggingface_hub import hf_hub_download

    from instella_reasoning.perturbations import PerturbationConfig, make_variant_suite
    from instella_reasoning.records import read_benchmark, write_jsonl
    from instella_reasoning.structural import make_structural_variants

    out = _run_dir()
    (out / "variants").mkdir(parents=True, exist_ok=True)
    repo = "GOVINDFROM/Instella-Reasoning"
    base = "experiments/runs/fullscale-S250-v2"
    cont = json.load(open(hf_hub_download(
        repo, f"{base}/analysis/containment_verified.json", repo_type="dataset"),
        encoding="utf-8"))["containment"]
    items = read_benchmark(hf_hub_download(
        repo, f"{base}/base/gsm8k_train.jsonl", repo_type="dataset"))

    def steps(it) -> int:
        return str((it.metadata or {}).get("rationale", "")).count("<<")

    high, low = [], []
    for it in items:
        rec = cont.get(f"train::{it.id}")
        if not isinstance(rec, dict):
            continue
        c = rec.get("containment")
        if c is None:
            continue
        (high if c >= 0.80 else low if c <= 0.10 else []).append((it, c))

    # Difficulty-match by reasoning-step count, taking equal numbers from each bin.
    by_steps_hi: dict[int, list] = defaultdict(list)
    by_steps_lo: dict[int, list] = defaultdict(list)
    for it, c in high:
        by_steps_hi[steps(it)].append((it, c))
    for it, c in low:
        by_steps_lo[steps(it)].append((it, c))

    picked: list = []
    for s in sorted(set(by_steps_hi) | set(by_steps_lo)):
        n = min(len(by_steps_hi.get(s, [])), len(by_steps_lo.get(s, [])))
        for pool, arm in ((by_steps_hi, "seen"), (by_steps_lo, "unseen")):
            for it, c in pool.get(s, [])[:n]:
                it.metadata = dict(it.metadata or {})
                it.metadata.update({"arm": arm, "containment": c, "n_steps": s})
                picked.append(it)

    variants = list(make_variant_suite(picked, PerturbationConfig(
        numeric_variants=numeric_k, seed=seed)))
    variants.extend(make_structural_variants(
        picked, seed=seed, premise_removal_k=premise_removal_k).variants)
    meta = {i.id: i.metadata for i in picked}
    for v in variants:
        src = meta.get(v.parent_id) or meta.get(v.id) or {}
        v.metadata = {**(v.metadata or {}), "arm": src.get("arm"),
                      "containment": src.get("containment"), "n_steps": src.get("n_steps")}

    write_jsonl(out / "variants" / f"{DOSE}.jsonl", variants)
    counts: dict[str, int] = {}
    for v in variants:
        counts[f"{v.metadata.get('arm')}|{v.variant_type}"] = counts.get(
            f"{v.metadata.get('arm')}|{v.variant_type}", 0) + 1
    n_hi = sum(1 for i in picked if i.metadata["arm"] == "seen")
    print(f"[dose] pools high={len(high)} low={len(low)} -> matched {n_hi}/arm, "
          f"rows={len(variants)}", flush=True)
    for k in sorted(counts):
        print(f"  {k:34s} {counts[k]:4d}", flush=True)
    runs_vol.commit()
    return {"matched_per_arm": n_hi, "rows": len(variants), "cells": counts}


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes=VOLUMES, secrets=SECRETS)
def generate(alias: str, benchmarks: str = "", overwrite: bool = False,
             types: str = "") -> dict:
    """Generate one checkpoint's completions across the suite.

    Invoked per checkpoint so the GPU budget is spent in controllable units and a failure
    costs one checkpoint rather than the run. Existing outputs are skipped unless
    ``overwrite``, so an interrupted run resumes instead of restarting.
    """
    import sys

    sys.path.insert(0, "/src")
    from instella_reasoning.records import read_benchmark

    out = _run_dir()
    gen_dir = out / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    names = [b.strip() for b in (benchmarks or ",".join(SUITE)).split(",") if b.strip()]

    llm = _load(alias)
    report: dict = {}
    for bench in names:
        target = gen_dir / f"{alias}__{bench}.jsonl"
        # With a type filter, an existing file is *merged* rather than replaced: only the
        # filtered variant types are re-generated and swapped in. Overwriting would
        # discard the conditions already paid for, which for the long-chain-of-thought
        # checkpoint is hours of GPU time.
        keep_existing: list[str] = []
        if target.exists() and not overwrite:
            if not types:
                print(f"[{alias}] {bench}: exists, skipping", flush=True)
                continue
            wanted = {t.strip() for t in types.split(",") if t.strip()}
            keep_existing = [ln for ln in target.read_text(encoding="utf-8").splitlines()
                             if ln.strip() and json.loads(ln)["variant_type"] not in wanted]
            print(f"[{alias}] {bench}: merging, retaining {len(keep_existing)} existing rows",
                  flush=True)
        items = read_benchmark(out / "variants" / f"{bench}.jsonl")
        if types:
            # Lets the long-chain-of-thought checkpoint pay only for the cells the
            # difference-in-differences needs. Surface variants feed the consistency
            # term, which is not part of that estimator.
            keep = {t.strip() for t in types.split(",") if t.strip()}
            items = [i for i in items if i.variant_type in keep]
            print(f"[{alias}] {bench}: filtered to {len(items)} rows ({sorted(keep)})",
                  flush=True)
        rows = _generate(llm, alias, [i.prompt for i in items])
        with target.open("w", encoding="utf-8") as fh:
            for ln in keep_existing:
                fh.write(ln + "\n")
            for item, row in zip(items, rows, strict=False):
                fh.write(json.dumps({
                    "id": item.id,
                    "parent_id": item.parent_id or item.id,
                    "benchmark": bench,
                    "skill": SKILL_OF.get(bench, "arithmetic"),
                    "arm": (item.metadata or {}).get("arm"),
                    "variant_type": item.variant_type,
                    "answer_changing": bool(item.metadata.get("answer_changing"))
                                       or item.variant_type in ("numeric_perturbation",
                                                                "gsm_symbolic"),
                    "gold": item.answer,
                    # Retained for premise_removal, whose rows have no gold: the metric is
                    # whether the model emits the parent's answer to a question the prompt
                    # no longer determines.
                    "parent_answer": (item.metadata or {}).get("parent_answer"),
                    "unanswerable": bool((item.metadata or {}).get("unanswerable")),
                    **row,
                }) + "\n")
        report[bench] = {
            "rows": len(rows),
            "truncated": sum(1 for r in rows if not r["finished"]),
        }
        runs_vol.commit()
        print(f"[{alias}] {bench}: wrote {len(rows)} rows", flush=True)
    del llm
    return report


def _correct(pred: str | None, gold: str | None) -> bool:
    """Exact-match with numeric normalisation, falling back to symbolic equivalence."""
    # A gold field that still carries its whole solution is unwrapped here rather than at
    # load time, so previously generated rows are rescored correctly without re-running.
    unwrapped = boxed_balanced(gold)
    if unwrapped is not None:
        gold = unwrapped
    if pred is None or gold is None:
        return False
    p, g = normalise(pred), normalise(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    if p.strip().upper() == g.strip().upper():
        return True
    try:  # MATH answers need symbolic equality (\frac, \sqrt, pi ...)
        from instella_reasoning.answer_equivalence import answers_equivalent

        # Argument order is (expected, predicted); reversing it silently changes the
        # LaTeX-normalisation path applied to each side.
        return bool(answers_equivalent(gold, pred))
    except Exception:
        return False


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def atlas() -> dict:
    """Score generations and build the reliability atlas.

    Reliability = accuracy x consistency, per checkpoint and reasoning subskill.
    Consistency is computed over answer-*preserving* variants only; answer-changing
    variants (numeric and structural) enter accuracy and the perturbation contrast, where
    comparing correctness labels across different gold answers would be meaningless.
    """
    import glob
    from collections import defaultdict

    out = _run_dir()
    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(str(out / "generations" / "*.jsonl"))):
        alias = pathlib.Path(path).name.split("__")[0]
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            r["correct"] = _correct(
                boxed_balanced(r.get("completion")) or r.get("extracted"), r.get("gold"))
            rows_by_model[alias].append(r)

    report: dict = {}
    for alias, rows in rows_by_model.items():
        per_skill: dict[str, dict] = {}
        by_skill: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_skill[r["skill"]].append(r)

        for skill, srows in by_skill.items():
            acc = sum(r["correct"] for r in srows) / max(1, len(srows))

            clusters: dict[str, list[dict]] = defaultdict(list)
            for r in srows:
                if not r["answer_changing"]:
                    clusters[r["parent_id"]].append(r)
            cons_vals = [
                sum(1 for r in c if r["correct"] == c[0]["correct"]) / len(c)
                for c in clusters.values()
                if len(c) >= 2
            ]
            cons = sum(cons_vals) / len(cons_vals) if cons_vals else None

            def _acc(pred, srows=srows) -> float | None:
                sel = [r for r in srows if pred(r)]
                return sum(x["correct"] for x in sel) / len(sel) if sel else None

            def _trunc(pred, srows=srows) -> float | None:
                """Truncation rate for a condition.

                Load bearing, not diagnostic. The structural variants compose an extra
                step onto the problem, so they generate longer; if a long-chain-of-thought
                checkpoint truncates more often on them, its accuracy falls for a reason
                that has nothing to do with recalling a solution program. The contrast is
                only interpretable when these rates are comparable across conditions.
                """
                sel = [r for r in srows if pred(r)]
                return (sum(0 if x["finished"] else 1 for x in sel) / len(sel)
                        if sel else None)

            def _by(*vtypes):
                """Match any of several names for one condition.

                The auto-derived numeric variants are tagged ``gsm_symbolic`` while the
                config-driven ones are ``numeric_perturbation``; matching only the latter
                silently reports the numeric column as absent.
                """
                names = set(vtypes)
                # noqa target: `r` is this lambda's own parameter and shadows the
                # outer loop name, which B023 does not model.
                return lambda r, names=names: r["variant_type"] in names  # noqa: B023

            per_skill[skill] = {
                "n_rows": len(srows),
                "n_clusters": len(clusters),
                "accuracy": round(acc, 4),
                "consistency": None if cons is None else round(cons, 4),
                "reliability": None if cons is None else round(acc * cons, 4),
                "acc_original": _acc(_by("original")),
                "acc_numeric": _acc(_by("numeric_perturbation", "gsm_symbolic")),
                # The discriminating contrast: structural perturbation sits outside the
                # augmentation's support, numeric perturbation inside it.
                "acc_depth": _acc(_by("depth_extension")),
                "acc_distractor": _acc(_by("distractor_quantity")),
                # Paired with the accuracies above so an accuracy gap can be checked
                # against the truncation gap that could have produced it.
                "trunc_original": _trunc(_by("original")),
                "trunc_numeric": _trunc(_by("numeric_perturbation", "gsm_symbolic")),
                "trunc_depth": _trunc(_by("depth_extension")),
                "trunc_distractor": _trunc(_by("distractor_quantity")),
                "truncated": sum(1 for r in srows if not r["finished"]),
                "extraction_tiers": {
                    t: sum(1 for r in srows if r["extraction_tier"] == t)
                    for t in {r["extraction_tier"] for r in srows}
                },
            }
        report[alias] = per_skill
        print(f"[atlas] {alias}: " + " ".join(
            f"{s}={v['reliability']}" for s, v in sorted(per_skill.items())), flush=True)

    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "atlas.json").write_text(json.dumps(report, indent=2), "utf-8")

    lines = ["# Reasoning Reliability Atlas", "",
             "Reliability = Accuracy x Consistency. `acc_numeric` perturbs values (inside "
             "the pretraining augmentation's support); `acc_depth` perturbs the solution "
             "program (outside it).", "",
             "| Model | Subskill | n | Acc | Cons | Reliability | orig | numeric | depth | distract |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for alias in sorted(report):
        for skill in sorted(report[alias]):
            c = report[alias][skill]
            lines.append(
                f"| {alias} | {skill} | {c['n_rows']} | {fmt(c['accuracy'])} | "
                f"{fmt(c['consistency'])} | {fmt(c['reliability'])} | {fmt(c['acc_original'])} | "
                f"{fmt(c['acc_numeric'])} | {fmt(c['acc_depth'])} | {fmt(c['acc_distractor'])} |")
    (out / "analysis" / "atlas.md").write_text("\n".join(lines), "utf-8")
    runs_vol.commit()
    print("\n".join(lines))
    return report


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def did(bench: str = ARMS, n_bootstrap: int = 4000, seed: int = 6198) -> dict:
    """Difference-in-differences over verified membership x perturbation condition.

    Two contrasts per checkpoint, sharing the ``original`` baseline:

        DiD_numeric = (seen - unseen | original) - (seen - unseen | numeric)
        DiD_depth   = (seen - unseen | original) - (seen - unseen | depth_extension)

    The numeric arm is the negative control. Its perturbation re-values parameters, which
    is the operation that generated the pretraining augmentation, so it lies inside the
    training distribution and should show no effect regardless of what the model does.
    The depth arm changes the solution program, which the augmentation never did.

    Intervals resample **parent problems** as indivisible clusters. The variants of one
    problem are not independent observations; resampling rows would contract intervals by
    roughly the design effect and produce confidently wrong error bars.
    """
    import glob
    import random
    from collections import defaultdict

    out = _run_dir()
    paths = sorted(glob.glob(str(out / "generations" / f"*__{bench}.jsonl")))
    if not paths:
        raise SystemExit("no arms generations; run build_arms then generate --alias X "
                         "--benchmarks arms")

    def cell_acc(rows: list[dict], arm: str, cond) -> float | None:
        sel = [r for r in rows if r["arm"] == arm and cond(r["variant_type"])]
        return sum(r["correct"] for r in sel) / len(sel) if sel else None

    report: dict = {}
    for path in paths:
        alias = pathlib.Path(path).name.split("__")[0]
        rows = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            r["correct"] = _correct(
                boxed_balanced(r.get("completion")) or r.get("extracted"), r.get("gold"))
            rows.append(r)

        by_parent: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_parent[r["parent_id"]].append(r)
        parents = list(by_parent)

        conditions = {
            "numeric": lambda v: v in ("numeric_perturbation", "gsm_symbolic"),
            "depth": lambda v: v == "depth_extension",
            "distractor": lambda v: v == "distractor_quantity",
        }
        orig = lambda v: v == "original"  # noqa: E731

        def estimate(sample: list[dict], cond, orig=orig) -> float | None:
            a = cell_acc(sample, "seen", orig)
            b = cell_acc(sample, "unseen", orig)
            c = cell_acc(sample, "seen", cond)
            d = cell_acc(sample, "unseen", cond)
            if None in (a, b, c, d):
                return None
            return (a - b) - (c - d)

        entry: dict = {"n_parents": len(parents), "n_rows": len(rows)}
        for name, cond in conditions.items():
            point = estimate(rows, cond)
            draws = []
            rng = random.Random(seed)
            for _ in range(n_bootstrap):
                sample: list[dict] = []
                for _ in range(len(parents)):
                    sample.extend(by_parent[parents[rng.randrange(len(parents))]])
                v = estimate(sample, cond)
                if v is not None:
                    draws.append(v)
            draws.sort()
            lo = draws[int(0.025 * len(draws))] if draws else None
            hi = draws[int(0.975 * len(draws))] if draws else None
            entry[f"did_{name}"] = None if point is None else round(point, 4)
            entry[f"ci_{name}"] = [None if lo is None else round(lo, 4),
                                   None if hi is None else round(hi, 4)]
            entry[f"excludes_zero_{name}"] = bool(
                lo is not None and hi is not None and (lo > 0 or hi < 0))

            # Cells and truncation, so an effect can be checked against the truncation
            # difference that could have manufactured it.
            for arm in ("seen", "unseen"):
                sel = [r for r in rows if r["arm"] == arm and cond(r["variant_type"])]
                entry[f"acc_{arm}_{name}"] = (
                    round(sum(r["correct"] for r in sel) / len(sel), 4) if sel else None)
                entry[f"trunc_{arm}_{name}"] = (
                    round(sum(0 if r["finished"] else 1 for r in sel) / len(sel), 4)
                    if sel else None)
            for arm in ("seen", "unseen"):
                sel = [r for r in rows if r["arm"] == arm and orig(r["variant_type"])]
                entry[f"acc_{arm}_original"] = (
                    round(sum(r["correct"] for r in sel) / len(sel), 4) if sel else None)

        # Answer-leakage filter. Generation rows do not carry the prompt, so it is joined
        # back from the variant file. An item whose parent answer still appears in the
        # pruned prompt is copying, not recall, and leakage correlates with containment
        # (6.9 vs 3.1 percent on the measured arms), so leaving it in biases the estimate
        # upward by roughly the size of the effect.
        leak_ids: set[str] = set()
        vpath = out / "variants" / f"{bench}.jsonl"
        if vpath.exists():
            for line in open(vpath, encoding="utf-8"):
                if not line.strip():
                    continue
                v = json.loads(line)
                if v.get("variant_type") != "premise_removal":
                    continue
                pa = (v.get("metadata") or {}).get("parent_answer")
                if pa is None:
                    continue
                if str(abs(int(pa))) in re.findall(r"\d+", v.get("prompt", "")):
                    leak_ids.add(v["id"])
        n_leak = sum(1 for r in rows
                     if r["variant_type"] == "premise_removal" and r["id"] in leak_ids)
        entry["n_leaked_excluded"] = n_leak
        rows = [r for r in rows
                if not (r["variant_type"] == "premise_removal" and r["id"] in leak_ids)]
        by_parent = defaultdict(list)
        for r in rows:
            by_parent[r["parent_id"]].append(r)
        parents = list(by_parent)

        # Recall rate on unanswerable items. This is the direct memorisation measurement:
        # the premise the answer depended on has been deleted, so reproducing the parent's
        # answer cannot come from the text. Compared across verified-membership arms.
        for arm in ("seen", "unseen"):
            sel = [r for r in rows
                   if r["arm"] == arm and r["variant_type"] == "premise_removal"]
            if sel:
                hits = sum(1 for r in sel
                           if r.get("parent_answer") is not None
                           and normalise(r.get("extracted")) == normalise(str(r["parent_answer"])))
                entry[f"recall_{arm}"] = round(hits / len(sel), 4)
                entry[f"n_removal_{arm}"] = len(sel)
            else:
                entry[f"recall_{arm}"] = None
        if entry.get("recall_seen") is not None and entry.get("recall_unseen") is not None:
            entry["recall_gap"] = round(entry["recall_seen"] - entry["recall_unseen"], 4)
            draws = []
            rng2 = random.Random(seed)
            for _ in range(n_bootstrap):
                sample: list[dict] = []
                for _ in range(len(parents)):
                    sample.extend(by_parent[parents[rng2.randrange(len(parents))]])
                vals = {}
                for arm in ("seen", "unseen"):
                    sel = [r for r in sample
                           if r["arm"] == arm and r["variant_type"] == "premise_removal"]
                    vals[arm] = (sum(1 for r in sel
                                     if r.get("parent_answer") is not None
                                     and normalise(r.get("extracted"))
                                     == normalise(str(r["parent_answer"]))) / len(sel)
                                 if sel else None)
                if vals["seen"] is not None and vals["unseen"] is not None:
                    draws.append(vals["seen"] - vals["unseen"])
            draws.sort()
            entry["ci_recall_gap"] = [round(draws[int(0.025 * len(draws))], 4),
                                      round(draws[int(0.975 * len(draws))], 4)] if draws else None

        # Paired within-parent contrast: for parents answered correctly on the numeric
        # variant, how often is the depth variant wrong, and vice versa? Discordant pairs
        # carry the whole signal, and the pairing controls item difficulty exactly.
        b_disc = c_disc = 0
        for _pid, prows in by_parent.items():
            num = [r for r in prows if conditions["numeric"](r["variant_type"])]
            dep = [r for r in prows if conditions["depth"](r["variant_type"])]
            if not num or not dep:
                continue
            n_ok = sum(r["correct"] for r in num) / len(num) >= 0.5
            d_ok = sum(r["correct"] for r in dep) / len(dep) >= 0.5
            if n_ok and not d_ok:
                b_disc += 1
            elif d_ok and not n_ok:
                c_disc += 1
        entry["paired_numeric_ok_depth_wrong"] = b_disc
        entry["paired_depth_ok_numeric_wrong"] = c_disc
        # McNemar without continuity correction; chi-square on 1 df.
        entry["mcnemar_chi2"] = (
            round((b_disc - c_disc) ** 2 / (b_disc + c_disc), 3)
            if (b_disc + c_disc) else None)

        report[alias] = entry
        print(f"[did] {alias}: numeric={entry['did_numeric']} {entry['ci_numeric']} | "
              f"depth={entry['did_depth']} {entry['ci_depth']} | "
              f"McNemar chi2={entry['mcnemar_chi2']} ({b_disc} vs {c_disc})", flush=True)

    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / f"did_{bench}.json").write_text(
        json.dumps(report, indent=2), "utf-8")
    runs_vol.commit()
    print(json.dumps(report, indent=2))
    return report


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def ladder(n_bootstrap: int = 2000, seed: int = 6198) -> dict:
    """Numeric versus structural fragility on an externally authored perturbation ladder.

    Uses the published GSM-Symbolic release rather than our own generators, which matters
    because the comparison would otherwise rest on perturbations we designed and could
    have tuned. Within it:

      * template head vs instances  -> the *numeric* effect (same program, new values)
      * main vs p1 vs p2            -> the *structural* effect (one and two added clauses)

    Both contrasts are measured on the same items by the same scorer, so the difference
    between them is a property of the model rather than of the two probes' construction.
    Intervals resample template clusters, since instances of one template are not
    independent observations.
    """
    import glob
    import random
    from collections import defaultdict

    out = _run_dir()
    report: dict = {}
    for alias in sorted({pathlib.Path(p).name.split("__")[0]
                         for p in glob.glob(str(out / "generations" / "*__gsm_symbolic*.jsonl"))}):
        entry: dict = {}
        for bench in ("gsm_symbolic", "gsm_symbolic_p1", "gsm_symbolic_p2"):
            path = out / "generations" / f"{alias}__{bench}.jsonl"
            if not path.exists():
                continue
            rows = []
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                r["correct"] = _correct(
                boxed_balanced(r.get("completion")) or r.get("extracted"), r.get("gold"))
                rows.append(r)
            clusters: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                clusters[r["parent_id"]].append(r)
            keys = list(clusters)

            def acc(sample: list[dict], pred) -> float | None:
                sel = [r for r in sample if pred(r)]
                return sum(r["correct"] for r in sel) / len(sel) if sel else None

            is_head = lambda r: r["variant_type"] == "original"  # noqa: E731
            is_inst = lambda r: r["variant_type"] != "original"  # noqa: E731

            point_head, point_inst = acc(rows, is_head), acc(rows, is_inst)
            draws = []
            rng = random.Random(seed)
            for _ in range(n_bootstrap):
                sample: list[dict] = []
                for _ in range(len(keys)):
                    sample.extend(clusters[keys[rng.randrange(len(keys))]])
                a, b = acc(sample, is_head), acc(sample, is_inst)
                if a is not None and b is not None:
                    draws.append(b - a)
            draws.sort()
            entry[bench] = {
                "n_templates": len(keys),
                "n_rows": len(rows),
                "acc_template_head": None if point_head is None else round(point_head, 4),
                "acc_instances": None if point_inst is None else round(point_inst, 4),
                "numeric_effect": (None if None in (point_head, point_inst)
                                   else round(point_inst - point_head, 4)),
                "ci_numeric_effect": ([round(draws[int(0.025 * len(draws))], 4),
                                       round(draws[int(0.975 * len(draws))], 4)]
                                      if draws else None),
            }
        # Structural effect: each added clause, measured on instances so the numeric
        # perturbation is already applied and cannot be the thing that moves.
        base = entry.get("gsm_symbolic", {}).get("acc_instances")
        for rung in ("gsm_symbolic_p1", "gsm_symbolic_p2"):
            val = entry.get(rung, {}).get("acc_instances")
            entry.setdefault("structural_effect", {})[rung] = (
                None if None in (base, val) else round(val - base, 4))
        report[alias] = entry
        print(f"[ladder] {alias}: "
              f"head={entry.get('gsm_symbolic', {}).get('acc_template_head')} "
              f"inst={entry.get('gsm_symbolic', {}).get('acc_instances')} "
              f"numeric={entry.get('gsm_symbolic', {}).get('numeric_effect')} "
              f"{entry.get('gsm_symbolic', {}).get('ci_numeric_effect')} | "
              f"structural={entry.get('structural_effect')}", flush=True)

    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "ladder.json").write_text(json.dumps(report, indent=2), "utf-8")
    runs_vol.commit()
    return report


@app.function(cpu=4.0, memory=8192, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def figures(did_source: str = "dose") -> dict:
    """Render every paper figure to both PDF (vector) and PNG.

    ``did_source`` selects which membership design drives the recall figure. It defaults
    to the within-train dose contrast rather than the train-versus-test arms, because that
    is the design without the generalisation-gap confound and therefore the one the claim
    should rest on.
    """
    import sys

    sys.path.insert(0, "/src")
    from instella_reasoning.analysis.atlas_figures import build_all

    out = _run_dir()
    written = build_all(
        out / "analysis" / "atlas.json",
        out / "figures",
        did_path=out / "analysis" / f"did_{did_source}.json",
    )
    runs_vol.commit()
    for name, paths in written.items():
        print(f"[fig] {name}: {[pathlib.Path(p).name for p in paths]}", flush=True)
    return {k: [str(p) for p in v] for k, v in written.items()}


@app.function(cpu=4.0, timeout=60 * 60, volumes=VOLUMES, secrets=SECRETS)
def push(repo: str = "GOVINDFROM/Instella-Reasoning") -> str:
    """Upload the run directory to the dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(_run_dir()),
        path_in_repo=f"experiments/runs/{RUN}",
        ignore_patterns=["**/__pycache__/**"],
    )
    print(f"[push] uploaded {RUN}", flush=True)
    return f"{repo}:{RUN}"


@app.local_entrypoint()
def main(stage: str = "smoke", aliases: str = "", n: int = 200) -> None:
    if stage == "smoke":
        smoke.remote(aliases or "M3-instruct,M3-math")
    elif stage == "validate":
        validate.remote(aliases or "M3-base,M3-instruct,M3-math", n)
    elif stage == "build":
        build.remote(n)
    elif stage == "build_arms":
        build_arms.remote()
    elif stage == "atlas":
        atlas.remote()
    elif stage == "did":
        did.remote()
    elif stage == "ladder":
        ladder.remote()
    elif stage == "build_dose":
        build_dose.remote()
    elif stage == "figures":
        figures.remote()
    elif stage == "push":
        push.remote()
    elif stage == "generate":
        for alias in [a.strip() for a in (aliases or "M3-instruct").split(",") if a.strip()]:
            generate.remote(alias)
    else:
        raise SystemExit(f"unknown stage: {stage}")
