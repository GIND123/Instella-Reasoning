"""Run the full-scale memorisation suite on a Modal A100, driven from a local terminal.

Why this exists
---------------
The suite previously ran inside a hosted notebook session. When that session was reclaimed
the run died with no signal and no way to reattach — the failure mode that cost hours. A
Modal *app* launched with ``--detach`` runs server-side: the local terminal is only a log
viewer, so closing the laptop, losing wifi, or Ctrl+C'ing does not stop the GPU.

Durability has three independent layers, because each one fails differently:

1. **Per-batch fsync** — ``generate`` flushes every batch to the run directory and skips
   already-generated ids on restart, so a kill mid-block resumes mid-block.
2. **Modal Volume** — the run directory lives on a persistent volume, committed every
   ``COMMIT_EVERY`` seconds and on exit. A container that dies keeps its work.
3. **HF push** — the suite pushes after each scored block. This is the off-Modal backup and
   the thing the analysis is ultimately read from.

The local working tree is baked into the image (``copy=True``), so what runs is exactly the
code sitting on this machine — no git remote, no token, no drift between local and remote.

Setup (once)
------------
    modal secret create instella-hf HF_TOKEN=hf_...

Usage
-----
    modal run --detach experiments/modal_fullscale.py             # full suite, resumes
    modal run experiments/modal_fullscale.py::status              # inventory, no GPU
    modal run experiments/modal_fullscale.py::probe               # 2-min GPU health check

    MODAL_GPU=A100-80GB modal run --detach experiments/modal_fullscale.py

Reattach to a detached run's logs at any time:
    modal app logs instella-fullscale
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time

import modal

# --- layout -----------------------------------------------------------------------------
REPO_LOCAL = pathlib.Path(__file__).resolve().parent.parent
REPO_REMOTE = "/root/Instella-Reasoning"
# The suite mirrors its *relative* run path to HF, so the run directory must stay at
# experiments/runs/<name> inside the repo. Mounting the volume there keeps the remote HF
# layout identical to every previous session's.
RUNS_REMOTE = f"{REPO_REMOTE}/experiments/runs"
CACHE_REMOTE = "/cache"

GPU = os.environ.get("MODAL_GPU", "A100-40GB")
TIMEOUT_H = float(os.environ.get("MODAL_HOURS", "12"))
COMMIT_EVERY = 300  # seconds between volume commits
MONITOR_EVERY = 60  # seconds between GPU telemetry lines

runs_vol = modal.Volume.from_name("instella-runs", create_if_missing=True)
cache_vol = modal.Volume.from_name("instella-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        # Exact pin, not a range. Two separate reasons:
        #  1. Instella ships remote modeling code (modeling_instella.py) written against the
        #     transformers 4.4x attention/cache_position API. 5.x removed it and silently
        #     degrades the forward pass into degenerate generations rather than erroring.
        #  2. stage1 of this run was generated under 4.56.0. The DiD compares checkpoints to
        #     each other, so a minor-version drift between stage1 and instruct would enter
        #     the estimate as if it were a model difference. Floating the version is the same
        #     confound as switching inference backends mid-run.
        "transformers==4.56.0",
        "torch>=2.3.0",
        "accelerate>=0.30.0",
        "datasets>=2.19.0",
        "huggingface-hub>=0.23.0",
        "safetensors>=0.4.3",
        "sympy>=1.12",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_REMOTE}/hf",
            "HF_HUB_DISABLE_XET": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            # The source tree is read-only in the image; stop Python trying to write .pyc
            # next to it and emitting a warning per import.
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{REPO_REMOTE}/src",
        }
    )
    # The suite shells out to the `instella-reasoning` console script. Providing it as a
    # two-line shim avoids a runtime `pip install -e .`, which would need a writable source
    # tree and would rebuild metadata on every container start.
    .run_commands(
        "cat > /usr/local/bin/instella-reasoning <<'EOF'\n"
        "#!/usr/bin/env python\n"
        "import sys\n"
        "from instella_reasoning.cli import main\n"
        "sys.exit(main())\n"
        "EOF",
        "chmod +x /usr/local/bin/instella-reasoning",
    )
    .add_local_dir(
        REPO_LOCAL,
        REPO_REMOTE,
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            # Shadowed by the volume at runtime; copying it would bloat the image with
            # every previous run's generations.
            "experiments/runs",
            "experiments/runs/**",
            "models",
            "models/**",
            "**/__pycache__",
            "**/__pycache__/**",
            "**/*.pyc",
            ".venv",
            ".venv/**",
            "**/.DS_Store",
        ],
    )
)

app = modal.App("instella-fullscale", image=image)

SECRETS = [modal.Secret.from_name("instella-hf")]
VOLUMES = {RUNS_REMOTE: runs_vol, CACHE_REMOTE: cache_vol}


# --- helpers ------------------------------------------------------------------------------
def _gpu_line() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return "nvidia-smi unavailable"
        util, used, total, temp, clk = (f.strip() for f in out.stdout.strip().split(","))
        return f"gpu={util}% mem={int(used)/1024:.1f}/{int(total)/1024:.0f}GiB {temp}C sm={clk}MHz"
    except Exception as exc:  # noqa: BLE001 - telemetry must never kill the run
        return f"nvidia-smi error: {exc}"


def _background(fn, *, name: str):
    t = threading.Thread(target=fn, name=name, daemon=True)
    t.start()
    return t


def _start_watchers(stop: threading.Event) -> None:
    """Telemetry + volume commits, both non-fatal by construction."""
    started = time.time()

    def monitor() -> None:
        while not stop.wait(MONITOR_EVERY):
            el = time.time() - started
            print(f"[monitor {el // 3600:.0f}h{(el % 3600) // 60:02.0f}m] {_gpu_line()}", flush=True)

    def committer() -> None:
        while not stop.wait(COMMIT_EVERY):
            try:
                runs_vol.commit()
                print("[volume] committed run directory", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[volume] commit failed ({exc}); work is still on disk", flush=True)

    _background(monitor, name="gpu-monitor")
    _background(committer, name="vol-committer")


# --- the GPU run --------------------------------------------------------------------------
@app.function(
    gpu=GPU,
    timeout=int(TIMEOUT_H * 3600),
    volumes=VOLUMES,
    secrets=SECRETS,
)
def run_suite(overrides: dict | None = None) -> str:
    """Execute run_fullscale_suite.sh. Idempotent: finished blocks are skipped on restart."""
    env = {**os.environ, **{k: str(v) for k, v in (overrides or {}).items()}}
    os.makedirs(f"{CACHE_REMOTE}/hf", exist_ok=True)

    import torch
    import transformers

    print("=" * 74)
    print(f" repo    : {REPO_REMOTE}")
    print(f" gpu     : {GPU}   |  {_gpu_line()}")
    # Printed into the log because the DiD compares checkpoints generated in different
    # sessions; the library versions are part of what makes those comparable.
    print(f" stack   : transformers {transformers.__version__} | torch {torch.__version__}")
    print(f" timeout : {TIMEOUT_H}h")
    print(f" run     : experiments/runs/fullscale-S{env.get('N_PER_ARM', '250')}")
    print(f" hf sync : {env.get('HF_SYNC', '1')} -> {env.get('HF_RESULTS_REPO', 'GOVINDFROM/Instella-Reasoning')}")
    print("=" * 74, flush=True)

    stop = threading.Event()
    _start_watchers(stop)
    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", "experiments/run_fullscale_suite.sh"],
            cwd=REPO_REMOTE,
            env=env,
        )
        rc = proc.returncode
    finally:
        stop.set()
        # Commit before returning regardless of outcome — a crashed suite still produced
        # generations worth keeping, and losing them is the expensive failure.
        try:
            runs_vol.commit()
            print("[volume] final commit done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[volume] final commit failed: {exc}", flush=True)

    elapsed = (time.time() - started) / 3600
    verdict = "OK" if rc == 0 else f"FAILED rc={rc}"
    print(f"\nsuite finished in {elapsed:.2f}h — {verdict}", flush=True)
    return verdict


# --- cheap inventory (no GPU) ---------------------------------------------------------------
@app.function(timeout=900, volumes=VOLUMES, secrets=SECRETS)
def status() -> None:
    """Per-block progress against the exact input row counts. Costs no GPU time."""
    import glob
    import json

    os.chdir(REPO_REMOTE)
    runs = sorted(glob.glob("experiments/runs/fullscale-S*"))
    if not runs:
        print("no run directory yet — the first `run_suite` will create it.")
        return
    out = pathlib.Path(runs[-1])
    print(f"run dir: {out}\n")

    def rows_and_ids(path: pathlib.Path) -> tuple[int, int]:
        n, ids = 0, set()
        if not path.exists():
            return 0, 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail from a killed writer
                n += 1
                for key in ("benchmark_id", "id", "item_id"):
                    if key in row:
                        ids.add(row[key])
                        break
        return n, len(ids)

    inputs = {
        "arms": out / "variants/arms_variants.jsonl",
        "gsmsym": out / "variants/gsm_symbolic_official.jsonl",
        "resample": out / "variants/resample_control.jsonl",
    }
    expected = {b: rows_and_ids(p)[0] for b, p in inputs.items()}
    for block, n in expected.items():
        print(f"input  {block:9s} {n:5d} rows")

    markers = sorted(p.name for p in (out / "markers").glob("*"))
    print(f"\nmarkers: {', '.join(markers) or '(none)'}\n")

    # The suite runs the decoding-noise control only for RESAMPLE_MODELS; counting the
    # skipped checkpoints as outstanding work overstates the remaining GPU time ~5x.
    resample_models = set(os.environ.get("RESAMPLE_MODELS", "instruct math").split())

    print(f"{'block':24s} {'done':>6s} {'/exp':>6s} {'%':>7s}  {'scored':>7s}  state")
    remaining = 0
    for tag in ("stage1", "stage2", "instruct"):
        for block, exp in expected.items():
            if exp == 0:
                continue
            if block == "resample" and tag not in resample_models:
                print(f"{tag + '/' + block:24s} {'-':>6s} {exp:6d} {'-':>7s}  {'-':>7s}  n/a (not in RESAMPLE_MODELS)")
                continue
            _, n_gen = rows_and_ids(out / f"generations/{tag}__{block}.jsonl")
            n_sco, _ = rows_and_ids(out / f"scores/{tag}__{block}.jsonl")
            if n_sco:
                state = "COMPLETE"
            elif n_gen >= exp:
                state = "needs scoring"
            else:
                state = "partial" if n_gen else "not started"
                remaining += exp - n_gen
            print(
                f"{tag + '/' + block:24s} {n_gen:6d} {exp:6d} "
                f"{100.0 * n_gen / exp:6.1f}%  {n_sco:7d}  {state}"
            )

    print(f"\nremaining generations: {remaining:,} rows")
    print(f"  @0.56 it/s -> {remaining / 0.56 / 3600:.1f}h    @1.7 it/s -> {remaining / 1.7 / 3600:.1f}h")


# --- GPU health probe -------------------------------------------------------------------------
@app.function(gpu=GPU, timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def probe(n: int = 16, batch: int = 8) -> float:
    """Measure real decode throughput before committing hours to a suspect GPU.

    Returns items/sec. The stage2/arms block on healthy hardware ran at ~0.56 it/s; a
    figure far below that means the container drew degraded hardware and should be killed
    rather than debugged.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"probe: {_gpu_line()}")
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    model_id = "amd/Instella-3B"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    prompt = (
        "Natalia sold clips to 48 friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether?\n"
        "Think step by step, then give the final answer.\n"
    )
    prompts = [prompt] * n

    t0 = time.time()
    for i in range(0, n, batch):
        enc = tok(prompts[i : i + batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=256, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        print(f"  batch {i // batch + 1}: {time.time() - t0:.1f}s  {_gpu_line()}", flush=True)
    rate = n / (time.time() - t0)

    print(f"\nthroughput: {rate:.2f} items/sec at batch={batch}, 256 new tokens")
    print("  reference: stage2/arms sustained ~0.56 it/s at 1024 tokens on healthy hardware.")
    print("  VERDICT:", "healthy" if rate >= 0.5 else "DEGRADED — kill and relaunch")
    return rate


@app.function(timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def extraction_audit(write: bool = False) -> dict:
    """Which extraction tier produced each answer, per checkpoint.

    ``write`` is off by default: this is safe to run *while the suite is generating*, and a
    second container committing the shared Volume mid-run can clobber the suite's writes.
    Pass ``write=True`` only once the suite has finished.

    `extract_numeric` falls back in three tiers: the `#### N` marker, then an
    "answer is ..." phrase, then the last number anywhere in the completion. Tier 3 is a
    guess -- "...so 72 clips, about 6 dozen" yields 6.

    stage1/stage2 emitted the marker on ~98% of completions; the DPO-tuned instruct
    checkpoint emits it on 34%. So the checkpoints are not being measured the same way, and
    an accuracy difference across the trajectory could be an artifact of which tier fired.
    The headline DiD is a within-checkpoint double difference and largely cancels this, but
    the raw trajectory does not. This quantifies the exposure instead of assuming it away.
    """
    import glob
    import json
    import re
    from collections import defaultdict

    os.chdir(REPO_REMOTE)
    runs = sorted(glob.glob("experiments/runs/fullscale-S*"))
    if not runs:
        print("no run directory yet")
        return {}
    out = pathlib.Path(runs[-1])

    from instella_reasoning.prompting import _ANSWER_IS, _HASH_ANSWER  # noqa: PLC2701

    def tier_of(completion: str) -> str:
        if _HASH_ANSWER.findall(completion):
            return "1_marker"
        if _ANSWER_IS.search(completion):
            return "2_phrase"
        if re.search(r"-?\d", completion):
            return "3_last_number"
        return "4_no_number"

    report: dict = {}
    print(f"{'checkpoint/block':26s} {'n':>6s}  {'tier1':>7s} {'tier2':>7s} {'tier3':>7s} {'none':>6s}   acc@tier3")
    for gen_path in sorted((out / "generations").glob("*__*.jsonl")):
        stem = gen_path.stem
        # Correctness lives in the score file, keyed by the same benchmark_id.
        correct: dict[str, bool] = {}
        score_path = out / "scores" / f"{stem}.jsonl"
        if score_path.exists():
            with score_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "benchmark_id" in row and "correct" in row:
                        correct[row["benchmark_id"]] = bool(row["correct"])

        tiers: dict[str, int] = defaultdict(int)
        t3_hits, t3_total = 0, 0
        with gen_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completion = row.get("completion") or ""
                tier = tier_of(completion)
                tiers[tier] += 1
                if tier == "3_last_number" and row.get("benchmark_id") in correct:
                    t3_total += 1
                    t3_hits += correct[row["benchmark_id"]]
        n = sum(tiers.values()) or 1
        acc3 = f"{t3_hits / t3_total:.3f}" if t3_total else "n/a"
        print(
            f"{stem:26s} {n:6d}  "
            f"{tiers['1_marker'] / n:6.1%} {tiers['2_phrase'] / n:6.1%} "
            f"{tiers['3_last_number'] / n:6.1%} {tiers['4_no_number'] / n:5.1%}   {acc3}"
        )
        report[stem] = {"n": n, "tiers": dict(tiers), "tier3_accuracy": acc3}

    # --- the part that actually threatens the DiD ------------------------------------
    # The double difference cancels extraction error only if the tier mix is the SAME in
    # all four cells. If a perturbed item makes the model drop its `####` marker more often
    # than the original does, the extractor becomes correlated with the treatment and the
    # cancellation argument fails. Cross tier against (arm x condition) to find out.
    import dataclasses

    from instella_reasoning.metrics import is_answer_changing
    from instella_reasoning.records import EvaluationRecord

    # EvaluationRecord is a plain dataclass (no from_dict); build it from known fields so a
    # schema addition in the score writer cannot break this audit.
    _EVAL_FIELDS = {f.name for f in dataclasses.fields(EvaluationRecord)}

    def to_eval(row: dict) -> EvaluationRecord:
        return EvaluationRecord(**{k: v for k, v in row.items() if k in _EVAL_FIELDS})

    print("\n" + "=" * 78)
    print("TIER MIX BY DiD CELL  — these four rows must look alike for the DiD to cancel")
    print("=" * 78)
    _order = {"stage1": 0, "stage2": 1, "sft": 2, "instruct": 3}
    for tag in sorted(
        {p.name.replace("__arms.jsonl", "") for p in (out / "scores").glob("*__arms.jsonl")},
        key=lambda t: (_order.get(t, 99), t),
    ):
        score_path = out / "scores" / f"{tag}__arms.jsonl"
        gen_path = out / "generations" / f"{tag}__arms.jsonl"
        if not (score_path.exists() and gen_path.exists()):
            continue
        completions: dict[str, str] = {}
        with gen_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completions[str(row.get("benchmark_id"))] = row.get("completion") or ""

        cells: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        acc: dict[tuple[str, str], list[int]] = defaultdict(list)
        with score_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = to_eval(json.loads(line))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                arm = rec.metadata.get("arm")
                if arm not in ("seen", "unseen"):
                    continue
                cond = "perturbed" if is_answer_changing(rec) else "original"
                key = (str(arm), cond)
                cells[key][tier_of(completions.get(rec.benchmark_id, ""))] += 1
                acc[key].append(int(bool(rec.correct)))

        if not cells:
            continue
        print(f"\n{tag}:")
        print(f"  {'cell':20s} {'n':>5s}  {'tier1':>7s} {'tier2':>7s} {'tier3':>7s}   {'acc':>6s}")
        for arm in ("seen", "unseen"):
            for cond in ("original", "perturbed"):
                key = (arm, cond)
                t = cells.get(key)
                if not t:
                    continue
                n = sum(t.values()) or 1
                a = sum(acc[key]) / len(acc[key]) if acc[key] else 0.0
                print(
                    f"  {arm + '/' + cond:20s} {n:5d}  {t['1_marker'] / n:6.1%} "
                    f"{t['2_phrase'] / n:6.1%} {t['3_last_number'] / n:6.1%}   {a:6.3f}"
                )
        # The bias that survives into the DiD is the SECOND difference of the tier-1 rate,
        # not its spread. A tier gap between seen and unseen that is identical under both
        # conditions subtracts out; only a gap that *changes* with the condition leaks in.
        def rate(arm: str, cond: str) -> float:
            t = cells.get((arm, cond))
            n = sum(t.values()) if t else 0
            return (t["1_marker"] / n) if n else 0.0

        def acc_of(arm: str, cond: str) -> float:
            v = acc.get((arm, cond)) or []
            return sum(v) / len(v) if v else 0.0

        tier_did = (rate("seen", "original") - rate("unseen", "original")) - (
            rate("seen", "perturbed") - rate("unseen", "perturbed")
        )
        acc_did = (acc_of("seen", "original") - acc_of("unseen", "original")) - (
            acc_of("seen", "perturbed") - acc_of("unseen", "perturbed")
        )
        report[f"{tag}__tier1_did"] = tier_did
        report[f"{tag}__accuracy_did"] = acc_did
        print(f"  -> accuracy DiD : {acc_did:+.4f}")
        print(f"  -> tier-1 DiD   : {tier_did:+.4f}  (extraction bias leaking into the DiD)")
        if abs(tier_did) <= 0.05:
            print("  -> OK: extraction artifact cancels in the double difference")
        else:
            print("  -> WARNING: extraction is confounded with the treatment; re-score with "
                  "a marker-independent extractor before believing this DiD")

    if write:
        dest = out / "analysis/extraction_tiers.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs_vol.commit()
        print(f"\nwrote {dest}")
    else:
        print("\n(read-only: pass write=True after the suite finishes to persist)")
    print("Read: a large tier-1 gap between checkpoints means the trajectory comparison is")
    print("partly an extraction artifact. Tier-3 accuracy far below tier-1 accuracy means")
    print("the fallback is guessing wrong, not just guessing.")
    return report


# Set in the parent before the worker Pool forks, so children inherit it copy-on-write
# instead of pickling ~200k gram entries per worker.
_CONTAINMENT_STATE: dict = {}


def _scan_shard(task: tuple) -> tuple[dict, dict]:
    """Scan one byte range of the corpus; return matched gram hashes per namespaced item."""
    import json as _json
    from collections import defaultdict as _dd

    from instella_reasoning.datasets.splits import (
        NGRAM_N,
        _gram_hashes,
        _token_ids,
        normalize_tokens,
    )

    path, start, end = task
    gram_owners = _CONTAINMENT_STATE["gram_owners"]
    # A *copy* of the item vocab: unknown corpus tokens get ids above the item range, so
    # they can never collide with an item token id and manufacture a false match.
    vocab = dict(_CONTAINMENT_STATE["vocab"])

    matched: dict[str, set] = _dd(set)
    example: dict[str, str] = {}
    with open(path, "rb") as fh:
        fh.seek(start)
        if start:
            fh.readline()  # the partial line at the boundary belongs to the previous shard
        while fh.tell() < end:
            raw = fh.readline()
            if not raw:
                break
            try:
                row = _json.loads(raw)
            except _json.JSONDecodeError:
                continue
            ids = _token_ids(normalize_tokens(row.get("text") or ""), vocab)
            for g in _gram_hashes(ids, NGRAM_N):
                owners = gram_owners.get(g)
                if not owners:
                    continue
                for owner in owners:
                    matched[owner].add(g)
                    example.setdefault(owner, row.get("source") or row.get("id"))
    return dict(matched), example


@app.function(cpu=8.0, memory=16384, timeout=3600, volumes=VOLUMES, secrets=SECRETS)
def containment_repair(write: bool = False, push: bool = False) -> dict:
    """Re-measure 13-gram containment with split-namespaced ids, and audit the built arms.

    `verify_containment` keys `item_grams` by `item.id`, but GSM8K train and test both use
    the id scheme `gsm8k_NNNNN`. Loading 1000 of each makes every id collide pairwise: the
    test twin overwrites the train twin's gram set while `gram_owners` retains both, so the
    numerator accumulates hits against grams that are not in the denominator and containment
    can exceed 1.0 -- which is meaningless for a "fraction of the item's n-grams found in the
    corpus" statistic, and indefensible in F4.

    This recomputes with keys `train::<id>` / `test::<id>` so the two splits never alias, then
    reports whether the arms that were actually generated still carry the right verdicts. The
    arms are the thing the DiD rests on; if they survive, the corrupted file was a reporting
    bug and no GPU needs to be re-spent.
    """
    import json
    import multiprocessing as mp
    from collections import defaultdict

    from instella_reasoning.datasets.splits import (
        NGRAM_N,
        SEEN_THRESHOLD,
        UNSEEN_THRESHOLD,
        _gram_hashes,
        _token_ids,
        normalize_tokens,
    )
    from instella_reasoning.records import read_benchmark

    os.chdir(REPO_REMOTE)
    out = pathlib.Path(sorted(__import__("glob").glob("experiments/runs/fullscale-S*"))[-1])
    corpus = out / "contamination/synthetic_corpus.jsonl"
    if not corpus.exists():
        raise SystemExit(f"corpus missing at {corpus}")

    train = read_benchmark(out / "base/gsm8k_train.jsonl")
    test = read_benchmark(out / "base/gsm8k_test.jsonl")
    overlap = {i.id for i in train} & {i.id for i in test}
    print(f"train={len(train)} test={len(test)} colliding ids={len(overlap)}")
    print(f"corpus={corpus.stat().st_size / 1e9:.2f} GB\n")

    vocab: dict[str, int] = {}
    gram_owners: dict[int, set[str]] = defaultdict(set)
    item_grams: dict[str, set[int]] = {}
    for split, items in (("train", train), ("test", test)):
        for item in items:
            key = f"{split}::{item.id}"
            grams = set(_gram_hashes(_token_ids(normalize_tokens(item.prompt), vocab), NGRAM_N))
            item_grams[key] = grams
            for g in grams:
                gram_owners[g].add(key)
    print(f"indexed {len(item_grams)} items, {len(gram_owners):,} distinct grams, "
          f"vocab={len(vocab):,}")

    _CONTAINMENT_STATE["gram_owners"] = dict(gram_owners)
    _CONTAINMENT_STATE["vocab"] = vocab

    n_shards = 8
    size = corpus.stat().st_size
    step = size // n_shards
    tasks = [(str(corpus), i * step, size if i == n_shards - 1 else (i + 1) * step)
             for i in range(n_shards)]
    print(f"scanning corpus across {n_shards} shards...", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(n_shards) as pool:
        results = pool.map(_scan_shard, tasks)
    matched: dict[str, set] = defaultdict(set)
    example: dict[str, str] = {}
    for part, ex in results:
        for k, v in part.items():
            matched[k] |= v
        for k, v in ex.items():
            example.setdefault(k, v)
    print(f"corpus scanned in {time.time() - t0:.1f}s\n")

    def verdict_of(key: str) -> tuple[float, str]:
        total = len(item_grams[key])
        if total == 0:
            return 0.0, "too_short"
        c = len(matched.get(key, ())) / total
        if c >= SEEN_THRESHOLD:
            return c, "seen"
        if c <= UNSEEN_THRESHOLD:
            return c, "unseen"
        return c, "ambiguous"

    fixed = {k: verdict_of(k) for k in item_grams}
    over_one = sum(1 for c, _ in fixed.values() if c > 1.0)
    print(f"containment > 1.0 after fix: {over_one}  (must be 0)")
    for split in ("train", "test"):
        counts: dict[str, int] = defaultdict(int)
        for k, (_, v) in fixed.items():
            if k.startswith(split + "::"):
                counts[v] += 1
        print(f"  {split:5s}: " + "  ".join(f"{v}={counts[v]}" for v in sorted(counts)))

    # The decisive question: do the arms that were actually generated still hold up?
    arms = read_benchmark(out / "arms/gsm8k_arms.jsonl")
    agree, disagree = 0, []
    for item in arms:
        arm = item.metadata.get("arm")
        split = item.metadata.get("gsm8k_split") or ("train" if arm == "seen" else "test")
        c, v = fixed.get(f"{split}::{item.id}", (float("nan"), "missing"))
        if v == arm:
            agree += 1
        else:
            disagree.append((item.id, arm, v, c))
    print(f"\narms audit: {len(arms)} items — {agree} keep their verdict, "
          f"{len(disagree)} disagree")
    for iid, was, now, c in disagree[:10]:
        print(f"   {iid}: built as {was}, corrected verdict {now} (containment {c:.3f})")

    pool_seen = sum(1 for k, (_, v) in fixed.items() if k.startswith("train::") and v == "seen")
    pool_unseen = sum(1 for k, (_, v) in fixed.items() if k.startswith("test::") and v == "unseen")
    print(f"\neligible pools under corrected measurement: seen={pool_seen} unseen={pool_unseen}")
    print(f"  -> supports up to {min(pool_seen, pool_unseen)} per arm "
          f"(currently built: {sum(1 for i in arms if i.metadata.get('arm') == 'seen')})")

    report = {
        "colliding_ids": len(overlap),
        "containment_over_one_after_fix": over_one,
        "arms_total": len(arms),
        "arms_verdict_agree": agree,
        "arms_verdict_disagree": len(disagree),
        "disagreements": [
            {"id": i, "built_as": w, "corrected": n, "containment": round(c, 6)}
            for i, w, n, c in disagree
        ],
        "pool_seen": pool_seen,
        "pool_unseen": pool_unseen,
        "containment": {k: {"containment": round(c, 6), "verdict": v} for k, (c, v) in fixed.items()},
    }
    if write:
        dest = out / "analysis/containment_verified.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {dest}")
        runs_vol.commit()
        if push:
            subprocess.call(
                [sys.executable, "experiments/hf_sync.py", "push", "--repo",
                 os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
                 "--path", str(out), "--message", "containment re-measured with namespaced ids"]
            )
    else:
        print("\n(read-only: pass write=True to persist, push=True to also sync HF)")
    return {k: v for k, v in report.items() if k != "containment"}


@app.function(cpu=4.0, timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def purified_did(write: bool = False, push: bool = False) -> dict:
    """Re-estimate the DiD with the seen arm restricted to *verifiably* seen items.

    The built seen arm carries 83/194 items whose corrected containment lands in the
    ambiguous band (0.39-0.79), below the 0.80 threshold the design requires. Ambiguous
    items are, by construction, ones the model may not have memorised -- so leaving them in
    the seen arm dilutes the treatment and pulls any real memorisation effect toward zero.
    A null DiD measured on a diluted arm is not evidence of no memorisation.

    This costs no GPU: every one of these items was already generated. It re-runs the same
    `difference_in_differences` (same cluster bootstrap over parent items) on the subset
    whose treatment label survives correct measurement, so the two estimates are directly
    comparable.
    """
    import dataclasses
    import glob
    import json

    from instella_reasoning.analysis.memorization import difference_in_differences
    from instella_reasoning.records import EvaluationRecord

    os.chdir(REPO_REMOTE)
    out = pathlib.Path(sorted(glob.glob("experiments/runs/fullscale-S*"))[-1])
    verified_path = out / "analysis/containment_verified.json"
    if not verified_path.exists():
        raise SystemExit("run containment_repair --write first")
    verified = json.loads(verified_path.read_text())["containment"]

    fields = {f.name for f in dataclasses.fields(EvaluationRecord)}

    def load(tag: str) -> list[EvaluationRecord]:
        path = out / f"scores/{tag}__arms.jsonl"
        recs = []
        if not path.exists():
            return recs
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(EvaluationRecord(**{k: v for k, v in json.loads(line).items() if k in fields}))
                except (json.JSONDecodeError, TypeError):
                    continue
        return recs

    def is_verified(rec: EvaluationRecord) -> bool:
        """True when the item's corrected verdict still matches the arm it was placed in."""
        arm = rec.metadata.get("arm")
        split = rec.metadata.get("gsm8k_split") or ("train" if arm == "seen" else "test")
        # Variants inherit their parent's treatment status -- the perturbation changes the
        # numbers, not whether the underlying problem was in the training corpus.
        entry = verified.get(f"{split}::{rec.parent_id}") or verified.get(f"{split}::{rec.benchmark_id}")
        return bool(entry) and entry["verdict"] == arm

    report: dict = {}
    # Discovered from disk, not hardcoded: a checkpoint added outside the suite's tier loop
    # (e.g. sft) must not be silently dropped from the trajectory it was generated for.
    order = {"stage1": 0, "stage2": 1, "sft": 2, "instruct": 3, "math_sft": 4, "math": 5}
    tags = sorted(
        {p.name.replace("__arms.jsonl", "") for p in (out / "scores").glob("*__arms.jsonl")},
        key=lambda t: (order.get(t, 99), t),
    )
    print(f"checkpoints: {', '.join(tags)}\n")
    print(f"{'checkpoint':10s} {'arm set':10s} {'seen n':>7s} {'unseen n':>9s} "
          f"{'DiD':>9s} {'CI95':>22s}  verdict")
    for tag in tags:
        recs = load(tag)
        if not recs:
            continue
        for label, subset in (("full", recs), ("purified", [r for r in recs if is_verified(r)])):
            if not subset:
                continue
            res = difference_in_differences(subset)
            d = res.to_dict()
            ci = d["cluster_bootstrap_ci95"]
            sig = "EXCLUDES ZERO" if d["excludes_zero"] else "spans zero"
            print(
                f"{tag:10s} {label:10s} {d['n_clusters']['seen']:7d} "
                f"{d['n_clusters']['unseen']:9d} {d['difference_in_differences']:+9.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {sig}"
            )
            report[f"{tag}__{label}"] = d
        print()

    if write:
        dest = out / "analysis/memorization_purified.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs_vol.commit()
        print(f"wrote {dest}")
        if push:
            subprocess.call(
                [sys.executable, "experiments/hf_sync.py", "push", "--repo",
                 os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
                 "--path", str(out), "--message", "DiD on verified-only seen arm"]
            )
    else:
        print("(read-only: pass write=True to persist)")
    return {k: {"did": v["difference_in_differences"], "ci": v["cluster_bootstrap_ci95"]}
            for k, v in report.items()}


@app.function(gpu=GPU, timeout=4 * 3600, volumes=VOLUMES, secrets=SECRETS)
def run_extra(tag: str, block: str = "arms", temperature: float = 0.0,
              run: str = "fullscale-S250-v2") -> dict:
    """Generate + score one (checkpoint, block) outside the suite's tier loop.

    The suite runs whatever ``tier(SUITE_TIER)`` returns, so reaching a single Tier-2
    checkpoint would drag in `math_sft` and `math` too -- and `math` is deliberately excluded
    for failing its completion gate. This runs exactly one block, writes the merged
    ``scores/<tag>__ALL.jsonl`` the analysis stage globs for, and pushes.

    Generation settings come from the checkpoint registry, not from arguments, so an added
    checkpoint is measured on the same terms as the existing three. Anything else would make
    a trajectory difference indistinguishable from a configuration difference.
    """
    import json

    from instella_reasoning.checkpoints import resolve
    from instella_reasoning.evaluation import (
        generate_with_transformers,
        score_generations,
        termination_report,
    )
    from instella_reasoning.records import GenerationRecord, read_benchmark, read_jsonl

    os.chdir(REPO_REMOTE)
    out = pathlib.Path("experiments/runs") / run
    bench = {
        "arms": out / "variants/arms_variants.jsonl",
        "gsmsym": out / "variants/gsm_symbolic_official.jsonl",
        "resample": out / "variants/resample_control.jsonl",
    }[block]
    gen_path = out / f"generations/{tag}__{block}.jsonl"
    score_path = out / f"scores/{tag}__{block}.jsonl"
    ckpt = resolve(tag)

    items = read_benchmark(bench)
    print("=" * 70)
    print(f" {tag}/{block}  model={ckpt.load_path}")
    print(f" items={len(items)} tokens={ckpt.max_new_tokens} shot={ckpt.n_shot} T={temperature}")
    print(f" {_gpu_line()}")
    print("=" * 70, flush=True)

    stop = threading.Event()
    _start_watchers(stop)
    started = time.time()
    try:
        generate_with_transformers(
            benchmark=items,
            model_name_or_path=ckpt.load_path,
            output_path=gen_path,
            max_new_tokens=ckpt.max_new_tokens,
            temperature=temperature,
            batch_size=int(os.environ.get("BATCH", "8")),
            use_chat_template=not ckpt.is_base,
            n_shot=ckpt.n_shot,
        )
    finally:
        stop.set()
        runs_vol.commit()

    gens = [GenerationRecord.from_dict(r) for r in read_jsonl(gen_path)]
    rep = termination_report(gens, threshold=0.85)
    print(f"\ntermination {rep.termination_rate:.1%} · marker {rep.marker_rate:.1%} · "
          f"median {rep.median_chars} chars · passes={rep.passes}")
    score_generations(items, gens, score_path)

    merged = out / f"scores/{tag}__ALL.jsonl"
    parts = []
    for b in ("arms", "gsmsym", "resample"):
        p = out / f"scores/{tag}__{b}.jsonl"
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").rstrip("\n"))
    merged.write_text("\n".join(x for x in parts if x) + "\n", encoding="utf-8")
    runs_vol.commit()

    subprocess.call(
        [sys.executable, "experiments/hf_sync.py", "push", "--repo",
         os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
         "--path", str(out), "--message", f"{tag}/{block}"]
    )
    print(f"\n{tag}/{block} done in {(time.time() - started) / 3600:.2f}h")
    return {"tag": tag, "block": block, "n": len(gens),
            "termination_rate": rep.termination_rate, "passes": rep.passes}


@app.function(cpu=4.0, timeout=1800)
def run_tests(path: str = "tests/") -> int:
    """Run the repo test suite in the container.

    The local interpreter on the operator's machine is 3.9 and the package requires 3.10+
    (`dataclass(slots=True)`), so shared-code changes cannot be verified locally. The image
    is 3.11, which makes this the only place the tests actually execute.
    """
    os.chdir(REPO_REMOTE)
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    rc = subprocess.call([sys.executable, "-m", "pytest", path, "-q"])
    print(f"\npytest exit={rc}")
    return rc


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes=VOLUMES, secrets=SECRETS)
def finalize(run: str = "fullscale-S250-v2", n_items: int = 250) -> None:
    """Re-run the audit, DiD, atlas and figures on CPU once all generation is done.

    Stages 6-8 of the suite are pure analysis. Running them through `run_suite` would hold an
    A100 idle for several minutes of matplotlib, so they get their own CPU container.
    """
    import glob

    os.chdir(REPO_REMOTE)
    out = pathlib.Path("experiments/runs") / run

    # Another container wrote the newest blocks; without this the mount can still show the
    # state it had at container start, and the analysis would quietly run on stale inputs.
    runs_vol.reload()

    def sh(*cmd: str, allow_fail: bool = False) -> None:
        print(f"\n$ {' '.join(cmd[:4])} ...", flush=True)
        rc = subprocess.call(list(cmd))
        # A silent non-zero once left a stale termination.json in place while the rest of the
        # analysis refreshed, so the audit described a different run than the numbers did.
        # `check-termination` is the one command whose non-zero is a *finding* rather than an
        # error -- it reports a block below the gate, and the suite likewise lets it through.
        if rc != 0 and not allow_fail:
            raise SystemExit(f"FAILED (rc={rc}): {' '.join(cmd[:3])}")
        if rc != 0:
            print(f"  (rc={rc}: a block is below the termination gate — see termination.json)")

    gens = sorted(glob.glob(f"{out}/generations/*.jsonl"))
    print(f"auditing {len(gens)} generation files")
    sh("instella-reasoning", "check-termination", "--generations", *gens,
       "--output", f"{out}/analysis/termination.json", "--min-rate", "0.85", allow_fail=True)
    # The DiD is defined on the arms block alone. `make_resample_suite` emits, per item, the
    # original *plus* N copies -- and that original carries the same benchmark_id as the arms
    # block's original while being generated at T>0. Feeding the merged __ALL file in gives
    # those items two rows apiece and mixes sampled decoding into a greedy contrast, for only
    # the checkpoints that ran the control.
    sh("instella-reasoning", "memorization", "--scores", *glob.glob(f"{out}/scores/*__arms.jsonl"),
       "--output", f"{out}/analysis/memorization.json")

    contam = out / "contamination/gsm8k_contam.jsonl"
    contam.parent.mkdir(parents=True, exist_ok=True)
    if not contam.exists():
        contam.write_text("", encoding="utf-8")
    for merged in sorted(glob.glob(f"{out}/scores/*__ALL.jsonl")):
        tag = pathlib.Path(merged).name.replace("__ALL.jsonl", "")
        sh("instella-reasoning", "atlas", "--scores", merged, "--contamination", str(contam),
           "--output", f"{out}/atlas/{tag}_atlas.json", "--markdown", f"{out}/atlas/{tag}_atlas.md")

    sh("instella-reasoning", "figures", "--scores", *glob.glob(f"{out}/scores/*__ALL.jsonl"),
       "--analysis", f"{out}/analysis/memorization.json", "--output-dir", f"{out}/figures",
       "--containment", f"{out}/analysis/containment.json",
       "--termination", f"{out}/analysis/termination.json", "--n-items", str(n_items))

    runs_vol.commit()
    subprocess.call(
        [sys.executable, "experiments/hf_sync.py", "push", "--repo",
         os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
         "--path", str(out), "--message", "final: analysis + atlas + figures"]
    )
    print("\nfinalize complete")


@app.function(cpu=2.0, timeout=900, volumes=VOLUMES, secrets=SECRETS)
def detector_falsepositive(run: str = "fullscale-S250", write: bool = False, push: bool = False) -> dict:
    """What a standard n-gram contamination heuristic would claim, against known ground truth.

    The corpus is `amd/Instella-GSM8K-synthetic`, derived from GSM8K **train**. GSM8K **test**
    is therefore absent from it by construction -- every test item is a known negative. That
    makes this a rare setting where a contamination detector's false-positive rate can be
    measured rather than argued about.

    The widely used rule (Brown et al., 2020 and descendants) flags an item when *any* n-gram
    of it appears in the corpus. This reports what that rule, and a sweep of stricter
    thresholds, would claim about items that cannot be contaminated. Overlap on a known
    negative is not memorisation -- GSM8K train and test share annotators and templates, so
    they share phrasing ("how many ... does ... have in total"). That conflation of template
    similarity with membership is the construct-validity failure this study is about.
    """
    import json
    from statistics import median

    os.chdir(REPO_REMOTE)
    out = pathlib.Path("experiments/runs") / run
    verified = json.loads((out / "analysis/containment_verified.json").read_text())["containment"]

    train = sorted(v["containment"] for k, v in verified.items() if k.startswith("train::"))
    test = sorted(v["containment"] for k, v in verified.items() if k.startswith("test::"))

    def pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0

    print(f"containment distribution   (n_train={len(train)}, n_test={len(test)})")
    print(f"{'':10s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s}")
    for name, xs in (("train", train), ("test", test)):
        print(f"{name:10s} {median(xs):8.3f} {pct(xs, 0.90):8.3f} {pct(xs, 0.99):8.3f} {max(xs):8.3f}")

    print(f"\n{'rule':28s} {'train flagged':>14s} {'test flagged':>13s} {'FPR':>8s}")
    rules = [("any n-gram match (Brown et al.)", 1e-9), ("containment >= 0.05", 0.05),
             ("containment >= 0.10", 0.10), ("containment >= 0.25", 0.25),
             ("containment >= 0.50", 0.50), ("containment >= 0.80 (this study)", 0.80)]
    report: dict = {"n_train": len(train), "n_test": len(test), "rules": {}}
    for label, thr in rules:
        n_tr = sum(1 for c in train if c >= thr)
        n_te = sum(1 for c in test if c >= thr)
        fpr = n_te / len(test) if test else 0.0
        print(f"{label:28s} {n_tr:6d} ({n_tr / len(train):5.1%}) {n_te:5d} ({fpr:5.1%}) {fpr:8.1%}")
        report["rules"][label] = {"threshold": thr, "train_flagged": n_tr,
                                  "test_flagged": n_te, "false_positive_rate": fpr}

    worst = sorted(((v["containment"], k) for k, v in verified.items() if k.startswith("test::")),
                   reverse=True)[:5]
    print("\nhighest-containment known negatives (test items, provably absent from the corpus):")
    for c, k in worst:
        print(f"   {k}: {c:.3f}")
    print("\nThese are template/phrasing overlap between GSM8K train and test, not membership.")
    print("A rule that flags them reports contamination where none can exist.")

    if write:
        dest = out / "analysis/detector_falsepositive.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs_vol.commit()
        print(f"\nwrote {dest}")
        if push:
            subprocess.call(
                [sys.executable, "experiments/hf_sync.py", "push", "--repo",
                 os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
                 "--path", str(out), "--message", "detector false-positive rate on known negatives"]
            )
    else:
        print("\n(read-only: pass write=True to persist)")
    return report


@app.function(gpu=GPU, timeout=4 * 3600, volumes=VOLUMES, secrets=SECRETS)
def repair_truncated(tag: str = "stage1", run: str = "fullscale-S250-v2",
                     max_new_tokens: int = 2048, write: bool = False) -> dict:
    """Regenerate only the rows that hit the token cap, at a larger budget.

    This is v1's documented procedure (261 stage-1 rows regenerated at 2048), reapplied so
    v2's block is not a mix of repaired 2048-token rows and unrepaired 1024-token ones. The
    85% gate exists to stop truncated text being scored as a weak model; the fix is to give
    the truncated items enough budget to finish, not to move the gate.

    Safe by construction: a row is only dropped if it never reached a semantic stop, so the
    procedure can only convert unscoreable text into a measurement. It is applied without
    reference to arm, and `termination_by_cell` separately confirms truncation is balanced
    across the DiD cells, so this cannot tilt the contrast.
    """
    import json

    from instella_reasoning.checkpoints import resolve
    from instella_reasoning.evaluation import generate_with_transformers, termination_report
    from instella_reasoning.records import GenerationRecord, read_benchmark

    os.chdir(REPO_REMOTE)
    out = pathlib.Path("experiments/runs") / run
    gen_path = out / f"generations/{tag}__arms.jsonl"
    bench_path = out / "variants/arms_variants.jsonl"

    kept, dropped = [], []
    with gen_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            (kept if (row.get("metadata") or {}).get("finished", True) else dropped).append(line)
    print(f"{tag}: {len(kept)} finished, {len(dropped)} truncated -> regenerate at {max_new_tokens}")
    if not write:
        print("(dry run: pass write=True to regenerate)")
        return {"finished": len(kept), "truncated": len(dropped)}
    if not dropped:
        return {"finished": len(kept), "truncated": 0}

    ckpt = resolve(tag)
    gen_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    runs_vol.commit()

    stop = threading.Event()
    _start_watchers(stop)
    try:
        # Resume skips every row still present, so only the dropped ids are regenerated.
        generate_with_transformers(
            benchmark=read_benchmark(bench_path),
            model_name_or_path=ckpt.load_path,
            output_path=gen_path,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            batch_size=int(os.environ.get("BATCH", "8")),
            use_chat_template=not ckpt.is_base,
            n_shot=ckpt.n_shot,
        )
    finally:
        stop.set()
        runs_vol.commit()

    rows = [GenerationRecord.from_dict(json.loads(line))
            for line in gen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rep = termination_report(rows, threshold=0.85)
    print(f"\ntermination after repair: {rep.termination_rate:.1%} "
          f"(marker {rep.marker_rate:.1%}, median {rep.median_chars} chars) "
          f"passes={rep.passes}")
    subprocess.call(
        [sys.executable, "experiments/hf_sync.py", "push", "--repo",
         os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
         "--path", str(out), "--message", f"{tag} truncated-row repair at {max_new_tokens}"]
    )
    return {"termination_rate": rep.termination_rate, "passes": rep.passes, "n": rep.n}


@app.function(cpu=2.0, timeout=900, volumes=VOLUMES, secrets=SECRETS)
def termination_by_cell(tag: str = "stage1", run: str = "fullscale-S250-v2") -> dict:
    """Termination rate split by DiD cell, and by whether the row is reused or newly generated.

    The 85% gate is a whole-block statistic, so it cannot distinguish "this model rambles"
    from "truncation is concentrated where it biases the estimate". Only the second is fatal.
    Truncation that is even across seen/unseen and original/perturbed attenuates every cell
    alike and largely cancels in the double difference; truncation correlated with the
    treatment does not, and no amount of extra tokens fixes an estimate built on it.

    Also splits reused (v1, some regenerated at 2048 tokens) from new (1024) rows, because a
    mixed token budget inside one block is itself a measurement inconsistency.
    """
    import glob
    import json

    os.chdir(REPO_REMOTE)
    out = pathlib.Path("experiments/runs") / run
    gen_path = out / f"generations/{tag}__arms.jsonl"
    src_path = pathlib.Path("experiments/runs/fullscale-S250") / f"generations/{tag}__arms.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"missing {gen_path}")

    reused: set[str] = set()
    if src_path.exists():
        with src_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    reused.add(str(json.loads(line)["benchmark_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    arms = {}
    for row in __import__("instella_reasoning.records", fromlist=["read_jsonl"]).read_jsonl(
        out / "variants/arms_variants.jsonl"
    ):
        arms[str(row.get("id"))] = (
            (row.get("metadata") or {}).get("arm"),
            str(row.get("variant_type") or "original"),
        )

    from instella_reasoning.metrics import ANSWER_CHANGING_VARIANTS

    cells: dict[tuple, list[int]] = {}
    origin: dict[str, list[int]] = {"reused(v1)": [], "new(1024)": []}
    with gen_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = str(row.get("benchmark_id"))
            fin = int(bool((row.get("metadata") or {}).get("finished", True)))
            arm, vt = arms.get(bid, (None, "original"))
            origin["reused(v1)" if bid in reused else "new(1024)"].append(fin)
            if arm in ("seen", "unseen"):
                cond = "perturbed" if vt in ANSWER_CHANGING_VARIANTS else "original"
                cells.setdefault((arm, cond), []).append(fin)

    print(f"{tag} / {run}\n")
    print(f"{'origin':14s} {'n':>6s} {'termination':>12s}")
    for k, v in origin.items():
        if v:
            print(f"{k:14s} {len(v):6d} {sum(v) / len(v):11.1%}")

    print(f"\n{'cell':20s} {'n':>6s} {'termination':>12s}")
    rates = {}
    for arm in ("seen", "unseen"):
        for cond in ("original", "perturbed"):
            v = cells.get((arm, cond)) or []
            if not v:
                continue
            rates[(arm, cond)] = sum(v) / len(v)
            print(f"{arm + '/' + cond:20s} {len(v):6d} {rates[(arm, cond)]:11.1%}")

    did = None
    if len(rates) == 4:
        did = (rates[("seen", "original")] - rates[("unseen", "original")]) - (
            rates[("seen", "perturbed")] - rates[("unseen", "perturbed")]
        )
        print(f"\ntermination second difference: {did:+.4f}")
        print("  -> " + ("truncation is balanced across cells; it attenuates the DiD "
                         "uniformly rather than biasing it"
                         if abs(did) <= 0.05 else
                         "truncation is correlated with the treatment; regenerate at a "
                         "larger budget before trusting this block"))
    return {"origin": {k: (sum(v) / len(v) if v else None) for k, v in origin.items()},
            "cells": {f"{a}/{c}": r for (a, c), r in rates.items()},
            "termination_did": did}


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes=VOLUMES, secrets=SECRETS)
def stage_v2(n_per_arm: int = 250, src_name: str = "fullscale-S250",
             dst_name: str = "fullscale-S250-v2", write: bool = False) -> dict:
    """Stage a corrected 250/arm run in its own directory, reusing v1's generations.

    Editing v1 in place does not survive: the suite's first action is ``_hf_pull``, and
    ``snapshot_download(local_dir=".")`` syncs local files *down* to match the remote, so any
    local edit is silently reverted before generation starts. Writing v2 to a fresh directory
    sidesteps that entirely -- its remote path is empty, so the pull is a no-op -- and leaves
    v1 intact on HF as the reproducible record of the first analysis.

    What v2 fixes, all downstream of the train/test id collision:

    * **Treatment labels** -- 83/194 v1 seen-arm items were actually ambiguous (containment
      0.39-0.79, under the 0.80 threshold). Only verdict-matching items are eligible here.
    * **Difficulty matching** -- ``assign_difficulty_bins`` is keyed by ``item.id`` too, so
      every train item inherited its test twin's difficulty and the arms were never really
      matched. Bins are recomputed over split-namespaced ids.
    * **Arm disjointness** -- with corrected verdicts a train item can be seen while its test
      twin is unseen; sharing a bare id they would collapse into one row at generation time.
      The unseen pool excludes every id the seen arm already took.
    * **F4** -- ``containment.json`` is rewritten from the corrected measurement, so the
      figure no longer plots containment above 1.0.
    """
    import glob
    import json
    import shutil
    from collections import defaultdict

    from instella_reasoning.difficulty import assign_difficulty_bins
    from instella_reasoning.gsm_symbolic import magnitude_report
    from instella_reasoning.perturbations import PerturbationConfig, make_variant_suite
    from instella_reasoning.records import BenchmarkItem, read_benchmark, write_jsonl

    os.chdir(REPO_REMOTE)
    src = pathlib.Path("experiments/runs") / src_name
    dst = pathlib.Path("experiments/runs") / dst_name
    verified = json.loads((src / "analysis/containment_verified.json").read_text())["containment"]
    train = read_benchmark(src / "base/gsm8k_train.jsonl")
    test = read_benchmark(src / "base/gsm8k_test.jsonl")

    ns = [BenchmarkItem(id=f"train::{i.id}", prompt=i.prompt, answer=i.answer, metadata=i.metadata)
          for i in train]
    ns += [BenchmarkItem(id=f"test::{i.id}", prompt=i.prompt, answer=i.answer, metadata=i.metadata)
           for i in test]
    bins = assign_difficulty_bins(ns, n_bins=3)

    seen_pool = [i for i in train if (verified.get(f"train::{i.id}") or {}).get("verdict") == "seen"]
    unseen_pool = [i for i in test if (verified.get(f"test::{i.id}") or {}).get("verdict") == "unseen"]
    print(f"eligible: seen={len(seen_pool)} unseen={len(unseen_pool)}")

    already: set[str] = set()
    for gen in (src / "generations").glob("*__arms.jsonl"):
        with gen.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    already.add(str(json.loads(line).get("benchmark_id", "")).split("__")[0])
                except json.JSONDecodeError:
                    continue

    def stratify(pool: list, prefix: str) -> dict[int, list]:
        g: dict[int, list] = defaultdict(list)
        for i in pool:
            g[bins.get(f"{prefix}::{i.id}", 0)].append(i)
        for b in g.values():
            b.sort(key=lambda i: (i.id not in already, i.id))  # already-generated first
        return g

    seen_bins = stratify(seen_pool, "train")
    per_bin = max(1, n_per_arm // max(1, len(seen_bins)))
    seen_sel: list = []
    for b in sorted(seen_bins):
        seen_sel.extend(seen_bins[b][:per_bin])
    for b in sorted(seen_bins):
        for i in seen_bins[b][per_bin:]:
            if len(seen_sel) >= n_per_arm:
                break
            seen_sel.append(i)
    seen_sel = seen_sel[:n_per_arm]

    taken = {i.id for i in seen_sel}
    unseen_bins = stratify([i for i in unseen_pool if i.id not in taken], "test")
    profile: dict[int, int] = defaultdict(int)
    for i in seen_sel:
        profile[bins.get(f"train::{i.id}", 0)] += 1
    unseen_sel: list = []
    for b, want in sorted(profile.items()):
        got = unseen_bins.get(b, [])[:want]
        if len(got) < want:
            print(f"  bin {b}: unseen short by {want - len(got)}")
        unseen_sel.extend(got)

    for i in seen_sel:
        i.metadata = {**i.metadata, "arm": "seen", "gsm8k_split": "train"}
    for i in unseen_sel:
        i.metadata = {**i.metadata, "arm": "unseen", "gsm8k_split": "test"}
    arms = [*seen_sel, *unseen_sel]
    assert len({i.id for i in arms}) == len(arms), "bare-id collision survived into the arms"

    suite = make_variant_suite(arms, PerturbationConfig(seed=6198, numeric_variants=2, max_variants=2))
    mag = magnitude_report(suite)
    print(f"arms: seen={len(seen_sel)} unseen={len(unseen_sel)}  variants={len(suite)}")
    print(f"difficulty profile: {dict(sorted(profile.items()))}")
    print(f"magnitude in_band={mag.in_band} median_ratio="
          f"{mag.to_dict().get('median_answer_magnitude_ratio')}")
    if not mag.in_band:
        raise SystemExit("ABORT: numeric variants are not magnitude-neutral; refusing to spend GPU.")

    new_ids = {i.id for i in suite}
    reuse_total = regen_total = 0
    print(f"\n{'checkpoint':12s} {'reusable':>9s} {'to generate':>12s}")
    per_tag: dict[str, list[str]] = {}
    for tag in ("stage1", "stage2", "instruct"):
        kept: list[str] = []
        gen = src / f"generations/{tag}__arms.jsonl"
        if gen.exists():
            with gen.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if str(json.loads(line)["benchmark_id"]) in new_ids:
                            kept.append(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
        per_tag[tag] = kept
        reuse_total += len(kept)
        regen_total += len(new_ids) - len(kept)
        print(f"{tag:12s} {len(kept):9d} {len(new_ids) - len(kept):12d}")
    print(f"{'TOTAL':12s} {reuse_total:9d} {regen_total:12d}")
    print(f"estimated GPU: {regen_total * 2.5 / 3600:.1f}h at 2.5 s/item")

    if not write:
        print("\n(dry run: pass write=True to stage v2)")
        return {"seen": len(seen_sel), "unseen": len(unseen_sel), "to_generate": regen_total}

    for sub in ("base", "arms", "variants", "generations", "scores", "analysis", "atlas",
                "contamination", "figures", "markers", "review"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    # Shared inputs. The 1.45 GB corpus is deliberately not copied: the stage1_corpus marker
    # means nothing reads it, and it is already archived under v1 on HF.
    for rel in ("base/gsm8k_train.jsonl", "base/gsm8k_test.jsonl",
                "variants/gsm_symbolic.jsonl", "variants/gsm_symbolic_p1.jsonl",
                "variants/gsm_symbolic_official.jsonl", "variants/resample_control.jsonl",
                "analysis/containment_verified.json"):
        if (src / rel).exists():
            shutil.copy2(src / rel, dst / rel)

    write_jsonl(dst / "arms/gsm8k_arms.jsonl", arms)
    write_jsonl(dst / "variants/arms_variants.jsonl", suite)

    # F4 reads {"items": [...]}; rebuild it from the corrected measurement.
    intended = {f"train::{i.id}": "seen" for i in train}
    intended.update({f"test::{i.id}": "unseen" for i in test})
    (dst / "analysis/containment.json").write_text(
        json.dumps({
            "summary": {"n_seen": len(seen_sel), "n_unseen": len(unseen_sel),
                        "source": "containment_verified.json (split-namespaced ids)"},
            "items": [
                {"benchmark_id": k, "containment": v["containment"], "verdict": v["verdict"],
                 "intended_arm": intended.get(k)}
                for k, v in sorted(verified.items())
            ],
        }, indent=2), encoding="utf-8")

    # Blocks that are unaffected by the arm rebuild carry their scores across, so the suite
    # skips them; the arms blocks arrive with generations but no score file and re-open.
    for tag in ("stage1", "stage2", "instruct"):
        (dst / f"generations/{tag}__arms.jsonl").write_text(
            "\n".join(per_tag[tag]) + ("\n" if per_tag[tag] else ""), encoding="utf-8")
        for block in ("gsmsym", "resample"):
            for kind in ("generations", "scores"):
                s = src / f"{kind}/{tag}__{block}.jsonl"
                if s.exists():
                    shutil.copy2(s, dst / f"{kind}/{tag}__{block}.jsonl")

    for marker in ("stage1_corpus", "stage2_splits", "stage3_variants", "stage4_aux"):
        (dst / "markers" / marker).touch()

    runs_vol.commit()
    print(f"\nstaged {dst}")
    print(f"launch with:  modal run --detach experiments/modal_fullscale.py --out {dst}")
    return {"seen": len(seen_sel), "unseen": len(unseen_sel), "variants": len(suite),
            "reusable": reuse_total, "to_generate": regen_total, "dst": str(dst)}


@app.local_entrypoint()
def main(n_per_arm: int = 250, tier: int = 1, hf_sync: int = 1, block: bool = False,
         out: str = "", min_termination: str = "") -> None:
    """Launch the suite. Fire-and-forget by default.

    ``.remote()`` blocks the local client for the whole run, and a cancellation of that
    local call propagates to the container — which is exactly how a previous 2h run died at
    instruct/resample. ``.spawn()`` returns a handle immediately, so with ``--detach`` there
    is no long-lived local process left that can take the run down with it.
    """
    args = {"N_PER_ARM": n_per_arm, "SUITE_TIER": tier, "HF_SYNC": hf_sync}
    if out:
        # The suite derives OUT from N_PER_ARM unless told otherwise; pointing it at the
        # staged v2 directory keeps v1 untouched on HF and makes the pull a no-op.
        args["OUT"] = out
    if min_termination:
        # Only ever lower this against a *quantified* truncation profile. stage1/arms sits at
        # 84.9% purely because v1 repaired 261 rows at 2048 tokens and v2's new rows were not;
        # termination_by_cell puts the second difference at -0.013, so the shortfall attenuates
        # the cells uniformly instead of biasing the contrast. repair_truncated restores the
        # block to >=85% afterwards. Never lower it to make an unexplained failure go away.
        args["MIN_TERMINATION"] = min_termination
    if block:
        print(run_suite.remote(args))
        return
    handle = run_suite.spawn(args)
    print(f"spawned run_suite — call id {handle.object_id}")
    print("the suite now runs server-side; this client can exit safely.")
    print("follow it with:  modal app logs <app-id from `modal app list`>")
