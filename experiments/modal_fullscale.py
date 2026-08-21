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
        def rate(arm: str, cond: str, cells=cells) -> float:
            t = cells.get((arm, cond))
            n = sum(t.values()) if t else 0
            return (t["1_marker"] / n) if n else 0.0

        def acc_of(arm: str, cond: str, acc=acc) -> float:
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


JUDGE_MODES = ("reference_free", "reference_based")


def _judge_prompt(question: str, completion: str, gold: str | None, mode: str) -> str:
    """Pointwise correctness grading, in the reference-based and reference-free settings.

    Reference-based supplies the gold answer, reducing the task to a comparison; it is the
    control, and should sit near ceiling regardless of membership. Reference-free withholds
    it, so the judge must evaluate the reasoning itself — which is the only setting where
    having memorised the problem could plausibly help, and therefore the only setting where a
    membership effect is interpretable.
    """
    # The final answer lives at the end of a chain of thought, so the tail is kept rather
    # than the head when a completion has to be truncated.
    tail = completion.strip()[-1200:]
    ref = f"\nReference answer: {gold}\n" if mode == "reference_based" and gold else ""
    return (
        "You are grading a solution to a grade-school math problem.\n\n"
        f"Problem:\n{question.strip()}\n"
        f"{ref}\n"
        f"Proposed solution:\n{tail}\n\n"
        "Is the final answer of the proposed solution correct? "
        "Reply with exactly one word: CORRECT or INCORRECT."
    )


@app.function(gpu=GPU, timeout=3 * 3600, volumes=VOLUMES, secrets=SECRETS)
def judge_run(tag: str = "instruct", mode: str = "reference_free",
              run: str = "fullscale-S250-v2", judge: str = "instruct",
              judge_model: str = "", judge_label: str = "",
              batch_size: int = 8, max_new_tokens: int = 24) -> dict:
    """Grade one checkpoint's arms generations with an Instella judge.

    Uses the same model family as the system under evaluation on purpose. Membership of each
    graded item in the *judge's* own training corpus is exactly verified, which is what makes
    the seen/unseen contrast interpretable; with an external judge that membership would be
    unknown and the contrast would mean nothing.

    Resumes on benchmark id, flushing per batch, so an interrupted run continues.
    """
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from instella_reasoning.checkpoints import resolve
    from instella_reasoning.records import read_jsonl

    if mode not in JUDGE_MODES:
        raise SystemExit(f"mode must be one of {JUDGE_MODES}")
    os.chdir(REPO_REMOTE)
    runs_vol.reload()
    out = pathlib.Path("experiments/runs") / run
    # An external judge writes under a label so a ladder of judge sizes coexists. The
    # in-family runs keep their original single-underscore names, and judge_analysis reads
    # both patterns, so earlier verdicts stay valid rather than needing regeneration.
    if judge_model:
        label = judge_label or judge_model.split("/")[-1].lower()
        dest = out / f"analysis/judge_{label}__{tag}__{mode}.jsonl"
    else:
        dest = out / f"analysis/judge_{tag}_{mode}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)

    items = {str(r["id"]): r for r in read_jsonl(out / "variants/arms_variants.jsonl")}
    gens = {str(r["benchmark_id"]): r for r in read_jsonl(out / f"generations/{tag}__arms.jsonl")}
    truth = {str(r["benchmark_id"]): bool(r["correct"])
             for r in read_jsonl(out / f"scores/{tag}__arms.jsonl")}

    done: set[str] = set()
    if dest.exists():
        for r in read_jsonl(dest):
            done.add(str(r["benchmark_id"]))
    pending = [b for b in gens if b in items and b in truth and b not in done]
    print(f"judge={judge_model or judge} target={tag}/arms mode={mode}")
    print(f"{len(done)} already graded, {len(pending)} pending", flush=True)
    if not pending:
        return {"graded": len(done), "pending": 0}

    judge_path = judge_model or resolve(judge).load_path
    print(f"loading judge {judge_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        judge_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def render(bid: str) -> str:
        item, gen = items[bid], gens[bid]
        text = _judge_prompt(item.get("prompt", ""), gen.get("completion", ""),
                             item.get("answer"), mode)
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)

    def parse(raw: str) -> bool | None:
        """None when the judge does not commit — an abstention, not a wrong answer."""
        upper = raw.upper()
        has_incorrect = "INCORRECT" in upper
        # "INCORRECT" contains "CORRECT", so the negative must be tested first.
        has_correct = "CORRECT" in upper.replace("INCORRECT", "")
        if has_incorrect and not has_correct:
            return False
        if has_correct and not has_incorrect:
            return True
        return None

    stop = threading.Event()
    _start_watchers(stop)
    started = time.time()
    n_written = 0
    try:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            enc = tok([render(b) for b in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            rows = []
            for bid, seq in zip(chunk, ids, strict=False):
                raw = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                item = items[bid]
                rows.append({
                    "benchmark_id": bid,
                    "parent_id": item.get("parent_id") or bid,
                    "arm": (item.get("metadata") or {}).get("arm"),
                    "variant_type": item.get("variant_type") or "original",
                    "judge_verdict": parse(raw),
                    "ground_truth": truth[bid],
                    "raw": raw.strip()[:120],
                })
            with dest.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            n_written += len(rows)
            if (i // batch_size) % 20 == 0:
                el = time.time() - started
                rate = n_written / el if el else 0
                print(f"  {n_written}/{len(pending)}  {rate:.2f} item/s  "
                      f"eta {(len(pending) - n_written) / max(rate, 1e-6) / 60:.0f} min",
                      flush=True)
                runs_vol.commit()
    finally:
        stop.set()
        runs_vol.commit()

    elapsed = (time.time() - started) / 3600
    print(f"\ngraded {n_written} in {elapsed:.2f}h")
    return {"graded": n_written + len(done), "hours": elapsed}


@app.function(cpu=4.0, memory=8192, timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def consistency_by_arm(run: str = "fullscale-S250-v2", n_bootstrap: int = 4000,
                       seed: int = 6198, write: bool = False, push: bool = False) -> dict:
    """Paraphrase consistency split by verified membership — a probe the DiD cannot make.

    The difference-in-differences uses *numeric* perturbation, which changes the arithmetic
    while preserving problem structure. It therefore detects a model that memorised an
    **answer** but is blind to one that memorised a **solution procedure**: a recalled
    template still executes correctly on new numbers, producing exactly the flat DiD observed.
    That is the main study's most serious open gap.

    Answer-*preserving* variants attack it from the other side. Rephrasing and distractor
    insertion leave the gold answer and the required procedure intact but change the surface
    form. A model reciting a memorised item should be *more* stable across those rewrites than
    one solving from scratch, so elevated consistency on verified-seen items is a memorisation
    signature that survives numeric perturbation.

    The resample block supplies the null. At temperature 0 there is no sampling variance, so
    raw consistency has no baseline; where a checkpoint has the control, consistency is
    reported against its own decoding-noise floor.
    """
    import glob
    import json
    import random
    from collections import defaultdict

    from instella_reasoning.metrics import cluster_consistency, decoding_noise_consistency
    from instella_reasoning.records import EvaluationRecord, read_jsonl

    os.chdir(REPO_REMOTE)
    runs_vol.reload()
    out = pathlib.Path("experiments/runs") / run
    fields = {f.name for f in __import__("dataclasses").fields(EvaluationRecord)}

    def load(path: pathlib.Path) -> list[EvaluationRecord]:
        if not path.exists():
            return []
        rows = []
        for r in read_jsonl(path):
            try:
                rows.append(EvaluationRecord(**{k: v for k, v in r.items() if k in fields}))
            except TypeError:
                continue
        return rows

    order = {"stage1": 0, "stage2": 1, "sft": 2, "instruct": 3}
    tags = sorted({pathlib.Path(p).name.replace("__arms.jsonl", "")
                   for p in glob.glob(f"{out}/scores/*__arms.jsonl")},
                  key=lambda t: (order.get(t, 99), t))

    report: dict = {}
    print("Modal-answer agreement over answer-preserving variants, by verified membership.")
    print("Elevated consistency on seen items indicates recall of a solution procedure,")
    print("which numeric perturbation cannot detect.\n")
    print(f"{'ckpt':9s} {'n_seen':>6s} {'n_unseen':>8s} {'C_seen':>7s} {'C_unseen':>9s} "
          f"{'delta':>8s} {'CI95':>20s} {'noise floor':>12s}")

    for tag in tags:
        recs = load(out / f"scores/{tag}__arms.jsonl")
        if not recs:
            continue
        by_parent: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for r in recs:
            by_parent[r.parent_id].append(r)

        arm_of, cons = {}, {}
        for pid, rows in by_parent.items():
            arms = {(r.metadata or {}).get("arm") for r in rows}
            arm = next((a for a in arms if a in ("seen", "unseen")), None)
            if arm is None:
                continue
            # A cluster needs at least two answer-preserving rows for agreement to exist.
            preserving = [r for r in rows if r.variant_type in
                          ("original", "rephrasing", "irrelevant_context", "entity_substitution")]
            if len(preserving) < 2:
                continue
            arm_of[pid] = arm
            cons[pid] = cluster_consistency(preserving)

        seen = [cons[p] for p in cons if arm_of[p] == "seen"]
        unseen = [cons[p] for p in cons if arm_of[p] == "unseen"]
        if not seen or not unseen:
            continue
        c_seen = sum(seen) / len(seen)
        c_unseen = sum(unseen) / len(unseen)
        delta = c_seen - c_unseen

        # Bootstrap over parent items, matching the main analysis.
        keys = list(cons)
        rng = random.Random(seed)
        draws = []
        for _ in range(n_bootstrap):
            s, u = [], []
            for _ in range(len(keys)):
                p = keys[rng.randrange(len(keys))]
                (s if arm_of[p] == "seen" else u).append(cons[p])
            if s and u:
                draws.append(sum(s) / len(s) - sum(u) / len(u))
        draws.sort()
        ci = [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]] if draws else [None, None]

        # Decoding-noise floor from the resample control, where the checkpoint has one.
        floor = None
        rs = load(out / f"scores/{tag}__resample.jsonl")
        if rs:
            grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
            for r in rs:
                grouped[r.parent_id].append(r)
            vals = [v for v, n in (decoding_noise_consistency(g) for g in grouped.values()) if v >= 0]
            floor = sum(vals) / len(vals) if vals else None

        excl = bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))
        print(f"{tag:9s} {len(seen):6d} {len(unseen):8d} {c_seen:7.4f} {c_unseen:9.4f} "
              f"{delta:+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"{'n/a' if floor is None else format(floor, '.4f'):>12s}"
              f"{'  *' if excl else ''}")

        # Accuracy is the confound. Seen items are solved more often (0.655 vs 0.455 at
        # stage2), and a model that is more often right is mechanically more self-consistent,
        # so an unstratified gap can be produced by accuracy alone with no recall involved.
        # Stratifying on whether the cluster's original was answered correctly holds that
        # constant: a gap surviving inside BOTH strata is not an accuracy artifact.
        correct_of = {}
        for pid, rows in by_parent.items():
            origs = [r for r in rows if r.variant_type == "original"]
            if origs:
                correct_of[pid] = bool(origs[0].correct)
        strata = {}
        for label, want in (("original_correct", True), ("original_wrong", False)):
            s = [cons[p] for p in cons if arm_of[p] == "seen" and correct_of.get(p) is want]
            u = [cons[p] for p in cons if arm_of[p] == "unseen" and correct_of.get(p) is want]
            if len(s) < 10 or len(u) < 10:
                strata[label] = {"n_seen": len(s), "n_unseen": len(u), "delta": None,
                                 "ci95": [None, None], "note": "too few clusters"}
                continue
            ks = list(cons)
            rng2 = random.Random(seed + 1)
            d2 = []
            for _ in range(n_bootstrap):
                ss, uu = [], []
                for _ in range(len(ks)):
                    p = ks[rng2.randrange(len(ks))]
                    if correct_of.get(p) is not want:
                        continue
                    (ss if arm_of[p] == "seen" else uu).append(cons[p])
                if ss and uu:
                    d2.append(sum(ss) / len(ss) - sum(uu) / len(uu))
            d2.sort()
            c2 = [d2[int(0.025 * len(d2))], d2[int(0.975 * len(d2))]] if d2 else [None, None]
            dd = sum(s) / len(s) - sum(u) / len(u)
            e2 = bool(c2[0] is not None and (c2[0] > 0 or c2[1] < 0))
            strata[label] = {"n_seen": len(s), "n_unseen": len(u),
                             "consistency_seen": sum(s) / len(s),
                             "consistency_unseen": sum(u) / len(u),
                             "delta": dd, "ci95": c2, "excludes_zero": e2}
            print(f"    {label:17s} n={len(s):3d}/{len(u):3d}  Δ={dd:+.4f} "
                  f"[{c2[0]:+.4f},{c2[1]:+.4f}]{'  *' if e2 else ''}")

        report[tag] = {
            "n_seen": len(seen), "n_unseen": len(unseen),
            "consistency_seen": c_seen, "consistency_unseen": c_unseen,
            "delta_seen_minus_unseen": delta, "delta_ci95": ci, "excludes_zero": excl,
            "decoding_noise_floor": floor,
            "accuracy_stratified": strata,
        }

    print("\nA positive delta excluding zero would be evidence of procedure-level recall on")
    print("verified-seen items, and would qualify the main result. A null strengthens it: the")
    print("seen advantage survives both numeric perturbation and surface rewriting.")

    if write:
        dest = out / "analysis/consistency_by_arm.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs_vol.commit()
        print(f"\nwrote {dest}")
        if push:
            subprocess.call(
                [sys.executable, "experiments/hf_sync.py", "push", "--repo",
                 os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
                 "--path", str(out), "--message", "paraphrase consistency by verified arm"]
            )
    else:
        print("\n(read-only: pass write=True to persist)")
    return report


def _balanced_accuracy(pairs: list[tuple[bool, bool]]) -> tuple[float, float, float] | None:
    """Sensitivity, specificity, and their mean, from (judge_verdict, ground_truth) pairs.

    Cohen's kappa cannot be compared across the two arms here. Kappa is a function of the
    marginals as well as the agreement, and the arms have very different ground-truth base
    rates -- instruct is correct on 81% of seen originals against 56% of unseen ones. An arm
    whose base rate sits near 0.5 has far more kappa headroom than a skewed one, so a kappa
    gap between arms is produced by the skew itself. That is the kappa paradox, and it would
    masquerade as a judge defect.

    Sensitivity and specificity are conditioned on the true class, so neither depends on the
    base rate, and their unweighted mean is comparable across arms by construction.
    """
    pos = [j for j, t in pairs if t]
    neg = [j for j, t in pairs if not t]
    if not pos or not neg:
        return None  # one true class absent: no base-rate-free comparison is possible
    sens = sum(1 for j in pos if j) / len(pos)
    spec = sum(1 for j in neg if not j) / len(neg)
    return sens, spec, (sens + spec) / 2


def _cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Chance-corrected agreement between judge verdict and exact-match ground truth.

    Raw agreement is not usable here: the arms are unbalanced in correctness (instruct is
    right on ~81% of originals), so a judge that always answered CORRECT would post high
    agreement while carrying no information. Kappa removes exactly that.
    """
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pj_true = sum(1 for a, _ in pairs if a) / n
    pt_true = sum(1 for _, b in pairs if b) / n
    pe = pj_true * pt_true + (1 - pj_true) * (1 - pt_true)
    if abs(1 - pe) < 1e-12:
        return None  # degenerate: one class absent, kappa undefined rather than perfect
    return (po - pe) / (1 - pe)


@app.function(cpu=4.0, memory=8192, timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def judge_analysis(run: str = "fullscale-S250-v2", n_bootstrap: int = 4000,
                   seed: int = 6198, write: bool = False, push: bool = False) -> dict:
    """Judge reliability split by verified training-set membership.

    The question is whether an LLM judge grades more accurately on problems that are
    verifiably in its own training corpus. A gap is a construct-validity failure in the
    *instrument* rather than in the model under test, which is what makes it reportable
    independently of the memorisation result.

    Reference-based grading is the control. Supplying the gold answer reduces the task to a
    comparison, so a membership effect there would indicate something other than memorisation
    (formatting familiarity, for instance). The effect is only interpretable as memorisation if
    it appears in the reference-free condition and not in the control.

    Intervals come from a bootstrap over parent items, matching the main analysis: the five
    rows of one problem are not independent observations.
    """
    import glob
    import json
    import random
    from collections import defaultdict

    from instella_reasoning.metrics import ANSWER_CHANGING_VARIANTS

    os.chdir(REPO_REMOTE)
    runs_vol.reload()
    out = pathlib.Path("experiments/runs") / run
    files = sorted(glob.glob(f"{out}/analysis/judge_*_*.jsonl"))
    if not files:
        raise SystemExit("no judge verdict files; run judge_run first")

    report: dict = {}
    print("balanced accuracy = mean(sensitivity, specificity); base-rate independent, so")
    print("comparable across arms in a way Cohen's kappa is not.\n")
    print(f"{'judge':22s} {'target':9s} {'mode':16s} {'n':>5s} {'abst':>6s} "
          f"{'BA_seen':>8s} {'BA_unseen':>9s} {'delta':>8s} {'CI95':>20s}")
    for path in files:
        stem = pathlib.Path(path).stem.replace("judge_", "", 1)
        if "__" in stem:
            jlabel, tag, mode = stem.split("__")
        else:
            # Original in-family runs, before the judge ladder existed.
            jlabel = "instella-3b-instruct"
            tag, mode = stem.split("_", 1)
        stem = f"{jlabel}__{tag}__{mode}"
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        n_total = len(rows)
        usable = [r for r in rows if r["judge_verdict"] is not None and r["arm"] in ("seen", "unseen")]
        abstain = 1 - len(usable) / n_total if n_total else 0.0

        by_arm: dict[str, list] = defaultdict(list)
        by_cond: dict[str, list] = defaultdict(list)
        clusters: dict[str, list] = defaultdict(list)
        for r in usable:
            pair = (bool(r["judge_verdict"]), bool(r["ground_truth"]))
            by_arm[r["arm"]].append(pair)
            cond = "perturbed" if r["variant_type"] in ANSWER_CHANGING_VARIANTS else "original"
            by_cond[f"{r['arm']}|{cond}"].append(pair)
            clusters[r["parent_id"]].append((r["arm"], pair))

        k_seen = _cohen_kappa(by_arm.get("seen", []))
        k_unseen = _cohen_kappa(by_arm.get("unseen", []))
        ba_seen = _balanced_accuracy(by_arm.get("seen", []))
        ba_unseen = _balanced_accuracy(by_arm.get("unseen", []))
        # Base rates are reported because they are the whole reason kappa is not comparable.
        base_seen = (sum(1 for _, t in by_arm.get("seen", []) if t)
                     / max(1, len(by_arm.get("seen", []))))
        base_unseen = (sum(1 for _, t in by_arm.get("unseen", []) if t)
                       / max(1, len(by_arm.get("unseen", []))))

        delta_k = (k_seen - k_unseen) if (k_seen is not None and k_unseen is not None) else None
        delta_ba = (ba_seen[2] - ba_unseen[2]) if (ba_seen and ba_unseen) else None

        # Resample parent items, not rows. Balanced accuracy is the headline; kappa is carried
        # only so the base-rate artifact remains visible next to the corrected number.
        def boot(metric, clusters=clusters) -> list:
            keys = list(clusters)
            rng = random.Random(seed)
            draws = []
            for _ in range(n_bootstrap):
                s, u = [], []
                for _ in range(len(keys)):
                    for arm, pair in clusters[keys[rng.randrange(len(keys))]]:
                        (s if arm == "seen" else u).append(pair)
                a, b = metric(s), metric(u)
                if a is not None and b is not None:
                    av = a[2] if isinstance(a, tuple) else a
                    bv = b[2] if isinstance(b, tuple) else b
                    draws.append(av - bv)
            if not draws:
                return [None, None]
            draws.sort()
            return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]

        ci_ba = boot(_balanced_accuracy) if delta_ba is not None else [None, None]
        ci_k = boot(_cohen_kappa) if delta_k is not None else [None, None]

        fmt = lambda v: " n/a  " if v is None else f"{v:+.4f}"  # noqa: E731
        cis = "n/a" if ci_ba[0] is None else f"[{ci_ba[0]:+.4f}, {ci_ba[1]:+.4f}]"
        excl_mark = "*" if (ci_ba[0] is not None and (ci_ba[0] > 0 or ci_ba[1] < 0)) else " "
        print(f"{jlabel[:22]:22s} {tag:9s} {mode:16s} {len(usable):5d} {abstain:5.1%} "
              f"{fmt(ba_seen[2] if ba_seen else None):>8s} "
              f"{fmt(ba_unseen[2] if ba_unseen else None):>9s} "
              f"{fmt(delta_ba):>8s} {cis:>20s} {excl_mark}")

        report[stem] = {
            "judge": jlabel,
            "target": tag, "mode": mode, "n_total": n_total, "n_usable": len(usable),
            "abstention_rate": abstain,
            "ground_truth_base_rate": {"seen": base_seen, "unseen": base_unseen},
            "balanced_accuracy": {
                "seen": None if not ba_seen else
                        {"sensitivity": ba_seen[0], "specificity": ba_seen[1], "balanced": ba_seen[2]},
                "unseen": None if not ba_unseen else
                          {"sensitivity": ba_unseen[0], "specificity": ba_unseen[1], "balanced": ba_unseen[2]},
                "delta_seen_minus_unseen": delta_ba,
                "delta_ci95": ci_ba,
                "excludes_zero": bool(ci_ba[0] is not None and (ci_ba[0] > 0 or ci_ba[1] < 0)),
            },
            # Retained deliberately: the kappa gap is an artifact of the differing base rates
            # above, and keeping both numbers side by side documents why it was discarded.
            "kappa_confounded": {
                "seen": k_seen, "unseen": k_unseen,
                "delta_seen_minus_unseen": delta_k, "delta_ci95": ci_k,
                "note": "not comparable across arms; base rates differ, see kappa paradox",
            },
            "cells": {
                k: {"n": len(v),
                    "balanced_accuracy": (_balanced_accuracy(v) or (None, None, None))[2]}
                for k, v in sorted(by_cond.items())
            },
            "accuracy_vs_truth": (
                sum(1 for v in by_arm.values() for a, b in v if a == b)
                / max(1, sum(len(v) for v in by_arm.values()))
            ),
        }

    print("\nReading: a delta whose interval excludes zero means judge reliability depends on")
    print("whether the graded item is in the judge's own training data. If that appears in")
    print("reference_free but not reference_based, memorisation is the parsimonious account.")

    if write:
        dest = out / "analysis/judge_reliability.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs_vol.commit()
        print(f"\nwrote {dest}")
        if push:
            subprocess.call(
                [sys.executable, "experiments/hf_sync.py", "push", "--repo",
                 os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
                 "--path", str(out), "--message",
                 "judge reliability by verified membership"]
            )
    else:
        print("\n(read-only: pass write=True to persist)")
    return report


@app.function(cpu=4.0, memory=8192, timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def supplementary_figures(run: str = "fullscale-S250-v2") -> list[str]:
    """Four supplementary panels the F1-F8 set does not cover.

    Each carries evidence that is currently only available as a table, and each answers a
    question a reviewer raises against a null result specifically: is the treatment label
    trustworthy (S1, S2), is the scoring comparable across checkpoints (S3), and is the
    absence of an effect visible in the raw cells rather than only in a derived statistic
    (S4). Styling follows the main set so the two are readable as one document.
    """
    import json

    from instella_reasoning.analysis.figures import (
        GRID,
        INK,
        INK_SECONDARY,
        SERIES,
        _mpl,
        _save,
    )

    plt = _mpl()
    if plt is None:
        raise SystemExit("matplotlib unavailable")

    os.chdir(REPO_REMOTE)
    runs_vol.reload()
    out = pathlib.Path("experiments/runs") / run
    figs = out / "figures"
    a = out / "analysis"
    verified = json.loads((a / "containment_verified.json").read_text())["containment"]
    det = json.loads((a / "detector_falsepositive.json").read_text())
    tiers = json.loads((a / "extraction_tiers.json").read_text())
    memo = json.loads((a / "memorization.json").read_text())

    NAMES = {"amd/Instella-3B-Stage1": "stage1", "amd/Instella-3B": "stage2",
             "amd/Instella-3B-SFT": "sft", "amd/Instella-3B-Instruct": "instruct"}
    ORDER = ["stage1", "stage2", "sft", "instruct"]
    written: list[str] = []

    def _clean(ax) -> None:
        ax.set_facecolor("none")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    # S1 -- containment distribution. The treatment label rests on these two distributions
    # being separable; a histogram shows separability that a threshold table asserts.
    train = [v["containment"] for k, v in verified.items() if k.startswith("train::")]
    test = [v["containment"] for k, v in verified.items() if k.startswith("test::")]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = [i / 20 for i in range(21)]
    ax.hist(train, bins=bins, color=SERIES[0], alpha=0.85, label=f"GSM8K train (n={len(train)})")
    ax.hist(test, bins=bins, color=SERIES[1], alpha=0.85, label=f"GSM8K test (n={len(test)})")
    ax.axvline(0.80, color=INK, lw=1.2, ls="--")
    ax.axvline(0.10, color=INK, lw=1.2, ls="--")
    ax.text(0.80, ax.get_ylim()[1] * 0.94, " seen threshold", fontsize=8, color=INK)
    ax.text(0.10, ax.get_ylim()[1] * 0.86, " unseen threshold", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("13-gram containment against the training corpus", color=INK, fontsize=10)
    ax.set_ylabel("items (log scale)", color=INK, fontsize=10)
    ax.set_title("S1  Treatment assignment is verified, not inferred", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _clean(ax)
    written.append(_save(fig, figs / "s1_containment_distribution.png", plt))

    # S2 -- detector error against known negatives. The corpus derives from GSM8K train, so
    # every test item is a negative by construction and the false-positive rate is measurable.
    rules = det["rules"]
    labels = list(rules)
    thr = [rules[k]["threshold"] for k in labels]
    fpr = [rules[k]["false_positive_rate"] * 100 for k in labels]
    tpr = [rules[k]["train_flagged"] / det["n_train"] * 100 for k in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(thr, tpr, marker="o", color=SERIES[0], lw=2, label="GSM8K train flagged")
    ax.plot(thr, fpr, marker="s", color=SERIES[1], lw=2, label="GSM8K test flagged (false positives)")
    for x, y in zip(thr, fpr, strict=False):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=8, color=INK_SECONDARY, ha="center")
    ax.set_xlabel("containment threshold for a contamination flag", color=INK, fontsize=10)
    ax.set_ylabel("percent of split flagged", color=INK, fontsize=10)
    ax.set_title("S2  Exact detection is accurate on items that cannot be contaminated",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax)
    written.append(_save(fig, figs / "s2_detector_false_positive.png", plt))

    # S3 -- extraction provenance. The marker rate collapses between checkpoints, so the
    # question is whether scoring is comparable; the composition makes the answer inspectable.
    keys = [f"{t}__arms" for t in ORDER if f"{t}__arms" in tiers]
    t1 = [tiers[k]["tiers"].get("1_marker", 0) / tiers[k]["n"] * 100 for k in keys]
    t2 = [tiers[k]["tiers"].get("2_phrase", 0) / tiers[k]["n"] * 100 for k in keys]
    t3 = [tiers[k]["tiers"].get("3_last_number", 0) / tiers[k]["n"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = range(len(keys))
    ax.bar(xs, t1, color=SERIES[0], label="tier 1  #### marker")
    ax.bar(xs, t2, bottom=t1, color=SERIES[2], label="tier 2  answer-is phrase")
    ax.bar(xs, t3, bottom=[a + b for a, b in zip(t1, t2, strict=False)], color=SERIES[3],
           label="tier 3  last number")
    for i, v in enumerate(t1):
        ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([k.replace("__arms", "") for k in keys], color=INK, fontsize=10)
    ax.set_ylabel("percent of completions", color=INK, fontsize=10)
    ax.set_title("S3  Answer extraction differs by checkpoint but cancels in the double difference",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    _clean(ax)
    written.append(_save(fig, figs / "s3_extraction_tiers.png", plt))

    # S4 -- the four cells per checkpoint. The DiD is a contrast of contrasts; showing the
    # cells lets a reader verify the null in the raw accuracies instead of trusting a scalar.
    by = {NAMES[k]: v["did"] for k, v in memo["per_model"].items() if k in NAMES}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    width = 0.2
    cellnames = ["seen|original", "unseen|original", "seen|perturbed", "unseen|perturbed"]
    pretty = ["seen orig", "unseen orig", "seen pert", "unseen pert"]
    for j, cell in enumerate(cellnames):
        vals = [by[t]["cells"][cell]["accuracy"] for t in ORDER if t in by]
        ax.bar([i + (j - 1.5) * width for i in range(len(vals))], vals, width,
               color=SERIES[j % len(SERIES)], label=pretty[j])
    for i, t in enumerate([t for t in ORDER if t in by]):
        d = by[t]["difference_in_differences"]
        ax.annotate(f"DiD {d:+.3f}", (i, 0.95), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len([t for t in ORDER if t in by])))
    ax.set_xticklabels([t for t in ORDER if t in by], color=INK, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy", color=INK, fontsize=10)
    ax.set_title("S4  The seen advantage does not shrink under perturbation at any checkpoint",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.10))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _clean(ax)
    written.append(_save(fig, figs / "s4_did_cells.png", plt))

    # --- J1-J3: judge reliability, rendered only once verdicts exist ---------------------
    judge_path = a / "judge_reliability.json"
    if judge_path.exists():
        jr = json.loads(judge_path.read_text())
        modes = ["reference_free", "reference_based"]
        tags = [t for t in ORDER if any(v["target"] == t for v in jr.values())]

        # J1 -- the primary contrast: kappa by arm, per mode, with the bootstrap interval on
        # the difference. Reference-based is the control panel.
        # Judge ladder, ordered by capability. Keys are judge__target__mode.
        LADDER = ["instella-3b-instruct", "qwen2.5-7b-instruct", "qwen2.5-14b-instruct"]
        SHORT = {"instella-3b-instruct": "Instella 3B", "qwen2.5-7b-instruct": "Qwen2.5 7B",
                 "qwen2.5-14b-instruct": "Qwen2.5 14B"}
        judges = [j for j in LADDER if any(v.get("judge") == j for v in jr.values())]

        def entry(j: str, t: str, m: str) -> dict | None:
            return jr.get(f"{j}__{t}__{m}")

        # J4 -- the dissociation. Supplying the reference answer removes the membership gap
        # once the judge is competent, while withholding it does not, and the reference-free
        # effect does not shrink as the judge scales. A single-condition design would report
        # the gap as a property of the judge; the control is what makes it interpretable.
        if judges:
            fig, axes4 = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
            for ax, target in zip(axes4, ["stage2", "instruct"], strict=False):
                xs = list(range(len(judges)))
                for k, mode in enumerate(modes):
                    ys, los, his = [], [], []
                    for j in judges:
                        e = entry(j, target, mode)
                        ba = (e or {}).get("balanced_accuracy") or {}
                        d, ci = ba.get("delta_seen_minus_unseen"), ba.get("delta_ci95") or [None, None]
                        ys.append(d if d is not None else float("nan"))
                        los.append((d - ci[0]) if (d is not None and ci[0] is not None) else 0.0)
                        his.append((ci[1] - d) if (d is not None and ci[1] is not None) else 0.0)
                    ax.errorbar([x + (k - 0.5) * 0.08 for x in xs], ys, yerr=[los, his],
                                marker="o" if mode == "reference_free" else "s",
                                color=SERIES[0] if mode == "reference_free" else SERIES[1],
                                lw=2, capsize=3, label=mode.replace("_", " "))
                ax.axhline(0.0, color=INK, lw=1.2, ls="--")
                ax.set_xticks(xs)
                ax.set_xticklabels([SHORT.get(j, j) for j in judges], color=INK, fontsize=9.5)
                ax.set_title(f"grading {target} outputs", color=INK, fontsize=10.5, loc="left")
                ax.grid(axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                _clean(ax)
            axes4[0].set_ylabel("balanced accuracy, seen minus unseen", color=INK, fontsize=10)
            axes4[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left")
            fig.suptitle("J4  Withholding the reference answer opens a membership gap that "
                         "judge scale does not close", color=INK, fontsize=11, x=0.02, ha="left")
            written.append(_save(fig, figs / "j4_judge_dissociation.png", plt))

            # J5 -- absolute reliability up the ladder, so the dissociation is read against
            # judges that are actually competent rather than against noise.
            fig, axes5 = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
            for ax, mode in zip(axes5, modes, strict=False):
                xs = list(range(len(judges)))
                for k, arm in enumerate(["seen", "unseen"]):
                    ys = []
                    for j in judges:
                        e = entry(j, "instruct", mode)
                        ba = ((e or {}).get("balanced_accuracy") or {}).get(arm) or {}
                        ys.append(ba.get("balanced", float("nan")))
                    ax.bar([x + (k - 0.5) * 0.34 for x in xs], ys, 0.34,
                           color=SERIES[k], label=f"verified {arm}")
                ax.axhline(0.5, color=INK, lw=1.2, ls="--")
                ax.text(len(judges) - 0.45, 0.515, "chance", fontsize=8, color=INK, ha="right")
                ax.set_xticks(xs)
                ax.set_xticklabels([SHORT.get(j, j) for j in judges], color=INK, fontsize=9.5)
                ax.set_ylim(0.45, 1.02)
                ax.set_title(mode.replace("_", " "), color=INK, fontsize=10.5, loc="left")
                ax.grid(axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                _clean(ax)
            axes5[0].set_ylabel("balanced accuracy, grading instruct", color=INK, fontsize=10)
            axes5[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
            fig.suptitle("J5  Judge competence up the ladder, by verified membership",
                         color=INK, fontsize=11, x=0.02, ha="left")
            written.append(_save(fig, figs / "j5_judge_ladder.png", plt))

        # J1 keeps the original per-target view, for the in-family judge only.
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
        for ax, mode in zip(axes, modes, strict=False):
            present = [t for t in tags if f"instella-3b-instruct__{t}__{mode}" in jr]
            xs = range(len(present))
            ba = [jr[f"instella-3b-instruct__{t}__{mode}"]["balanced_accuracy"] for t in present]
            ks = [(e["seen"] or {}).get("balanced", 0.0) for e in ba]
            ku = [(e["unseen"] or {}).get("balanced", 0.0) for e in ba]
            ax.bar([x - 0.18 for x in xs], ks, 0.36, color=SERIES[0], label="verified seen")
            ax.bar([x + 0.18 for x in xs], ku, 0.36, color=SERIES[1], label="verified unseen")
            # 0.5 is chance for a base-rate-independent binary metric. Without the line a
            # reader cannot see that most of these bars sit essentially on it.
            ax.axhline(0.5, color=INK, lw=1.2, ls="--")
            ax.text(len(present) - 0.45, 0.505, "chance", fontsize=8, color=INK, ha="right")
            for i, e in enumerate(ba):
                d, ci = e["delta_seen_minus_unseen"], e["delta_ci95"]
                if d is not None and ci[0] is not None:
                    mark = "*" if e["excludes_zero"] else ""
                    ax.annotate(f"Δ{d:+.3f}{mark}", (i, max(ks[i], ku[i]) + 0.012),
                                ha="center", fontsize=8, color=INK)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(present, color=INK, fontsize=9.5)
            ax.set_ylim(0.45, 0.72)
            ax.set_title(mode.replace("_", " "), color=INK, fontsize=10.5, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            _clean(ax)
        axes[0].set_ylabel("balanced accuracy vs exact-match truth", color=INK, fontsize=10)
        axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
        fig.suptitle("J1  A 3B judge is close to chance, and the seen gap survives the "
                     "reference-based control", color=INK, fontsize=11, x=0.02, ha="left")
        written.append(_save(fig, figs / "j1_judge_balanced_accuracy.png", plt))

        # J2 -- abstention and headline agreement. A judge that declines to commit is not a
        # judge that disagrees, and conflating the two would inflate apparent reliability.
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        keys = [f"instella-3b-instruct__{t}__{m}" for m in modes for t in tags
                if f"instella-3b-instruct__{t}__{m}" in jr]
        xs = range(len(keys))
        ax.bar(xs, [jr[k]["abstention_rate"] * 100 for k in keys], color=SERIES[4],
               label="abstention rate")
        ax.plot(list(xs), [jr[k]["accuracy_vs_truth"] * 100 for k in keys], marker="o",
                color=SERIES[0], lw=2, label="raw agreement with ground truth")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([k.split("__")[1] + "\n" + k.split("__")[2][:9] for k in keys],
                           color=INK, fontsize=8.5)
        ax.set_ylabel("percent", color=INK, fontsize=10)
        ax.set_title("J2  Abstention and raw agreement, reported separately",
                     color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        _clean(ax)
        written.append(_save(fig, figs / "j2_judge_abstention.png", plt))

        # J3 -- kappa across the four cells. Paraphrase robustness is a listed workshop topic,
        # and answer-preserving rewrites leave the gold answer intact, so any movement between
        # original and perturbed within an arm is judge fragility rather than task difficulty.
        cellnames = ["seen|original", "unseen|original", "seen|perturbed", "unseen|perturbed"]
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        width = 0.2
        for j, cell in enumerate(cellnames):
            vals = []
            for t in tags:
                e = jr.get(f"instella-3b-instruct__{t}__reference_free")
                v = (e or {}).get("cells", {}).get(cell, {}).get("balanced_accuracy")
                vals.append(v if v is not None else 0.5)
            ax.bar([i + (j - 1.5) * width for i in range(len(vals))], vals, width,
                   color=SERIES[j % len(SERIES)], label=cell)
        ax.axhline(0.5, color=INK, lw=1.2, ls="--")
        ax.set_ylim(0.45, 0.72)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, color=INK, fontsize=9.5)
        ax.set_ylabel("balanced accuracy", color=INK, fontsize=10)
        ax.set_title("J3  Judge reliability per cell, reference-free grading "
                     "(dashed line is chance)", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, ncol=4,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        _clean(ax)
        written.append(_save(fig, figs / "j3_judge_cells.png", plt))
    else:
        print("(no judge_reliability.json yet — J1-J3 skipped)")

    runs_vol.commit()
    subprocess.call(
        [sys.executable, "experiments/hf_sync.py", "push", "--repo",
         os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"),
         "--path", str(out), "--message", "supplementary and judge figures"]
    )
    for w in written:
        print(w)
    return written


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
