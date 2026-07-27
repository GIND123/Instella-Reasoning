#!/usr/bin/env python3
"""Pre-run gate for the full-scale suite. Nothing launches until this is green.

A 15-GPU-hour run has exactly one expensive failure mode: it completes, and only then do
you discover that something upstream was wrong — a truncated model, a mislabelled variant
set, an HF token without write access, a run directory that silently skipped every stage
because a stale marker said it was done. Each of those has already happened once on this
project. This script checks all of them in a few minutes of CPU, plus an optional short
GPU smoke.

Checks, in order of how much they would cost if skipped:

  1  imports + package installed
  2  HF auth, and write access proven by an actual round-trip
  3  protected remote paths are not the target
  4  run directory is fresh (or its markers are consistent)
  5  every model in the tier resolves and is reachable
  6  every dataset is reachable and not script-only
  7  numeric variants are magnitude-neutral and the template yield is sane
  8  the four known-mislabelled pilot items are still rejected  (regression guard)
  9  disk headroom
 10  GPU smoke: each model generates a few items and clears the termination gate

Exit code 0 = safe to launch. Non-zero = do not spend the budget.

    python scripts/preflight_fullscale.py                 # checks 1-9
    python scripts/preflight_fullscale.py --smoke         # + the GPU smoke (10)
    python scripts/preflight_fullscale.py --out experiments/runs/fullscale-S250
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> str:
    _results.append((name, status, detail))
    icon = {PASS: "  ok ", FAIL: "FAIL ", WARN: "warn ", SKIP: "skip "}[status]
    tqdm.write(f"[{icon}] {name}" + (f" — {detail}" if detail else ""))
    return status


# -- 1. imports ------------------------------------------------------------------


def check_imports() -> str:
    try:
        import instella_reasoning  # noqa: F401
        from instella_reasoning import checkpoints, gsm_symbolic  # noqa: F401
        from instella_reasoning.analysis import figures, memorization, mixed_models  # noqa: F401
        from instella_reasoning.datasets import splits  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return record("package imports", FAIL, f"{type(exc).__name__}: {exc}")
    optional = []
    for name in ("datasets", "transformers", "sentence_transformers", "matplotlib", "scipy"):
        try:
            __import__(name)
        except ImportError:
            optional.append(name)
    if optional:
        return record(
            "package imports", WARN, f"missing optional: {', '.join(optional)} (some stages will skip)"
        )
    return record("package imports", PASS)


# -- 2/3. Hugging Face auth, write round-trip, protected paths -------------------


def check_hf(repo: str, out_dir: str) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return record("HF token", FAIL, "set HF_TOKEN (Colab: userdata.get('hf'), WRITE scope)")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return record("HF token", FAIL, "huggingface_hub not installed (pip install -e '.[hf]')")

    api = HfApi()
    try:
        who = api.whoami(token=token)
        record("HF auth", PASS, f"user {who.get('name')}")
    except Exception as exc:  # noqa: BLE001
        return record("HF auth", FAIL, f"{type(exc).__name__}: {exc}")

    sys.path.insert(0, str(REPO / "experiments"))
    from hf_sync import PROTECTED_PREFIXES, _is_protected

    if _is_protected(out_dir):
        return record(
            "protected paths",
            FAIL,
            f"{out_dir!r} is protected ({', '.join(PROTECTED_PREFIXES)}); choose a new OUT",
        )
    record("protected paths", PASS, f"{out_dir} does not collide with kept runs")

    # Prove write access with a real upload rather than trusting the token's advertised
    # scope — a read-only token authenticates fine and then fails 14 hours in.
    try:
        from huggingface_hub import create_repo, upload_file

        create_repo(repo, repo_type="dataset", token=token, exist_ok=True, private=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("preflight write probe\n")
            probe = handle.name
        upload_file(
            path_or_fileobj=probe,
            path_in_repo=f"{out_dir}/markers/_preflight_write_probe.txt",
            repo_id=repo,
            repo_type="dataset",
            token=token,
            commit_message="[preflight] write probe",
        )
        os.unlink(probe)
        return record("HF write access", PASS, f"round-trip to {repo} succeeded")
    except Exception as exc:  # noqa: BLE001
        return record("HF write access", FAIL, f"{type(exc).__name__}: {exc}")


# -- 4. run directory ------------------------------------------------------------


def check_run_dir(out_dir: str) -> str:
    path = REPO / out_dir
    if not path.exists():
        return record("run directory", PASS, f"{out_dir} is new")
    markers = list((path / "markers").glob("*")) if (path / "markers").exists() else []
    scores = list((path / "scores").glob("*.jsonl")) if (path / "scores").exists() else []
    if markers or scores:
        return record(
            "run directory",
            WARN,
            f"{out_dir} already has {len(markers)} stage markers and {len(scores)} score files; "
            "the run will RESUME and skip those. Delete it for a clean start.",
        )
    return record("run directory", PASS, f"{out_dir} exists but is empty")


# -- 5. models -------------------------------------------------------------------


def check_models(suite_tier: int) -> str:
    from instella_reasoning.checkpoints import tier

    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
    except ImportError:
        return record("models reachable", SKIP, "huggingface_hub missing")

    problems = []
    for ckpt in tier(suite_tier):
        try:
            api.model_info(ckpt.hf_id, token=token)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{ckpt.tag} ({ckpt.hf_id}): {type(exc).__name__}")
    if problems:
        return record("models reachable", FAIL, "; ".join(problems))
    tags = ", ".join(c.tag for c in tier(suite_tier))
    return record("models reachable", PASS, tags)


# -- 6. datasets -----------------------------------------------------------------


def check_datasets() -> str:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
    except ImportError:
        return record("datasets reachable", SKIP, "huggingface_hub missing")

    token = os.environ.get("HF_TOKEN")
    needed = ["openai/gsm8k", "apple/GSM-Symbolic", "amd/Instella-GSM8K-synthetic"]
    problems, script_only = [], []
    for repo in needed:
        try:
            files = api.list_repo_files(repo, repo_type="dataset", token=token)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{repo}: {type(exc).__name__}")
            continue
        # A repo whose only loader is a .py script is unusable under datasets>=3.
        if any(f.endswith(".py") for f in files) and not any(
            f.endswith((".parquet", ".jsonl", ".json")) for f in files
        ):
            script_only.append(repo)
    if problems:
        return record("datasets reachable", FAIL, "; ".join(problems))
    if script_only:
        return record("datasets reachable", FAIL, f"script-only (datasets>=3 cannot load): {script_only}")
    return record("datasets reachable", PASS, ", ".join(needed))


# -- 7/8. variant generator ------------------------------------------------------


_PILOT_MISLABELLED = {
    # Hand-adjudicated in docs/AUDIT_2026-07-26.md: value-based substitution conflated a
    # question quantity with a structural constant ("half" = 2 in "2/2"), producing a
    # wrong gold answer. These must never template again.
    "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?": (
        "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3",
        "3",
    ),
    "Cynthia eats one serving of ice cream every night. She buys cartons of ice cream with 15 servings of ice cream per carton at a cost of $4.00 per carton. After 60 days, how much will she spend on ice cream?": (
        "Each container has 15 servings and she eats 1 a night so after 60 days she needs 60/15 = 4 containers\nIf each carton costs $4.00 and she needs 4 containers then it will cost her 4*4 = $<<4*4=16>>16.00\n#### 16",
        "16",
    ),
}


def check_variant_generator() -> str:
    import random

    from instella_reasoning.gsm_symbolic import magnitude_report, make_numeric_variants
    from instella_reasoning.records import BenchmarkItem

    # 8. regression guard on the known-bad pilot items
    leaked = []
    for prompt, (rationale, answer) in _PILOT_MISLABELLED.items():
        item = BenchmarkItem(
            id="probe", prompt=prompt, answer=answer, parent_id="probe",
            metadata={"benchmark": "gsm8k", "rationale": rationale},
        )
        if make_numeric_variants(item, k=2, rng=random.Random(0)).n_templated:
            leaked.append(prompt[:44])
    if leaked:
        record("mislabel regression guard", FAIL, f"re-templated known-bad items: {leaked}")
    else:
        record("mislabel regression guard", PASS, "known-mislabelled pilot items still rejected")

    # 7. magnitude neutrality + yield on whatever real data is available
    pilot = REPO / "experiments/runs/reliability-B120-K5-clean/variants/gsm8k_variants.jsonl"
    if not pilot.exists():
        return record("variant magnitude + yield", SKIP, "no pilot data to sample")
    from instella_reasoning.records import read_benchmark

    originals = [i for i in read_benchmark(pilot) if i.variant_type == "original"]
    produced, templated = [], 0
    for item in originals:
        result = make_numeric_variants(item, k=2, rng=random.Random(f"pf:{item.id}"))
        templated += result.n_templated
        produced.extend(result.items)
    report = magnitude_report(produced)
    yield_rate = templated / len(originals) if originals else 0.0
    detail = (
        f"yield {yield_rate:.0%} ({templated}/{len(originals)}), "
        f"median answer ratio {report.median_ratio:.2f}x, "
        f"{report.share_larger:.0%} larger"
    )
    if not report.in_band:
        return record("variant magnitude + yield", FAIL, detail + " — NOT magnitude-neutral")
    if yield_rate < 0.25:
        return record("variant magnitude + yield", WARN, detail + " — low yield, check rejections")
    return record("variant magnitude + yield", PASS, detail)


# -- 9. disk ---------------------------------------------------------------------


def check_disk(min_gb: float = 12.0) -> str:
    free_gb = shutil.disk_usage(REPO).free / 1e9
    detail = f"{free_gb:.1f} GB free"
    if free_gb < min_gb:
        return record("disk headroom", FAIL, detail + f" (need ~{min_gb:.0f} GB for corpus + weights)")
    return record("disk headroom", PASS, detail)


# -- 10. GPU smoke ---------------------------------------------------------------


def check_gpu_smoke(suite_tier: int, n_items: int) -> str:
    try:
        import torch
    except ImportError:
        return record("GPU smoke", SKIP, "torch not installed")
    if not torch.cuda.is_available():
        return record("GPU smoke", WARN, "no CUDA device — the suite will be unusably slow")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    record("GPU present", PASS, f"{name}, {total:.0f} GB")

    from instella_reasoning.checkpoints import tier
    from instella_reasoning.datasets import load_benchmark_to_jsonl
    from instella_reasoning.evaluation import generate_with_transformers, termination_report
    from instella_reasoning.records import read_benchmark

    with tempfile.TemporaryDirectory() as tmp:
        bench_path = Path(tmp) / "smoke.jsonl"
        try:
            load_benchmark_to_jsonl("gsm8k", bench_path, limit=n_items)
        except Exception as exc:  # noqa: BLE001
            return record("GPU smoke", FAIL, f"benchmark load failed: {exc}")
        items = read_benchmark(bench_path)

        failures = []
        checkpoints = tier(suite_tier)
        with tqdm(
            checkpoints,
            desc="GPU smoke",
            unit="checkpoint",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for ckpt in progress:
                progress.set_postfix_str(f"{ckpt.tag}: load + generate", refresh=True)
                out = Path(tmp) / f"{ckpt.tag}.jsonl"
                try:
                    if ckpt.local_assembly:
                        assembly = REPO / ckpt.local_assembly
                        progress.set_postfix_str(f"{ckpt.tag}: assemble checkpoint", refresh=True)
                        subprocess.run(
                            [
                                sys.executable,
                                str(REPO / "experiments/fetch_instella_math_hf.py"),
                                "--repo",
                                ckpt.hf_id,
                                "--out",
                                str(assembly),
                            ],
                            cwd=REPO,
                            check=True,
                        )
                        progress.set_postfix_str(f"{ckpt.tag}: load + generate", refresh=True)
                    rows = generate_with_transformers(
                        benchmark=items,
                        model_name_or_path=ckpt.load_path,
                        output_path=out,
                        max_new_tokens=ckpt.max_new_tokens,
                        batch_size=min(4, n_items),
                        n_shot=ckpt.n_shot,
                        use_chat_template=not ckpt.is_base,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{ckpt.tag}: {type(exc).__name__}: {str(exc)[:70]}")
                    continue
                report = termination_report(rows)
                status = PASS if report.passes else FAIL
                record(
                    f"  smoke {ckpt.tag}",
                    status,
                    f"completed {report.termination_rate:.0%}, marker {report.marker_rate:.0%}, "
                    f"median {report.median_chars} chars",
                )
                if not report.passes:
                    failures.append(
                        f"{ckpt.tag}: only {report.termination_rate:.0%} completed at "
                        f"{ckpt.max_new_tokens} tokens — raise its budget in checkpoints.py"
                    )
        if failures:
            return record("GPU smoke", FAIL, "; ".join(failures))
    return record("GPU smoke", PASS, "all checkpoints generate and terminate")


# -- main ------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.environ.get("OUT", "experiments/runs/fullscale-S250"))
    parser.add_argument("--repo", default=os.environ.get("HF_RESULTS_REPO", "GOVINDFROM/Instella-Reasoning"))
    parser.add_argument("--tier", type=int, default=int(os.environ.get("SUITE_TIER", "1")))
    parser.add_argument("--smoke", action="store_true", help="also run the short GPU generation smoke")
    parser.add_argument("--smoke-items", type=int, default=4)
    parser.add_argument("--json", default=None, help="write the report as JSON")
    args = parser.parse_args()

    print("=" * 74)
    print(" PREFLIGHT — full-scale memorisation suite")
    print(f" run dir: {args.out}   repo: {args.repo}   tier: {args.tier}")
    print("=" * 74)

    checks = (
        ("package imports", check_imports, ()),
        ("Hugging Face access", check_hf, (args.repo, args.out)),
        ("run directory", check_run_dir, (args.out,)),
        ("model registry", check_models, (args.tier,)),
        ("datasets", check_datasets, ()),
        ("variant generator", check_variant_generator, ()),
        ("disk headroom", check_disk, ()),
    )
    with tqdm(
        total=len(checks) + 1,
        desc="Preflight",
        unit="check",
        dynamic_ncols=True,
        leave=True,
    ) as progress:
        for label, check, check_args in checks:
            progress.set_postfix_str(label, refresh=True)
            check(*check_args)
            progress.update()

        progress.set_postfix_str("GPU smoke", refresh=True)
        if args.smoke:
            check_gpu_smoke(args.tier, args.smoke_items)
        else:
            record("GPU smoke", SKIP, "pass --smoke to verify generation before the real run")
        progress.update()

    fails = [r for r in _results if r[1] == FAIL]
    warns = [r for r in _results if r[1] == WARN]
    print("=" * 74)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps([{"check": n, "status": s, "detail": d} for n, s, d in _results], indent=2),
            encoding="utf-8",
        )
    if fails:
        print(f" NOT SAFE TO LAUNCH — {len(fails)} check(s) failed:")
        for name, _, detail in fails:
            print(f"   - {name}: {detail}")
        return 1
    print(f" SAFE TO LAUNCH ({len(warns)} warning(s))")
    if not args.smoke:
        print(" Strongly recommended before the real run:  --smoke")
    print("   bash experiments/run_fullscale_suite.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
