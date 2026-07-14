#!/usr/bin/env python3
"""Preflight the compute environment before downloading or loading a model.

Run this FIRST on any new Colab/host. It answers the one question that decided the
whole run: is there a usable CUDA GPU, or is torch a CPU-only build? Downloading the
6 GB Instella checkpoint only to discover you are on CPU wastes time and disk.

    python scripts/preflight.py                 # just report
    python scripts/preflight.py --require-gpu    # exit 1 if no CUDA GPU (for CI/gating)

Exit code is 0 when a CUDA GPU is available, 1 otherwise (or when --require-gpu and
no GPU). Prints `nvidia-smi` output when the driver is present.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess


def _nvidia_smi() -> None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        print("[nvidia-smi] not found on PATH -> no NVIDIA driver / GPU runtime.")
        return
    print("[nvidia-smi]")
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        print(out.stdout.strip() or "(no GPUs reported)")
        if out.returncode != 0 and out.stderr.strip():
            print(out.stderr.strip())
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"(nvidia-smi failed: {exc})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU / environment preflight.")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Exit non-zero if no CUDA GPU is available.")
    args = parser.parse_args(argv)

    _nvidia_smi()

    try:
        from instella_reasoning.evaluation import describe_device

        info = describe_device()
    except Exception as exc:  # noqa: BLE001
        print(f"[torch] could not import torch/instella_reasoning: {exc}")
        return 1

    print("\n[torch] " + ", ".join(f"{k}={v}" for k, v in info.items()))

    if info["cpu_only_build"]:
        print("\nWARNING: this is a CPU-only torch build (version ends in '+cpu').")
        print("  -> --load-in-4bit will be ignored and generation runs on CPU (slow).")
        print("  -> In Colab: Runtime > Change runtime type > GPU (T4), then reinstall.")
    if not info["cuda_available"]:
        print("\nNo CUDA GPU visible. For a real generation run, switch to a GPU runtime.")
        print("On CPU, only do a tiny load/smoke check: --limit 2 --max-new-tokens 64.")

    ok = info["cuda_available"]
    if args.require_gpu and not ok:
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
