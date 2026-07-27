#!/usr/bin/env python
"""Sync a run directory to/from a Hugging Face dataset repo for periodic saving + resume.

The suite writes everything for one run under a single directory (e.g.
``experiments/runs/fullscale-S250/``). This script mirrors that directory to a HF dataset
repo so that:

- **periodic save** — ``push`` after each (model, stage) uploads new scores/generations,
  so a reclaimed Colab VM never loses completed work;
- **cross-machine resume** — ``pull`` at the start downloads whatever a previous session
  already finished, and the suite's per-stage marker files let it skip those pieces — no
  GPU hours are re-spent.

The remote layout mirrors the local path, so several runs coexist without collision.

Two policies that exist because of specific, expensive failures
--------------------------------------------------------------

**Generations are uploaded by default.** They used to be excluded as "bulky and
re-derivable". They are not re-derivable without re-running the GPU: when a scoring bug
was found after a completed run, the raw text was gone and the entire run had to be
repeated. At ~15k generations the upload is a couple of hundred MB — far cheaper than the
GPU hours it insures. Set ``HF_INCLUDE_GENERATIONS=0`` to opt out.

**Existing runs are protected.** ``PROTECTED_PREFIXES`` lists remote paths that this
script refuses to write, so a new run can never overwrite results that are being kept for
reference. Resume still works normally — it only ever writes inside its own run directory.

Auth: reads the token from ``HF_TOKEN`` (set it from your Colab secret, e.g.
``os.environ["HF_TOKEN"] = userdata.get("hf")``). Every failure is non-fatal and printed
as a warning — a sync hiccup must never kill a multi-hour GPU run.

Usage:
    python experiments/hf_sync.py pull   --repo GOVINDFROM/Instella-Reasoning --path experiments/runs/<name>
    python experiments/hf_sync.py push   --repo GOVINDFROM/Instella-Reasoning --path experiments/runs/<name> --message "scores instruct"
    python experiments/hf_sync.py status --repo GOVINDFROM/Instella-Reasoning
"""

from __future__ import annotations

import argparse
import os
import sys

#: Remote paths this script will never write. The B120/B15 pilot runs are kept as the
#: reference the audit was written against; a new run must land in a new directory.
PROTECTED_PREFIXES = (
    "experiments/runs/reliability-B120-K5",
    "experiments/runs/reliability-B120-K5-clean",
    "experiments/runs/reliability-B15-K5",
)


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _warn(msg: str) -> int:
    print(f"[hf_sync] WARNING: {msg}", file=sys.stderr)
    return 0  # non-fatal by design


def _normalize(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _is_protected(path: str) -> bool:
    norm = _normalize(path)
    return any(norm == p or norm.startswith(p + "/") for p in PROTECTED_PREFIXES)


def _include_generations(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("HF_INCLUDE_GENERATIONS", "1") != "0"


def pull(repo: str, path: str, repo_type: str) -> int:
    """Download the run directory (if it exists remotely) into the local tree."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        return _warn("huggingface_hub not installed; skipping pull (install the .[hf] extra).")

    norm = _normalize(path)
    try:
        snapshot_download(
            repo_id=repo,
            repo_type=repo_type,
            local_dir=".",
            allow_patterns=[f"{norm}/**"],
            token=_token(),
        )
        print(f"[hf_sync] pulled {repo}:{norm} -> ./{norm}")
    except RepositoryNotFoundError:
        print(f"[hf_sync] {repo} not found yet (first run) — nothing to resume.")
    except Exception as exc:  # noqa: BLE001 - sync must never crash the GPU run
        return _warn(f"pull failed ({exc}); continuing without remote resume.")
    return 0


def push(repo: str, path: str, repo_type: str, message: str, include_generations: bool) -> int:
    """Create the repo if needed and upload the run directory."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        return _warn("huggingface_hub not installed; skipping push (install the .[hf] extra).")

    norm = _normalize(path)
    if _is_protected(norm):
        return _warn(
            f"refusing to write protected path {norm!r} (kept for reference). "
            "Use a new run directory."
        )
    if not os.path.isdir(norm):
        return _warn(f"local path {norm} does not exist; nothing to push.")
    token = _token()
    if not token:
        return _warn("no HF_TOKEN in env; skipping push (set it from your 'hf' Colab secret).")

    ignore = None if include_generations else ["generations/**", "**/generations/**"]
    try:
        # Experiment artifacts can include model outputs and licensed benchmark material.
        # Create the shared results repository privately by default; an owner can make it
        # public later as an explicit publication decision.
        create_repo(repo, repo_type=repo_type, token=token, exist_ok=True, private=True)
        HfApi().upload_folder(
            folder_path=norm,
            path_in_repo=norm,
            repo_id=repo,
            repo_type=repo_type,
            token=token,
            ignore_patterns=ignore,
            commit_message=f"[fullscale] {message}",
        )
        gen = "with" if include_generations else "without"
        print(f"[hf_sync] pushed ./{norm} -> {repo}:{norm} ({gen} generations) — {message}")
    except Exception as exc:  # noqa: BLE001 - sync must never crash the GPU run
        return _warn(f"push failed ({exc}); results remain local and will retry next checkpoint.")
    return 0


def status(repo: str, repo_type: str) -> int:
    """List what is already stored remotely, grouped by run — the resume overview."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return _warn("huggingface_hub not installed.")
    try:
        files = HfApi().list_repo_files(repo, repo_type=repo_type, token=_token())
    except Exception as exc:  # noqa: BLE001
        return _warn(f"status failed ({exc}).")

    runs: dict[str, int] = {}
    for name in files:
        parts = name.split("/")
        if len(parts) >= 3 and parts[0] == "experiments" and parts[1] == "runs":
            runs["/".join(parts[:3])] = runs.get("/".join(parts[:3]), 0) + 1
    print(f"[hf_sync] {repo} — {len(files)} files across {len(runs)} run(s):")
    for run, count in sorted(runs.items()):
        mark = "  (protected)" if _is_protected(run) else ""
        print(f"   {count:5d} files  {run}{mark}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pull", "push", "status"])
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. GOVINDFROM/Instella-Reasoning")
    parser.add_argument("--path", default=None, help="local run dir; mirrored as the remote path")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--message", default="checkpoint", help="commit message (push only)")
    parser.add_argument(
        "--include-generations",
        action="store_true",
        help="force-upload raw generations/ (default on; disable with HF_INCLUDE_GENERATIONS=0)",
    )
    args = parser.parse_args()

    if args.action == "status":
        return status(args.repo, args.repo_type)
    if not args.path:
        parser.error("--path is required for pull/push")
    if args.action == "pull":
        return pull(args.repo, args.path, args.repo_type)
    return push(
        args.repo,
        args.path,
        args.repo_type,
        args.message,
        _include_generations(args.include_generations),
    )


if __name__ == "__main__":
    raise SystemExit(main())
