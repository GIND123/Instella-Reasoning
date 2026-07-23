#!/usr/bin/env python
"""Sync a run directory to/from a Hugging Face dataset repo for periodic saving + resume.

The GPU suite writes everything for one run under a single directory (e.g.
``experiments/runs/reliability-B120-K5/``). This script mirrors that directory to a HF
dataset repo so that:

- **periodic save** — ``push`` after each (model, benchmark) uploads the new scores/atlas,
  so a reclaimed Colab VM never loses completed work;
- **cross-machine resume** — ``pull`` at the start of a run downloads whatever a previous
  session (or a teammate) already finished, and the suite's per-benchmark score files are
  its completion markers, so generation resumes exactly where it stopped — no GPU hours
  are re-spent.

The remote layout mirrors the local path: a run at ``experiments/runs/<name>`` lives at
``experiments/runs/<name>`` inside the dataset repo, so pull/push are symmetric and several
runs coexist without collision.

Auth: reads the token from ``HF_TOKEN`` (set it from your Colab secret, e.g.
``os.environ["HF_TOKEN"] = userdata.get("hf")``). Every failure is non-fatal and printed as
a warning — a sync hiccup must never kill a multi-hour GPU run.

Usage:
    python experiments/hf_sync.py pull --repo GOVINDFROM/Instella-Reasoning --path experiments/runs/<name>
    python experiments/hf_sync.py push --repo GOVINDFROM/Instella-Reasoning --path experiments/runs/<name> --message "scores instruct/gsm8k"
"""

from __future__ import annotations

import argparse
import os
import sys

# Raw model generations are bulky and fully re-derivable from scores; scores are the resume
# markers. Exclude generations by default to keep uploads fast (override with --include-generations).
_DEFAULT_IGNORE = ["generations/**", "**/generations/**"]


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _warn(msg: str) -> int:
    print(f"[hf_sync] WARNING: {msg}", file=sys.stderr)
    return 0  # non-fatal by design


def pull(repo: str, path: str, repo_type: str) -> int:
    """Download the run directory (if it exists remotely) into the local tree."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        return _warn("huggingface_hub not installed; skipping pull (install the .[hf] extra).")

    norm = path.replace("\\", "/").rstrip("/")
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
    """Create the repo if needed and upload the run directory (minus bulky generations)."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        return _warn("huggingface_hub not installed; skipping push (install the .[hf] extra).")

    norm = path.replace("\\", "/").rstrip("/")
    if not os.path.isdir(norm):
        return _warn(f"local path {norm} does not exist; nothing to push.")
    token = _token()
    if not token:
        return _warn("no HF_TOKEN in env; skipping push (set it from your 'hf' Colab secret).")

    ignore = None if include_generations else list(_DEFAULT_IGNORE)
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
            commit_message=f"[reliability-suite] {message}",
        )
        print(f"[hf_sync] pushed ./{norm} -> {repo}:{norm} ({message})")
    except Exception as exc:  # noqa: BLE001 - sync must never crash the GPU run
        return _warn(f"push failed ({exc}); results remain local and will retry next checkpoint.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pull", "push"])
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. GOVINDFROM/Instella-Reasoning")
    parser.add_argument("--path", required=True, help="local run dir; mirrored as the remote path")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--message", default="checkpoint", help="commit message (push only)")
    parser.add_argument(
        "--include-generations",
        action="store_true",
        help="also upload raw generations/ (bulky; scores alone are enough to resume)",
    )
    args = parser.parse_args()

    if args.action == "pull":
        return pull(args.repo, args.path, args.repo_type)
    return push(args.repo, args.path, args.repo_type, args.message, args.include_generations)


if __name__ == "__main__":
    raise SystemExit(main())
