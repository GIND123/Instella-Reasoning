"""Run-provenance manifest — make every result reproducible and auditable.

A reviewer's first question about a headline number is *"which model checkpoint, which
benchmark revision, which package versions produced this?"* Without a recorded answer the
result is not reproducible. :func:`collect_provenance` captures that context — git commit +
dirty flag, Python/OS, the versions of the scientific packages actually installed, and the
run config — and :func:`write_manifest` drops it beside the outputs as ``manifest.json``.

Everything is best-effort and dependency-free: a missing git binary, a non-repo working
directory, or an uninstalled optional package degrades to ``None`` rather than raising, so
the manifest is always written.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# Packages whose versions materially affect results (or explain a bad run, e.g. a
# transformers 5.x that breaks Instella's remote code). Absent ones record as None.
_TRACKED_PACKAGES = (
    "instella-reasoning",
    "transformers",
    "torch",
    "datasets",
    "accelerate",
    "sentence-transformers",
    "faiss-cpu",
    "numpy",
    "scipy",
    "sympy",
    "bitsandbytes",
)


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_state() -> dict[str, Any]:
    """Return {commit, branch, dirty} for the working tree, or Nones outside a repo."""
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(packages: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each tracked package (None if not installed)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_provenance(
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full provenance record for a run."""
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if config is not None:
        manifest["config"] = config
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect provenance and write it as pretty JSON; return the manifest dict."""
    manifest = collect_provenance(config=config, extra=extra)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
