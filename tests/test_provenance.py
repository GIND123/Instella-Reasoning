from __future__ import annotations

import json
from pathlib import Path

from instella_reasoning.provenance import (
    collect_provenance,
    git_state,
    package_versions,
    write_manifest,
)


def test_collect_provenance_has_required_fields() -> None:
    manifest = collect_provenance(config={"model": "amd/Instella-3B"}, extra={"note": "smoke"})
    assert manifest["manifest_version"] == 1
    assert "generated_at_utc" in manifest
    assert set(manifest["git"]) == {"commit", "branch", "dirty"}
    assert manifest["python_version"]
    assert manifest["config"] == {"model": "amd/Instella-3B"}
    assert manifest["extra"] == {"note": "smoke"}


def test_package_versions_returns_entry_per_tracked_package() -> None:
    versions = package_versions(("instella-reasoning", "definitely-not-installed-xyz"))
    assert "instella-reasoning" in versions
    # Own package is installed (editable) in the test env; unknown one is None.
    assert versions["definitely-not-installed-xyz"] is None


def test_git_state_is_graceful() -> None:
    state = git_state()
    # In this repo commit is a hash; outside a repo it would be None — either is valid.
    assert set(state) == {"commit", "branch", "dirty"}


def test_write_manifest_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    written = write_manifest(path, config={"seed": 6198})
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["config"] == {"seed": 6198}
