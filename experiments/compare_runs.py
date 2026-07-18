#!/usr/bin/env python3
"""Compare two CPU-suite run directories and report whether they agree.

Usage:
    python experiments/compare_runs.py <reference_run_dir> <new_run_dir>

Checks, per benchmark:
  - variant validation: n_variants, answer-preservation counts/rate, numeric counts
  - contamination: hit-row count and C/PC label distribution
  - attribution: profile count and mean top-1 cosine (tolerance 1e-3)

Exit code 0 = runs agree, 1 = differences found. Stdlib only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

TOL = 1e-3
diffs: list[str] = []
oks: list[str] = []


def note(ok: bool, message: str) -> None:
    (oks if ok else diffs).append(message)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare_validations(ref: Path, new: Path) -> None:
    keys = ("n_variants", "n_answer_preserving", "n_preserved_ok",
            "answer_preservation_rate", "n_answer_changing", "n_degenerate_text")
    for ref_file in sorted((ref / "variants").glob("*_validation.json")):
        bench = ref_file.name.replace("_validation.json", "")
        new_file = new / "variants" / ref_file.name
        if not new_file.exists():
            note(False, f"[variants/{bench}] missing in new run")
            continue
        a, b = json.loads(ref_file.read_text()), json.loads(new_file.read_text())
        mismatched = [k for k in keys if a.get(k) != b.get(k)]
        if mismatched:
            detail = ", ".join(f"{k}: {a.get(k)} != {b.get(k)}" for k in mismatched)
            note(False, f"[variants/{bench}] {detail}")
        else:
            note(True, f"[variants/{bench}] identical ({a['n_variants']} variants, rate {a['answer_preservation_rate']})")


def compare_contamination(ref: Path, new: Path) -> None:
    for ref_file in sorted((ref / "contamination").glob("*_contam.jsonl")):
        bench = ref_file.name.replace("_contam.jsonl", "")
        new_file = new / "contamination" / ref_file.name
        if not new_file.exists():
            note(False, f"[contamination/{bench}] missing in new run")
            continue
        a, b = load_jsonl(ref_file), load_jsonl(new_file)
        la, lb = Counter(r.get("label") for r in a), Counter(r.get("label") for r in b)
        if len(a) != len(b) or la != lb:
            note(False, f"[contamination/{bench}] rows {len(a)} vs {len(b)}, labels {dict(la)} vs {dict(lb)}")
        else:
            note(True, f"[contamination/{bench}] identical ({len(a)} hit-rows, labels {dict(la) or 'none'})")


def compare_attribution(ref: Path, new: Path) -> None:
    for ref_file in sorted((ref / "attribution").glob("*_attrib.jsonl")):
        bench = ref_file.name.replace("_attrib.jsonl", "")
        new_file = new / "attribution" / ref_file.name
        if not new_file.exists():
            note(False, f"[attribution/{bench}] missing in new run")
            continue
        a, b = load_jsonl(ref_file), load_jsonl(new_file)
        if len(a) != len(b):
            note(False, f"[attribution/{bench}] profile count {len(a)} vs {len(b)}")
            continue
        mean_a = sum(r.get("top1_cosine", 0.0) for r in a) / max(len(a), 1)
        mean_b = sum(r.get("top1_cosine", 0.0) for r in b) / max(len(b), 1)
        if abs(mean_a - mean_b) > TOL:
            note(False, f"[attribution/{bench}] mean top1_cosine {mean_a:.4f} vs {mean_b:.4f} (tol {TOL})")
        else:
            note(True, f"[attribution/{bench}] {len(a)} profiles, mean top1_cosine {mean_a:.4f} ≈ {mean_b:.4f}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref, new = Path(sys.argv[1]), Path(sys.argv[2])
    for path in (ref, new):
        if not path.is_dir():
            print(f"error: {path} is not a directory")
            return 2
    compare_validations(ref, new)
    compare_contamination(ref, new)
    compare_attribution(ref, new)

    print(f"Comparing\n  reference: {ref}\n  new:       {new}\n")
    for line in oks:
        print(f"  OK   {line}")
    for line in diffs:
        print(f"  DIFF {line}")
    print(f"\n{len(oks)} agreements, {len(diffs)} differences")
    if diffs:
        print("Result: RUNS DIFFER — inspect the DIFF lines above.")
        return 1
    print("Result: runs agree — everything looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
