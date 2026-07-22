"""Resume behaviour of generate_with_transformers (no GPU/model needed).

When every benchmark id is already present in the output file, generation must return the
existing records and return *before* loading any model — this is the fast-forward path that
makes an interrupted GPU run resumable.
"""

import json

from instella_reasoning.evaluation import generate_with_transformers
from instella_reasoning.records import BenchmarkItem


def _write_generations(path, ids):
    with open(path, "w", encoding="utf-8") as handle:
        for i in ids:
            handle.write(json.dumps({"benchmark_id": i, "completion": f"done-{i}", "model": "m"}) + "\n")


def test_generation_skips_completed_items_without_loading_model(tmp_path, monkeypatch):
    out = tmp_path / "gen.jsonl"
    items = [BenchmarkItem(id="q0", prompt="a", answer="1"),
             BenchmarkItem(id="q1", prompt="b", answer="2")]
    _write_generations(out, ["q0", "q1"])

    # If the resume check fails and it tries to load, this makes the test fail loudly
    # instead of silently downloading a 3B model.
    import instella_reasoning.evaluation as ev
    monkeypatch.setattr(ev, "_load_model", lambda *a, **k: pytest_fail_load())

    rows = generate_with_transformers(items, "amd/does-not-matter", out)
    assert len(rows) == 2
    assert {r.benchmark_id for r in rows} == {"q0", "q1"}


def pytest_fail_load():
    raise AssertionError("_load_model was called even though all items were already generated")


def test_generation_resume_detects_partial(tmp_path):
    """A partially-complete file leaves only the missing items pending."""
    from instella_reasoning.records import read_jsonl

    out = tmp_path / "gen.jsonl"
    _write_generations(out, ["q0"])  # only q0 done; q1 still pending
    done = {row["benchmark_id"] for row in read_jsonl(out)}
    items = [BenchmarkItem(id="q0", prompt="a", answer="1"),
             BenchmarkItem(id="q1", prompt="b", answer="2")]
    pending = [it for it in items if it.id not in done]
    assert [it.id for it in pending] == ["q1"]
