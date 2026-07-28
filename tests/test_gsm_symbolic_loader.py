from instella_reasoning.datasets import loaders
from instella_reasoning.records import read_jsonl


def _dataset():
    return [
        {
            "id": 7,
            "instance": 0,
            "original_question": "Original question?",
            "original_answer": "Work. #### 3",
            "original_id": 42,
            "question": "Changed question?",
            "answer": "Work. #### 4",
            "canary": "test-canary",
        }
    ]


def test_gsm_symbolic_configs_have_disjoint_ids(tmp_path, monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return _dataset()

    monkeypatch.setattr(loaders, "_require_datasets", lambda: fake_load_dataset)

    paths = []
    for benchmark in ("gsm_symbolic", "gsm_symbolic_p1"):
        path = tmp_path / f"{benchmark}.jsonl"
        assert loaders.load_benchmark_to_jsonl(benchmark, path) == 2
        paths.append(path)

    main = list(read_jsonl(paths[0]))
    p1 = list(read_jsonl(paths[1]))
    main_ids = {row["id"] for row in main}
    p1_ids = {row["id"] for row in p1}

    assert main_ids.isdisjoint(p1_ids)
    assert {row["metadata"]["benchmark_config"] for row in main} == {"main"}
    assert {row["metadata"]["benchmark_config"] for row in p1} == {"p1"}
