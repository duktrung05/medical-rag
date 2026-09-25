"""The dense benchmark must send the same normalized text as production retrieval."""

import json

from scripts import run_dense_baseline


def test_dense_baseline_adapts_queries_before_search(tmp_path, monkeypatch):
    split = tmp_path / "dev"
    split.mkdir()
    (split / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","doc_id":"d1","chunk_index":0,"language":"vi","text":"đau tim"}\n',
        encoding="utf-8",
    )
    (split / "queries.jsonl").write_text(
        '{"id":"q1","query":"ĐAU   TIM??"}\n', encoding="utf-8"
    )
    (split / "ground_truth.jsonl").write_text(
        '{"id":"q1","relevant_docs":["d1"],"relevant_chunks":["c1"]}\n',
        encoding="utf-8",
    )
    index = tmp_path / "dense"
    index.mkdir()
    (index / "manifest.json").write_text("{}", encoding="utf-8")
    (index / "passage_embeddings.npy").write_bytes(b"fixture")
    config = tmp_path / "config.yaml"
    config.write_text(
        "experiment_name: fixture\nbackend: dense\ncorpus: dev/chunks.jsonl\n"
        "retrieval:\n  dense:\n    enabled: true\n    model_name: fixture-model\n"
        "    index_path: dense\n",
        encoding="utf-8",
    )

    searched = []

    class FakeDenseRetriever:
        model_name = "fixture-model"
        model_revision = "f" * 40
        tokenizer_revision_resolved = "f" * 40
        device = "cpu"

        def __init__(self, **kwargs):
            pass

        def search_batch(self, queries, top_k):
            searched.extend(queries)
            return [[("c1", 1.0)] for _ in queries]

    monkeypatch.setattr(run_dense_baseline, "DenseRetriever", FakeDenseRetriever)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dense_baseline",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    run_dense_baseline.main()

    assert searched == ["đau tim?"]
    assert (
        json.loads((tmp_path / "out" / "metrics.json").read_text())["1"]["macro_f2"]
        == 1.0
    )
