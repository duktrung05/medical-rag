"""End-to-end CLI fixtures for every selectable retrieval backend."""

import json

import pytest

import scripts.retrieve as retrieve_script
from src.config import PipelineConfig, RuntimeConfig, load_config
from src.config import SparseRetrievalConfig
from src.data.loader import DataLoader
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.demo import DemoRetriever
from src.retrieval.hybrid import HybridRetriever
from src.reranking.bge_reranker import BGEReranker
from src.service import build_pipeline


def _write_fixture_files(tmp_path, backend):
    selected_backend = "hybrid" if backend == "hybrid-rerank" else backend
    (tmp_path / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","doc_id":"d1","chunk_index":0,"language":"vi","text":"tim mach"}\n',
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        '{"id":"q1","query":"tim mach"}\n', encoding="utf-8"
    )
    if backend == "demo":
        config_text = "backend: demo\ncorpus: chunks.jsonl\nmax_chunks: 5\n"
    else:
        sparse = selected_backend in {"sparse", "hybrid"}
        dense = selected_backend in {"dense", "hybrid"}
        if sparse:
            sparse_index = tmp_path / "sparse-index"
            sparse_index.mkdir()
            BM25Retriever.build_index(
                DataLoader.load_chunks(tmp_path / "chunks.jsonl"),
                sparse_index,
                SparseRetrievalConfig(enabled=True, index_path=sparse_index),
            )
        if dense:
            (tmp_path / "dense-index").mkdir()
        sparse_config = (
            "    enabled: true\n    index_path: sparse-index\n"
            if sparse
            else "    enabled: false\n"
        )
        dense_config = (
            "    enabled: true\n    model_name: fixture-model\n    index_path: dense-index\n"
            if dense
            else "    enabled: false\n"
        )
        reranker_config = (
            "reranker:\n  enabled: true\n  model_name: fixture-reranker\n"
            if backend == "hybrid-rerank"
            else ""
        )
        config_text = (
            f"experiment_name: fixture-{backend}\nbackend: {selected_backend}\ncorpus: chunks.jsonl\n"
            f"retrieval:\n  sparse:\n{sparse_config}  dense:\n{dense_config}"
            f"fusion:\n  enabled: {str(selected_backend == 'hybrid').lower()}\n"
            f"{reranker_config}"
            "selection:\n  chunk:\n    threshold: 0\n    relative_delta: 1\n    min_k: 0\n    max_k: 5\n"
            "  document:\n    threshold: 0\n    relative_delta: 1\n    min_k: 0\n    max_k: 5\n"
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    ("backend", "expected_type", "expected_reranker"),
    [
        ("demo", DemoRetriever, None),
        ("sparse", BM25Retriever, None),
        ("dense", DenseRetriever, None),
        ("hybrid", HybridRetriever, None),
        ("hybrid-rerank", HybridRetriever, BGEReranker),
    ],
)
def test_retrieve_cli_uses_selected_backend(
    backend, expected_type, expected_reranker, tmp_path, monkeypatch
):
    config_path = _write_fixture_files(tmp_path, backend)
    output_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("src.retrieval.factory._require_dependency", lambda *_: None)
    class FixtureDenseRetriever(DenseRetriever):
        def __init__(self, **kwargs):
            self.model_name = kwargs["model_name"]
            self.index_path = kwargs["index_path"]

    monkeypatch.setattr("src.retrieval.factory.DenseRetriever", FixtureDenseRetriever)
    monkeypatch.setattr("src.reranking.factory.find_spec", lambda _: object())
    monkeypatch.setattr(
        BM25Retriever,
        "search",
        lambda self, query, top_k=200: [("c1", 0.9)],
    )
    monkeypatch.setattr(
        DenseRetriever,
        "search",
        lambda self, query, top_k=300: [("c1", 0.9)],
    )
    monkeypatch.setattr(
        BGEReranker,
        "rerank",
        lambda self, query, candidates, top_k=100: [(chunk_id, 0.9) for chunk_id, _ in candidates],
    )

    config = load_config(config_path)
    pipeline = build_pipeline(config)
    assert isinstance(config, RuntimeConfig if backend == "demo" else PipelineConfig)
    assert isinstance(pipeline.retriever, expected_type)
    if expected_reranker is not None:
        assert isinstance(pipeline.reranker, expected_reranker)
    else:
        assert pipeline.reranker is None

    monkeypatch.setattr(
        "sys.argv",
        [
            "scripts.retrieve",
            "--config",
            str(config_path),
            "--queries",
            str(tmp_path / "queries.jsonl"),
            "--output",
            str(output_path),
        ],
    )
    retrieve_script.main()
    predictions = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert predictions == [
        {"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]}
    ]
