"""Integration checks for the demo runtime and its data boundaries."""

import pytest

from src.adapters import adapt_query
from src.config import RuntimeConfig, load_config
from src.data.schema import QueryRecord
from src.service import build_pipeline


def test_demo_pipeline_returns_consistent_results(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(
        '{"chunk_id":"c1","doc_id":"d1","chunk_index":0,"language":"vi","text":"tài liệu tim mạch"}\n',
        encoding="utf-8",
    )
    config_file = tmp_path / "demo.yaml"
    config_file.write_text("corpus: chunks.jsonl\n", encoding="utf-8")
    pipeline = build_pipeline(load_config(config_file))
    query = adapt_query(QueryRecord(id="q1", query="tim mạch"))
    prediction = pipeline.run_query(query.query_id, query.text)
    assert prediction.relevant_chunks == ["c1"]
    assert prediction.relevant_docs == ["d1"]
    query = adapt_query(QueryRecord(id="q2", query="xyz"))
    assert pipeline.run_query(query.query_id, query.text).relevant_chunks == []


def test_demo_pipeline_does_not_require_retrieval_dependencies(tmp_path, monkeypatch):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text(
        '{"chunk_id":"c1","doc_id":"d1","chunk_index":0,"language":"vi","text":"tim mach"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("src.retrieval.factory.find_spec", lambda _: None)
    monkeypatch.setattr("src.reranking.factory.find_spec", lambda _: None)

    pipeline = build_pipeline(RuntimeConfig(corpus=corpus))
    query = adapt_query(QueryRecord(id="q1", query="tim mach"))
    prediction = pipeline.run_query(query.query_id, query.text)

    assert prediction.relevant_chunks == ["c1"]
    assert prediction.relevant_docs == ["d1"]


def test_empty_corpus_rejected(tmp_path):
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        build_pipeline(RuntimeConfig(corpus=corpus))


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        RuntimeConfig(corpus="unused.jsonl", backend="dense")
