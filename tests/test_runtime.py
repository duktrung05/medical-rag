"""Integration checks for the demo runtime and its data boundaries."""

import pytest

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
    prediction = pipeline.run_query(QueryRecord(id="q1", query="tim mạch"))
    assert prediction.relevant_chunks == ["c1"]
    assert prediction.relevant_docs == ["d1"]
    assert pipeline.run_query(QueryRecord(id="q2", query="xyz")).relevant_chunks == []


def test_empty_corpus_rejected(tmp_path):
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        build_pipeline(RuntimeConfig(corpus=corpus))


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        RuntimeConfig(corpus="unused.jsonl", backend="dense")
