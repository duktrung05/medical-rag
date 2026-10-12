import json
import sqlite3
import sys
import zipfile
from types import SimpleNamespace

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.retrieval_v2.core import (
    assert_grounded,
    build_sparse,
    fuse,
    passage_id,
    score_and_select,
    sparse_search,
    tokens,
    windows,
)


def test_windows_preserve_exact_source_and_identity():
    source = "Alpha beta. Gamma delta. Epsilon zeta."
    parts = list(windows(source, target_words=3, overlap=0.25))
    assert len(parts) > 1
    for start, end, passage in parts:
        assert source[start:end] == passage
        assert passage_id(7, 9, start, end, passage) == passage_id(7, 9, start, end, passage)
    assert tokens("治疗糖尿病")


def test_three_sources_find_deep_passage_and_ground_submission(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"doc_id": 10, "chunk_index": 0, "title": "Heart", "text": "Common introduction."},
        {"doc_id": 10, "chunk_index": 9, "title": "Heart", "text": "Rare myocarditis treatment evidence."},
        {"doc_id": 20, "chunk_index": 0, "title": "Other", "text": "Unrelated information."},
    ]), source)
    db = tmp_path / "pilot.sqlite"
    stats = build_sparse(source, db, target_words=192)
    assert stats["source_rows"] == 3
    con = sqlite3.connect(db)
    doc_hits = sparse_search(con, "myocarditis", "document", 10)
    passage_hits = sparse_search(con, "myocarditis", "passage", 10)
    assert doc_hits[0][0] == 10
    assert passage_hits[0][1] == 10
    assert con.execute("SELECT chunk_index FROM passages WHERE rowid=?", (passage_hits[0][0],)).fetchone()[0] == 9
    # Dense branch can independently surface doc 20, absent from lexical results.
    pool = fuse(con, "myocarditis", [(3, 0.8)], budget=3)
    assert {x["doc_id"] for x in pool} == {10, 20}
    assert any("passage_dense" in x["sources"] for x in pool)
    prediction = score_and_select(pool, max_docs=2, max_chunks=2)
    assert_grounded(con, prediction)
    prediction["relevant_chunks"][0]["chunk_text"] = "fabricated"
    with pytest.raises(ValueError, match="exact source"):
        assert_grounded(con, prediction)
    con.close()


def test_dense_faiss_id_maps_to_persisted_passage_row(tmp_path, monkeypatch):
    from src.retrieval_v2.core import build_dense, dense_search

    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"doc_id": 1, "chunk_index": 0, "title": "", "text": "alpha"},
        {"doc_id": 2, "chunk_index": 8, "title": "", "text": "beta"},
    ]), source)
    db = tmp_path / "source.sqlite"
    build_sparse(source, db)

    class FakeEncoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return np.asarray([[1.0, 0.0] if "alpha" in x else [0.0, 1.0] for x in texts], dtype="float32")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeEncoder))
    index_path = tmp_path / "dense.faiss"
    result = build_dense(db, index_path, tmp_path / "fake_model", batch_size=1)
    assert result["vectors"] == 2
    assert faiss.read_index(str(index_path)).ntotal == 2
    assert dense_search(index_path, np.asarray([0.0, 1.0]), 1)[0][0] == 2
    assert not index_path.with_suffix(".progress.json").exists()


def test_document_scoring_uses_second_passage_when_available():
    candidates = [
        {"doc_id": 1, "text": "first", "fusion_score": 0.2, "fusion_rank": 1, "rerank_logit": 2.0},
        {"doc_id": 1, "text": "second", "fusion_score": 0.1, "fusion_rank": 3, "rerank_logit": 1.0},
        {"doc_id": 2, "text": "other", "fusion_score": 0.2, "fusion_rank": 2, "rerank_logit": 2.05},
    ]
    result = score_and_select(candidates, max_docs=1, max_chunks=2)
    assert result["relevant_docs"] == [1]
    assert len(result["relevant_chunks"]) == 2


def test_official_vibiomir_zip_contract(tmp_path):
    from r2ai.validate import validate

    queries = tmp_path / "queries.parquet"
    corpus = tmp_path / "links.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": 1}, {"id": 2}]), queries)
    pq.write_table(pa.Table.from_pylist([{"id": 10}, {"id": 20}]), corpus)
    records = [
        {"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "Exact evidence"}]},
        {"id": 2, "relevant_docs": [20], "relevant_chunks": [{"doc_id": 20, "chunk_text": "Other evidence"}]},
    ]
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(records))
    assert validate(archive, query_path=queries, corpus_path=corpus) == []
