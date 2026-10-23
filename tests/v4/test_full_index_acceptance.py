"""Failure gates for full-index acceptance helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.v4.smoke_full_index import audit_scores
from scripts.v4.verify_full_index import check_links
from scripts.v4.verify_release_dense import check_mapping_structure, check_vectors
from src.medrag.core.contracts import Passage, Source
from src.medrag.ranking.score_cache import ScoreCache


def _matrix(path, data, *, dtype=np.float32):
    rows = np.asarray(data, dtype=dtype)
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=rows.dtype,
                                       shape=rows.shape)
    matrix[:] = rows
    matrix.flush()
    del matrix


def test_vector_verifier_scans_tail_and_rejects_bad_dtype(tmp_path):
    path = tmp_path / "vectors.npy"
    _matrix(path, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert check_vectors(path, expected_rows=3, dimension=2, block_rows=2)["rows"] == 3
    _matrix(path, [[1.0, 0.0], [0.0, 1.0], [float("nan"), 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        check_vectors(path, expected_rows=3, dimension=2, block_rows=2)
    _matrix(path, [[1, 0]], dtype=np.float64)
    with pytest.raises(ValueError, match="shape/dtype"):
        check_vectors(path, expected_rows=1, dimension=2)


def test_mapping_verifier_rejects_gap_despite_correct_count(tmp_path):
    path = tmp_path / "mapping.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);"
                         "CREATE TABLE rows(row_id INTEGER PRIMARY KEY,official_id INTEGER,"
                         "passage_id TEXT UNIQUE,input_sha256 TEXT);")
        db.executemany("INSERT INTO meta VALUES(?,?)", [
            ("format", "medrag-v4-dense-fp32-v1"), ("status", "complete"),
            ("fingerprint", "key"), ("rows", "2"), ("expected_rows", "2"),
            ("dimension", "2")])
        db.executemany("INSERT INTO rows VALUES(?,?,?,?)", [
            (0, 7, "p0", "a" * 64), (2, 7, "p2", "b" * 64)])
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        with pytest.raises(ValueError, match="count/sequence"):
            check_mapping_structure(db, fingerprint="key", expected_rows=2,
                                    dimension=2, expected_sources=1)


def test_combined_verifier_rejects_wrong_symlink(tmp_path):
    sparse, dense, index = (tmp_path / name for name in ("sparse", "dense", "index"))
    for name in ("corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json"):
        (sparse / name).parent.mkdir(exist_ok=True)
        (sparse / name).write_bytes(b"source")
    (dense / "dense").mkdir(parents=True)
    index.mkdir()
    for name in ("corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json"):
        (index / name).symlink_to((sparse / name).resolve())
    (index / "dense").symlink_to((dense / "dense").resolve(), target_is_directory=True)
    assert len(check_links(index, sparse, dense)) == 5
    (index / "dense").unlink()
    (index / "dense").symlink_to(sparse.resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="points elsewhere"):
        check_links(index, sparse, dense)


def test_smoke_score_audit_rejects_tampered_cached_pair(tmp_path):
    source = Source("release", 7, "https://example.test", "Title", "alpha beta", "test", "raw")
    passage = Passage("p7", "release", 7, source.url, source.text, 0, len(source.text),
                      source.text_sha256, "en", "test")

    class Store:
        def source(self, ident):
            assert ident == 7
            return source

        def passage(self, ident):
            assert ident == "p7"
            return passage

    cache = ScoreCache(tmp_path / "scores.sqlite", {"model": "fixture"}, create=True)

    class Scorer:
        def score(self, query, payloads):
            return [0.5]

    cache.score("query", [(7, "p7", "Title\nalpha beta")], Scorer())
    trace = {"rerank_pairs": 1, "ranked_candidates": [{
        "official_id": 7, "score": 0.1,
        "evidence": [{"passage_id": "p7", "text": "alpha beta", "start": 0, "end": 10,
                      "score_status": "reranked", "rerank_score": 0.5}]}]}
    assert audit_scores("query", trace, cache, Store())["reranked_pairs_audited"] == 1
    trace["ranked_candidates"][0]["evidence"][0]["rerank_score"] = 0.75
    with pytest.raises(ValueError, match="Score cache differs"):
        audit_scores("query", trace, cache, Store())
