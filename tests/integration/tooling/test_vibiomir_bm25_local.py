"""Verify document mapping, unknown queries, and preservation during selective restore."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bm25s
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from bm25s.tokenization import Tokenized

from r2ai.bm25index import tokenize
from scripts.runtime.query_vibiomir_bm25 import attach_urls, query_index
from scripts.tooling.download_vibiomir_bm25 import download


def test_saved_index_returns_official_ids_urls_and_no_zero_score_matches(tmp_path):
    index = tmp_path / "testidx"
    texts = ["diabetes insulin", "asthma wheezing", "diabetes glucose"]
    vocab = {term: i for i, term in enumerate(sorted({t for text in texts for t in tokenize(text)}))}
    retriever = bm25s.BM25(method="lucene")
    retriever.index(Tokenized(ids=[[vocab[t] for t in tokenize(text)] for text in texts], vocab=vocab),
                    show_progress=False)
    retriever.save(str(index), corpus=None)
    np.save(tmp_path / "testidx_ids.npy", np.array([42, 7, 99], dtype=np.int64))
    (tmp_path / "testidx_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    corpus = tmp_path / "links.parquet"
    pq.write_table(pa.table({"id": [99, 7, 42], "url": ["url99", "url7", "url42"]}), corpus)
    result = query_index(index, "diabetes insulin", 3)
    attach_urls(result, corpus)
    assert result["results"][0]["doc_id"] == 42
    assert result["results"][0]["url"] == "url42"
    assert {r["doc_id"] for r in result["results"]} == {42, 99}
    assert query_index(index, "unknownword", 3)["results"] == []
    with pytest.raises(ValueError, match="positive"):
        query_index(index, "diabetes", 0)
    np.save(tmp_path / "testidx_ids.npy", np.array([42], dtype=np.int64))
    with pytest.raises(ValueError, match="mapping"):
        query_index(index, "diabetes", 3)


def test_download_pins_revision_checksums_and_preserves_different_local_files(tmp_path):
    payloads = {"data/vibiomir/idx/params.index.json": b"{}",
                "data/vibiomir/idx_ids.npy": b"ids",
                "data/vibiomir/idx_vocab.json": b"{}"}
    siblings = [SimpleNamespace(rfilename=name, size=len(data),
                               lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
                for name, data in payloads.items()]
    info = SimpleNamespace(sha="pinned-revision", siblings=siblings)
    args = SimpleNamespace(repo_id="owner/repo", revision="main", index_name="idx",
                           index_only=True, destination=tmp_path, max_gib=1)
    revisions = []

    def fake_download(repo_id, name, *, repo_type, revision, local_dir):
        revisions.append(revision)
        target = Path(local_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[name])
        return str(target)

    with patch("scripts.tooling.download_vibiomir_bm25.HfApi") as api, patch(
            "scripts.tooling.download_vibiomir_bm25.hf_hub_download", side_effect=fake_download):
        api.return_value.repo_info.return_value = info
        download(args)
        assert revisions == ["pinned-revision"] * 3
        download(args)
        assert len(revisions) == 3
        target = tmp_path / "data/vibiomir/idx_ids.npy"
        target.write_bytes(b"bad")
        with pytest.raises(FileExistsError):
            download(args)
        assert target.read_bytes() == b"bad"
