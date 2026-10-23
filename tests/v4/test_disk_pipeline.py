"""Disk canary keeps full-source and tail-passage V4 semantics."""
from __future__ import annotations

import hashlib
import json

import pytest

from src.medrag.core.contracts import Passage, Source
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.core.pipeline import Pipeline
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.retrieval.disk_bm25 import DiskBM25


def test_disk_pipeline_matches_memory_and_fails_on_corruption(tmp_path):
    text = "unrelated " * 50 + "unique tail therapy"
    source = Source("release", 10, "https://example.test/10", "Unrelated title", text,
                    "test", "raw", {"normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    start = text.index("unique tail therapy")
    passage = Passage("10:p8", "release", 10, source.url, text[start:], start, len(text),
                      source.text_sha256, "en", "test")
    config = {"selection": {"max_docs": 10, "max_chunks": 10, "chunks_per_doc": 1}}
    memory = Pipeline([source], [passage], selection_policy=None)
    store = DiskCorpusStore(tmp_path / "corpus.sqlite", "release", "fingerprint", create=True)
    store.append_sources([source])
    store.append_passages([passage])
    store.complete()
    doc = DiskBM25(tmp_path / "documents.sqlite", "document_bm25", "fingerprint", create=True)
    doc.append_batch([(10, None, f"{source.title} {source.text}")])
    doc.complete()
    chunks = DiskBM25(tmp_path / "passages.sqlite", "passage_bm25", "fingerprint", create=True)
    chunks.append_batch([(10, passage.passage_id, passage.text)])
    chunks.complete()
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in
             ("corpus.sqlite", "documents.sqlite", "passages.sqlite")}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "medrag-v4-disk-index-v1", "status": "complete",
        "corpus_version": "release", "fingerprint": "fingerprint", "source_count": 1,
        "passage_count": 1, "files": files,
    }))
    disk = DiskPipeline(tmp_path, config)
    assert disk.retrieve("unique tail therapy") == memory.retrieve("unique tail therapy")
    assert disk.retrieve("unique tail therapy")[0]["evidence"][0]["passage_id"] == "10:p8"
    independent = DiskPipeline(tmp_path, {"selection": {"document_selection": "independent"}})
    assert independent.selection_policy.document_selection == "independent"
    assert independent.retrieve("unique tail therapy")[0]["document_evidence"]["passage_id"] == "10:p8"
    context = DiskPipeline(tmp_path, {"selection": {"document_selection": "independent",
                                                     "chunk_context": "adjacent"}})
    context_result = context.retrieve("unique tail therapy")[0]
    assert context_result["evidence"][0]["seed_passage_id"] == "10:p8"
    assert context_result["evidence"][0]["text"] == passage.text
    assert context_result["document_evidence"]["passage_id"] == "10:p8"
    assert store.passages_for_parent(10) == [passage]
    with pytest.raises(ValueError, match="no dense search backend"):
        DiskPipeline(tmp_path, {"dense": {"enabled": True}})
    with pytest.raises(ValueError, match="no reranker"):
        DiskPipeline(tmp_path, {"reranker": {"enabled": True}})
    with (tmp_path / "manifest.json").open("r+", encoding="utf-8") as stream:
        bad = json.load(stream)
        bad["files"]["corpus.sqlite"] = "wrong"
        stream.seek(0)
        json.dump(bad, stream)
        stream.truncate()
    with pytest.raises(ValueError, match="checksum mismatch"):
        DiskPipeline(tmp_path, config)


def test_disk_store_rolls_back_only_pending_partition(tmp_path):
    a = Source("release", 1, "https://example.test/1", "A", "first text", "test", "raw")
    b = Source("release", 2, "https://example.test/2", "B", "pending text", "test", "raw")
    pa = Passage("1:p", "release", 1, a.url, a.text, 0, len(a.text), a.text_sha256, "en", "test")
    pb = Passage("2:p", "release", 2, b.url, b.text, 0, len(b.text), b.text_sha256, "en", "test")
    store = DiskCorpusStore(tmp_path / "store.sqlite", "release", "lock", create=True)
    store.append_sources([a])
    store.append_passages([pa])
    store.append_sources([b])
    store.append_passages([pb])
    assert store.rollback_part({2}) == (1, 1)
    store.append_sources([b])
    store.append_passages([pb])
    assert store.complete() == {"status": "complete", "sources": 2, "passages": 2}
    assert store.source(1).text == a.text
    assert store.passage("2:p").text == b.text
