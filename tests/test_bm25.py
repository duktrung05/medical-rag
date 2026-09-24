"""Persistence, Unicode, determinism, and batching checks for BM25."""

import pytest

from src.config import SparseRetrievalConfig
from src.data.schema import ChunkRecord
from src.retrieval.bm25 import BM25Retriever


def _chunks():
    return [
        ChunkRecord(
            chunk_id="c-b",
            doc_id="d2",
            chunk_index=0,
            language="vi",
            text="Đau tim và mạch máu",
        ),
        ChunkRecord(
            chunk_id="c-a",
            doc_id="d1",
            chunk_index=0,
            language="vi",
            text="ĐAU TIM và mạch máu",
        ),
        ChunkRecord(
            chunk_id="c-zh",
            doc_id="d3",
            chunk_index=0,
            language="zh",
            text="心脏和血管",
        ),
    ]


def test_bm25_persists_metadata_unicode_and_deterministic_ties(tmp_path):
    index_path = tmp_path / "sparse-index"
    config = SparseRetrievalConfig(enabled=True, index_path=index_path)
    BM25Retriever.build_index(_chunks(), index_path, config)
    retriever = BM25Retriever(index_path)

    first = retriever.search("ĐAU tim", top_k=10)
    second = retriever.search("đau tim", top_k=10)
    assert first == second
    assert [chunk_id for chunk_id, _ in first] == ["c-a", "c-b"]
    assert retriever.chunk_metadata["c-zh"] == {
        "doc_id": "d3",
        "language": "zh",
        "text": "心脏和血管",
    }
    assert retriever.search_batch(["đau tim", "心脏"], top_k=1) == [
        first[:1],
        retriever.search("心脏", top_k=1),
    ]


def test_bm25_rejects_corpus_and_tokenizer_mismatch(tmp_path):
    index_path = tmp_path / "sparse-index"
    config = SparseRetrievalConfig(enabled=True, index_path=index_path)
    BM25Retriever.build_index(_chunks(), index_path, config)
    with pytest.raises(ValueError, match="does not match the configured corpus"):
        BM25Retriever(index_path, expected_corpus_hash="wrong-hash")
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        BM25Retriever(index_path, tokenizer="char_ngram")
    with pytest.raises(ValueError, match="k1 does not match"):
        BM25Retriever(index_path, k1=1.8)


def test_char_ngram_tokenizer_is_a_separate_index_experiment(tmp_path):
    index_path = tmp_path / "char-index"
    config = SparseRetrievalConfig(
        enabled=True,
        index_path=index_path,
        tokenizer="char_ngram",
        char_ngram_size=3,
    )
    BM25Retriever.build_index(_chunks(), index_path, config)
    retriever = BM25Retriever(index_path, tokenizer="char_ngram", char_ngram_size=3)
    assert retriever.index.tokenizer == "char_ngram"
    assert retriever.search("đau tim", top_k=1)[0][0] == "c-a"
