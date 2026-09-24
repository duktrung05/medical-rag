"""Tests for the retrieval and reranker construction factories."""

from src.config import DenseRetrievalConfig, FusionConfig, RerankerConfig, SparseRetrievalConfig
from src.data.schema import ChunkRecord
from src.reranking.bge_reranker import BGEReranker
from src.reranking.factory import create_reranker
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.factory import (
    create_dense_retriever,
    create_hybrid_retriever,
    create_sparse_retriever,
)
from src.retrieval.hybrid import HybridRetriever


def test_retriever_factories_create_configured_types(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.factory._require_dependency", lambda *_: None)
    class FixtureDenseRetriever(DenseRetriever):
        def __init__(self, **kwargs):
            self.model_name = kwargs["model_name"]
            self.index_path = kwargs["index_path"]

    monkeypatch.setattr("src.retrieval.factory.DenseRetriever", FixtureDenseRetriever)
    sparse_index = tmp_path / "sparse"
    dense_index = tmp_path / "dense"
    sparse_index.mkdir()
    dense_index.mkdir()
    sparse_config = SparseRetrievalConfig(enabled=True, index_path=sparse_index)
    BM25Retriever.build_index(
        [ChunkRecord(chunk_id="c1", doc_id="d1", chunk_index=0, language="vi", text="tim mach")],
        sparse_index,
        sparse_config,
    )

    sparse = create_sparse_retriever(
        sparse_config
    )
    dense = create_dense_retriever(
        DenseRetrievalConfig(
            enabled=True,
            model_name="test-model",
            index_path=dense_index,
        )
    )
    hybrid = create_hybrid_retriever(sparse, dense, FusionConfig(enabled=True))

    assert isinstance(sparse, BM25Retriever)
    assert isinstance(dense, DenseRetriever)
    assert isinstance(hybrid, HybridRetriever)


def test_reranker_factory_creates_configured_reranker(monkeypatch):
    monkeypatch.setattr("src.reranking.factory.find_spec", lambda _: object())
    reranker = create_reranker(RerankerConfig(enabled=True, model_name="test-reranker"))
    assert isinstance(reranker, BGEReranker)
    assert create_reranker(RerankerConfig(enabled=False)) is None
