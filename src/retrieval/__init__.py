"""Retrieval module."""

from src.retrieval.bm25 import BaseRetriever, BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.factory import (
    create_dense_retriever,
    create_hybrid_retriever,
    create_retriever,
    create_sparse_retriever,
)

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "create_sparse_retriever",
    "create_dense_retriever",
    "create_hybrid_retriever",
    "create_retriever",
]
