"""Hybrid retriever fusing BM25 sparse and Dense semantic candidates."""

from typing import List, Tuple
from src.retrieval.bm25 import BaseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):
    """Combines BM25 and Dense retriever via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        rrf_k: int = 60,
    ):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        sparse_results = self.sparse_retriever.search(query, top_k=top_k)
        dense_results = self.dense_retriever.search(query, top_k=top_k)
        return reciprocal_rank_fusion([sparse_results, dense_results], k=self.rrf_k, top_k=top_k)
