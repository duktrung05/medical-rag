"""Hybrid retriever fusing sparse and dense candidates via RRF."""

from src.retrieval.bm25 import BaseRetriever
from src.retrieval.fusion import FusedCandidate, reciprocal_rank_fusion_with_provenance


class HybridRetriever(BaseRetriever):
    def __init__(self, sparse_retriever: BaseRetriever, dense_retriever: BaseRetriever,
                 rrf_k: int = 60, sparse_top_k: int | None = None, dense_top_k: int | None = None):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k

    def search_with_provenance(self, query: str, top_k: int = 200) -> list[FusedCandidate]:
        sparse = self.sparse_retriever.search(query, top_k=self.sparse_top_k or top_k)
        dense = self.dense_retriever.search(query, top_k=self.dense_top_k or top_k)
        return reciprocal_rank_fusion_with_provenance([sparse, dense], k=self.rrf_k, top_k=top_k)

    def positive_coverage(self, query: str, positive_chunk_ids: set[str]) -> dict[str, list[str]]:
        sparse = {cid for cid, _ in self.sparse_retriever.search(query, self.sparse_top_k or 200)}
        dense = {cid for cid, _ in self.dense_retriever.search(query, self.dense_top_k or 200)}
        out = {key: [] for key in ("bm25_only", "dense_only", "both", "missed")}
        for cid in sorted(positive_chunk_ids):
            out["both" if cid in sparse and cid in dense else "bm25_only" if cid in sparse else "dense_only" if cid in dense else "missed"].append(cid)
        return out

    def search(self, query: str, top_k: int = 200):
        return [(item.chunk_id, item.score) for item in self.search_with_provenance(query, top_k)]
