"""BM25 sparse retrieval module."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseRetriever(ABC):
    """Unified interface for all retrievers (BM25, Dense, Hybrid)."""

    @abstractmethod
    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """Returns list of (chunk_id, relevance_score) sorted descending by score."""
        pass


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever baseline."""

    def __init__(self, index_path: str = "artifacts/sparse_index"):
        self.index_path = index_path

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        # Ready for Day 2 full implementation
        return []
