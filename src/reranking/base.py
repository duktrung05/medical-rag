"""Abstract base interface for cross-encoder rerankers."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseReranker(ABC):
    """Abstract interface for reranking candidate chunks with a cross-encoder."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # List of (chunk_id, chunk_text)
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Scores (query, chunk_text) pairs and returns (chunk_id, score) sorted descending."""
        pass
