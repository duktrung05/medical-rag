"""Dense indexing interfaces (FAISS HNSW / IVFPQ)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
import numpy as np


class BaseDenseIndex(ABC):
    """Abstract interface for dense vector index."""

    @abstractmethod
    def build(self, chunk_ids: List[str], embeddings: np.ndarray, output_path: Path) -> None:
        """Builds index from vector embeddings."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 300) -> List[Tuple[str, float]]:
        """Searches vector index and returns (chunk_id, similarity_score)."""
        pass
