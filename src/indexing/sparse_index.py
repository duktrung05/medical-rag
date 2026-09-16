"""Sparse indexing interfaces (Tantivy / BM25)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
from src.data.schema import ChunkRecord


class BaseSparseIndex(ABC):
    """Abstract interface for sparse lexical index."""

    @abstractmethod
    def build(self, chunks: List[ChunkRecord], output_path: Path) -> None:
        """Builds index from chunks and persists to output path."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """Searches index and returns list of (chunk_id, score)."""
        pass
