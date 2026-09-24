"""Deterministic, Unicode-aware BM25 sparse retrieval."""

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

from src.config import SparseRetrievalConfig
from src.data.schema import ChunkRecord
from src.indexing.sparse_index import SparseBM25Index, corpus_sha256


class BaseRetriever(ABC):
    """Unified interface for all retrievers (BM25, Dense, Hybrid)."""

    @abstractmethod
    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """Returns list of (chunk_id, relevance_score) sorted descending by score."""
        pass


class BM25Retriever(BaseRetriever):
    """BM25 retriever loading a metadata-rich index built from ChunkRecords."""

    def __init__(
        self,
        index_path: str | Path = "artifacts/sparse_index",
        *,
        expected_corpus_hash: str | None = None,
        tokenizer: str | None = None,
        char_ngram_size: int | None = None,
        k1: float | None = None,
        b: float | None = None,
    ):
        self.index_path = Path(index_path)
        self.index = SparseBM25Index.load(
            self.index_path,
            expected_corpus_hash=expected_corpus_hash,
            tokenizer=tokenizer,
            char_ngram_size=char_ngram_size,
            k1=k1,
            b=b,
        )
        self.chunk_metadata = {
            item["chunk_id"]: {
                "doc_id": item["doc_id"],
                "language": item["language"],
                "text": item["text"],
            }
            for item in self.index.documents
        }

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        return self.index.search(query, top_k=top_k)

    def search_batch(
        self, queries: Sequence[str], top_k: int = 200
    ) -> list[list[tuple[str, float]]]:
        """Search a batch while retaining the same deterministic ranking contract."""
        return [self.search(query, top_k=top_k) for query in queries]

    @staticmethod
    def build_index(
        chunks: Sequence[ChunkRecord],
        index_path: str | Path,
        config: SparseRetrievalConfig,
    ) -> str:
        """Build and persist a BM25 index; return its source corpus hash."""
        index = SparseBM25Index(
            tokenizer=config.tokenizer,
            char_ngram_size=config.char_ngram_size,
            k1=config.k1,
            b=config.b,
        )
        index.build(chunks, Path(index_path))
        return corpus_sha256(chunks)
