"""Dense multilingual retriever module using BGE-M3."""

from typing import List, Tuple
from src.retrieval.bm25 import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retriever using multilingual sentence embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-m3", index_path: str = "artifacts/dense_index"):
        self.model_name = model_name
        self.index_path = index_path

    def search(self, query: str, top_k: int = 300) -> List[Tuple[str, float]]:
        # Ready for Day 3 full implementation
        return []
