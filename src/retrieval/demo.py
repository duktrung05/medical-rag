"""Small lexical demo for plumbing checks; not a multilingual search model."""

import re

from src.data.preprocess import normalize_text
from src.data.schema import ChunkRecord
from src.retrieval.bm25 import BaseRetriever


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(text)))


class DemoRetriever(BaseRetriever):
    def __init__(self, chunks: list[ChunkRecord]):
        self.entries = [(chunk.chunk_id, tokens(chunk.text)) for chunk in chunks]

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        query_tokens = tokens(query)
        if not query_tokens or top_k <= 0:
            return []
        results = [
            (chunk_id, len(query_tokens & chunk_tokens) / len(query_tokens))
            for chunk_id, chunk_tokens in self.entries
            if query_tokens & chunk_tokens
        ]
        return sorted(results, key=lambda item: (-item[1], item[0]))[:top_k]
