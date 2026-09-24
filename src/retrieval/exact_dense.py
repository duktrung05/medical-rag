"""Small-corpus dense retrieval with exact cosine/dot-product search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    score: float
    rank: int


def exact_search(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    chunk_ids: Sequence[str],
    top_k: int,
) -> list[list[RankedChunk]]:
    """Return deterministic exact inner-product rankings.

    With L2-normalized inputs, inner product is cosine similarity. Equal scores
    are resolved by chunk ID so repeated runs produce the same ranking.
    """
    if query_embeddings.ndim != 2 or passage_embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional arrays.")
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Query and passage embedding dimensions do not match.")
    if passage_embeddings.shape[0] != len(chunk_ids):
        raise ValueError("Number of passage embeddings must match chunk IDs.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique for deterministic dense ranking.")
    if not np.isfinite(query_embeddings).all() or not np.isfinite(passage_embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    scores = query_embeddings @ passage_embeddings.T
    limit = min(top_k, len(chunk_ids))
    ids = np.asarray(chunk_ids)
    results: list[list[RankedChunk]] = []
    for row in scores:
        order = np.lexsort((ids, -row))[:limit]
        results.append(
            [
                RankedChunk(
                    chunk_id=str(chunk_ids[index]),
                    score=float(row[index]),
                    rank=rank,
                )
                for rank, index in enumerate(order, start=1)
            ]
        )
    return results
