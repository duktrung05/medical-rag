"""Reciprocal rank fusion with deterministic output and source ranks."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    score: float
    sparse_rank: int | None
    dense_rank: int | None

    @property
    def sources(self):
        return tuple(name for name, rank in (("bm25", self.sparse_rank), ("dense", self.dense_rank)) if rank is not None)


def reciprocal_rank_fusion_with_provenance(rankings, *, k=60, top_k=200):
    if k < 1 or top_k < 0:
        raise ValueError("k must be >= 1 and top_k must be >= 0")
    scores = defaultdict(float)
    ranks = defaultdict(dict)
    names = ("bm25", "dense") if len(rankings) == 2 else tuple(f"retriever_{i}" for i in range(len(rankings)))
    for name, ranking in zip(names, rankings, strict=True):
        seen = set()
        rank = 0
        for cid, _ in ranking:
            if cid in seen:
                continue
            seen.add(cid)
            rank += 1
            ranks[cid][name] = rank
            scores[cid] += 1.0 / (k + rank)
    result = [FusedCandidate(cid, score, ranks[cid].get("bm25"), ranks[cid].get("dense")) for cid, score in scores.items()]
    result.sort(key=lambda item: (-item.score, item.chunk_id))
    return result[:top_k]


def reciprocal_rank_fusion(rankings, k=60, top_k=200):
    return [(item.chunk_id, item.score) for item in reciprocal_rank_fusion_with_provenance(rankings, k=k, top_k=top_k)]
