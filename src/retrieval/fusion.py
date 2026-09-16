"""Reciprocal Rank Fusion (RRF) and candidate rank combination."""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Sequence[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 200,
) -> List[Tuple[str, float]]:
    """Combines multiple ranked candidate lists using Reciprocal Rank Fusion (RRF):

    RRF(d) = sum_{r} 1 / (k + rank_r(d))
    where rank_r(d) is 1-indexed.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for rank_list in rankings:
        for rank_idx, (doc_id, _) in enumerate(rank_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank_idx)

    sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_candidates[:top_k]
