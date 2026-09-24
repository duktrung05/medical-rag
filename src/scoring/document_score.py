"""Document score aggregation strategies from constituent chunk retrieval scores."""

from typing import Dict, List, Literal
import numpy as np


def aggregate_doc_scores_max(chunk_scores: Dict[str, float], chunk_to_doc: Dict[str, str]) -> Dict[str, float]:
    """Score(doc) = max_{c in doc} Score(c)

    Computes the maximum chunk relevance score for each parent document.
    """
    doc_scores: Dict[str, float] = {}
    for chunk_id, score in chunk_scores.items():
        doc_id = chunk_to_doc.get(chunk_id)
        if not doc_id:
            continue
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = float(score)
    return doc_scores


def aggregate_doc_scores_hybrid(
    chunk_scores: Dict[str, float],
    chunk_to_doc: Dict[str, str],
    alpha: float = 0.8,
    beta: float = 0.2,
    top_n: int = 3,
) -> Dict[str, float]:
    """Score(doc) = alpha * Max + beta * Mean(Top_N)"""
    grouped: Dict[str, List[float]] = {}
    for chunk_id, score in chunk_scores.items():
        doc_id = chunk_to_doc.get(chunk_id)
        if not doc_id:
            continue
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].append(score)

    doc_scores: Dict[str, float] = {}
    for doc_id, scores in grouped.items():
        sorted_scores = sorted(scores, reverse=True)
        max_score = sorted_scores[0]
        mean_top_n = float(np.mean(sorted_scores[:top_n]))
        doc_scores[doc_id] = float(alpha * max_score + beta * mean_top_n)

    return doc_scores


def aggregate_doc_scores(
    chunk_scores: Dict[str, float],
    chunk_to_doc: Dict[str, str],
    method: Literal["max", "mean", "hybrid_mean"] = "max",
    alpha: float = 0.8,
    beta: float = 0.2,
    top_n: int = 3,
) -> Dict[str, float]:
    """Dispatches document score aggregation based on configuration."""
    if method == "max":
        return aggregate_doc_scores_max(chunk_scores, chunk_to_doc)
    elif method == "mean":
        grouped: Dict[str, List[float]] = {}
        for chunk_id, score in chunk_scores.items():
            doc_id = chunk_to_doc.get(chunk_id)
            if doc_id:
                grouped.setdefault(doc_id, []).append(float(score))
        return {doc_id: float(np.mean(scores)) for doc_id, scores in grouped.items()}
    elif method == "hybrid_mean":
        return aggregate_doc_scores_hybrid(chunk_scores, chunk_to_doc, alpha=alpha, beta=beta, top_n=top_n)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
