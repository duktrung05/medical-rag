"""Scoring and aggregation module."""

from src.scoring.document_score import aggregate_doc_scores, aggregate_doc_scores_max, aggregate_doc_scores_hybrid
from src.scoring.chunk_score import normalize_min_max, sort_candidates

__all__ = [
    "aggregate_doc_scores",
    "aggregate_doc_scores_max",
    "aggregate_doc_scores_hybrid",
    "normalize_min_max",
    "sort_candidates",
]
