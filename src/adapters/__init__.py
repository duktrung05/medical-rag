"""Adapters between external input records and the retrieval core."""

from src.adapters.input import NormalizedQuery, adapt_query

__all__ = ["NormalizedQuery", "adapt_query"]
