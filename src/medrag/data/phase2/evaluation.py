"""Aggregate deterministic pilot metrics."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def passage_metrics(passages: list[dict], document_count: int,
                    min_chunk_tokens: int) -> dict:
    tokens = [int(row["token_count"]) for row in passages]
    per_document: dict[int, int] = {}
    overlap_chars = 0
    total_chars = 0
    previous: dict[int, tuple[int, int]] = {}
    for row in passages:
        doc_id = int(row["doc_id"])
        per_document[doc_id] = per_document.get(doc_id, 0) + 1
        span = (int(row["char_start"]), int(row["char_end"]))
        if doc_id in previous:
            overlap_chars += max(0, previous[doc_id][1] - span[0])
        total_chars += span[1] - span[0]
        previous[doc_id] = span
    per_doc = list(per_document.values())
    return {
        "document_count": document_count,
        "passage_count": len(passages),
        "tokens_mean": statistics.fmean(tokens) if tokens else 0.0,
        "tokens_median": statistics.median(tokens) if tokens else 0.0,
        "tokens_p95": percentile(tokens, 0.95),
        "tokens_p99": percentile(tokens, 0.99),
        "small_passages": sum(value < min_chunk_tokens for value in tokens),
        "overlap_ratio_chars": overlap_chars / total_chars if total_chars else 0.0,
        "passages_per_document_mean": statistics.fmean(per_doc) if per_doc else 0.0,
        "passages_per_document_p95": percentile(per_doc, 0.95),
    }
