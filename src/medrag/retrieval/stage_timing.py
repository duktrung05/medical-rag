"""Non-invasive stage timers for the frozen Pipeline.prepare_rerank path."""
from __future__ import annotations

import time
from collections import defaultdict

import src.medrag.core.pipeline as pipeline_module


class TimedSearch:
    def __init__(self, backend, stage: str, measurements: dict):
        self.backend = backend
        self.stage = stage
        self.measurements = measurements

    def search(self, query: str, limit: int, **kwargs):
        started = time.perf_counter()
        try:
            return self.backend.search(query, limit, **kwargs)
        finally:
            self.measurements[f"{self.stage}_seconds"] += time.perf_counter() - started
            self.measurements[f"{self.stage}_calls"] += 1

    def search_parent(self, query: str, parent_id: int, limit: int = 1):
        started = time.perf_counter()
        try:
            return self.backend.search_parent(query, parent_id, limit)
        finally:
            self.measurements["parent_fallback_seconds"] += time.perf_counter() - started
            self.measurements["parent_fallback_calls"] += 1


def prepare_rerank_with_stage_timing(pipeline, query: str, limit: int):
    """Call the unmodified pipeline while timing its retrieval boundaries.

    Fusion is patched only for the duration of this synchronous call. The
    residual is reported as pair construction and includes candidate grouping,
    corpus lookups, verification, allocation, and payload creation.
    """
    measurements = defaultdict(float)
    original_doc = pipeline.doc_bm25
    original_passage = pipeline.passage_bm25
    original_dense = pipeline.dense
    original_fuse = pipeline_module.fuse

    def timed_fuse(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_fuse(*args, **kwargs)
        finally:
            measurements["fusion_seconds"] += time.perf_counter() - started
            measurements["fusion_calls"] += 1

    pipeline.doc_bm25 = TimedSearch(original_doc, "document_bm25", measurements)
    pipeline.passage_bm25 = TimedSearch(original_passage, "passage_bm25", measurements)
    if original_dense is not None:
        pipeline.dense = TimedSearch(original_dense, "dense_total", measurements)
    pipeline_module.fuse = timed_fuse
    started = time.perf_counter()
    try:
        result = pipeline.prepare_rerank(query, limit)
    finally:
        total = time.perf_counter() - started
        pipeline.doc_bm25 = original_doc
        pipeline.passage_bm25 = original_passage
        pipeline.dense = original_dense
        pipeline_module.fuse = original_fuse

    if original_dense is not None and hasattr(original_dense, "last_timing"):
        dense = original_dense.last_timing
        measurements["query_embedding_seconds"] = dense["query_embedding_seconds"]
        measurements["dense_search_seconds"] = dense["dense_search_seconds"]
        measurements["dense_total_seconds"] = dense["dense_total_seconds"]
    exclusive = (
        measurements["document_bm25_seconds"]
        + measurements["passage_bm25_seconds"]
        + measurements["parent_fallback_seconds"]
        + measurements["dense_total_seconds"]
        + measurements["fusion_seconds"]
    )
    measurements["pair_construction_seconds"] = max(0.0, total - exclusive)
    measurements["prepare_total_seconds"] = total
    timing = dict(measurements)
    for name in ("document_bm25_calls", "passage_bm25_calls", "parent_fallback_calls",
                 "dense_total_seconds", "query_embedding_seconds", "dense_search_seconds",
                 "fusion_calls"):
        timing.setdefault(name, 0)
    return (*result, timing)
