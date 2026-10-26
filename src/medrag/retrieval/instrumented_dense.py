"""Timing-only DiskDense variant with an identical search implementation."""
from __future__ import annotations

import math
import sqlite3
import time

import numpy as np

from src.medrag.core.contracts import Hit
from src.medrag.retrieval.disk_dense import DiskDense


class InstrumentedDiskDense(DiskDense):
    """Expose query-encoding and matrix-search time without changing arithmetic."""

    def search(self, query: str, limit: int = 20) -> list[Hit]:
        total_started = time.perf_counter()
        if limit <= 0:
            raise ValueError("Dense result limit must be positive")
        meta = self._check(require_complete=True)
        encode_query = getattr(self.encoder, "encode_query", None)
        if encode_query is None:
            encode_query = self.encoder.encode
        encode_started = time.perf_counter()
        query_vector = np.asarray(encode_query([query])[0], dtype=np.float64)
        query_embedding_seconds = time.perf_counter() - encode_started
        search_started = time.perf_counter()
        if query_vector.shape != (int(meta["dimension"]),) or not np.isfinite(query_vector).all():
            raise ValueError("Dense query dimension/finite check failed")
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 0 or not math.isfinite(query_norm):
            raise ValueError("Dense query vector has zero/invalid norm")
        matrix = np.load(self.matrix_path, mmap_mode="r")
        candidates: list[tuple[float, int]] = []
        count = int(meta["rows"])
        for start in range(0, count, self.block_rows):
            end = min(start + self.block_rows, count)
            block = np.asarray(matrix[start:end], dtype=np.float64)
            scores = block @ query_vector / (np.linalg.norm(block, axis=1) * query_norm)
            keep = min(limit, len(scores))
            if keep == len(scores):
                local = np.arange(len(scores))
            else:
                threshold = float(np.partition(scores, len(scores) - keep)[len(scores) - keep])
                local = np.flatnonzero(scores >= threshold)
            candidates.extend((float(scores[index]), start + int(index)) for index in local)
        with sqlite3.connect(f"file:{self.mapping}?mode=ro", uri=True) as db:
            identities = {}
            for offset in range(0, len(candidates), 800):
                row_ids = [row_id for _, row_id in candidates[offset:offset + 800]]
                query_sql = (
                    "SELECT row_id,official_id,passage_id FROM rows WHERE row_id IN "
                    f"({','.join('?' for _ in row_ids)})"
                )
                identities.update((row_id, (parent, passage_id)) for row_id, parent, passage_id
                                  in db.execute(query_sql, row_ids))
        if len(identities) != len(candidates):
            raise ValueError("Dense mapping is missing candidate rows")
        ordered = sorted(candidates, key=lambda item: (-item[0], identities[item[1]][0],
                                                        identities[item[1]][1]))[:limit]
        result = [Hit(*identities[row_id], score, "dense", rank)
                  for rank, (score, row_id) in enumerate(ordered, 1)]
        dense_search_seconds = time.perf_counter() - search_started
        self.last_timing = {
            "query_embedding_seconds": query_embedding_seconds,
            "dense_search_seconds": dense_search_seconds,
            "dense_total_seconds": time.perf_counter() - total_started,
        }
        return result
