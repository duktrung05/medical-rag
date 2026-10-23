"""Opt-in BM25 experiment; the frozen DiskPipeline never imports this backend.

Only the parent-filtered SQL traversal changes. Global search still delegates
to DiskBM25. In particular, do not sort the query's token set here: its iteration
order determines floating-point accumulation in the frozen implementation.
"""
from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from src.medrag.core.contracts import Hit
from src.medrag.retrieval.core import tokens
from src.medrag.retrieval.disk_bm25 import DiskBM25


PARENT_SQL = (
    "SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id "
    "FROM docs d INDEXED BY docs_parent CROSS JOIN postings p "
    "WHERE d.official_id=? AND p.term=? AND p.row_id=d.row_id"
)
IMPLEMENTATION_VERSION = "p5-parent-first-bm25-v1"


class ParentFirstDiskBM25(DiskBM25):
    """Read-only experimental reader using existing parent/posting indexes."""

    def __init__(self, path: Path, channel: str, fingerprint: str):
        super().__init__(path, channel, fingerprint)

    def search(self, query: str, limit: int = 20, *, parent_id: int | None = None) -> list[Hit]:
        if parent_id is None:
            return super().search(query, limit)
        if limit <= 0:
            raise ValueError("BM25 result limit must be positive")
        meta = self._check(require_complete=True)
        n = int(meta["rows"])
        if not n:
            return []
        avgdl = int(meta["total_length"]) / n
        scores: dict[int, float] = defaultdict(float)
        identities: dict[int, tuple[int, str | None]] = {}
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            for term in set(tokens(query)):
                df_row = db.execute("SELECT count FROM df WHERE term=?", (term,)).fetchone()
                if df_row is None:
                    continue
                df = int(df_row["count"])
                idf = math.log(1 + (n - df + .5) / (df + .5))
                for row in db.execute(PARENT_SQL, (parent_id, term)):
                    tf = int(row["tf"])
                    length = int(row["length"])
                    row_id = int(row["row_id"])
                    identities[row_id] = (int(row["official_id"]), row["passage_id"])
                    scores[row_id] += idf * tf * 2.2 / (
                        tf + 1.2 * (.25 + .75 * length / max(avgdl, 1)))
            if not scores:
                return []
        ordered = sorted(scores, key=lambda row_id: (
            -scores[row_id], identities[row_id][0], str(identities[row_id][1])))[:limit]
        return [Hit(*identities[row_id], scores[row_id], self.channel, rank)
                for rank, row_id in enumerate(ordered, 1)]
