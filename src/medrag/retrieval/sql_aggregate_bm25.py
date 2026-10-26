"""Read-only SQL-aggregate BM25 search for large immutable V4 indexes."""
from __future__ import annotations

import math
import sqlite3

from src.medrag.core.contracts import Hit
from src.medrag.retrieval.core import tokens
from src.medrag.retrieval.disk_bm25 import DiskBM25


class SQLAggregateDiskBM25(DiskBM25):
    """Execute score aggregation and top-k inside SQLite instead of Python."""

    def __init__(self, *args, max_query_terms: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_query_terms is not None and (type(max_query_terms) is not int or max_query_terms <= 0):
            raise ValueError('max_query_terms must be a positive integer or None')
        self.max_query_terms = max_query_terms

    def search(self, query: str, limit: int = 20, *, parent_id: int | None = None) -> list[Hit]:
        if parent_id is not None:
            return super().search(query, limit, parent_id=parent_id)
        if limit <= 0:
            raise ValueError('BM25 result limit must be positive')
        meta = self._check(require_complete=True)
        count = int(meta['rows'])
        if not count:
            return []
        average = int(meta['total_length']) / count
        terms = sorted(set(tokens(query)))
        if not terms:
            return []
        with sqlite3.connect(f'file:{self.path}?mode=ro', uri=True) as db:
            dfs = dict(db.execute(
                f"SELECT term,count FROM df WHERE term IN ({','.join('?' for _ in terms)})", terms))
            weighted = [(term, math.log(1 + (count - int(dfs[term]) + .5)
                                            / (int(dfs[term]) + .5)))
                        for term in terms if term in dfs]
            if self.max_query_terms is not None:
                weighted = sorted(weighted, key=lambda item: (-item[1], item[0]))[
                    :self.max_query_terms]
            if not weighted:
                return []
            values = ','.join('(?,?)' for _ in weighted)
            parameters = [value for pair in weighted for value in pair] + [average, limit]
            sql = f"""
                WITH query_terms(term,idf) AS (VALUES {values})
                SELECT d.official_id,d.passage_id,
                       SUM(q.idf * p.tf * 2.2 /
                           (p.tf + 1.2 * (0.25 + 0.75 * d.length / ?))) AS score
                FROM query_terms q
                JOIN postings p ON p.term=q.term
                JOIN docs d ON d.row_id=p.row_id
                GROUP BY d.row_id
                ORDER BY score DESC,d.official_id ASC,CAST(d.passage_id AS TEXT) ASC
                LIMIT ?
            """
            rows = db.execute(sql, parameters).fetchall()
        return [Hit(int(doc), passage, float(score), self.channel, rank)
                for rank, (doc, passage, score) in enumerate(rows, 1)]
