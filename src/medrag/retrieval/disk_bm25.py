"""Disk-backed BM25 with the same token and score formula as V4's in-memory index."""
from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.medrag.core.contracts import Hit
from src.medrag.retrieval.core import tokens


class DiskBM25:
    def __init__(self, path: Path, channel: str, fingerprint: str, *, create: bool = False):
        self.path = path
        self.channel = channel
        self.fingerprint = fingerprint
        if create:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE docs(row_id INTEGER PRIMARY KEY, official_id INTEGER NOT NULL,
                                      passage_id TEXT, length INTEGER NOT NULL);
                    CREATE INDEX docs_parent ON docs(official_id);
                    CREATE TABLE postings(term TEXT NOT NULL, row_id INTEGER NOT NULL, tf INTEGER NOT NULL,
                                          PRIMARY KEY(term, row_id)) WITHOUT ROWID;
                    CREATE TABLE df(term TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID;
                """)
                db.executemany("INSERT INTO meta(key,value) VALUES(?,?)", [
                    ("format", "v4-bm25-disk-v1"), ("status", "building"),
                    ("channel", channel), ("fingerprint", fingerprint),
                    ("rows", "0"), ("total_length", "0"),
                ])
        self._check(require_complete=False)

    def _check(self, *, require_complete: bool) -> dict[str, str]:
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key,value FROM meta"))
        if (meta.get("format") != "v4-bm25-disk-v1" or meta.get("channel") != self.channel
                or meta.get("fingerprint") != self.fingerprint):
            raise ValueError("BM25 disk index identity/fingerprint mismatch")
        if require_complete and meta.get("status") != "complete":
            raise ValueError("BM25 disk index is incomplete")
        return meta

    def append_batch(self, records: Iterable[tuple[int, str | None, str]]) -> int:
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("Cannot append to a completed BM25 index")
        added = 0
        next_row = int(meta["rows"])
        total_length = int(meta["total_length"])
        with sqlite3.connect(self.path) as db:
            for official_id, passage_id, text in records:
                if type(official_id) is not int or not isinstance(text, str):
                    raise ValueError("BM25 records need integer parent IDs and string text")
                counts = Counter(tokens(text))
                length = sum(counts.values())
                db.execute("INSERT INTO docs(row_id,official_id,passage_id,length) VALUES(?,?,?,?)",
                           (next_row, official_id, passage_id, length))
                db.executemany("INSERT INTO postings(term,row_id,tf) VALUES(?,?,?)",
                               ((term, next_row, tf) for term, tf in counts.items()))
                db.executemany("INSERT INTO df(term,count) VALUES(?,1) "
                               "ON CONFLICT(term) DO UPDATE SET count=count+1",
                               ((term,) for term in counts))
                next_row += 1
                total_length += length
                added += 1
            db.execute("UPDATE meta SET value=? WHERE key='rows'", (str(next_row),))
            db.execute("UPDATE meta SET value=? WHERE key='total_length'", (str(total_length),))
        return added

    def complete(self) -> dict:
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("BM25 index is already complete")
        with sqlite3.connect(self.path) as db:
            actual_rows = db.execute("SELECT count(*) FROM docs").fetchone()[0]
            actual_length = db.execute("SELECT coalesce(sum(length),0) FROM docs").fetchone()[0]
            if actual_rows != int(meta["rows"]) or actual_length != int(meta["total_length"]):
                raise ValueError("BM25 document table differs from metadata")
            db.execute("UPDATE meta SET value='complete' WHERE key='status'")
        return {"format": "v4-bm25-disk-v1", "status": "complete", "channel": self.channel,
                "fingerprint": self.fingerprint, "rows": actual_rows,
                "total_length": actual_length}

    def truncate_to_rows(self, rows: int) -> int:
        """Discard rows beyond the last committed partition checkpoint."""
        meta = self._check(require_complete=False)
        current = int(meta["rows"])
        if meta["status"] != "building" or not 0 <= rows <= current:
            raise ValueError("Invalid BM25 checkpoint rollback")
        if rows == current:
            return rows
        with sqlite3.connect(self.path) as db:
            removed_terms = db.execute(
                "SELECT term,count(*) FROM postings WHERE row_id>=? GROUP BY term", (rows,)).fetchall()
            db.executemany("UPDATE df SET count=count-? WHERE term=?",
                           ((count, term) for term, count in removed_terms))
            db.execute("DELETE FROM df WHERE count<=0")
            db.execute("DELETE FROM postings WHERE row_id>=?", (rows,))
            db.execute("DELETE FROM docs WHERE row_id>=?", (rows,))
            total_length = db.execute("SELECT coalesce(sum(length),0) FROM docs").fetchone()[0]
            db.execute("UPDATE meta SET value=? WHERE key='rows'", (str(rows),))
            db.execute("UPDATE meta SET value=? WHERE key='total_length'", (str(total_length),))
        return rows

    def search(self, query: str, limit: int = 20, *, parent_id: int | None = None) -> list[Hit]:
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
                if parent_id is None:
                    rows = db.execute("SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id FROM postings p "
                                      "JOIN docs d ON d.row_id=p.row_id WHERE p.term=?", (term,))
                else:
                    rows = db.execute("SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id FROM postings p "
                                      "JOIN docs d ON d.row_id=p.row_id "
                                      "WHERE p.term=? AND d.official_id=?", (term, parent_id))
                for row in rows:
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

    def search_parent(self, query: str, parent_id: int, limit: int = 1) -> list[Hit]:
        return self.search(query, limit, parent_id=parent_id)
