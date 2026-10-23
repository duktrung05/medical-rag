"""Durable V4 pair-score cache keyed by exact model input and scoring contract."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path


def _canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


class ScoreCache:
    def __init__(self, path: Path, contract: dict, *, create: bool = False):
        self.path = path
        self.contract = contract
        self.fingerprint = hashlib.sha256(_canonical(contract).encode()).hexdigest()
        if create:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                    CREATE TABLE scores(cache_key TEXT PRIMARY KEY, score REAL NOT NULL,
                                        query_sha256 TEXT NOT NULL, official_id INTEGER NOT NULL,
                                        passage_id TEXT NOT NULL, input_sha256 TEXT NOT NULL);
                """)
                db.executemany("INSERT INTO meta VALUES(?,?)", [
                    ("format", "medrag-v4-score-cache-v1"), ("contract", _canonical(contract)),
                    ("fingerprint", self.fingerprint),
                ])
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key,value FROM meta"))
        if (meta.get("format") != "medrag-v4-score-cache-v1"
                or meta.get("fingerprint") != self.fingerprint
                or meta.get("contract") != _canonical(contract)):
            raise ValueError("Score cache contract/fingerprint mismatch")

    def _key(self, query: str, official_id: int, passage_id: str, payload: str) -> str:
        return hashlib.sha256(_canonical({"contract": self.fingerprint, "query": query,
                                          "official_id": official_id, "passage_id": passage_id,
                                          "payload": payload}).encode()).hexdigest()

    def score(self, query: str, rows: list[tuple[int, str, str]], reranker) -> list[float]:
        if len({(official_id, passage_id) for official_id, passage_id, _ in rows}) != len(rows):
            raise ValueError("Duplicate reranker pair identity")
        keys = [self._key(query, doc_id, passage_id, payload)
                for doc_id, passage_id, payload in rows]
        cached = {}
        with sqlite3.connect(self.path) as db:
            for key in keys:
                result = db.execute("SELECT score FROM scores WHERE cache_key=?", (key,)).fetchone()
                if result is not None:
                    cached[key] = float(result[0])
        missing = [(index, rows[index], key) for index, key in enumerate(keys) if key not in cached]
        if missing:
            values = reranker.score(query, [row[2] for _, row, _ in missing])
            if len(values) != len(missing):
                raise ValueError("Reranker returned a different number of scores")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError("Reranker returned a non-finite score")
            query_sha = hashlib.sha256(query.encode()).hexdigest()
            with sqlite3.connect(self.path) as db:
                for (_, (doc_id, passage_id, payload), key), value in zip(missing, values, strict=True):
                    input_sha = hashlib.sha256(payload.encode()).hexdigest()
                    score = float(value)
                    db.execute("INSERT INTO scores VALUES(?,?,?,?,?,?)",
                               (key, score, query_sha, doc_id, passage_id, input_sha))
                    cached[key] = score
        return [cached[key] for key in keys]
