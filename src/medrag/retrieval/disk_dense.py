"""Exact passage dense search from a resumable, identity-mapped FP32 matrix."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from src.medrag.core.contracts import Hit


class DiskDense:
    def __init__(self, path: Path, encoder, fingerprint: str, *, create: bool = False,
                 expected_rows: int | None = None, dimension: int | None = None,
                 block_rows: int = 32768):
        self.path = path
        self.encoder = encoder
        self.fingerprint = fingerprint
        self.block_rows = block_rows
        self.mapping = path / "mapping.sqlite"
        self.matrix_path = path / "vectors.npy"
        if create:
            if path.exists():
                raise FileExistsError(path)
            if not expected_rows or not dimension or expected_rows <= 0 or dimension <= 0:
                raise ValueError("Dense create requires positive expected rows and dimension")
            path.mkdir(parents=True)
            matrix = np.lib.format.open_memmap(self.matrix_path, mode="w+", dtype="float32",
                                               shape=(expected_rows, dimension))
            matrix.flush()
            del matrix
            with sqlite3.connect(self.mapping) as db:
                db.executescript("""
                    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE rows(row_id INTEGER PRIMARY KEY, official_id INTEGER NOT NULL,
                                      passage_id TEXT NOT NULL UNIQUE, input_sha256 TEXT NOT NULL);
                    CREATE INDEX rows_parent ON rows(official_id);
                """)
                db.executemany("INSERT INTO meta VALUES(?,?)", [
                    ("format", "medrag-v4-dense-fp32-v1"), ("status", "building"),
                    ("fingerprint", fingerprint), ("rows", "0"),
                    ("expected_rows", str(expected_rows)), ("dimension", str(dimension)),
                ])
        self.meta = self._check(require_complete=False)
        if block_rows <= 0:
            raise ValueError("Dense block rows must be positive")
        shape = np.load(self.matrix_path, mmap_mode="r").shape
        if shape != (int(self.meta["expected_rows"]), int(self.meta["dimension"])):
            raise ValueError("Dense matrix shape differs from mapping metadata")

    def _check(self, *, require_complete: bool) -> dict[str, str]:
        with sqlite3.connect(f"file:{self.mapping}?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key,value FROM meta"))
        if meta.get("format") != "medrag-v4-dense-fp32-v1" or meta.get("fingerprint") != self.fingerprint:
            raise ValueError("Dense mapping fingerprint/format mismatch")
        if require_complete and meta.get("status") != "complete":
            raise ValueError("Dense mapping is incomplete")
        return meta

    def count(self) -> int:
        return int(self._check(require_complete=False)["rows"])

    def identity(self, row_id: int) -> tuple[int, str, str]:
        with sqlite3.connect(f"file:{self.mapping}?mode=ro", uri=True) as db:
            row = db.execute("SELECT official_id,passage_id,input_sha256 FROM rows WHERE row_id=?",
                             (row_id,)).fetchone()
        if row is None:
            raise KeyError(row_id)
        return int(row[0]), str(row[1]), str(row[2])

    def append_batch(self, records: list[tuple[int, str, str]], vectors) -> int:
        """Commit vectors before mapping rows so a crash can safely overwrite uncommitted rows."""
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("Cannot append to a completed dense index")
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (len(records), int(meta["dimension"])) or not np.isfinite(array).all():
            raise ValueError("Dense vector count/dimension/finite check failed")
        norms = np.linalg.norm(array, axis=1)
        if len(records) and (np.any(norms < 0.99) or np.any(norms > 1.01)):
            raise ValueError("Dense vectors must be L2 normalized")
        start = int(meta["rows"])
        end = start + len(records)
        if end > int(meta["expected_rows"]):
            raise ValueError("Dense index exceeds expected passage count")
        for official_id, passage_id, input_sha in records:
            if type(official_id) is not int or not passage_id or len(input_sha) != 64:
                raise ValueError("Dense row identity/input hash is invalid")
        matrix = np.load(self.matrix_path, mmap_mode="r+")
        matrix[start:end] = array
        matrix.flush()
        del matrix
        with sqlite3.connect(self.mapping) as db:
            db.executemany("INSERT INTO rows VALUES(?,?,?,?)", (
                (start + index, parent, passage_id, input_sha)
                for index, (parent, passage_id, input_sha) in enumerate(records)))
            db.execute("UPDATE meta SET value=? WHERE key='rows'", (str(end),))
        return len(records)

    def complete(self) -> dict:
        meta = self._check(require_complete=False)
        count = int(meta["rows"])
        if meta["status"] != "building" or count != int(meta["expected_rows"]):
            raise ValueError("Dense index is incomplete or already complete")
        with sqlite3.connect(self.mapping) as db:
            actual = db.execute("SELECT count(*) FROM rows").fetchone()[0]
            maximum = db.execute("SELECT max(row_id) FROM rows").fetchone()[0]
            if actual != count or maximum != count - 1:
                raise ValueError("Dense row mapping count/sequence mismatch")
            db.execute("UPDATE meta SET value='complete' WHERE key='status'")
        return {"status": "complete", "rows": count, "dimension": int(meta["dimension"]),
                "vectors_sha256": _sha(self.matrix_path), "mapping_sha256": _sha(self.mapping)}

    def search(self, query: str, limit: int = 20) -> list[Hit]:
        if limit <= 0:
            raise ValueError("Dense result limit must be positive")
        meta = self._check(require_complete=True)
        encode_query = getattr(self.encoder, "encode_query", None)
        if encode_query is None:
            encode_query = self.encoder.encode
        query_vector = np.asarray(encode_query([query])[0], dtype=np.float64)
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
            # Match the reference cosine, including each stored vector's actual norm.
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
                query_sql = f"SELECT row_id,official_id,passage_id FROM rows WHERE row_id IN ({','.join('?' for _ in row_ids)})"
                identities.update((row_id, (parent, passage_id)) for row_id, parent, passage_id
                                  in db.execute(query_sql, row_ids))
        if len(identities) != len(candidates):
            raise ValueError("Dense mapping is missing candidate rows")
        ordered = sorted(candidates, key=lambda item: (-item[0], identities[item[1]][0],
                                                        identities[item[1]][1]))[:limit]
        return [Hit(*identities[row_id], score, "dense", rank)
                for rank, (score, row_id) in enumerate(ordered, 1)]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload(text: str, title: str) -> str:
    return f"{title}\n{text}" if title else text


def input_sha256(text: str, title: str) -> str:
    return hashlib.sha256(payload(text, title).encode("utf-8")).hexdigest()
