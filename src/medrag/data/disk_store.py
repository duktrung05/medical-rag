"""V4 source and passage lookup keyed by immutable release identities."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.medrag.core.contracts import Passage, Source
from src.medrag.evidence.verifier import verify


class DiskCorpusStore:
    def __init__(self, path: Path, corpus_version: str, fingerprint: str, *, create: bool = False):
        self.path = path
        self.corpus_version = corpus_version
        self.fingerprint = fingerprint
        if create:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE sources(official_id INTEGER PRIMARY KEY, url TEXT NOT NULL,
                                         title TEXT NOT NULL, text TEXT NOT NULL, origin TEXT NOT NULL,
                                         raw_sha256 TEXT NOT NULL, metadata TEXT NOT NULL);
                    CREATE TABLE passages(passage_id TEXT PRIMARY KEY, official_id INTEGER NOT NULL,
                                          url TEXT NOT NULL, text TEXT NOT NULL,
                                          source_start INTEGER NOT NULL, source_end INTEGER NOT NULL,
                                          source_sha256 TEXT NOT NULL, language TEXT NOT NULL,
                                          provenance TEXT NOT NULL,
                                          FOREIGN KEY(official_id) REFERENCES sources(official_id));
                    CREATE INDEX passages_parent ON passages(official_id);
                """)
                db.executemany("INSERT INTO meta(key,value) VALUES(?,?)", [
                    ("format", "medrag-v4-disk-store-v1"), ("status", "building"),
                    ("corpus_version", corpus_version), ("fingerprint", fingerprint),
                    ("sources", "0"), ("passages", "0"),
                ])
        self._check(require_complete=False)

    def _check(self, *, require_complete: bool) -> dict[str, str]:
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key,value FROM meta"))
        if (meta.get("format") != "medrag-v4-disk-store-v1"
                or meta.get("corpus_version") != self.corpus_version
                or meta.get("fingerprint") != self.fingerprint):
            raise ValueError("V4 disk store identity/fingerprint mismatch")
        if require_complete and meta.get("status") != "complete":
            raise ValueError("V4 disk store is incomplete")
        return meta

    def append_sources(self, sources: list[Source]) -> int:
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("Cannot append sources to a complete store")
        with sqlite3.connect(self.path) as db:
            for source in sources:
                if source.corpus_version != self.corpus_version:
                    raise ValueError("Source corpus version mismatch")
                db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?)", (
                    source.official_id, source.url, source.title, source.text, source.origin,
                    source.raw_sha256, json.dumps(source.metadata, ensure_ascii=False, sort_keys=True)))
            db.execute("UPDATE meta SET value=? WHERE key='sources'",
                       (str(int(meta["sources"]) + len(sources)),))
        return len(sources)

    def append_passages(self, passages: list[Passage]) -> int:
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("Cannot append passages to a complete store")
        parents = {p.official_id for p in passages}
        with sqlite3.connect(self.path) as db:
            source_rows = db.execute(
                f"SELECT * FROM sources WHERE official_id IN ({','.join('?' for _ in parents)})",
                list(parents)).fetchall() if parents else []
            db.row_factory = sqlite3.Row
            source_by_id = {int(row[0]): Source(self.corpus_version, int(row[0]), row[1], row[2],
                                                row[3], row[4], row[5], json.loads(row[6]))
                            for row in source_rows}
            for passage in passages:
                source = source_by_id.get(passage.official_id)
                if source is None or not verify(passage, source):
                    raise ValueError(f"Passage does not match its source: {passage.passage_id}")
                db.execute("INSERT INTO passages VALUES(?,?,?,?,?,?,?,?,?)", (
                    passage.passage_id, passage.official_id, passage.url, passage.text,
                    passage.source_start, passage.source_end, passage.source_sha256,
                    passage.language, passage.provenance))
            db.execute("UPDATE meta SET value=? WHERE key='passages'",
                       (str(int(meta["passages"]) + len(passages)),))
        return len(passages)

    def complete(self) -> dict:
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("V4 disk store is already complete")
        with sqlite3.connect(self.path) as db:
            sources = db.execute("SELECT count(*) FROM sources").fetchone()[0]
            passages = db.execute("SELECT count(*) FROM passages").fetchone()[0]
            parents = db.execute("SELECT count(DISTINCT official_id) FROM passages").fetchone()[0]
            if (sources != int(meta["sources"]) or passages != int(meta["passages"])
                    or parents != sources):
                raise ValueError("Disk store count/parent coverage mismatch")
            db.execute("UPDATE meta SET value='complete' WHERE key='status'")
        return {"status": "complete", "sources": sources, "passages": passages}

    def rollback_part(self, official_ids: set[int]) -> tuple[int, int]:
        """Remove only one uncommitted release partition before retrying it."""
        meta = self._check(require_complete=False)
        if meta["status"] != "building":
            raise ValueError("Cannot roll back a completed store")
        with sqlite3.connect(self.path) as db:
            for offset in range(0, len(official_ids), 800):
                batch = list(sorted(official_ids))[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                db.execute(f"DELETE FROM passages WHERE official_id IN ({placeholders})", batch)
                db.execute(f"DELETE FROM sources WHERE official_id IN ({placeholders})", batch)
            sources = db.execute("SELECT count(*) FROM sources").fetchone()[0]
            passages = db.execute("SELECT count(*) FROM passages").fetchone()[0]
            db.execute("UPDATE meta SET value=? WHERE key='sources'", (str(sources),))
            db.execute("UPDATE meta SET value=? WHERE key='passages'", (str(passages),))
        return sources, passages

    def source(self, official_id: int) -> Source:
        self._check(require_complete=True)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            row = db.execute("SELECT url,title,text,origin,raw_sha256,metadata FROM sources "
                             "WHERE official_id=?", (official_id,)).fetchone()
        if row is None:
            raise KeyError(official_id)
        return Source(self.corpus_version, official_id, row[0], row[1], row[2], row[3], row[4],
                      json.loads(row[5]))

    def sources_batch(self, official_ids: set[int]) -> dict[int, Source]:
        self._check(require_complete=True)
        found = {}
        ordered = sorted(official_ids)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            for offset in range(0, len(ordered), 800):
                batch = ordered[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                for row in db.execute(
                        f"SELECT official_id,url,title,text,origin,raw_sha256,metadata FROM sources "
                        f"WHERE official_id IN ({placeholders})", batch):
                    found[int(row[0])] = Source(self.corpus_version, int(row[0]), row[1], row[2],
                                                row[3], row[4], row[5], json.loads(row[6]))
        if set(found) != official_ids:
            raise ValueError("Source lookup is missing eligible parent IDs")
        return found

    def passage(self, passage_id: str) -> Passage:
        self._check(require_complete=True)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            row = db.execute("SELECT official_id,url,text,source_start,source_end,source_sha256,"
                             "language,provenance FROM passages WHERE passage_id=?",
                             (passage_id,)).fetchone()
        if row is None:
            raise KeyError(passage_id)
        return Passage(passage_id, self.corpus_version, *row)

    def passages_batch(self, passage_ids: list[str]) -> dict[str, Passage]:
        self._check(require_complete=True)
        found = {}
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            for offset in range(0, len(passage_ids), 800):
                batch = passage_ids[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                for row in db.execute(
                        f"SELECT passage_id,official_id,url,text,source_start,source_end,"
                        f"source_sha256,language,provenance FROM passages "
                        f"WHERE passage_id IN ({placeholders})", batch):
                    found[row[0]] = Passage(row[0], self.corpus_version, *row[1:])
        if set(found) != set(passage_ids):
            raise ValueError("Passage lookup is missing eligible passage IDs")
        return found

    def passages_for_parent(self, official_id: int) -> list[Passage]:
        self._check(require_complete=True)
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT passage_id,official_id,url,text,source_start,source_end,"
                "source_sha256,language,provenance FROM passages WHERE official_id=? "
                "ORDER BY source_start,source_end,passage_id", (official_id,)).fetchall()
        return [Passage(row[0], self.corpus_version, *row[1:]) for row in rows]

    def counts(self) -> tuple[int, int]:
        meta = self._check(require_complete=True)
        return int(meta["sources"]), int(meta["passages"])


class SourceLookup:
    def __init__(self, store: DiskCorpusStore):
        self.store = store

    def __getitem__(self, key: int) -> Source:
        return self.store.source(key)

    def __len__(self) -> int:
        return self.store.counts()[0]


class PassageLookup:
    def __init__(self, store: DiskCorpusStore):
        self.store = store

    def __getitem__(self, key: str) -> Passage:
        return self.store.passage(key)

    def __len__(self) -> int:
        return self.store.counts()[1]

    def for_parent(self, official_id: int) -> list[Passage]:
        return self.store.passages_for_parent(official_id)
