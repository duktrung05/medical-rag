"""Bounded retrieval V2: source windows, two lexical branches and dense fusion."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HAN = re.compile(r"[\u3400-\u9fff]")
WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    out = []
    for match in WORDS.finditer(text.casefold()):
        word = match.group()
        if HAN.search(word):
            out.extend(word[i:i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                out.append(word)
        else:
            out.append(word)
    return out


def lexical_text(text: str) -> str:
    return " ".join(tokens(text))


def windows(text: str, target_words: int = 192, overlap: float = 0.15):
    """Return exact substring windows; offsets refer only to the extracted chunk."""
    if target_words <= 0 or not 0 <= overlap < 1:
        raise ValueError("invalid window settings")
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not spans:
        return
    if len(spans) <= target_words:
        yield 0, len(text), text
        return
    # Prefer sentence/paragraph ends close to the target; never cross a source chunk.
    ends = {m.end() for m in re.finditer(r"[.!?。！？;；\n]", text)}
    start_word = 0
    while start_word < len(spans):
        stop_word = min(len(spans), start_word + target_words)
        candidates = [i for i in range(max(start_word + 1, stop_word - target_words // 4), stop_word)
                      if spans[i - 1][1] in ends]
        if candidates and stop_word < len(spans):
            stop_word = candidates[-1]
        start = spans[start_word][0]
        end = spans[stop_word - 1][1]
        yield start, end, text[start:end]
        if stop_word == len(spans):
            break
        start_word = max(start_word + 1, stop_word - int(target_words * overlap))


def passage_id(doc_id: int, chunk_index: int, start: int, end: int, text: str) -> str:
    key = f"{doc_id}:{chunk_index}:{start}:{end}:".encode() + text.encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:24]


def build_sparse(source: Path, db_path: Path, *, target_words: int = 192) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing V2 index: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        con.executescript("""
        CREATE TABLE source_chunks(doc_id INTEGER, chunk_index INTEGER, title TEXT, text TEXT,
          PRIMARY KEY(doc_id,chunk_index));
        CREATE VIRTUAL TABLE document_fts USING fts5(content, tokenize='unicode61');
        CREATE TABLE document_rows(rowid INTEGER PRIMARY KEY, doc_id INTEGER, chunk_index INTEGER);
        CREATE TABLE passages(rowid INTEGER PRIMARY KEY, passage_id TEXT UNIQUE, doc_id INTEGER,
          chunk_index INTEGER, start_char INTEGER, end_char INTEGER, title TEXT, text TEXT);
        CREATE VIRTUAL TABLE passage_fts USING fts5(content, tokenize='unicode61');
        CREATE INDEX passages_doc ON passages(doc_id);
        """)
        doc_rows = passage_rows = invalid = duplicate = 0
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                doc_id, chunk_index = row.get("doc_id"), row.get("chunk_index")
                title, body = row.get("title") or "", row.get("text")
                if not isinstance(doc_id, int) or not isinstance(chunk_index, int) or not isinstance(body, str) or not body.strip():
                    invalid += 1
                    continue
                try:
                    cur = con.execute("INSERT INTO source_chunks VALUES(?,?,?,?)", (doc_id, chunk_index, title, body))
                except sqlite3.IntegrityError:
                    duplicate += 1
                    continue
                doc_rows += 1
                rowid = cur.lastrowid
                con.execute("INSERT INTO document_rows VALUES(?,?,?)", (rowid, doc_id, chunk_index))
                con.execute("INSERT INTO document_fts(rowid,content) VALUES(?,?)",
                            (rowid, lexical_text(title + " " + body)))
                for start, end, passage in windows(body, target_words):
                    pid = passage_id(doc_id, chunk_index, start, end, passage)
                    cur = con.execute("INSERT INTO passages(passage_id,doc_id,chunk_index,start_char,end_char,title,text) VALUES(?,?,?,?,?,?,?)",
                                      (pid, doc_id, chunk_index, start, end, title, passage))
                    con.execute("INSERT INTO passage_fts(rowid,content) VALUES(?,?)",
                                (cur.lastrowid, lexical_text(title + " " + passage)))
                    passage_rows += 1
            con.commit()
        return {"source_rows": doc_rows, "passages": passage_rows, "invalid": invalid, "duplicate": duplicate}
    finally:
        con.close()


def _match_query(query: str) -> str:
    terms = list(dict.fromkeys(tokens(query)))[:32]
    return " OR ".join('"' + x.replace('"', '""') + '"' for x in terms)


def sparse_search(con: sqlite3.Connection, query: str, branch: str, top_k: int) -> list[tuple]:
    match = _match_query(query)
    if not match:
        return []
    if branch == "passage":
        return con.execute("""SELECT p.rowid,p.doc_id,-bm25(passage_fts) AS score
          FROM passage_fts JOIN passages p ON p.rowid=passage_fts.rowid
          WHERE passage_fts MATCH ? ORDER BY score DESC LIMIT ?""", (match, top_k)).fetchall()
    if branch == "document":
        rows = con.execute("""SELECT d.doc_id,-bm25(document_fts) AS score
          FROM document_fts JOIN document_rows d ON d.rowid=document_fts.rowid
          WHERE document_fts MATCH ? ORDER BY score DESC LIMIT ?""", (match, top_k * 8)).fetchall()
        best: dict[int, float] = {}
        for doc_id, score in rows:
            best[doc_id] = max(best.get(doc_id, -float("inf")), score)
        return sorted(best.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    raise ValueError(branch)


def dense_search(index_path: Path, query_vector: np.ndarray, top_k: int, index=None) -> list[tuple[int, float]]:
    import faiss
    if index is None:
        index = faiss.read_index(str(index_path))
    vec = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    scores, ids = index.search(vec, top_k)
    return [(int(i) + 1, float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]


def build_dense(db_path: Path, index_path: Path, model_path: Path, batch_size: int = 8,
                device: str = "cpu") -> dict:
    """Bounded exact pilot with batch encoding and resumable FAISS checkpoints."""
    import faiss
    from sentence_transformers import SentenceTransformer
    if index_path.exists():
        raise FileExistsError(index_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    con = sqlite3.connect(db_path)
    total = con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    if not total:
        con.close()
        raise ValueError("no passages")
    partial = index_path.with_suffix(".partial.faiss")
    checkpoint = index_path.with_suffix(".progress.json")
    db_stat = db_path.stat()
    source_key = {"db_bytes": db_stat.st_size, "db_mtime_ns": db_stat.st_mtime_ns,
                  "total_passages": total, "model_path": str(model_path), "device": device}
    if checkpoint.exists() != partial.exists():
        con.close()
        raise RuntimeError("dense checkpoint and partial index must both exist or both be absent")
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["source"] != source_key:
            con.close()
            raise RuntimeError("dense resume source/model fingerprint changed")
        index = faiss.read_index(str(partial))
        processed = state["processed"]
        if index.ntotal != processed:
            con.close()
            raise RuntimeError("dense checkpoint count mismatch")
    else:
        index = None
        processed = 0
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    cursor = con.execute("SELECT rowid,title,text FROM passages WHERE rowid>? ORDER BY rowid", (processed,))
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        if rows[0][0] != processed + 1 or rows[-1][0] != processed + len(rows):
            con.close()
            raise RuntimeError("passage rowids must be contiguous for FAISS mapping")
        texts = [f"{title}\n{text}" for _, title, text in rows]
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True).astype("float32")
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        processed += len(rows)
        if processed % (batch_size * 16) == 0 or processed == total:
            faiss.write_index(index, str(partial))
            checkpoint.write_text(json.dumps({"source": source_key, "processed": processed}), encoding="utf-8")
    con.close()
    if processed != total:
        raise RuntimeError("dense index incomplete")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, index_path)
    checkpoint.unlink()
    return {"vectors": processed, "dimension": index.d, "normalized": True,
            "metric": "inner_product", "model_path": str(model_path),
            "rowid_mapping": "FAISS zero-based id + 1 = passages.rowid",
            "checkpoint_interval_batches": 16, "device": device}


def fuse(con: sqlite3.Connection, query: str, dense_hits: list[tuple[int, float]],
         *, doc_k: int = 200, passage_k: int = 500, budget: int = 200,
         diverse: bool = True, rrf_k: int = 60) -> list[dict]:
    docs = sparse_search(con, query, "document", doc_k)
    passages = sparse_search(con, query, "passage", passage_k)
    evidence: dict[int, dict] = {}

    def add(rowid: int, doc_id: int, source: str, rank: int):
        item = evidence.setdefault(rowid, {"rowid": rowid, "doc_id": doc_id,
                                          "sources": {}, "fusion_score": 0.0})
        if source not in item["sources"]:
            item["sources"][source] = rank
            item["fusion_score"] += 1.0 / (rrf_k + rank)

    for rank, (rowid, doc_id, _) in enumerate(passages, 1):
        add(rowid, doc_id, "passage_bm25", rank)
    for rank, (rowid, _) in enumerate(dense_hits, 1):
        doc_id = con.execute("SELECT doc_id FROM passages WHERE rowid=?", (rowid,)).fetchone()
        if doc_id:
            add(rowid, doc_id[0], "passage_dense", rank)
    for rank, (doc_id, _) in enumerate(docs, 1):
        # Promote the best direct passage hit, or a source passage for this document.
        candidate = next((r for r, d, _ in passages if d == doc_id), None)
        if candidate is None:
            row = con.execute("SELECT rowid FROM passages WHERE doc_id=? ORDER BY rowid LIMIT 1", (doc_id,)).fetchone()
            candidate = row[0] if row else None
        if candidate is not None:
            add(candidate, doc_id, "document_bm25", rank)
    ranked = sorted(evidence.values(), key=lambda x: (-x["fusion_score"], x["doc_id"], x["rowid"]))
    if diverse:
        seen = set()
        first = []
        rest = []
        for item in ranked:
            (rest if item["doc_id"] in seen else first).append(item)
            seen.add(item["doc_id"])
        ranked = (first + rest)[:budget]
    else:
        ranked = ranked[:budget]
    for rank, item in enumerate(ranked, 1):
        row = con.execute("SELECT passage_id,chunk_index,start_char,end_char,title,text FROM passages WHERE rowid=?",
                          (item["rowid"],)).fetchone()
        item.update(zip(("passage_id", "source_chunk_index", "start_char", "end_char", "title", "text"), row))
        item["fusion_rank"] = rank
    return ranked


def score_and_select(candidates: list[dict], *, max_docs: int = 10, max_chunks: int = 10) -> dict:
    by_doc = defaultdict(list)
    for item in candidates:
        by_doc[item["doc_id"]].append(item)
    doc_scores = []
    for doc_id, items in by_doc.items():
        scores = sorted((x.get("rerank_logit", x["fusion_score"]) for x in items), reverse=True)
        # Simple rank-based support bonus; no uncalibrated BM25/cosine addition.
        first_rank = min(x["fusion_rank"] for x in items)
        score = scores[0] + (0.1 * scores[1] if len(scores) > 1 else 0) + 1 / (60 + first_rank)
        doc_scores.append((doc_id, score))
    docs = [doc_id for doc_id, _ in sorted(doc_scores, key=lambda x: (-x[1], x[0]))[:max_docs]]
    accepted = []
    seen = set()
    for item in sorted(candidates, key=lambda x: (-x.get("rerank_logit", x["fusion_score"]), x["fusion_rank"])):
        key = item["doc_id"], item["text"]
        if item["doc_id"] in docs and key not in seen:
            accepted.append({"doc_id": item["doc_id"], "chunk_text": item["text"]})
            seen.add(key)
        if len(accepted) >= max_chunks:
            break
    return {"relevant_docs": docs, "relevant_chunks": accepted}


def assert_grounded(con: sqlite3.Connection, prediction: dict) -> None:
    docs = set(prediction["relevant_docs"])
    for chunk in prediction["relevant_chunks"]:
        if chunk["doc_id"] not in docs:
            raise ValueError("evidence parent missing")
        rows = con.execute("SELECT text FROM source_chunks WHERE doc_id=?", (chunk["doc_id"],))
        if not any(chunk["chunk_text"] in source for (source,) in rows):
            raise ValueError("evidence is not an exact source substring")
