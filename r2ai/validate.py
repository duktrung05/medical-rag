"""Validate a submission ZIP against the competition spec before uploading.

A malformed submission is simply not scored, and the public phase only allows 10
a day, so every upload is checked here first. Rules enforced come from the task's
Submission Instructions page.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/vibiomir"
EXPECTED_QUERIES = 1200
# Uploads above this are rejected by the platform as "Bad or empty zip file",
# which looks like corruption rather than a size limit. Observed: 37.8 MB
# accepted, 57.3 MB rejected.
UPLOAD_CAP_MB = 50.0


def _as_doc_id(value: object, allow_string: bool) -> int | None:
    """Normalise a doc id to int, optionally tolerating the string spelling."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if allow_string and isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def validate(zip_path: Path, strict_ids: bool = True,
             allow_string_ids: bool = False, *, query_path: Path | None = None,
             corpus_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            errors.append(f"ZIP must contain exactly one file, found {len(names)}: {names}")
            return errors
        if "/" in names[0]:
            errors.append(f"File must not be inside a subdirectory: {names[0]}")
        payload = zf.read(names[0])

    try:
        records = json.loads(payload)
    except json.JSONDecodeError as exc:
        return [f"Not valid JSON: {exc}"]
    if not isinstance(records, list):
        return ["Top level must be a JSON array"]

    expected = set(pq.read_table(query_path or RAW / "query.parquet", columns=["id"]).to_pydict()["id"])
    corpus_ids = set(pq.read_table(corpus_path or RAW / "links_corpus.parquet", columns=["id"]).to_pydict()["id"])

    seen: set[int] = set()
    empty_docs = empty_chunks = 0
    doc_counts: list[int] = []
    chunk_counts: list[int] = []
    unknown_docs = 0

    for i, rec in enumerate(records):
        where = f"record[{i}]"
        if not isinstance(rec, dict):
            errors.append(f"{where}: not an object")
            continue
        qid = rec.get("id")
        if not isinstance(qid, int) or isinstance(qid, bool):
            errors.append(f"{where}: 'id' must be an integer, got {type(qid).__name__}")
            continue
        if qid in seen:
            errors.append(f"{where}: duplicate query id {qid}")
        seen.add(qid)

        docs = rec.get("relevant_docs")
        if not isinstance(docs, list):
            errors.append(f"{where} (id={qid}): 'relevant_docs' must be a list")
            docs = []
        for d in docs:
            norm = _as_doc_id(d, allow_string_ids)
            if norm is None:
                errors.append(f"{where} (id={qid}): doc id {d!r} is not an integer")
            elif strict_ids and norm not in corpus_ids:
                unknown_docs += 1
        normalized_docs = [_as_doc_id(d, allow_string_ids) for d in docs]
        valid_docs = {d for d in normalized_docs if d is not None}
        if len(valid_docs) != len([d for d in normalized_docs if d is not None]):
            errors.append(f"id={qid}: duplicate doc ids in relevant_docs")
        doc_counts.append(len(docs))
        if not docs:
            empty_docs += 1

        chunks = rec.get("relevant_chunks")
        if not isinstance(chunks, list):
            errors.append(f"{where} (id={qid}): 'relevant_chunks' must be a list")
            chunks = []
        seen_chunks: set[tuple[int, str]] = set()
        for j, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                errors.append(f"{where} (id={qid}) chunk[{j}]: not an object")
                continue
            cd, text = ch.get("doc_id"), ch.get("chunk_text")
            norm = _as_doc_id(cd, allow_string_ids)
            if norm is None:
                errors.append(f"{where} (id={qid}) chunk[{j}]: doc_id must be an integer")
            elif strict_ids and norm not in corpus_ids:
                unknown_docs += 1
            if norm is not None and norm not in valid_docs:
                errors.append(f"{where} (id={qid}) chunk[{j}]: parent document {norm} absent from relevant_docs")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{where} (id={qid}) chunk[{j}]: chunk_text must be a non-empty string")
            elif norm is not None:
                key = norm, text
                if key in seen_chunks:
                    errors.append(f"{where} (id={qid}) chunk[{j}]: duplicate evidence chunk")
                seen_chunks.add(key)
            if "chunk_order" in ch:
                order = ch["chunk_order"]
                if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                    errors.append(f"{where} (id={qid}) chunk[{j}]: chunk_order must be a non-negative int")
        chunk_counts.append(len(chunks))
        if not chunks:
            empty_chunks += 1

    missing = expected - seen
    if missing:
        errors.append(f"Missing {len(missing)} of {len(expected)} query ids "
                      f"(e.g. {sorted(missing)[:8]}) -- submission will not be scored")
    extra = seen - expected
    if extra:
        errors.append(f"{len(extra)} query ids not in query.parquet (e.g. {sorted(extra)[:8]})")
    if unknown_docs:
        errors.append(f"{unknown_docs} doc ids are not present in links_corpus.parquet "
                      "(they can never count as positives)")

    size_mb = zip_path.stat().st_size / 1e6
    if size_mb > UPLOAD_CAP_MB:
        errors.append(f"ZIP is {size_mb:.1f} MB, over the ~{UPLOAD_CAP_MB:.0f} MB upload "
                      "cap -- the platform rejects it as 'Bad or empty zip file'. "
                      "Trim relevant_chunks (r2ai/reslice.py); doc ids cost almost nothing.")
    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    print(f"records            : {len(records)}")
    print(f"unique query ids   : {len(seen)} / {len(expected)}")
    print(f"docs per query     : avg {avg(doc_counts):.2f}  (empty: {empty_docs})")
    print(f"chunks per query   : avg {avg(chunk_counts):.2f}  (empty: {empty_chunks})")
    print(f"zip size           : {size_mb:.1f} MB")
    for w in warnings[:10]:
        print(f"WARN  {w}")
    if empty_docs:
        print(f"WARN  {empty_docs} queries have no predicted docs -> contribute 0 to Doc F2")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--no-strict-ids", action="store_true")
    ap.add_argument("--allow-string-ids", action="store_true",
                    help="accept doc ids spelled as JSON strings (for the A/B test)")
    ap.add_argument("--query-parquet", type=Path, help="Override expected query IDs for a labeled subset")
    ap.add_argument("--corpus-parquet", type=Path, help="Override the document ID corpus")
    a = ap.parse_args()
    problems = validate(a.zip_path, strict_ids=not a.no_strict_ids,
                        allow_string_ids=a.allow_string_ids, query_path=a.query_parquet,
                        corpus_path=a.corpus_parquet)
    if problems:
        print(f"\nINVALID ({len(problems)} problems):")
        for p in problems[:25]:
            print(f"  - {p}")
        sys.exit(1)
    print("\nVALID")
