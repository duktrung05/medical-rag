"""Self-contained ranking cache and CPU-only submission IO."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

from r2ai.evidence_selector import EvidenceCandidate, QueryRanking

CACHE_VERSION = 1


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_ranking_cache(path: Path, rankings: list[QueryRanking], metadata: dict) -> None:
    if len({r.query_id for r in rankings}) != len(rankings):
        raise ValueError("Duplicate query IDs in cache")
    if path.exists():
        raise FileExistsError(f"Ranking cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary)
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as stream:
            header = {"kind": "r2ai_evidence_rankings", "version": CACHE_VERSION,
                      "num_queries": len(rankings), "metadata": metadata}
            stream.write(json.dumps(header, ensure_ascii=False, allow_nan=False) + "\n")
            for ranking in rankings:
                stream.write(json.dumps(asdict(ranking), ensure_ascii=False, allow_nan=False) + "\n")
        # Exclusive publication prevents overwriting an existing experiment.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_ranking_cache(path: Path) -> tuple[list[QueryRanking], dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream, "{}"))
        if header.get("kind") != "r2ai_evidence_rankings" or header.get("version") != CACHE_VERSION:
            raise ValueError("Unsupported ranking cache schema")
        rankings = []
        for line in stream:
            row = json.loads(line)
            rankings.append(QueryRanking(row["query_id"], row["document_order"],
                                         [EvidenceCandidate(**c) for c in row["candidates"]],
                                         row.get("query_text", "")))
    if len(rankings) != header.get("num_queries") or len({r.query_id for r in rankings}) != len(rankings):
        raise ValueError("Truncated cache or duplicate query IDs")
    if not isinstance(header.get("metadata"), dict):
        raise TypeError("Cache metadata must be an object")
    return rankings, header["metadata"]


def write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False)
                                 for r in predictions) + "\n]\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("predictions.json", payload)


def selection_summary(predictions: list[dict]) -> dict:
    from collections import Counter

    chunks = Counter(len(p["relevant_chunks"]) for p in predictions)
    docs = Counter(len(p["relevant_docs"]) for p in predictions)
    n = len(predictions)
    return {"num_queries": n, "average_chunks": sum(k * v for k, v in chunks.items()) / n if n else 0,
            "average_docs": sum(k * v for k, v in docs.items()) / n if n else 0,
            "empty_chunk_rate": chunks[0] / n if n else 0,
            "empty_doc_rate": docs[0] / n if n else 0,
            "chunk_count_distribution": dict(sorted(chunks.items())),
            "doc_count_distribution": dict(sorted(docs.items()))}
