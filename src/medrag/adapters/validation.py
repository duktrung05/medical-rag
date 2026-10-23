"""Strict V4 submission validation and reproducible ZIP packaging."""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

from src.medrag.adapters.vibiomir_submission import records
from src.medrag.evidence.context import derive_adjacent_evidence, passages_for_parent
from src.medrag.evidence.selector import SelectionPolicy
from src.medrag.evidence.verifier import verify


def canonical_payload(predictions: list[dict]) -> bytes:
    return json.dumps(predictions, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def parse_payload(payload: bytes) -> list[dict]:
    parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
                        parse_constant=_reject_constant)
    if not isinstance(parsed, list):
        raise ValueError("Prediction payload must be a JSON array")
    return parsed


def _validate_registered_evidence(item: dict, doc_id: int, source, store, *,
                                  reranker_enabled: bool) -> None:
    passage = store.passage(item["passage_id"])
    if (passage.official_id != doc_id or item["text"] != passage.text
            or item["start"] != passage.source_start or item["end"] != passage.source_end
            or not verify(passage, source)):
        raise ValueError(f"Ungrounded evidence: query document={doc_id}, passage={passage.passage_id}")
    if reranker_enabled:
        value = item.get("rerank_score")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Missing or invalid rerank score: {passage.passage_id}")
        if item.get("score_status") != "reranked":
            raise ValueError(f"Unscored evidence selected: {passage.passage_id}")


def _validate_derived_evidence(item: dict, doc_id: int, source, store,
                               policy: SelectionPolicy, *, reranker_enabled: bool) -> None:
    if item.get("evidence_kind") != "adjacent_source_span_v1":
        raise ValueError(f"Invalid derived evidence kind: document={doc_id}")
    seed = item.get("seed_evidence")
    if not isinstance(seed, dict):
        raise ValueError(f"Missing derived evidence seed: document={doc_id}")
    _validate_registered_evidence(seed, doc_id, source, store,
                                  reranker_enabled=reranker_enabled)
    expected = derive_adjacent_evidence(
        seed, source, passages_for_parent(store, doc_id), policy)
    if item != expected:
        raise ValueError(f"Derived evidence contract mismatch: document={doc_id}")
    if reranker_enabled:
        value = item.get("rerank_score")
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value != seed["rerank_score"]):
            raise ValueError(f"Derived evidence score differs from seed: document={doc_id}")


def validate_selected(query_ids: list[int], selected: dict[int, list[dict]], store,
                      policy: SelectionPolicy, *, result_limit: int,
                      reranker_enabled: bool, official_ids: set[int]) -> list[dict]:
    """Check V4 identities, exact grounding, caps and selected/serialized equality."""
    if not query_ids or any(type(qid) is not int for qid in query_ids):
        raise ValueError("Query IDs must be nonempty integers")
    if len(query_ids) != len(set(query_ids)) or set(selected) != set(query_ids):
        raise ValueError("Selected query coverage differs from official query IDs")
    if result_limit <= 0:
        raise ValueError("Result limit must be positive")
    predictions = records(query_ids, selected)
    for qid, prediction in zip(query_ids, predictions, strict=True):
        documents = selected[qid]
        if len(documents) > min(result_limit, policy.max_docs):
            raise ValueError(f"Document cap exceeded: query={qid}")
        if sum(len(doc["evidence"]) for doc in documents) > policy.max_chunks:
            raise ValueError(f"Chunk cap exceeded: query={qid}")
        seen_docs: set[int] = set()
        seen_text: set[tuple[int, str]] = set()
        expected_docs = []
        expected_chunks = []
        for document in documents:
            doc_id = document["official_id"]
            if type(doc_id) is not int or doc_id not in official_ids or doc_id in seen_docs:
                raise ValueError(f"Invalid or duplicate document ID: query={qid}, doc={doc_id}")
            evidence = document["evidence"]
            if (len(evidence) > policy.chunks_per_doc
                    or (not evidence and policy.document_selection == "with_chunks")):
                raise ValueError(f"Invalid evidence count: query={qid}, doc={doc_id}")
            source = store.source(doc_id)
            if policy.document_selection == "independent":
                anchor = document.get("document_evidence")
                if not isinstance(anchor, dict):
                    raise ValueError(f"Missing document evidence: query={qid}, doc={doc_id}")
                _validate_registered_evidence(anchor, doc_id, source, store,
                                              reranker_enabled=reranker_enabled)
            seen_docs.add(doc_id)
            expected_docs.append(doc_id)
            for item in evidence:
                if policy.chunk_context == "adjacent":
                    _validate_derived_evidence(item, doc_id, source, store, policy,
                                               reranker_enabled=reranker_enabled)
                else:
                    _validate_registered_evidence(item, doc_id, source, store,
                                                  reranker_enabled=reranker_enabled)
                identity = (doc_id, item["text"])
                if identity in seen_text:
                    raise ValueError(f"Duplicate parent/text evidence: query={qid}, doc={doc_id}")
                seen_text.add(identity)
                expected_chunks.append({"doc_id": doc_id, "chunk_text": item["text"]})
        if (prediction != {"id": qid, "relevant_docs": expected_docs,
                           "relevant_chunks": expected_chunks}):
            raise ValueError(f"Selected/exported mismatch: query={qid}")
    return predictions


def validate_payload(payload: bytes, *, query_ids: set[int], official_ids: set[int],
                     policy: SelectionPolicy, result_limit: int) -> list[dict]:
    predictions = parse_payload(payload)
    if len(predictions) != len(query_ids):
        raise ValueError("Prediction count differs from official query count")
    seen_queries = set()
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"id", "relevant_docs", "relevant_chunks"}:
            raise ValueError("Prediction record has invalid keys")
        qid = row["id"]
        if type(qid) is not int or qid in seen_queries or qid not in query_ids:
            raise ValueError("Invalid or duplicate query ID")
        seen_queries.add(qid)
        docs, chunks = row["relevant_docs"], row["relevant_chunks"]
        if not isinstance(docs, list) or not isinstance(chunks, list):
            raise ValueError("Prediction docs/chunks must be lists")
        if (len(docs) > min(result_limit, policy.max_docs)
                or len(chunks) > policy.max_chunks):
            raise ValueError("Prediction output cap exceeded")
        if (any(type(doc_id) is not int or doc_id not in official_ids for doc_id in docs)
                or len(set(docs)) != len(docs)):
            raise ValueError("Invalid or duplicate document ID")
        counts = Counter()
        seen_chunks = set()
        for chunk in chunks:
            if not isinstance(chunk, dict) or set(chunk) != {"doc_id", "chunk_text"}:
                raise ValueError("Invalid chunk keys")
            doc_id, chunk_text = chunk["doc_id"], chunk["chunk_text"]
            if (type(doc_id) is not int or doc_id not in docs
                    or not isinstance(chunk_text, str) or not chunk_text.strip()):
                raise ValueError("Invalid chunk parent or text")
            if (doc_id, chunk_text) in seen_chunks:
                raise ValueError("Duplicate parent/text evidence")
            seen_chunks.add((doc_id, chunk_text))
            counts[doc_id] += 1
            if counts[doc_id] > policy.chunks_per_doc:
                raise ValueError("Per-document evidence cap exceeded")
        if policy.document_selection == "with_chunks" and set(docs) != set(counts):
            raise ValueError("Document without selected evidence")
    if seen_queries != query_ids:
        raise ValueError("Missing official query IDs")
    return predictions


def package_zip(path: Path, payload: bytes) -> str:
    """Write a single-entry ZIP with stable metadata; never replace an existing file."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("predictions.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_zip(path: Path, *, query_ids: set[int], official_ids: set[int],
                 policy: SelectionPolicy, result_limit: int,
                 max_bytes: int = 50_000_000) -> dict:
    if path.stat().st_size > max_bytes:
        raise ValueError("ZIP exceeds configured operational size cap")
    with zipfile.ZipFile(path, "r") as archive:
        if archive.namelist() != ["predictions.json"]:
            raise ValueError("ZIP must contain only predictions.json at root")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        payload = archive.read("predictions.json")
    parsed = validate_payload(payload, query_ids=query_ids, official_ids=official_ids,
                              policy=policy, result_limit=result_limit)
    return {"status": "pass", "query_count": len(parsed),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "zip_bytes": path.stat().st_size}
