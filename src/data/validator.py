"""Corpus and dataset validation checks."""

from typing import Iterable, List
from src.data.schema import ChunkRecord, QueryRecord


def validate_chunks_integrity(chunks: Iterable[ChunkRecord]) -> List[str]:
    """Inspects chunk collection for integrity issues: duplicate chunk_ids, empty text, invalid indexes."""
    errors = []
    seen_chunks = set()
    doc_chunk_indices = {}

    for c in chunks:
        if c.chunk_id in seen_chunks:
            errors.append(f"Duplicate chunk_id: {c.chunk_id}")
        seen_chunks.add(c.chunk_id)

        if not c.text.strip():
            errors.append(f"Empty text in chunk_id: {c.chunk_id}")

        if c.doc_id not in doc_chunk_indices:
            doc_chunk_indices[c.doc_id] = set()
        if c.chunk_index in doc_chunk_indices[c.doc_id]:
            errors.append(f"Duplicate chunk_index {c.chunk_index} for doc_id {c.doc_id}")
        doc_chunk_indices[c.doc_id].add(c.chunk_index)

    return errors


def validate_queries_integrity(queries: Iterable[QueryRecord]) -> List[str]:
    """Inspects query collection for integrity issues: duplicate query IDs or empty queries."""
    errors = []
    seen_queries = set()

    for q in queries:
        if q.id in seen_queries:
            errors.append(f"Duplicate query ID: {q.id}")
        seen_queries.add(q.id)

        if not q.query.strip():
            errors.append(f"Empty query string in query ID: {q.id}")

    return errors
