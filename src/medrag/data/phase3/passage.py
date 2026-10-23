"""Deterministic content identity for Phase 3 passage occurrences."""

from __future__ import annotations

import re

import pyarrow as pa


PASSAGE_GROUPING_POLICY_VERSION = "vibiomir-phase3-passage-content-v1"
PASSAGE_CONTENT_SCHEMA_VERSION = "vibiomir-phase3-passage-content-groups-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


SOURCE_OFFSETS_TYPE = pa.struct([
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
])


PASSAGE_CONTENT_SCHEMA = pa.schema([
    ("passage_id", pa.string()),
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("passage_index", pa.int32()),
    ("content_sha256", pa.string()),
    ("normalized_content_hash", pa.string()),
    ("raw_content_group_id", pa.string()),
    ("passage_content_group_id", pa.string()),
    ("raw_content_group_size", pa.int32()),
    ("content_group_size", pa.int32()),
    ("distinct_document_count", pa.int32()),
    ("cross_document_content", pa.bool_()),
    ("source_offsets", SOURCE_OFFSETS_TYPE),
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
    ("source_text_sha256", pa.string()),
    ("text_representation_sha256", pa.string()),
    ("chunker_version", pa.string()),
    ("normalization_version", pa.string()),
    ("phase2_release_id", pa.string()),
    ("grouping_scope", pa.string()),
    ("grouping_policy_version", pa.string()),
])


def validate_sha256(value: str, *, field: str) -> str:
    """Return a canonical SHA-256 string or fail closed."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def raw_content_group_id(content_sha256: str) -> str:
    digest = validate_sha256(content_sha256, field="content_sha256")
    return f"pcraw_{digest}"


def passage_content_group_id(normalized_content_hash: str) -> str:
    digest = validate_sha256(
        normalized_content_hash, field="normalized_content_hash"
    )
    return f"pcg_{digest}"


def passage_group_row(
    passage: dict,
    *,
    content_sha256: str,
    normalized_content_hash: str,
    raw_group_size: int,
    content_group_size: int,
    distinct_document_count: int,
    normalization_version: str,
    phase2_release_id: str,
    grouping_scope: str,
) -> dict:
    """Create one immutable grouping record for one source occurrence."""
    raw_hash = validate_sha256(content_sha256, field="content_sha256")
    normalized_hash = validate_sha256(
        normalized_content_hash, field="normalized_content_hash"
    )
    source_hash = validate_sha256(
        passage["source_text_sha256"], field="source_text_sha256"
    )
    representation_hash = validate_sha256(
        passage["text_representation_sha256"],
        field="text_representation_sha256",
    )
    if raw_group_size < 1 or content_group_size < 1:
        raise ValueError("Passage group sizes must be positive")
    if not 1 <= distinct_document_count <= content_group_size:
        raise ValueError("Invalid distinct document count")
    char_start = int(passage["char_start"])
    char_end = int(passage["char_end"])
    if char_start < 0 or char_end <= char_start:
        raise ValueError("Invalid passage source offsets")
    passage_id = passage["passage_id"]
    official_url = passage["official_url"]
    if not isinstance(passage_id, str) or not passage_id:
        raise ValueError("Missing passage_id")
    if not isinstance(official_url, str) or not official_url:
        raise ValueError("Missing official_url")
    return {
        "passage_id": passage_id,
        "doc_id": int(passage["doc_id"]),
        "official_url": official_url,
        "passage_index": int(passage["passage_index"]),
        "content_sha256": raw_hash,
        "normalized_content_hash": normalized_hash,
        "raw_content_group_id": raw_content_group_id(raw_hash),
        "passage_content_group_id": passage_content_group_id(normalized_hash),
        "raw_content_group_size": int(raw_group_size),
        "content_group_size": int(content_group_size),
        "distinct_document_count": int(distinct_document_count),
        "cross_document_content": distinct_document_count > 1,
        "source_offsets": {"char_start": char_start, "char_end": char_end},
        "char_start": char_start,
        "char_end": char_end,
        "source_text_sha256": source_hash,
        "text_representation_sha256": representation_hash,
        "chunker_version": passage["chunker_version"],
        "normalization_version": normalization_version,
        "phase2_release_id": phase2_release_id,
        "grouping_scope": grouping_scope,
        "grouping_policy_version": PASSAGE_GROUPING_POLICY_VERSION,
    }


def verify_group_content(
    baselines: dict[bytes, tuple[str, str]],
    digest: bytes,
    *,
    passage_id: str,
    text: str,
    label: str,
) -> None:
    """Verify equality for repeated hashes and fail closed on collisions."""
    previous = baselines.get(digest)
    if previous is None:
        baselines[digest] = (passage_id, text)
        return
    previous_passage_id, previous_text = previous
    if text != previous_text:
        raise ValueError(
            f"SHA-256 collision or corrupt {label} group: "
            f"passage_id={previous_passage_id} differs from "
            f"passage_id={passage_id}"
        )
