"""Deterministic exact-document grouping for ViBioMIR Phase 3."""

from __future__ import annotations

import re

import pyarrow as pa


GROUPING_POLICY_VERSION = "vibiomir-phase3-exact-document-v1"
EXACT_DOCUMENT_SCHEMA_VERSION = "vibiomir-phase3-exact-document-groups-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


EXACT_DOCUMENT_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("source_text_sha256", pa.string()),
    ("normalized_text_sha256", pa.string()),
    ("raw_exact_group_id", pa.string()),
    ("normalized_exact_group_id", pa.string()),
    ("document_content_group_id", pa.string()),
    ("raw_exact_group_size", pa.int32()),
    ("normalized_exact_group_size", pa.int32()),
    ("raw_exact_duplicate", pa.bool_()),
    ("normalized_exact_duplicate", pa.bool_()),
    ("exact_match_type", pa.string()),
    ("normalization_version", pa.string()),
    ("phase1_release_id", pa.string()),
    ("grouping_scope", pa.string()),
    ("grouping_policy_version", pa.string()),
])


def validate_sha256(value: str, *, field: str) -> str:
    """Return a canonical SHA-256 string or fail closed."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def raw_group_id(source_text_sha256: str) -> str:
    digest = validate_sha256(source_text_sha256, field="source_text_sha256")
    return f"deraw_{digest}"


def normalized_group_id(normalized_text_sha256: str) -> str:
    digest = validate_sha256(normalized_text_sha256, field="normalized_text_sha256")
    return f"denorm_{digest}"


def content_group_id(normalized_text_sha256: str) -> str:
    digest = validate_sha256(normalized_text_sha256, field="normalized_text_sha256")
    return f"dcg_{digest}"


def exact_match_type(raw_size: int, normalized_size: int) -> str:
    """Classify one occurrence without merging or removing it."""
    if raw_size < 1 or normalized_size < 1:
        raise ValueError("Group sizes must be positive")
    if raw_size > 1 and normalized_size > 1:
        return "raw_and_normalized_exact"
    if raw_size > 1:
        return "raw_exact_only"
    if normalized_size > 1:
        return "normalized_exact_only"
    return "unique"


def exact_group_row(
    document: dict,
    *,
    raw_size: int,
    normalized_size: int,
    normalization_version: str,
    phase1_release_id: str,
    grouping_scope: str,
) -> dict:
    """Create the immutable one-row-per-document grouping record."""
    source_hash = validate_sha256(
        document["source_text_sha256"], field="source_text_sha256"
    )
    normalized_hash = validate_sha256(
        document["normalized_text_sha256"], field="normalized_text_sha256"
    )
    return {
        "doc_id": int(document["doc_id"]),
        "official_url": document["official_url"],
        "source_text_sha256": source_hash,
        "normalized_text_sha256": normalized_hash,
        "raw_exact_group_id": raw_group_id(source_hash),
        "normalized_exact_group_id": normalized_group_id(normalized_hash),
        "document_content_group_id": content_group_id(normalized_hash),
        "raw_exact_group_size": raw_size,
        "normalized_exact_group_size": normalized_size,
        "raw_exact_duplicate": raw_size > 1,
        "normalized_exact_duplicate": normalized_size > 1,
        "exact_match_type": exact_match_type(raw_size, normalized_size),
        "normalization_version": normalization_version,
        "phase1_release_id": phase1_release_id,
        "grouping_scope": grouping_scope,
        "grouping_policy_version": GROUPING_POLICY_VERSION,
    }


def verify_group_content(
    baselines: dict[bytes, tuple[int, str]],
    digest: bytes,
    *,
    doc_id: int,
    text: str,
    label: str,
) -> None:
    """Verify exact string equality for every repeated SHA-256 group."""
    previous = baselines.get(digest)
    if previous is None:
        baselines[digest] = (doc_id, text)
        return
    previous_doc_id, previous_text = previous
    if text != previous_text:
        raise ValueError(
            f"SHA-256 collision or corrupt {label} group: "
            f"doc_id={previous_doc_id} differs from doc_id={doc_id}"
        )

