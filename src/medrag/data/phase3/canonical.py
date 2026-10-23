"""Fail-closed canonical selection for verified Phase 3 content groups."""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa


CANONICAL_POLICY_VERSION = "vibiomir-phase3-canonical-selection-v1"
CANONICAL_SCHEMA_VERSION = "vibiomir-phase3-canonical-representatives-v1"
MEMBERSHIP_SCHEMA_VERSION = "vibiomir-phase3-canonical-membership-v1"


CANONICAL_REPRESENTATIVE_SCHEMA = pa.schema([
    ("entity_type", pa.string()),
    ("content_group_id", pa.string()),
    ("group_size", pa.int32()),
    ("canonical_doc_id", pa.int64()),
    ("canonical_passage_id", pa.string()),
    ("canonical_official_url", pa.string()),
    ("canonical_char_start", pa.int64()),
    ("canonical_char_end", pa.int64()),
    ("text_quality_review_required", pa.bool_()),
    ("quality_flag_count", pa.int32()),
    ("text_completeness_chars", pa.int64()),
    ("provenance_score", pa.int32()),
    ("selection_reason", pa.string()),
    ("selection_reason_chain", pa.list_(pa.string())),
    ("selection_policy_version", pa.string()),
])


CANONICAL_MEMBERSHIP_SCHEMA = pa.schema([
    ("entity_type", pa.string()),
    ("content_group_id", pa.string()),
    ("group_size", pa.int32()),
    ("member_rank", pa.int32()),
    ("is_canonical", pa.bool_()),
    ("doc_id", pa.int64()),
    ("passage_id", pa.string()),
    ("official_url", pa.string()),
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
    ("source_text_sha256", pa.string()),
    ("normalized_content_sha256", pa.string()),
    ("text_quality_review_required", pa.bool_()),
    ("quality_flag_count", pa.int32()),
    ("text_completeness_chars", pa.int64()),
    ("provenance_score", pa.int32()),
    ("selection_reason", pa.string()),
    ("selection_policy_version", pa.string()),
])


def non_whitespace_length(value: str | None) -> int:
    return sum(not character.isspace() for character in (value or ""))


def provenance_score(document: dict) -> int:
    """Count independently useful provenance fields without source favoritism."""
    fields = (
        "phase0_raw_sha256",
        "raw_gzip_sha256",
        "raw_body_sha256",
        "source_document_sha256",
        "phase0_raw_path",
        "source_document_path",
        "pmid",
        "extractor_version",
    )
    return sum(bool(document.get(field)) for field in fields)


def document_selection_key(document: dict) -> tuple:
    """Lower sorts first; the final doc_id makes the choice total and stable."""
    return (
        bool(document.get("quality_review_required", True)),
        len(document.get("quality_flags") or []),
        -non_whitespace_length(document.get("normalized_text")),
        -provenance_score(document),
        -(float(document.get("language_confidence") or 0.0)),
        str(document.get("source_kind") or ""),
        int(document["doc_id"]),
    )


def passage_selection_key(passage: dict) -> tuple:
    document = passage["document"]
    return (
        *document_selection_key(document)[:-1],
        -non_whitespace_length(passage.get("passage_text")),
        int(passage.get("passage_index", 0)),
        int(document["doc_id"]),
        str(passage["passage_id"]),
    )


def _first_difference(left: tuple, right: tuple, labels: tuple[str, ...]) -> str:
    for index, label in enumerate(labels):
        if left[index] != right[index]:
            return label
    return labels[-1]


def select_representative(
    members: list[dict], *, entity_type: str
) -> tuple[dict, list[dict], str, list[str]]:
    """Select one member from a verified exact group and explain the decision."""
    if entity_type == "document":
        key: Callable[[dict], tuple] = document_selection_key
        labels = (
            "quality_review_status",
            "quality_flag_count",
            "text_completeness",
            "source_provenance",
            "language_confidence",
            "source_kind_stable_order",
            "lowest_doc_id_tiebreak",
        )
    elif entity_type == "passage":
        key = passage_selection_key
        labels = (
            "parent_quality_review_status",
            "parent_quality_flag_count",
            "parent_text_completeness",
            "parent_source_provenance",
            "parent_language_confidence",
            "parent_source_kind_stable_order",
            "passage_text_completeness",
            "lowest_passage_index",
            "lowest_doc_id_tiebreak",
            "lexical_passage_id_tiebreak",
        )
    else:
        raise ValueError(f"Unsupported entity_type: {entity_type}")
    if len(members) < 2:
        raise ValueError("Canonical selection requires a verified duplicate group")
    ranked = sorted(members, key=key)
    reason = _first_difference(key(ranked[0]), key(ranked[1]), labels)
    chain = [f"{label}={value!r}" for label, value in zip(labels, key(ranked[0]))]
    return ranked[0], ranked, reason, chain
