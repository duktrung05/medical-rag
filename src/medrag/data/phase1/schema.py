"""Arrow contracts for the versioned Phase 1 clean-document corpus."""

from __future__ import annotations

import pyarrow as pa


PHASE1_SCHEMA_VERSION = "vibiomir-phase1-clean-document-v3"

CLEAN_DOCUMENT_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("source_kind", pa.string()),
    ("content_group", pa.string()),
    ("title", pa.string()),
    ("source_text", pa.string()),
    ("normalized_text", pa.string()),
    ("source_text_sha256", pa.string()),
    ("normalized_text_sha256", pa.string()),
    ("normalization_version", pa.string()),
    ("normalization_changed", pa.bool_()),
    ("quality_flags", pa.list_(pa.string())),
    ("quality_reasons", pa.list_(pa.string())),
    ("quality_review_required", pa.bool_()),
    ("quality_validation_version", pa.string()),
    ("phase0_raw_path", pa.string()),
    ("phase0_raw_sha256", pa.string()),
    ("raw_path", pa.string()),
    ("raw_gzip_sha256", pa.string()),
    ("raw_body_sha256", pa.string()),
    ("encoding", pa.string()),
    ("encoding_source", pa.string()),
    ("extraction_method", pa.string()),
    ("language_metadata", pa.string()),
    ("language_metadata_normalized", pa.list_(pa.string())),
    ("detected_language", pa.string()),
    ("language_confidence", pa.float32()),
    ("language_detection_method", pa.string()),
    ("language_detection_version", pa.string()),
    ("language_metadata_agrees", pa.bool_()),
    ("pmid", pa.string()),
    ("extractor_version", pa.string()),
    ("source_document_path", pa.string()),
    ("source_document_sha256", pa.string()),
])
