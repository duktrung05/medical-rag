"""Versioned Arrow contracts and strategy definitions for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


CHUNKER_VERSION = "vibiomir-hierarchical-chunker-v2"
PASSAGE_SCHEMA_VERSION = "vibiomir-phase2-passage-v1"


@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    target_tokens: int
    max_tokens: int
    min_chunk_tokens: int
    overlap_tokens: int

    def __post_init__(self) -> None:
        if not 0 < self.min_chunk_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Require 0 < min <= target <= max tokens")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("Require 0 <= overlap < target tokens")


STRATEGIES = (
    ChunkingStrategy("A", 192, 256, 32, 32),
    ChunkingStrategy("B", 256, 384, 40, 48),
    ChunkingStrategy("C", 384, 512, 48, 64),
)


PASSAGE_SCHEMA = pa.schema([
    ("passage_id", pa.string()),
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("passage_index", pa.int32()),
    ("passage_text", pa.string()),
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
    ("source_text_sha256", pa.string()),
    ("text_representation", pa.string()),
    ("text_representation_sha256", pa.string()),
    ("token_count", pa.int32()),
    ("language", pa.string()),
    ("section_id", pa.string()),
    ("section_heading", pa.string()),
    ("strategy_id", pa.string()),
    ("chunker_version", pa.string()),
    ("tokenizer_name", pa.string()),
    ("tokenizer_revision", pa.string()),
    ("phase1_release_id", pa.string()),
    ("corpus_revision", pa.string()),
    ("processing_run_id", pa.string()),
])


PILOT_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("source_kind", pa.string()),
    ("domain", pa.string()),
    ("language", pa.string()),
    ("length_chars", pa.int64()),
    ("length_bin", pa.string()),
    ("has_heading", pa.bool_()),
    ("paragraph_profile", pa.string()),
    ("mixed_language", pa.bool_()),
    ("medical_measurement", pa.bool_()),
    ("selection_score", pa.uint64()),
])


MAPPING_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("strategy_id", pa.string()),
    ("passage_count", pa.int32()),
    ("first_passage_id", pa.string()),
    ("last_passage_id", pa.string()),
    ("covered_non_whitespace_chars", pa.int64()),
    ("source_non_whitespace_chars", pa.int64()),
])


FAILURE_SCHEMA = pa.schema([
    ("strategy_id", pa.string()),
    ("doc_id", pa.int64()),
    ("passage_id", pa.string()),
    ("failure_type", pa.string()),
    ("details", pa.string()),
])
