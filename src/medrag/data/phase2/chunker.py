"""Hierarchical passage construction from immutable normalized text."""

from __future__ import annotations

from .provenance import passage_id, sha256_text
from .schema import CHUNKER_VERSION, ChunkingStrategy
from .segmentation import segment_range
from .structure import find_sections
from .tokenizer import TOKENIZER_NAME, TOKENIZER_REVISION, BgeM3Tokenizer


def _repair_short_section_passages(text, values, tokenizer, strategy):
    """Attach a tiny section orphan backward, or forward when backward is too large."""
    if len(values) < 2:
        return values
    values = list(values)
    index = 0
    while index < len(values):
        section, span = values[index]
        if span.token_count >= strategy.min_chunk_tokens:
            index += 1
            continue
        if section.heading is not None:
            index += 1
            continue
        if index > 0:
            previous_section, previous = values[index - 1]
            count = tokenizer.count(text[previous.start:span.end])
            if count <= strategy.max_tokens:
                values[index - 1] = (previous_section, type(span)(previous.start, span.end, count))
                values.pop(index)
                index = max(0, index - 1)
                continue
        if index + 1 < len(values):
            _, following = values[index + 1]
            count = tokenizer.count(text[span.start:following.end])
            if count <= strategy.max_tokens:
                values[index + 1] = (section, type(span)(span.start, following.end, count))
                values.pop(index)
                continue
        index += 1
    return values


def chunk_document(
    row: dict,
    tokenizer: BgeM3Tokenizer,
    strategy: ChunkingStrategy,
    *,
    phase1_release_id: str,
    corpus_revision: str,
    processing_run_id: str,
) -> list[dict]:
    text = row["normalized_text"]
    expected_hash = row["normalized_text_sha256"]
    if sha256_text(text) != expected_hash:
        raise ValueError(f"Normalized text checksum mismatch for doc_id={row['doc_id']}")
    output: list[dict] = []
    values = []
    for section in find_sections(text, row.get("title")):
        for span in segment_range(text, section.start, section.end, tokenizer, strategy):
            values.append((section, span))
    values = _repair_short_section_passages(text, values, tokenizer, strategy)
    for section, span in values:
        value = text[span.start:span.end]
        index = len(output)
        output.append({
                "passage_id": passage_id(
                    chunker_version=CHUNKER_VERSION,
                    strategy_id=strategy.strategy_id,
                    doc_id=int(row["doc_id"]),
                    passage_index=index,
                    source_sha256=expected_hash,
                    char_start=span.start,
                    char_end=span.end,
                ),
                "doc_id": int(row["doc_id"]),
                "official_url": row["official_url"],
                "passage_index": index,
                "passage_text": value,
                "char_start": span.start,
                "char_end": span.end,
                "source_text_sha256": row["source_text_sha256"],
                "text_representation": "normalized_text",
                "text_representation_sha256": expected_hash,
                "token_count": span.token_count,
                "language": row["detected_language"],
                "section_id": section.section_id,
                "section_heading": section.heading,
                "strategy_id": strategy.strategy_id,
                "chunker_version": CHUNKER_VERSION,
                "tokenizer_name": TOKENIZER_NAME,
                "tokenizer_revision": TOKENIZER_REVISION,
                "phase1_release_id": phase1_release_id,
                "corpus_revision": corpus_revision,
                "processing_run_id": processing_run_id,
        })
    return output
