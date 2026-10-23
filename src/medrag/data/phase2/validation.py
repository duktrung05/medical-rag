"""Passage integrity and structural coverage checks."""

from __future__ import annotations

from .schema import ChunkingStrategy


def covered_non_whitespace(text: str, passages: list[dict]) -> tuple[int, int]:
    covered = bytearray(len(text))
    for passage in passages:
        covered[passage["char_start"]:passage["char_end"]] = b"\x01" * (
            passage["char_end"] - passage["char_start"]
        )
    source = sum(not char.isspace() for char in text)
    count = sum(bool(covered[index]) for index, char in enumerate(text) if not char.isspace())
    return count, source


def validate_document_passages(row: dict, passages: list[dict], tokenizer,
                               strategy: ChunkingStrategy) -> list[dict]:
    failures: list[dict] = []
    text = row["normalized_text"]
    seen_ids: set[str] = set()
    previous_index = -1
    for passage in passages:
        ident = passage["passage_id"]
        if ident in seen_ids:
            failures.append({"passage_id": ident, "failure_type": "duplicate_passage_id",
                             "details": "Duplicate within document"})
        seen_ids.add(ident)
        if passage["passage_index"] != previous_index + 1:
            failures.append({"passage_id": ident, "failure_type": "passage_order",
                             "details": "Non-contiguous passage_index"})
        previous_index = passage["passage_index"]
        start, end = passage["char_start"], passage["char_end"]
        if not 0 <= start < end <= len(text) or text[start:end] != passage["passage_text"]:
            failures.append({"passage_id": ident, "failure_type": "source_slice",
                             "details": f"Invalid [{start}:{end}] source mapping"})
            continue
        exact_count = tokenizer.count(passage["passage_text"])
        if exact_count != passage["token_count"]:
            failures.append({"passage_id": ident, "failure_type": "token_count",
                             "details": f"stored={passage['token_count']} exact={exact_count}"})
        if exact_count > strategy.max_tokens:
            failures.append({"passage_id": ident, "failure_type": "max_tokens",
                             "details": f"tokens={exact_count} max={strategy.max_tokens}"})
    covered, source = covered_non_whitespace(text, passages)
    if covered != source:
        failures.append({"passage_id": "", "failure_type": "structural_coverage",
                         "details": f"covered={covered} source={source}"})
    return failures
