"""Stable passage identity and content fingerprints."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def passage_id(*, chunker_version: str, strategy_id: str, doc_id: int,
               passage_index: int, source_sha256: str, char_start: int,
               char_end: int) -> str:
    payload = (
        f"{chunker_version}\0{strategy_id}\0{doc_id}\0{passage_index}\0"
        f"{source_sha256}\0{char_start}\0{char_end}"
    ).encode("ascii")
    return "p2_" + hashlib.sha256(payload).hexdigest()
