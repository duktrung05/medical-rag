"""Offline BGE-M3 tokenizer adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path


TOKENIZER_NAME = "BAAI/bge-m3"
TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class BgeM3Tokenizer:
    def __init__(self, tokenizer_json: Path):
        from tokenizers import Tokenizer

        tokenizer_json = tokenizer_json.resolve()
        if not tokenizer_json.is_file():
            raise FileNotFoundError(f"BGE-M3 tokenizer is not cached: {tokenizer_json}")
        self.path = tokenizer_json
        self.sha256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def encode_ids(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids


def default_tokenizer_path(repo_root: Path) -> Path:
    return (
        repo_root
        / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots"
        / TOKENIZER_REVISION
        / "tokenizer.json"
    )
