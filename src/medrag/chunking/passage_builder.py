from __future__ import annotations

import re
from hashlib import sha256

from src.medrag.core.contracts import Passage, Source
from src.medrag.data.metadata import language_hint

TOKEN = re.compile(r"[\u4e00-\u9fff]|[\w]+|[^\w\s]", re.UNICODE)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in TOKEN.finditer(text)]


def build_passages(source: Source, max_tokens: int = 256, overlap_tokens: int = 32) -> list[Passage]:
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Invalid token window")
    spans = token_spans(source.text)
    language = language_hint(source.text).value
    output = []
    step = max_tokens - overlap_tokens
    for i in range(0, len(spans), step):
        batch = spans[i:i + max_tokens]
        start, end = batch[0][0], batch[-1][1]
        text = source.text[start:end]
        digest = sha256(f"{source.corpus_version}:{source.official_id}:{start}:{end}".encode()).hexdigest()[:16]
        output.append(Passage(f"v4:{digest}", source.corpus_version, source.official_id, source.url,
                              text, start, end, source.text_sha256, language, "extracted_source"))
        if i + max_tokens >= len(spans):
            break
    return output

