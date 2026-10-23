from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256


@dataclass(frozen=True)
class Source:
    corpus_version: str
    official_id: int
    url: str
    title: str
    text: str
    origin: str
    raw_sha256: str
    metadata: dict = field(default_factory=dict)

    @property
    def text_sha256(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Passage:
    passage_id: str
    corpus_version: str
    official_id: int
    url: str
    text: str
    source_start: int | None
    source_end: int | None
    source_sha256: str
    language: str
    provenance: str


@dataclass(frozen=True)
class Hit:
    official_id: int
    passage_id: str | None
    score: float
    channel: str
    rank: int


@dataclass(frozen=True)
class Fused:
    official_id: int
    passage_id: str | None
    score: float
    ranks: dict[str, int]

