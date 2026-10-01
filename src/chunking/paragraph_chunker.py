"""Paragraph-first chunking with exact character offsets and token limits."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 256
    max_tokens: int = 384
    overlap_tokens: int = 48
    min_chunk_tokens: int = 40

    def __post_init__(self) -> None:
        if not (0 < self.min_chunk_tokens <= self.target_tokens <= self.max_tokens):
            raise ValueError("Require 0 < min_chunk_tokens <= target_tokens <= max_tokens")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("Require 0 <= overlap_tokens < target_tokens")


@dataclass(frozen=True)
class ChunkSpan:
    char_start: int
    char_end: int
    token_count: int


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int


_PARAGRAPH = re.compile(r"[^\S\n]*\S[\s\S]*?(?=\n\s*\n|\Z)")
_SENTENCE_END = re.compile(r"[.!?。！？；;]+(?:[\"'”’）)]*)\s*")
_VI_MARKS = set("ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


def detect_language(text: str) -> str:
    han = sum("\u4e00" <= c <= "\u9fff" for c in text)
    latin = sum("LATIN" in unicodedata.name(c, "") for c in text)
    if han > latin and han >= 5:
        return "zh"
    if latin > han and any(c in _VI_MARKS for c in text):
        return "vi"
    return "unknown"


def _trim(text: str, start: int, end: int) -> _Unit | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Unit(start, end) if start < end else None


def _split_by_tokens(text: str, start: int, end: int, counter: TokenCounter,
                     maximum: int) -> list[_Unit]:
    pieces = []
    cursor = start
    while cursor < end:
        while cursor < end and text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break
        lo, hi = cursor + 1, end + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if counter.count(text[cursor:mid]) <= maximum:
                lo = mid + 1
            else:
                hi = mid
        boundary = lo - 1
        if boundary <= cursor:
            raise ValueError("A single character exceeds max_tokens")
        if boundary < end:
            space = max(text.rfind(" ", cursor + 1, boundary + 1),
                        text.rfind("\n", cursor + 1, boundary + 1))
            if space > cursor and space - cursor >= (boundary - cursor) // 2:
                boundary = space
        unit = _trim(text, cursor, boundary)
        if unit:
            pieces.append(unit)
        cursor = boundary
    return pieces


def _units(text: str, counter: TokenCounter, maximum: int) -> list[_Unit]:
    units = []
    for paragraph in _PARAGRAPH.finditer(text):
        unit = _trim(text, *paragraph.span())
        if unit is None:
            continue
        if counter.count(text[unit.start:unit.end]) <= maximum:
            units.append(unit)
            continue
        cursor = unit.start
        for match in _SENTENCE_END.finditer(text, unit.start, unit.end):
            sentence = _trim(text, cursor, match.end())
            if sentence:
                if counter.count(text[sentence.start:sentence.end]) <= maximum:
                    units.append(sentence)
                else:
                    units.extend(_split_by_tokens(text, sentence.start, sentence.end,
                                                  counter, maximum))
            cursor = match.end()
        tail = _trim(text, cursor, unit.end)
        if tail:
            if counter.count(text[tail.start:tail.end]) <= maximum:
                units.append(tail)
            else:
                units.extend(_split_by_tokens(text, tail.start, tail.end, counter, maximum))
    return units


def chunk_text(text: str, counter: TokenCounter,
               config: ChunkingConfig | None = None) -> list[ChunkSpan]:
    """Return overlapping contiguous source spans; do not rewrite the text."""
    config = config or ChunkingConfig()
    units = _units(text, counter, config.max_tokens)
    if not units:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(units):
        end = start + 1
        while end < len(units):
            prospective = text[units[start].start:units[end].end]
            if counter.count(prospective) > config.target_tokens:
                break
            end += 1
        if end == start + 1 and counter.count(text[units[start].start:units[start].end]) > config.target_tokens:
            end = start + 1
        spans.append((start, end))
        if end == len(units):
            break
        next_start = end
        for possible in range(end - 1, start - 1, -1):
            overlap = text[units[possible].start:units[end - 1].end]
            if counter.count(overlap) <= config.overlap_tokens:
                next_start = possible
            else:
                break
        start = next_start
    if len(spans) > 1:
        last_start, last_end = spans[-1]
        prev_start, _ = spans[-2]
        last_tokens = counter.count(text[units[last_start].start:units[last_end - 1].end])
        merged_tokens = counter.count(text[units[prev_start].start:units[last_end - 1].end])
        if last_tokens < config.min_chunk_tokens and merged_tokens <= config.max_tokens:
            spans[-2] = (prev_start, last_end)
            spans.pop()
    output = []
    seen = set()
    for first, last in spans:
        start_char, end_char = units[first].start, units[last - 1].end
        value = text[start_char:end_char]
        if not value.strip():
            continue
        while value in seen and start_char > 0 and counter.count(text[start_char - 1:end_char]) <= config.max_tokens:
            start_char -= 1
            value = text[start_char:end_char]
        while value in seen and end_char < len(text) and counter.count(text[start_char:end_char + 1]) <= config.max_tokens:
            end_char += 1
            value = text[start_char:end_char]
        if value in seen:
            raise ValueError("Repeated source text cannot form unique chunks within max_tokens")
        count = counter.count(value)
        if count > config.max_tokens:
            raise ValueError("Chunk exceeds max_tokens")
        seen.add(value)
        output.append(ChunkSpan(start_char, end_char, count))
    return output
