"""Paragraph-first, sentence-aware, token-bounded exact source spans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .schema import ChunkingStrategy


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    token_count: int


_PARAGRAPH = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")
# A full stop followed by a digit is part of a decimal or dotted numeric value,
# not a sentence boundary (for example 5.963 or 3.272.256).
_SENTENCE_END = re.compile(r"(?:[!?。！？；;]+|\.+(?!\d))(?:[\"'”’）)]*)\s*")
_QA_ANSWER_START = re.compile(
    r"^(?:answer|the answer|diagnosis|the diagnosis|result|conclusion|"
    r"đáp án|trả lời|chẩn đoán|kết luận|答案|回答|诊断|结论)\b",
    re.I,
)


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _hard_split(text: str, start: int, end: int, counter: TokenCounter,
                maximum: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        while cursor < end and text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break
        lo, hi = cursor + 1, end + 1
        while lo < hi:
            middle = (lo + hi) // 2
            if counter.count(text[cursor:middle]) <= maximum:
                lo = middle + 1
            else:
                hi = middle
        boundary = lo - 1
        if boundary <= cursor:
            raise ValueError("A single source character exceeds max_tokens")
        if boundary < end:
            candidates = [text.rfind(mark, cursor + 1, boundary + 1)
                          for mark in (" ", "\n", "，", "、")]
            comma = text.rfind(",", cursor + 1, boundary + 1)
            if comma > cursor and not (
                comma > 0 and comma + 1 < len(text)
                and text[comma - 1].isdigit() and text[comma + 1].isdigit()
            ):
                candidates.append(comma)
            natural = max(candidates)
            if natural > cursor and natural - cursor >= (boundary - cursor) // 2:
                boundary = natural + (1 if text[natural] in ",，、" else 0)
        # BPE token counts are not strictly monotonic by character: extending
        # text can merge adjacent pieces and occasionally reduce the count.
        # Binary search therefore needs an exact post-condition check.
        while boundary > cursor and counter.count(text[cursor:boundary]) > maximum:
            boundary -= 1
        if boundary <= cursor:
            raise ValueError("Unable to form a token-bounded source span")
        trimmed = _trim(text, cursor, boundary)
        if trimmed:
            output.append(trimmed)
        cursor = boundary
    return output


def _atomic_units(text: str, start: int, end: int, counter: TokenCounter,
                  target: int, maximum: int) -> list[tuple[int, int]]:
    units: list[tuple[int, int]] = []
    for paragraph in _PARAGRAPH.finditer(text, start, end):
        unit = _trim(text, paragraph.start(), paragraph.end())
        if unit is None:
            continue
        if counter.count(text[unit[0]:unit[1]]) <= target:
            units.append(unit)
            continue
        cursor = unit[0]
        for sentence_end in _SENTENCE_END.finditer(text, unit[0], unit[1]):
            sentence = _trim(text, cursor, sentence_end.end())
            if sentence:
                if counter.count(text[sentence[0]:sentence[1]]) <= maximum:
                    units.append(sentence)
                else:
                    units.extend(_hard_split(text, *sentence, counter, maximum))
            cursor = sentence_end.end()
        tail = _trim(text, cursor, unit[1])
        if tail:
            if counter.count(text[tail[0]:tail[1]]) <= maximum:
                units.append(tail)
            else:
                units.extend(_hard_split(text, *tail, counter, maximum))
    return units


def _separates_question_answer(text: str, units: list[tuple[int, int]],
                               first: int, boundary: int) -> bool:
    if boundary <= first or boundary >= len(units):
        return False
    left = text[units[first][0]:units[boundary - 1][1]].strip()
    right = text[units[boundary][0]:units[boundary][1]].strip()
    if not left or not right or not any(mark in left for mark in ("?", "？")):
        return False
    return bool(_QA_ANSWER_START.match(right))


def _repair_short_groups(text: str, units: list[tuple[int, int]],
                         groups: list[tuple[int, int]], counter: TokenCounter,
                         strategy: ChunkingStrategy) -> list[tuple[int, int]]:
    if len(groups) < 2:
        return groups
    index = 0
    while index < len(groups):
        first, last = groups[index]
        count = counter.count(text[units[first][0]:units[last - 1][1]])
        if count >= strategy.min_chunk_tokens:
            index += 1
            continue
        if index > 0:
            previous_first, _ = groups[index - 1]
            merged = text[units[previous_first][0]:units[last - 1][1]]
            if counter.count(merged) <= strategy.max_tokens:
                groups[index - 1] = (previous_first, last)
                groups.pop(index)
                index = max(0, index - 1)
                continue
        if index + 1 < len(groups):
            _, following_last = groups[index + 1]
            merged = text[units[first][0]:units[following_last - 1][1]]
            if counter.count(merged) <= strategy.max_tokens:
                groups[index + 1] = (first, following_last)
                groups.pop(index)
                continue
        index += 1
    return groups


def segment_range(text: str, start: int, end: int, counter: TokenCounter,
                  strategy: ChunkingStrategy) -> list[TextSpan]:
    units = _atomic_units(
        text, start, end, counter, strategy.target_tokens, strategy.max_tokens,
    )
    if not units:
        return []
    groups: list[tuple[int, int]] = []
    first = 0
    while first < len(units):
        last = first + 1
        while last < len(units):
            if counter.count(text[units[first][0]:units[last][1]]) > strategy.target_tokens:
                if (
                    _separates_question_answer(text, units, first, last)
                    and counter.count(text[units[first][0]:units[last][1]])
                    <= strategy.max_tokens
                ):
                    last += 1
                break
            last += 1
        groups.append((first, last))
        if last == len(units):
            break
        next_first = last
        # Excluding ``first`` guarantees forward progress even when the whole
        # current group is smaller than the requested overlap.
        for possible in range(last - 1, first, -1):
            overlap = text[units[possible][0]:units[last - 1][1]]
            if counter.count(overlap) <= strategy.overlap_tokens:
                next_first = possible
            else:
                break
        first = next_first

    groups = _repair_short_groups(text, units, groups, counter, strategy)

    passages = []
    for group_start, group_end in groups:
        char_start = units[group_start][0]
        char_end = units[group_end - 1][1]
        count = counter.count(text[char_start:char_end])
        if count > strategy.max_tokens:
            raise ValueError("Passage exceeds max_tokens")
        passages.append(TextSpan(char_start, char_end, count))
    return passages
