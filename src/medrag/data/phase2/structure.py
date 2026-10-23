"""Conservative source-grounded document structure recognition."""

from __future__ import annotations

import re
from dataclasses import dataclass


_BLOCK = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)])\s+\S", re.I)
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")
_TERMINAL = tuple(".!?。！？;；")


@dataclass(frozen=True)
class SectionSpan:
    section_id: str
    start: int
    end: int
    heading: str | None
    heading_start: int | None = None
    heading_end: int | None = None


def _is_heading(block: str, title: str | None, first: bool) -> bool:
    value = block.strip()
    if not value or "\n" in value or len(value) > 160:
        return False
    if first and title and value == title.strip():
        return True
    if _MARKDOWN_HEADING.match(value) or _NUMBERED_HEADING.match(value):
        return True
    if value.endswith(":") and len(value) <= 100:
        return True
    words = value.split()
    if len(words) <= 12 and not value.endswith(_TERMINAL):
        letters = [char for char in value if char.isalpha()]
        uppercase = sum(char.isupper() for char in letters)
        return bool(letters) and uppercase / len(letters) >= 0.7
    return False


def find_sections(text: str, title: str | None = None) -> list[SectionSpan]:
    blocks = list(_BLOCK.finditer(text))
    headings = [
        match for index, match in enumerate(blocks)
        if _is_heading(match.group(), title, index == 0)
    ]
    if not headings:
        return [SectionSpan("section-0000", 0, len(text), None)] if text.strip() else []

    sections: list[SectionSpan] = []
    if headings[0].start() > 0 and text[:headings[0].start()].strip():
        sections.append(SectionSpan("section-0000", 0, headings[0].start(), None))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append(SectionSpan(
            f"section-{len(sections):04d}", heading.start(), end,
            heading.group().strip(), heading.start(), heading.end(),
        ))
    return sections


def has_reliable_heading(text: str, title: str | None = None) -> bool:
    return any(section.heading is not None for section in find_sections(text, title))
