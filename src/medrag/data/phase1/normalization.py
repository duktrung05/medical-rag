"""Conservative, traceable normalization for multilingual medical text."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


NORMALIZATION_VERSION = "unicode-nfc-whitespace-v1"
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+", re.UNICODE)
_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")


@dataclass(frozen=True)
class NormalizedText:
    source_text: str
    normalized_text: str
    source_text_sha256: str
    normalized_text_sha256: str
    version: str = NORMALIZATION_VERSION

    @property
    def changed(self) -> bool:
        return self.source_text != self.normalized_text


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_medical_text(text: str) -> NormalizedText:
    """Normalize Unicode and whitespace without changing lexical content.

    Newline conventions are unified, horizontal whitespace is collapsed within
    each line, and runs of blank lines become one paragraph separator. Case,
    punctuation, Vietnamese diacritics, numbers, units, and negation are kept.
    """
    if not isinstance(text, str):
        raise TypeError("medical text must be a string")
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = "\n".join(_HORIZONTAL_WHITESPACE.sub(" ", line).strip()
                           for line in normalized.split("\n"))
    normalized = _BLANK_LINES.sub("\n\n", normalized).strip("\n")
    return NormalizedText(
        source_text=text,
        normalized_text=normalized,
        source_text_sha256=text_sha256(text),
        normalized_text_sha256=text_sha256(normalized),
    )
