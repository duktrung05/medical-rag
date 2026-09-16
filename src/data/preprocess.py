"""Text preprocessing and normalization routines."""

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapses consecutive whitespace characters into a single space and trims."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    """Collapses consecutive repeated punctuation marks like '???' to '?' while keeping essential marks."""
    # Collapse multiple question marks
    text = re.sub(r"\?+", "?", text)
    # Collapse multiple exclamation marks
    text = re.sub(r"!+", "!", text)
    # Collapse multiple periods to maximum 3 (ellipsis)
    text = re.sub(r"\.{4,}", "...", text)
    # Normalize varied quotation marks to standard ascii quotes
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)
    return text


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Applies baseline normalization:

    1. Unicode normalization (NFC)
    2. Lowercase (optional)
    3. Normalizes repeated punctuation
    4. Trims and collapses multiple whitespaces
    Preserves medical entities, units, and drug names without destructive stemming.
    """
    if not text:
        return ""

    # Canonical NFC decomposition/composition for Vietnamese accents
    text = unicodedata.normalize("NFC", text)

    if lowercase:
        text = text.lower()

    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    return text
