"""Extract searchable text from a corpus URL, with no network access.

Vietnamese sites encode the article title in the path slug without diacritics;
Chinese wiki-style sites percent-encode the title. Numeric-id sites yield nothing.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote, urlsplit

_EXT = re.compile(r"\.(html?|aspx?|php|jsp|shtml)$", re.I)
_SEP = re.compile(r"[-_/+.,:;!?()\[\]{}'\"~|]+")
_DIGIT_RUN = re.compile(r"\d{3,}")
_WS = re.compile(r"\s+")
# Path segments that are site furniture rather than content.
_STOP_SEG = {
    "a", "w", "bai-viet", "tin-tuc", "question", "vie", "books", "herbs",
    "index", "html", "www", "com", "vn", "cn", "net", "articles", "article",
    "post", "posts", "news", "detail", "p", "c", "d", "view", "show", "id",
}


def strip_diacritics(text: str) -> str:
    """Fold Vietnamese diacritics so queries match undiacriticized URL slugs."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    )


def has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def url_text(url: str) -> str:
    """Return the human-meaningful text encoded in a URL path, or ''."""
    parts = urlsplit(url)
    raw = parts.path
    try:
        raw = unquote(raw, errors="strict")
    except (UnicodeDecodeError, ValueError):
        raw = unquote(raw, errors="ignore")
    raw = _EXT.sub("", raw)
    tokens: list[str] = []
    for segment in raw.split("/"):
        if not segment or segment.lower() in _STOP_SEG:
            continue
        segment = _DIGIT_RUN.sub(" ", segment)
        for token in _SEP.split(segment):
            token = token.strip()
            if not token or token.isdigit():
                continue
            if token.lower() in _STOP_SEG:
                continue
            tokens.append(token)
    return _WS.sub(" ", " ".join(tokens)).strip()
