"""Decode and extract the main text from an already downloaded HTML response."""

from __future__ import annotations

import codecs
import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag
from charset_normalizer import from_bytes


class DecodeError(ValueError):
    """No usable character encoding was found."""


@dataclass(frozen=True)
class ExtractedHtml:
    title: str
    text: str
    encoding: str
    encoding_source: str
    extraction_method: str
    paragraph_count: int


_CONTENT_TYPE_CHARSET = re.compile(r"charset\s*=\s*['\"]?([a-zA-Z0-9._-]+)", re.IGNORECASE)
_META_CHARSET = re.compile(rb"<meta\b[^>]*charset\s*=\s*['\"]?([a-zA-Z0-9._-]+)", re.IGNORECASE)
_META_HTTP_EQUIV = re.compile(
    rb"<meta\b[^>]*content\s*=\s*['\"][^'\"]*charset\s*=\s*([a-zA-Z0-9._-]+)",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")
_DROP_NAME = re.compile(
    r"(^|[-_\s])(nav|menu|footer|header|sidebar|advert|advertisement|ads|"
    r"social|share|related|recommend|comment|breadcrumb|popup|cookie|"
    r"subscription|suggest|similar|other|toolbar)([-_\s]|$)",
    re.IGNORECASE,
)
_CONTENT_NAME = re.compile(
    r"(article[-_]?body|article[-_]?content|article[-_]?detail|entry[-_]?content|"
    r"viewcontent|bodycontent|post[-_]?content|answercont|mastercms_content|"
    r"(^|[-_])content($|[-_])|(^|[-_])entry($|[-_]))",
    re.IGNORECASE,
)
_REMOVE_TAGS = (
    "script", "style", "noscript", "svg", "form", "nav", "footer", "aside",
    "header", "iframe", "template", "canvas", "button", "input", "select",
)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", _SPACE.sub(" ", value).strip())


def decode_html(raw: bytes, content_type: str | None = None) -> tuple[str, str, str]:
    """Decode with HTTP charset, HTML meta charset, then charset-normalizer."""
    candidates: list[tuple[str, str]] = []
    if content_type and (match := _CONTENT_TYPE_CHARSET.search(content_type)):
        candidates.append((match.group(1), "http_content_type"))
    head = raw[:65536]
    for pattern in (_META_CHARSET, _META_HTTP_EQUIV):
        if match := pattern.search(head):
            candidates.append((match.group(1).decode("ascii"), "html_meta"))
            break
    for encoding, source in candidates:
        for trial in (encoding, "gb18030") if encoding.lower() in {"gb2312", "gbk"} else (encoding,):
            try:
                canonical = codecs.lookup(trial).name
                return raw.decode(canonical, errors="strict"), canonical, source
            except (LookupError, UnicodeDecodeError):
                continue
    detected = from_bytes(raw).best()
    if detected is None or not detected.encoding:
        raise DecodeError("Could not decode HTML with declared or detected charset")
    try:
        encoding = codecs.lookup(detected.encoding).name
        return raw.decode(encoding, errors="strict"), encoding, "charset_normalizer"
    except (LookupError, UnicodeDecodeError) as exc:
        raise DecodeError(str(exc)) from exc


def _clean(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(_REMOVE_TAGS)):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.parent is None:
            continue
        if tag.name in {"html", "body"}:
            continue
        label = " ".join([str(tag.get("id") or ""), *(tag.get("class") or [])])
        style = str(tag.get("style") or "").lower().replace(" ", "")
        hidden = tag.has_attr("hidden") or str(tag.get("aria-hidden", "")).lower() == "true"
        hidden |= "display:none" in style or "visibility:hidden" in style
        if hidden or _DROP_NAME.search(label):
            tag.decompose()


def _paragraphs(container: Tag) -> list[str]:
    # A block is emitted only if it has no nested block of the same kind.
    blocks = container.find_all(["p", "h2", "h3", "h4", "blockquote", "li"])
    paragraphs = []
    for block in blocks:
        if block.find(["p", "h2", "h3", "h4", "blockquote", "li"]):
            continue
        value = _normalize(block.get_text(" ", strip=True))
        if value.startswith(("上一篇：", "下一篇：")):
            continue
        if value and (len(value) >= 12 or block.name in {"h2", "h3", "h4"}):
            paragraphs.append(value)
    if not paragraphs:
        value = _normalize(container.get_text(" ", strip=True))
        if value:
            paragraphs.append(value)
    return paragraphs


def _candidate_score(tag: Tag) -> tuple[float, int]:
    text = _normalize(tag.get_text(" ", strip=True))
    length = len(text)
    if length < 8:
        return -1.0, length
    links = sum(len(_normalize(a.get_text(" ", strip=True))) for a in tag.find_all("a"))
    density = links / max(length, 1)
    if density > 0.55:
        return -1.0, length
    paragraph_count = len(_paragraphs(tag))
    label = " ".join([str(tag.get("id") or ""), *(tag.get("class") or [])])
    name_bonus = 300 if _CONTENT_NAME.search(label) else 0
    if tag.name == "article":
        name_bonus += 180
    elif tag.name == "main":
        name_bonus += 80
    # Give focused containers a chance against ancestors containing recommendations.
    score = min(length, 5000) * 0.2 + min(paragraph_count, 30) * 18
    score += name_bonus - density * 1000
    score -= max(length - 5000, 0) * 0.35
    score -= max(len(tag.find_all("a")) - 10, 0) * 8
    return score, length


def extract_html(raw: bytes, content_type: str | None = None) -> ExtractedHtml:
    decoded, encoding, source = decode_html(raw, content_type)
    soup = BeautifulSoup(decoded, "html.parser")
    og = soup.find("meta", attrs={"property": "og:title"})
    og_title = _normalize(str(og.get("content") or "")) if isinstance(og, Tag) else ""
    article_title = soup.select_one("article h1") or soup.find("h1")
    title_tag = soup.find("title")
    title = og_title or (
        _normalize(article_title.get_text(" ", strip=True)) if article_title else ""
    ) or (_normalize(title_tag.get_text(" ", strip=True)) if title_tag else "")
    _clean(soup)
    root = soup.body or soup
    candidates = [root, *root.find_all(["article", "main", "div", "section", "td"])]
    ranked = [(score, tag, size) for tag in candidates if (score := _candidate_score(tag))[0] != -1
              for size in [score[1]]]
    if not ranked:
        return ExtractedHtml(title, "", encoding, source, "none", 0)
    substantive = [item for item in ranked if item[2] >= 80]
    if substantive:
        ranked = substantive
    _, winner, _ = max(ranked, key=lambda item: item[0][0])
    paragraphs = _paragraphs(winner)
    text = "\n\n".join(paragraphs)
    label = " ".join([str(winner.get("id") or ""), *(winner.get("class") or [])]).strip()
    method = f"{winner.name}:{label}" if label else winner.name
    return ExtractedHtml(title, text, encoding, source, method, len(paragraphs))
