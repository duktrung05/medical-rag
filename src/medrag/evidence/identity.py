"""Exact judgment identities and normalization with original source spans."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judgment_key(query_id: int, doc_id: int, text: str) -> tuple[int, int, str]:
    # Deliberately do not normalize: a changed context needs a new judgment.
    return int(query_id), int(doc_id), text_sha256(text)


def normalize(text: str) -> str:
    return " ".join(html.unescape(unicodedata.normalize("NFKC", text)).lower().split())


def normalized_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Map every normalized character to a half-open original source span.

    Combining clusters (including Hangul composition), compatibility expansions,
    HTML references and lowercase expansions retain both source boundaries.
    """
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    start = 0
    cluster = ""
    for index, char in enumerate(text):
        normalized_char = unicodedata.normalize("NFKC", char)
        joins = bool(cluster) and (
            unicodedata.combining(char)
            or (normalized_char and unicodedata.combining(normalized_char[0]))
            or unicodedata.normalize("NFKC", cluster + char)
            != unicodedata.normalize("NFKC", cluster) + normalized_char
        )
        if cluster and not joins:
            value = unicodedata.normalize("NFKC", cluster)
            chars.extend(value)
            spans.extend([(start, index)] * len(value))
            cluster = ""
        if not cluster:
            start = index
        cluster += char
    if cluster:
        value = unicodedata.normalize("NFKC", cluster)
        chars.extend(value)
        spans.extend([(start, len(text))] * len(value))
    expanded = "".join(chars)
    if expanded != unicodedata.normalize("NFKC", text):
        raise ValueError("NFKC source mapping could not be verified")

    # Same reference grammar as html.unescape, including optional semicolons.
    reference = re.compile(r"&(?:#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)")
    decoded: list[str] = []
    decoded_spans: list[tuple[int, int]] = []
    cursor = 0
    for match in reference.finditer(expanded):
        decoded.extend(expanded[cursor:match.start()])
        decoded_spans.extend(spans[cursor:match.start()])
        value = html.unescape(match.group())
        if value == match.group():
            decoded.extend(value)
            decoded_spans.extend(spans[match.start():match.end()])
        else:
            # A named reference may consume only a prefix (e.g. &notit;).
            consumed = len(match.group())
            for length in range(1, len(match.group()) + 1):
                prefix = match.group()[:length]
                if html.unescape(prefix) != prefix and (
                    html.unescape(prefix) + match.group()[length:] == value
                ):
                    consumed = length
                    break
            prefix_value = html.unescape(match.group()[:consumed])
            decoded.extend(prefix_value)
            decoded_spans.extend([(spans[match.start()][0],
                                   spans[match.start() + consumed - 1][1])] * len(prefix_value))
            decoded.extend(match.group()[consumed:])
            decoded_spans.extend(spans[match.start() + consumed:match.end()])
        cursor = match.end()
    decoded.extend(expanded[cursor:])
    decoded_spans.extend(spans[cursor:])
    decoded_text = "".join(decoded)
    lowered = decoded_text.lower()  # Preserve context-dependent Greek sigma.
    lower_spans = [span for char, span in zip(decoded, decoded_spans, strict=True)
                   for _ in char.lower()]
    if len(lowered) != len(lower_spans):
        raise ValueError("Lowercase source mapping could not be verified")

    output: list[str] = []
    output_spans: list[tuple[int, int]] = []
    whitespace: tuple[int, int] | None = None
    for char, span in zip(lowered, lower_spans, strict=True):
        if char.isspace():
            if output:
                whitespace = (whitespace[0] if whitespace else span[0], span[1])
            continue
        if whitespace:
            output.append(" ")
            output_spans.append(whitespace)
            whitespace = None
        output.append(char)
        output_spans.append(span)
    result = "".join(output)
    if result != normalize(text):
        raise ValueError("Normalized source mapping could not be verified")
    return result, output_spans


def containment(text: str, evidence: str) -> tuple[bool, int | None, int | None]:
    haystack, spans = normalized_spans(text)
    needle = normalize(evidence)
    if not needle:
        return False, None, None
    pos = haystack.find(needle)
    while pos >= 0:
        start = min(span[0] for span in spans[pos:pos + len(needle)])
        end = max(span[1] for span in spans[pos:pos + len(needle)])
        if normalize(text[start:end]) == needle:
            return True, start, end
        pos = haystack.find(needle, pos + 1)
    # Containment without a round-trippable source span must not claim offsets.
    return False, None, None
