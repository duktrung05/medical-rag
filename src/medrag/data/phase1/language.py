"""Dependency-free, auditable language detection for Phase 1 documents.

The detector is intentionally limited to corpus routing and quality review. It
does not infer medical entities and never rewrites document text. PubMed's
``language_metadata`` describes the source publication and is therefore only a
fallback hint: translated English abstracts must still be detected as English.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


LANGUAGE_DETECTION_VERSION = "script-stopwords-metadata-v2"

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_VIETNAMESE_DISTINCTIVE = set("ăđơưĂĐƠƯ")
_VIETNAMESE_WORDS = {
    "bệnh", "bị", "các", "cho", "có", "của", "cũng", "đã", "đang", "đến",
    "điều", "được", "do", "giúp", "hôm", "không", "khi", "là", "một", "ngày",
    "người", "này", "những", "như", "sau", "sẽ", "sức", "tại", "theo", "thuốc",
    "trị", "triệu", "trong", "trước", "từ", "và", "về", "với",
}
_LATIN_STOPWORDS = {
    "en": {"a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "of", "on", "patient", "patients", "that", "the", "this", "to", "was", "were", "with"},
    "de": {"auf", "bei", "das", "der", "die", "eine", "für", "ist", "mit", "und", "von", "zu"},
    "fr": {"avec", "dans", "des", "du", "est", "et", "la", "le", "les", "pour", "une"},
    "es": {"con", "del", "el", "en", "es", "la", "las", "los", "para", "por", "una", "y"},
    "pt": {"com", "da", "de", "do", "em", "e", "os", "para", "por", "que", "uma"},
    "it": {"che", "con", "dei", "del", "della", "e", "gli", "in", "la", "per", "una"},
}
_METADATA_CODES = {
    "eng": "en", "en": "en", "vie": "vi", "vi": "vi", "chi": "zh",
    "zho": "zh", "zh": "zh", "ger": "de", "deu": "de", "de": "de",
    "fre": "fr", "fra": "fr", "fr": "fr", "spa": "es", "es": "es",
    "por": "pt", "pt": "pt", "ita": "it", "it": "it", "jpn": "ja",
    "ja": "ja", "kor": "ko", "ko": "ko", "rus": "ru", "ru": "ru",
    "ukr": "uk", "uk": "uk", "heb": "he", "he": "he", "ara": "ar",
    "ar": "ar",
}


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float
    method: str
    version: str = LANGUAGE_DETECTION_VERSION
    metadata_languages: tuple[str, ...] = ()
    metadata_agrees: bool | None = None


def normalize_language_metadata(value: str | None) -> tuple[str, ...]:
    """Convert comma/semicolon-separated ISO-like metadata to stable codes."""
    if not value:
        return ()
    output = []
    for item in re.split(r"[,;\s]+", value.casefold().strip()):
        if not item:
            continue
        code = _METADATA_CODES.get(item, item)
        if code not in output:
            output.append(code)
    return tuple(output)


def _finish(language: str, confidence: float, method: str,
            metadata: tuple[str, ...]) -> LanguageDetection:
    agrees = None if not metadata or language in {"unknown", "mixed"} else language in metadata
    return LanguageDetection(language, round(confidence, 4), method,
                             metadata_languages=metadata, metadata_agrees=agrees)


def detect_document_language(text: str, metadata_hint: str | None = None) -> LanguageDetection:
    """Detect content language using scripts and lexical evidence.

    Metadata is considered only after strong script/lexical rules. This avoids
    treating PubMed publication-language metadata as the abstract language.
    """
    if not isinstance(text, str):
        raise TypeError("language detection input must be a string")
    metadata = normalize_language_metadata(metadata_hint)
    if not text.strip():
        return _finish("unknown", 0.0, "empty", metadata)

    letters = sum(char.isalpha() for char in text)
    han = sum("\u3400" <= char <= "\u9fff" for char in text)
    kana = sum("\u3040" <= char <= "\u30ff" for char in text)
    hangul = sum("\uac00" <= char <= "\ud7af" for char in text)
    cyrillic = sum("CYRILLIC" in unicodedata.name(char, "") for char in text)
    hebrew = sum("HEBREW" in unicodedata.name(char, "") for char in text)
    arabic = sum("ARABIC" in unicodedata.name(char, "") for char in text)
    latin = sum("LATIN" in unicodedata.name(char, "") for char in text)
    denominator = max(letters, 1)

    if kana >= 3 and kana / denominator >= 0.03:
        return _finish("ja", min(0.99, 0.8 + kana / denominator), "script:kana", metadata)
    if hangul >= 3 and hangul / denominator >= 0.08:
        return _finish("ko", min(0.99, 0.8 + hangul / denominator), "script:hangul", metadata)
    if han >= 20 and latin >= 20 and han / denominator >= 0.03 and latin / denominator >= 0.15:
        return _finish("mixed", 0.82, "script:mixed_han_latin", metadata)
    if han >= 5 and han / denominator >= 0.12:
        return _finish("zh", min(0.99, 0.78 + han / denominator), "script:han", metadata)
    if cyrillic >= 4 and cyrillic / denominator >= 0.15:
        code = next((code for code in metadata if code in {"ru", "uk"}), "ru")
        return _finish(code, min(0.96, 0.72 + cyrillic / denominator), "script:cyrillic", metadata)
    if hebrew >= 4 and hebrew / denominator >= 0.15:
        return _finish("he", min(0.96, 0.72 + hebrew / denominator), "script:hebrew", metadata)
    if arabic >= 4 and arabic / denominator >= 0.15:
        return _finish("ar", min(0.96, 0.72 + arabic / denominator), "script:arabic", metadata)

    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    token_set = set(tokens)
    vi_hits = len(token_set.intersection(_VIETNAMESE_WORDS))
    vi_distinctive = sum(char in _VIETNAMESE_DISTINCTIVE for char in text)
    if vi_hits >= 2 and (vi_distinctive >= 1 or vi_hits >= 5):
        return _finish("vi", min(0.99, 0.72 + 0.035 * vi_hits + 0.015 * vi_distinctive),
                       "lexical:vi", metadata)

    scores = {code: len(token_set.intersection(words)) for code, words in _LATIN_STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score >= 3 and (best_score >= second_score + 2 or best_score >= 7):
        return _finish(best, min(0.97, 0.65 + 0.035 * best_score),
                       f"lexical:{best}", metadata)

    compatible_hint = next((code for code in metadata if code in _LATIN_STOPWORDS), None)
    if compatible_hint and latin / denominator >= 0.5:
        return _finish(compatible_hint, 0.58, "metadata_fallback:latin", metadata)
    if latin >= 4 and latin / denominator >= 0.5:
        return _finish("unknown", 0.35, "script:latin_undetermined", metadata)
    return _finish("unknown", 0.2, "insufficient_evidence", metadata)


def language_stratum(language: str) -> str:
    """Reduce language codes to the stable QA pilot strata."""
    return language if language in {"vi", "en", "zh"} else (
        "unknown" if language == "unknown" else "other"
    )
