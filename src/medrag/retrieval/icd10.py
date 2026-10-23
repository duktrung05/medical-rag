"""Optional ICD-10 query mapping, expansion, and candidate reranking.

The WHO classification remains unchanged.  Multilingual aliases are project
metadata and are deliberately kept separate from WHO titles.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9À-ỹ]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def normalize_term(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFC", value).casefold()).strip()


def _contains(text: str, term: str, language: str) -> bool:
    if language == "zh" or any("\u4e00" <= char <= "\u9fff" for char in term):
        return term in text
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


@dataclass(frozen=True)
class ICDMatch:
    code: str
    title: str
    matched_alias: str
    alias_language: str
    confidence: float
    method: str = "verified_dictionary"


class ICD10QueryMapper:
    """Conservative dictionary mapper with explicit UNKNOWN behaviour."""

    def __init__(self, taxonomy_rows: Iterable[Mapping], alias_rows: Iterable[Mapping]):
        taxonomy = {str(row["code"]): dict(row) for row in taxonomy_rows if row.get("code")}
        aliases = []
        by_code: dict[str, list[dict]] = {}
        for source in alias_rows:
            row = dict(source)
            code = str(row["code"])
            if code not in taxonomy:
                raise ValueError(f"Alias refers to an unknown ICD-10 code: {code}")
            row["normalized_alias"] = normalize_term(str(row["alias"]))
            aliases.append(row)
            by_code.setdefault(code, []).append(row)
        self.taxonomy = taxonomy
        self.aliases = sorted(
            aliases, key=lambda row: (-len(row["normalized_alias"]), row["code"], row["alias"])
        )
        self.by_code = by_code

    def map_query(self, text: str, language: str, *, max_matches: int = 3) -> list[ICDMatch]:
        normalized = normalize_term(text)
        selected: list[ICDMatch] = []
        occupied_codes: set[str] = set()
        for row in self.aliases:
            alias_language = str(row["language"])
            if alias_language not in {language, "mul"}:
                continue
            alias = row["normalized_alias"]
            if row["code"] in occupied_codes or not _contains(normalized, alias, alias_language):
                continue
            title = str(self.taxonomy[row["code"]]["title"])
            selected.append(ICDMatch(
                code=str(row["code"]), title=title, matched_alias=str(row["alias"]),
                alias_language=alias_language, confidence=float(row.get("confidence", 1.0)),
            ))
            occupied_codes.add(str(row["code"]))
            if len(selected) >= max_matches:
                break
        return selected

    def expansion(self, text: str, language: str, matches: Sequence[ICDMatch]) -> str:
        additions: list[str] = []
        seen = {normalize_term(text)}
        for match in matches:
            for value in [match.title] + [
                str(row["alias"]) for row in self.by_code[match.code]
                if row["language"] in {language, "en"}
            ]:
                normalized = normalize_term(value)
                if normalized and normalized not in seen:
                    additions.append(value)
                    seen.add(normalized)
        return text if not additions else f"{text} {' '.join(additions)}"

    def codes_in_text(self, text: str, language: str) -> set[str]:
        return {match.code for match in self.map_query(text, language, max_matches=8)}


INTENT_TERMS = {
    "diagnosis": {
        "vi": ("chẩn đoán", "bệnh gì", "dấu hiệu của bệnh"),
        "en": ("diagnosis", "diagnosed", "what condition"),
        "zh": ("诊断", "什么病", "确诊"),
    },
    "symptoms": {
        "vi": ("triệu chứng", "biểu hiện", "dấu hiệu"),
        "en": ("symptom", "sign"),
        "zh": ("症状", "表现", "体征"),
    },
    "causes": {
        "vi": ("nguyên nhân", "do đâu", "vì sao", "tại sao"),
        "en": ("cause", "factor", "why"),
        "zh": ("原因", "为什么", "病因"),
    },
    "treatment": {
        "vi": ("điều trị", "chữa", "thuốc gì", "phẫu thuật"),
        "en": ("treatment", "treated", "therapy"),
        "zh": ("治疗", "用药", "手术"),
    },
    "complications": {
        "vi": ("biến chứng", "ảnh hưởng", "nguy hiểm"),
        "en": ("complication", "outcome", "risk"),
        "zh": ("并发症", "影响", "风险"),
    },
    "prevention": {
        "vi": ("phòng ngừa", "phòng tránh", "ngăn ngừa"),
        "en": ("prevention", "prevent"),
        "zh": ("预防", "防止"),
    },
}


def detect_intents(text: str, language: str) -> set[str]:
    normalized = normalize_term(text)
    return {
        intent for intent, languages in INTENT_TERMS.items()
        if any(_contains(normalized, normalize_term(term), language) for term in languages[language])
    }


def rerank_candidates(
    candidates: Sequence[dict], *, query_codes: set[str], query_intents: set[str],
    mapper: ICD10QueryMapper, language: str, concept_weight: float,
    intent_weight: float = 0.0,
) -> list[dict]:
    """Soft-rerank a fixed candidate pool; candidates are never filtered."""
    if not candidates:
        return []
    maximum = max(float(row["baseline_score"]) for row in candidates) or 1.0
    output = []
    for row in candidates:
        passage_codes = mapper.codes_in_text(str(row["passage_text"]), language)
        concept_match = bool(query_codes & passage_codes)
        passage_intents = detect_intents(str(row["passage_text"]), language)
        intent_match = bool(query_intents & passage_intents)
        score = float(row["baseline_score"]) + maximum * (
            concept_weight * float(concept_match) + intent_weight * float(intent_match)
        )
        output.append({
            **row, "score": score, "icd_concept_match": concept_match,
            "intent_match": intent_match,
        })
    output.sort(key=lambda row: (-row["score"], int(row["doc_id"]), str(row["passage_id"])))
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    return output


def rerank_candidates_cached(
    candidates: Sequence[dict], *, query_codes: set[str],
    passage_codes: Mapping[str, frozenset[str] | set[str]], concept_weight: float,
    rescore_depth: int,
) -> list[dict]:
    """Soft-rerank a prefix using precomputed features while preserving top-100 coverage."""
    if not candidates or not query_codes or concept_weight <= 0 or rescore_depth <= 0:
        return [{**row} for row in candidates]
    depth = min(rescore_depth, len(candidates))
    maximum = max(float(row["baseline_score"]) for row in candidates) or 1.0
    head = []
    for row in candidates[:depth]:
        codes = passage_codes.get(str(row["passage_id"]), frozenset())
        matched = bool(query_codes & set(codes))
        head.append({
            **row,
            "score": float(row["baseline_score"]) + maximum * concept_weight * float(matched),
            "icd_concept_match": matched,
        })
    head.sort(key=lambda row: (-float(row["score"]), int(row["doc_id"]), str(row["passage_id"])))
    # A non-negative bonus means no rescored head candidate can fall below the untouched
    # tail.  Appending the tail therefore retains exactly the original candidate set.
    output = head + [{**row} for row in candidates[depth:]]
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    return output
