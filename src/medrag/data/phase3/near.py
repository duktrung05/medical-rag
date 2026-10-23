"""Language-aware near-document candidate generation and verification."""

from __future__ import annotations

import hashlib
import re
import struct
from collections import defaultdict

import pyarrow as pa


NEAR_POLICY_VERSION = "vibiomir-phase3-near-document-v1"
NEAR_SCHEMA_VERSION = "vibiomir-phase3-near-duplicate-pairs-v1"
MINHASH_PRIME = 4_294_967_291
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|kg|ml|l|mmol|mmhg|iu|%)"
    r"(?:\s*/\s*(?:day|ngày|kg|min|phút))?\b",
    re.IGNORECASE,
)


NEAR_DUPLICATE_SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("left_doc_id", pa.int64()),
    ("right_doc_id", pa.int64()),
    ("left_official_url", pa.string()),
    ("right_official_url", pa.string()),
    ("left_language", pa.string()),
    ("right_language", pa.string()),
    ("shingle_strategy", pa.string()),
    ("shingle_size", pa.int16()),
    ("left_shingles", pa.int32()),
    ("right_shingles", pa.int32()),
    ("shared_shingles", pa.int32()),
    ("minhash_similarity", pa.float32()),
    ("jaccard_similarity", pa.float32()),
    ("left_containment", pa.float32()),
    ("right_containment", pa.float32()),
    ("threshold_passes", pa.list_(pa.float32())),
    ("guard_reasons", pa.list_(pa.string())),
    ("review_status", pa.string()),
    ("auto_merge_allowed", pa.bool_()),
    ("phase1_release_id", pa.string()),
    ("policy_version", pa.string()),
])


def _word_tokens(text: str) -> list[str]:
    return [token.casefold() for token in WORD_PATTERN.findall(text)]


def _character_tokens(text: str) -> list[str]:
    return [char.casefold() for char in text if char.isalnum()]


def shingle_text(text: str, language: str | None) -> tuple[set[str], str, int]:
    """Return language-aware shingles plus the applied strategy contract."""
    language = (language or "unknown").casefold()
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if han >= 20 and latin >= 20:
        tokens = _character_tokens(text)
        strategy, size = "mixed_character", 7
    elif language.startswith("zh") or han >= 20:
        tokens = _character_tokens(text)
        strategy, size = "chinese_character", 5
    elif language.startswith("vi"):
        tokens = _word_tokens(text)
        strategy, size = "vietnamese_syllable", 5
    elif language.startswith("en"):
        tokens = _word_tokens(text)
        strategy, size = "english_word", 5
    else:
        tokens = _word_tokens(text)
        strategy, size = "generic_word", 5
    if not tokens:
        return set(), strategy, size
    if len(tokens) < size:
        return {"\x1f".join(tokens)}, strategy, size
    return {
        "\x1f".join(tokens[index:index + size])
        for index in range(len(tokens) - size + 1)
    }, strategy, size


def hash_shingles(shingles: set[str], *, maximum: int = 20_000) -> frozenset[int]:
    """Hash and deterministically cap a shingle set for bounded memory."""
    values = {
        int.from_bytes(
            hashlib.blake2b(
                value.encode("utf-8"), digest_size=8, person=b"vbm3shng"
            ).digest(),
            "big",
        )
        for value in shingles
    }
    if len(values) > maximum:
        values = set(sorted(values)[:maximum])
    return frozenset(values)


def _coefficients(num_perm: int) -> tuple[tuple[int, int], ...]:
    result = []
    for index in range(num_perm):
        digest = hashlib.sha256(
            f"{NEAR_POLICY_VERSION}\0{index}".encode("ascii")
        ).digest()
        a = int.from_bytes(digest[:8], "big") % (MINHASH_PRIME - 1) + 1
        b = int.from_bytes(digest[8:16], "big") % MINHASH_PRIME
        result.append((a, b))
    return tuple(result)


def minhash_signature(shingles: frozenset[int], num_perm: int = 64) -> tuple[int, ...]:
    if not shingles:
        return (MINHASH_PRIME,) * num_perm
    signature = [MINHASH_PRIME] * num_perm
    coefficients = _coefficients(num_perm)
    for value in shingles:
        reduced = value % MINHASH_PRIME
        for index, (a, b) in enumerate(coefficients):
            candidate = (a * reduced + b) % MINHASH_PRIME
            if candidate < signature[index]:
                signature[index] = candidate
    return tuple(signature)


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("MinHash signatures must have the same non-zero length")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def lsh_candidates(
    records: list[dict],
    *,
    bands: int = 16,
    rows_per_band: int = 4,
    maximum_bucket_size: int = 200,
) -> tuple[set[tuple[int, int]], int]:
    """Generate deterministic candidate index pairs with bounded buckets."""
    if bands < 1 or rows_per_band < 1:
        raise ValueError("LSH dimensions must be positive")
    required = bands * rows_per_band
    buckets: dict[tuple[str, int, bytes], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    overflow_buckets = 0
    for record_index, record in enumerate(records):
        signature = record["signature"]
        if len(signature) != required:
            raise ValueError("Signature length differs from LSH band layout")
        for band in range(bands):
            start = band * rows_per_band
            values = signature[start:start + rows_per_band]
            key = (
                record["shingle_strategy"],
                band,
                struct.pack(f">{rows_per_band}Q", *values),
            )
            bucket = buckets[key]
            if len(bucket) >= maximum_bucket_size:
                overflow_buckets += 1
                continue
            for previous in bucket:
                candidates.add((previous, record_index))
            bucket.append(record_index)
    return candidates, overflow_buckets


def similarity_scores(
    left: frozenset[int], right: frozenset[int]
) -> tuple[int, float, float, float]:
    shared = len(left & right)
    union = len(left | right)
    jaccard = shared / union if union else 1.0
    left_containment = shared / len(left) if left else 0.0
    right_containment = shared / len(right) if right else 0.0
    return shared, jaccard, left_containment, right_containment


def negation_present(text: str) -> bool:
    lowered = text.casefold()
    terms = (
        " not ", " no ", " without ", "contraindicat", "không", "chống chỉ định",
        " không được ", "不", "无", "未", "禁忌",
    )
    padded = f" {lowered} "
    return any(term in padded for term in terms)


def dosage_signature(text: str) -> frozenset[str]:
    return frozenset(
        re.sub(r"\s+", "", match.group(0).casefold())
        for match in DOSAGE_PATTERN.finditer(text)
    )


def intent_signature(text: str) -> frozenset[str]:
    lowered = text.casefold()
    vocabulary = {
        "symptom": ("symptom", "triệu chứng", "症状"),
        "treatment": ("treatment", "therapy", "điều trị", "治疗"),
        "diagnosis": ("diagnos", "chẩn đoán", "诊断"),
        "prevention": ("prevent", "phòng ngừa", "预防"),
        "contraindication": ("contraindicat", "chống chỉ định", "禁忌"),
        "dosage": ("dosage", "dose", "liều", "剂量"),
    }
    return frozenset(
        label for label, terms in vocabulary.items() if any(term in lowered for term in terms)
    )


def medical_guard_reasons(left: str, right: str) -> list[str]:
    reasons = []
    if negation_present(left) != negation_present(right):
        reasons.append("medical_negation_mismatch")
    left_dosage, right_dosage = dosage_signature(left), dosage_signature(right)
    if (left_dosage or right_dosage) and left_dosage != right_dosage:
        reasons.append("drug_dosage_mismatch")
    left_intent, right_intent = intent_signature(left), intent_signature(right)
    if left_intent and right_intent and left_intent.isdisjoint(right_intent):
        reasons.append("medical_intent_mismatch")
    return reasons


def pair_id(left_doc_id: int, right_doc_id: int) -> str:
    left, right = sorted((int(left_doc_id), int(right_doc_id)))
    digest = hashlib.sha256(
        f"{NEAR_POLICY_VERSION}\0{left}\0{right}".encode("ascii")
    ).hexdigest()
    return f"ndp_{digest}"


def threshold_passes(
    jaccard: float,
    left_containment: float,
    right_containment: float,
    thresholds: tuple[float, ...],
    *,
    containment_threshold: float,
    containment_jaccard_floor: float,
) -> list[float]:
    maximum_containment = max(left_containment, right_containment)
    return [
        threshold
        for threshold in thresholds
        if jaccard >= threshold
        or (
            maximum_containment >= containment_threshold
            and jaccard >= containment_jaccard_floor
        )
    ]

