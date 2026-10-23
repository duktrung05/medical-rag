"""Partial passage overlap contracts and conservative classification."""

from __future__ import annotations

import hashlib
import re

import pyarrow as pa


OVERLAP_POLICY_VERSION = "vibiomir-phase3-passage-overlap-v1"
OVERLAP_SCHEMA_VERSION = "vibiomir-phase3-passage-overlap-pairs-v1"

INTENTIONAL_CHUNK_OVERLAP = "INTENTIONAL_CHUNK_OVERLAP"
CROSS_DOCUMENT_OVERLAP = "CROSS_DOCUMENT_OVERLAP"
PARTIAL_CONTENT_REUSE = "PARTIAL_CONTENT_REUSE"

SOURCE_OFFSETS_TYPE = pa.struct([
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
])

PASSAGE_OVERLAP_SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("left_passage_id", pa.string()),
    ("right_passage_id", pa.string()),
    ("left_doc_id", pa.int64()),
    ("right_doc_id", pa.int64()),
    ("left_official_url", pa.string()),
    ("right_official_url", pa.string()),
    ("left_source_offsets", SOURCE_OFFSETS_TYPE),
    ("right_source_offsets", SOURCE_OFFSETS_TYPE),
    ("left_language", pa.string()),
    ("right_language", pa.string()),
    ("candidate_sources", pa.list_(pa.string())),
    ("shingle_strategy", pa.string()),
    ("shingle_size", pa.int16()),
    ("left_shingles", pa.int32()),
    ("right_shingles", pa.int32()),
    ("shared_shingles", pa.int32()),
    ("minhash_similarity", pa.float32()),
    ("jaccard_similarity", pa.float32()),
    ("left_containment", pa.float32()),
    ("right_containment", pa.float32()),
    ("source_overlap_chars", pa.int64()),
    ("overlap_type", pa.string()),
    ("classification_reasons", pa.list_(pa.string())),
    ("guard_reasons", pa.list_(pa.string())),
    ("boilerplate_signals", pa.list_(pa.string())),
    ("review_status", pa.string()),
    ("auto_merge_allowed", pa.bool_()),
    ("policy_version", pa.string()),
])


_BOILERPLATE_PATTERNS = {
    "privacy_or_terms": re.compile(
        r"privacy policy|terms (?:of use|and conditions)|"
        r"chính sách bảo mật|điều khoản sử dụng|隐私政策|使用条款",
        re.IGNORECASE,
    ),
    "copyright": re.compile(
        r"all rights reserved|copyright|bản quyền|版权所有", re.IGNORECASE
    ),
    "navigation": re.compile(
        r"read more|related articles?|share this|follow us|subscribe|"
        r"xem thêm|bài viết liên quan|chia sẻ|đăng ký|"
        r"阅读更多|相关文章|分享|订阅",
        re.IGNORECASE,
    ),
    "cookie_notice": re.compile(r"\bcookies?\b|cookie", re.IGNORECASE),
    "comment_form": re.compile(
        r"ý kiến của bạn sẽ được biên tập|vui lòng gõ tiếng việt có dấu|"
        r"your comment will be moderated",
        re.IGNORECASE,
    ),
    "content_template_placeholder": re.compile(
        r"\{(?:name|time|body)\}", re.IGNORECASE
    ),
    "symptom_directory_navigation": re.compile(
        r"症状查询专题|点击部位名称|按症状名称开头字母",
        re.IGNORECASE,
    ),
    "disease_directory_navigation": re.compile(
        r"按疾病名称开头字母|按医学分科查询疾病|这里只列出常见疾病",
        re.IGNORECASE,
    ),
    "drug_directory_navigation": re.compile(
        r"肝胆胰用药[\s\S]{0,300}胃肠药|"
        r"解热镇痛药[\s\S]{0,300}神经系统用药",
        re.IGNORECASE,
    ),
    "news_feed_navigation": re.compile(
        r"baophutho\.vn|(?:20\d{2}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[\s\S]*){2}",
        re.IGNORECASE,
    ),
    "skin_directory_navigation": re.compile(
        r"细菌性皮肤病[\s\S]{0,300}病毒性皮肤病[\s\S]{0,300}皮肤真菌病",
        re.IGNORECASE,
    ),
    "editorial_source_disclaimer": re.compile(
        r"nguồn tham khảo\s*/\s*source[\s\S]{0,500}quy trình biên tập",
        re.IGNORECASE,
    ),
    "manufacturer_directory_navigation": re.compile(
        r"制药企业分区域查询|制药企业名录列表|全国制药企业列表",
        re.IGNORECASE,
    ),
}


def overlap_pair_id(left_passage_id: str, right_passage_id: str) -> str:
    left, right = sorted((left_passage_id, right_passage_id))
    if not left or not right or left == right:
        raise ValueError("Overlap pair requires two distinct passage IDs")
    digest = hashlib.sha256(
        f"{OVERLAP_POLICY_VERSION}\0{left}\0{right}".encode("utf-8")
    ).hexdigest()
    return f"pop_{digest}"


def boilerplate_signals(text: str) -> list[str]:
    return sorted(
        label for label, pattern in _BOILERPLATE_PATTERNS.items() if pattern.search(text)
    )


def is_boilerplate_only(
    left_text: str,
    right_text: str,
    *,
    shared_shingles: int,
    maximum_containment: float,
) -> tuple[bool, list[str]]:
    left = boilerplate_signals(left_text)
    right = boilerplate_signals(right_text)
    common = sorted(set(left) & set(right))
    strong_template_signals = {
        "comment_form",
        "content_template_placeholder",
        "symptom_directory_navigation",
        "disease_directory_navigation",
        "drug_directory_navigation",
        "news_feed_navigation",
        "skin_directory_navigation",
        "editorial_source_disclaimer",
        "manufacturer_directory_navigation",
    }
    one_sided_strong_template = bool(
        (set(left) | set(right)) & strong_template_signals
    ) and maximum_containment >= 0.75
    only = (
        bool(common)
        and (shared_shingles <= 20 or maximum_containment >= 0.65)
    ) or one_sided_strong_template
    signals = common or sorted(
        (set(left) | set(right)) & strong_template_signals
    )
    return only, signals


def exact_source_overlap(left: dict, right: dict) -> tuple[int, bool]:
    """Return overlap length and whether both passage slices contain identical source text."""
    if int(left["doc_id"]) != int(right["doc_id"]):
        return 0, False
    start = max(int(left["char_start"]), int(right["char_start"]))
    end = min(int(left["char_end"]), int(right["char_end"]))
    if end <= start:
        return 0, False
    left_start = start - int(left["char_start"])
    right_start = start - int(right["char_start"])
    length = end - start
    left_value = left["passage_text"][left_start:left_start + length]
    right_value = right["passage_text"][right_start:right_start + length]
    return length, left_value == right_value


def classify_overlap(
    left: dict,
    right: dict,
    *,
    shared_shingles: int,
    jaccard: float,
    left_containment: float,
    right_containment: float,
    minimum_shared_shingles: int = 5,
    jaccard_threshold: float = 0.45,
    containment_threshold: float = 0.75,
    containment_jaccard_floor: float = 0.20,
) -> tuple[str | None, list[str], int]:
    """Classify verified overlap; return None when evidence is insufficient."""
    source_chars, exact_source = exact_source_overlap(left, right)
    adjacent = (
        int(left["doc_id"]) == int(right["doc_id"])
        and abs(int(left["passage_index"]) - int(right["passage_index"])) == 1
    )
    if adjacent and source_chars > 0 and exact_source:
        return (
            INTENTIONAL_CHUNK_OVERLAP,
            [
                "same_document",
                "adjacent_passage_index",
                "overlapping_source_offsets",
                "overlap_text_exact",
            ],
            source_chars,
        )
    maximum_containment = max(left_containment, right_containment)
    if shared_shingles < minimum_shared_shingles:
        return None, ["insufficient_shared_shingles"], source_chars
    if not (
        jaccard >= jaccard_threshold
        or (
            maximum_containment >= containment_threshold
            and jaccard >= containment_jaccard_floor
        )
    ):
        return None, ["below_verified_overlap_threshold"], source_chars
    minimum_containment = min(left_containment, right_containment)
    size_ratio = max(left["shingle_count"], right["shingle_count"]) / max(
        1, min(left["shingle_count"], right["shingle_count"])
    )
    if int(left["doc_id"]) == int(right["doc_id"]):
        return PARTIAL_CONTENT_REUSE, ["same_document_non_adjacent_reuse"], source_chars
    if (
        maximum_containment >= 0.80
        and (minimum_containment <= 0.65 or size_ratio >= 1.5)
    ):
        return (
            PARTIAL_CONTENT_REUSE,
            ["cross_document", "asymmetric_containment"],
            source_chars,
        )
    return (
        CROSS_DOCUMENT_OVERLAP,
        ["cross_document", "bidirectional_verified_overlap"],
        source_chars,
    )
