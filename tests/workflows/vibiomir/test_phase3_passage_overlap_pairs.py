from __future__ import annotations

from scripts.workflows.vibiomir.build_phase3_passage_overlap_pairs import (
    ROOT,
    _load_prepared_cache,
    _rows_digest,
    _safe_output,
    _write_prepared_cache,
)
from src.medrag.data.phase3.near import medical_guard_reasons, shingle_text
from src.medrag.data.phase3.overlap import (
    CROSS_DOCUMENT_OVERLAP,
    INTENTIONAL_CHUNK_OVERLAP,
    PARTIAL_CONTENT_REUSE,
    classify_overlap,
    is_boilerplate_only,
    overlap_pair_id,
)


def _passage(
    passage_id: str,
    doc_id: int,
    index: int,
    text: str,
    start: int,
    *,
    shingle_count: int = 100,
) -> dict:
    return {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "passage_index": index,
        "passage_text": text,
        "char_start": start,
        "char_end": start + len(text),
        "shingle_count": shingle_count,
    }


def test_intentional_chunk_overlap_requires_adjacent_exact_source_span():
    source = "alpha beta gamma delta epsilon"
    left = _passage("p1", 1, 0, source[:22], 0)
    right = _passage("p2", 1, 1, source[11:], 11)
    overlap_type, reasons, overlap_chars = classify_overlap(
        left,
        right,
        shared_shingles=8,
        jaccard=0.2,
        left_containment=0.4,
        right_containment=0.4,
    )
    assert overlap_type == INTENTIONAL_CHUNK_OVERLAP
    assert "overlap_text_exact" in reasons
    assert overlap_chars > 0


def test_cross_document_bidirectional_overlap():
    left = _passage("p1", 1, 0, "same medical content", 0)
    right = _passage("p2", 2, 0, "same medical content plus", 0)
    overlap_type, _, _ = classify_overlap(
        left,
        right,
        shared_shingles=60,
        jaccard=0.6,
        left_containment=0.75,
        right_containment=0.75,
    )
    assert overlap_type == CROSS_DOCUMENT_OVERLAP


def test_asymmetric_containment_is_partial_content_reuse():
    left = _passage("p1", 1, 0, "short reused content", 0, shingle_count=50)
    right = _passage("p2", 2, 0, "long article containing short reused content", 0,
                     shingle_count=150)
    overlap_type, reasons, _ = classify_overlap(
        left,
        right,
        shared_shingles=45,
        jaccard=0.29,
        left_containment=0.90,
        right_containment=0.30,
    )
    assert overlap_type == PARTIAL_CONTENT_REUSE
    assert "asymmetric_containment" in reasons


def test_boilerplate_only_is_not_medical_equivalence_evidence():
    left = "Privacy policy. All rights reserved. Read more."
    right = "Privacy policy. All rights reserved. Read more articles."
    excluded, signals = is_boilerplate_only(
        left, right, shared_shingles=8, maximum_containment=0.9
    )
    assert excluded
    assert "privacy_or_terms" in signals


def test_multilingual_site_templates_are_boilerplate_only():
    vi = "{name} - {time} {body} Ý kiến của bạn sẽ được biên tập. Vui lòng gõ tiếng Việt có dấu"
    vi_copy = "{name} - {time} {body} Ý kiến của bạn sẽ được biên tập trước khi đăng"
    excluded, signals = is_boilerplate_only(
        vi, vi_copy, shared_shingles=30, maximum_containment=0.95
    )
    assert excluded
    assert "comment_form" in signals
    zh = "症状查询专题 点击部位名称可查询更多相关疾病 按症状名称开头字母查询症状"
    excluded, signals = is_boilerplate_only(
        zh, zh + " 参看", shared_shingles=30, maximum_containment=0.95
    )
    assert excluded
    assert "symptom_directory_navigation" in signals
    disease_directory = "多指畸形 骨折 按疾病名称开头字母查询疾病 按医学分科查询疾病"
    excluded, signals = is_boilerplate_only(
        disease_directory, disease_directory + " 更多疾病", shared_shingles=30,
        maximum_containment=0.95,
    )
    assert excluded
    assert "disease_directory_navigation" in signals
    feed = "2026-10-01 15:07:00 baophutho.vn bài một 2026-09-30 07:14:00 bài hai"
    excluded, signals = is_boilerplate_only(
        feed, feed + " bài ba", shared_shingles=30, maximum_containment=0.95
    )
    assert excluded
    assert "news_feed_navigation" in signals
    directory = "细菌性皮肤病 病毒性皮肤病 皮肤真菌病 皮炎 湿疹 皮肤肿瘤"
    directory_tail = "皮炎 湿疹 皮肤肿瘤"
    excluded, signals = is_boilerplate_only(
        directory, directory_tail, shared_shingles=25, maximum_containment=1.0
    )
    assert excluded
    assert "skin_directory_navigation" in signals
    manufacturer_directory = "制药企业分区域查询 制药企业名录列表可以按省份查看 全国制药企业列表"
    excluded, signals = is_boilerplate_only(
        manufacturer_directory,
        "制药企业名录列表可以按省份查看 全国制药企业列表",
        shared_shingles=25,
        maximum_containment=1.0,
    )
    assert excluded
    assert "manufacturer_directory_navigation" in signals


def test_medical_negation_and_dosage_conflicts_are_guarded():
    left = "This medicine is recommended at a dose of 5 mg daily."
    right = "This medicine is not recommended at a dose of 10 mg daily."
    reasons = medical_guard_reasons(left, right)
    assert "medical_negation_mismatch" in reasons
    assert "drug_dosage_mismatch" in reasons


def test_language_aware_shingling_covers_vi_en_zh():
    assert shingle_text("Bệnh nhân cần theo dõi đường huyết mỗi ngày", "vi")[1] == "vietnamese_syllable"
    assert shingle_text("Patients should monitor blood glucose every day", "en")[1] == "english_word"
    assert shingle_text("糖尿病患者应定期监测血糖水平", "zh")[1] == "chinese_character"


def test_pair_identity_and_output_digest_are_deterministic():
    assert overlap_pair_id("p-left", "p-right") == overlap_pair_id("p-right", "p-left")
    rows = [{"pair_id": "one", "score": 0.5}]
    assert _rows_digest(rows) == _rows_digest(rows)


def test_output_namespace_cannot_overlap_prior_releases():
    output = ROOT / "outputs/vibiomir_phase3_task5_test"
    assert _safe_output(output, []) == output.resolve()
    protected = ROOT / "outputs/vibiomir_phase2_full_v2"
    try:
        _safe_output(protected, [protected])
    except ValueError as error:
        assert "phase3" in str(error).casefold() or "overlaps" in str(error)
    else:
        raise AssertionError("Protected release was accepted as Task 5 output")


def test_task5_prepared_checkpoint_can_resume_without_recomputing(tmp_path):
    record = {
        "passage_id": "p-checkpoint",
        "doc_id": 1,
        "official_url": "https://example.test/1",
        "passage_index": 0,
        "passage_text": "Checkpoint passage",
        "char_start": 0,
        "char_end": 18,
        "source_text_sha256": "a" * 64,
        "language": "en",
        "chunker_version": "chunker-v2",
        "normalized_content_hash": "b" * 64,
        "shingle_strategy": "english_word",
        "shingle_size": 5,
        "shingles": [1, 2, 3],
        "signature": list(range(64)),
    }
    path = tmp_path / "prepared.parquet"
    fingerprint = _write_prepared_cache(path, [record])
    resumed = _load_prepared_cache(path, fingerprint)
    assert resumed[0]["passage_id"] == record["passage_id"]
    assert resumed[0]["shingles"] == frozenset(record["shingles"])
    assert resumed[0]["signature"] == tuple(record["signature"])
