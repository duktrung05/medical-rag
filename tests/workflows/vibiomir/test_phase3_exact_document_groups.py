from __future__ import annotations

import hashlib

import pytest

from scripts.workflows.vibiomir.build_phase3_exact_document_groups import (
    ROOT,
    _safe_output,
    _select_parts,
    _verify_full_gate,
)
from src.medrag.data.phase3.exact import (
    GROUPING_POLICY_VERSION,
    content_group_id,
    exact_group_row,
    exact_match_type,
    verify_group_content,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(doc_id: int, source: str, normalized: str) -> dict:
    return {
        "doc_id": doc_id,
        "official_url": f"https://example.test/{doc_id}",
        "source_text_sha256": _sha(source),
        "normalized_text_sha256": _sha(normalized),
    }


def test_exact_document_duplicates_keep_every_occurrence():
    left = _document(1, "Same text", "Same text")
    right = _document(2, "Same text", "Same text")
    rows = [
        exact_group_row(
            value,
            raw_size=2,
            normalized_size=2,
            normalization_version="n1",
            phase1_release_id="release",
            grouping_scope="test",
        )
        for value in (left, right)
    ]
    assert [row["doc_id"] for row in rows] == [1, 2]
    assert rows[0]["raw_exact_group_id"] == rows[1]["raw_exact_group_id"]
    assert rows[0]["document_content_group_id"] == rows[1]["document_content_group_id"]
    assert {row["exact_match_type"] for row in rows} == {"raw_and_normalized_exact"}


def test_normalized_document_duplicates_remain_distinguishable_from_raw_exact():
    left = _document(1, "Same   text", "Same text")
    right = _document(2, "Same text", "Same text")
    rows = [
        exact_group_row(
            value,
            raw_size=1,
            normalized_size=2,
            normalization_version="n1",
            phase1_release_id="release",
            grouping_scope="test",
        )
        for value in (left, right)
    ]
    assert rows[0]["raw_exact_group_id"] != rows[1]["raw_exact_group_id"]
    assert rows[0]["normalized_exact_group_id"] == rows[1]["normalized_exact_group_id"]
    assert {row["exact_match_type"] for row in rows} == {"normalized_exact_only"}
    assert all(row["grouping_policy_version"] == GROUPING_POLICY_VERSION for row in rows)


def test_content_group_id_is_deterministic_and_normalized_based():
    digest = _sha("normalized")
    assert content_group_id(digest) == content_group_id(digest)
    assert content_group_id(digest).endswith(digest)
    assert exact_match_type(1, 1) == "unique"


def test_duplicate_hash_candidates_receive_exact_string_verification():
    digest = bytes.fromhex(_sha("same"))
    baselines = {}
    verify_group_content(baselines, digest, doc_id=1, text="same", label="source")
    verify_group_content(baselines, digest, doc_id=2, text="same", label="source")
    with pytest.raises(ValueError, match="collision or corrupt"):
        verify_group_content(baselines, digest, doc_id=3, text="different", label="source")


def test_full_build_requires_separate_gate_and_dry_run_is_bounded():
    parts = ["part-00000", "part-00001"]
    assert _select_parts(parts, 1, False) == ["part-00000"]
    with pytest.raises(ValueError, match="fewer than all"):
        _select_parts(parts, 2, False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_parts(parts, 1, True)
    with pytest.raises(ValueError, match="pilot-acceptance"):
        _verify_full_gate(None)


def test_output_must_use_a_dedicated_phase3_namespace():
    output = ROOT / "outputs/vibiomir_phase3_test"
    assert _safe_output(output, []) == output.resolve()
    with pytest.raises(ValueError, match="phase3"):
        _safe_output(ROOT / "outputs/not_phase3", [])
