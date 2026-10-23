from __future__ import annotations

import pytest
import sqlite3

from scripts.workflows.vibiomir.phase3_scalable_canonical import (
    _memberships as scalable_memberships,
    _representatives as scalable_representatives,
    _schema as scalable_schema,
)
from scripts.workflows.vibiomir.select_phase3_canonical_content import ROOT, _safe_output
from src.medrag.data.phase3.canonical import (
    CANONICAL_POLICY_VERSION,
    document_selection_key,
    select_representative,
)


def _document(
    doc_id: int,
    *,
    review: bool = False,
    flags: list[str] | None = None,
    text: str = "same text",
    provenance: bool = True,
) -> dict:
    return {
        "doc_id": doc_id,
        "official_url": f"https://example.test/{doc_id}",
        "source_kind": "pubmed",
        "normalized_text": text,
        "quality_review_required": review,
        "quality_flags": flags or [],
        "language_confidence": 0.99,
        "source_document_sha256": "a" * 64 if provenance else None,
        "phase0_raw_sha256": "b" * 64 if provenance else None,
        "pmid": str(doc_id) if provenance else None,
    }


def test_quality_precedes_completeness_and_provenance():
    clean = _document(20, text="short", provenance=False)
    review = _document(10, review=True, text="a much longer text", provenance=True)
    canonical, _, reason, _ = select_representative(
        [review, clean], entity_type="document"
    )
    assert canonical["doc_id"] == 20
    assert reason == "quality_review_status"


def test_completeness_then_provenance_then_id_are_deterministic():
    short = _document(1, text="short", provenance=True)
    long = _document(9, text="a longer complete text", provenance=False)
    assert document_selection_key(long) < document_selection_key(short)
    canonical, ranked, reason, chain = select_representative(
        [short, long], entity_type="document"
    )
    assert canonical["doc_id"] == 9
    assert [row["doc_id"] for row in ranked] == [9, 1]
    assert reason == "text_completeness"
    assert chain


def test_exact_tie_uses_lowest_doc_id_independent_of_input_order():
    high, low = _document(9), _document(3)
    first = select_representative([high, low], entity_type="document")
    second = select_representative([low, high], entity_type="document")
    assert first[0]["doc_id"] == second[0]["doc_id"] == 3
    assert first[2] == "lowest_doc_id_tiebreak"
    assert CANONICAL_POLICY_VERSION in first[3][0] or first[3]


def test_passage_selection_prefers_better_parent_and_keeps_occurrence_identity():
    weak = _document(1, review=True)
    strong = _document(2)
    members = [
        {"passage_id": "p1", "doc_id": 1, "passage_index": 0,
         "passage_text": "same", "document": weak},
        {"passage_id": "p2", "doc_id": 2, "passage_index": 0,
         "passage_text": "same", "document": strong},
    ]
    canonical, ranked, reason, _ = select_representative(
        members, entity_type="passage"
    )
    assert canonical["passage_id"] == "p2"
    assert {row["passage_id"] for row in ranked} == {"p1", "p2"}
    assert reason == "parent_quality_review_status"


def test_singleton_and_unknown_relationships_fail_closed():
    with pytest.raises(ValueError, match="verified duplicate"):
        select_representative([_document(1)], entity_type="document")
    with pytest.raises(ValueError, match="Unsupported"):
        select_representative([_document(1), _document(2)], entity_type="near")


def test_output_namespace_is_new_and_bounded():
    expected = ROOT / "outputs/vibiomir_phase3_task6_test"
    assert _safe_output(expected) == expected.resolve()
    with pytest.raises(ValueError, match="phase3"):
        _safe_output(ROOT / "outputs/task6_test")


def test_disk_backed_full_canonical_matches_selection_policy():
    connection = sqlite3.connect(":memory:")
    scalable_schema(connection, "input", "config")
    quality = (
        "pubmed", "same exact content", 0, "[]", 0.99, None, "a" * 64,
        None, None, "pmid", "extractor", "source", "b" * 64,
    )
    connection.executemany(
        "INSERT INTO document_quality VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(2, *quality), (1, *quality)],
    )
    connection.executemany(
        "INSERT INTO document_members VALUES (?, ?, ?, ?, ?, ?)",
        [("dcg", 2, 2, "u2", "c" * 64, "d" * 64),
         ("dcg", 2, 1, "u1", "c" * 64, "d" * 64)],
    )
    connection.executemany(
        "INSERT INTO passage_members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [("pcg", 2, "p2", 2, "u2", 0, 0, 4, "c" * 64, "d" * 64, "same"),
         ("pcg", 2, "p1", 1, "u1", 0, 0, 4, "c" * 64, "d" * 64, "same")],
    )
    representatives = list(scalable_representatives(connection))
    memberships = list(scalable_memberships(connection))
    assert [(row["entity_type"], row["canonical_doc_id"]) for row in representatives] == [
        ("document", 1), ("passage", 1)
    ]
    assert len(memberships) == 4
    assert all(sum(
        item["is_canonical"] for item in memberships
        if item["content_group_id"] == group
    ) == 1 for group in {"dcg", "pcg"})
    connection.close()
