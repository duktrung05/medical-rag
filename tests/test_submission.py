"""Tests for competition submission validator."""

from src.data.loader import DocumentChunkMap
from src.data.schema import ChunkRecord
from src.submission.validate import validate_submission_records


def setup_test_doc_map() -> DocumentChunkMap:
    chunks = [
        ChunkRecord(chunk_id="C1", doc_id="DOC_A", chunk_index=0, language="vi", text="A0"),
        ChunkRecord(chunk_id="C2", doc_id="DOC_B", chunk_index=0, language="en", text="B0"),
        ChunkRecord(chunk_id="C3", doc_id="DOC_C", chunk_index=0, language="zh", text="C0"),
    ]
    return DocumentChunkMap.from_chunks(chunks)


def test_valid_submission():
    doc_map = setup_test_doc_map()
    records = [
        {"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C1"]},
        {"id": "Q2", "relevant_docs": ["DOC_B"], "relevant_chunks": ["C2"]},
    ]
    report = validate_submission_records(records, doc_map, expected_query_ids={"Q1", "Q2"})
    assert report.is_valid is True
    assert len(report.duplicate_queries) == 0
    assert len(report.missing_queries) == 0
    assert len(report.unknown_docs) == 0
    assert len(report.unknown_chunks) == 0
    assert len(report.parent_violations) == 0


def test_submission_duplicate_queries():
    doc_map = setup_test_doc_map()
    records = [
        {"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C1"]},
        {"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C1"]},
    ]
    report = validate_submission_records(records, doc_map, expected_query_ids={"Q1"})
    assert report.is_valid is False
    assert "Q1" in report.duplicate_queries


def test_submission_missing_query():
    doc_map = setup_test_doc_map()
    records = [
        {"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C1"]},
    ]
    report = validate_submission_records(records, doc_map, expected_query_ids={"Q1", "Q2"})
    assert report.is_valid is False
    assert "Q2" in report.missing_queries


def test_submission_unknown_doc_and_chunk():
    doc_map = setup_test_doc_map()
    records = [
        {"id": "Q1", "relevant_docs": ["DOC_UNKNOWN"], "relevant_chunks": ["C_UNKNOWN"]},
    ]
    report = validate_submission_records(records, doc_map, expected_query_ids={"Q1"})
    assert report.is_valid is False
    assert "DOC_UNKNOWN" in report.unknown_docs
    assert "C_UNKNOWN" in report.unknown_chunks


def test_submission_parent_consistency_violation():
    doc_map = setup_test_doc_map()
    # C2 belongs to DOC_B, but DOC_B is not in relevant_docs
    records = [
        {"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C2"]},
    ]
    report = validate_submission_records(records, doc_map, expected_query_ids={"Q1"})
    assert report.is_valid is False
    assert len(report.parent_violations) == 1
    assert "DOC_B" in report.parent_violations[0]
