"""Tests for competition submission validator."""

import subprocess
import sys

import pytest

from src.data.loader import DocumentChunkMap
from src.data.schema import ChunkRecord
from src.submission.validate import (
    validate_submission_file,
    validate_submission_records,
)


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


def test_submission_reports_extra_query_ids():
    report = validate_submission_records(
        [{"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C1"]},
         {"id": "Q2", "relevant_docs": [], "relevant_chunks": []}],
        setup_test_doc_map(), expected_query_ids={"Q1"},
    )
    assert not report.is_valid
    assert report.extra_queries == ["Q2"]


def test_submitted_expected_queries_path_must_exist(tmp_path):
    submission = tmp_path / "submission.jsonl"
    submission.write_text('{"id":"Q1","relevant_docs":["DOC_A"],"relevant_chunks":["C1"]}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Expected queries file does not exist"):
        validate_submission_file(submission, setup_test_doc_map(), tmp_path / "missing.jsonl")


def test_expected_queries_must_have_unique_ids(tmp_path):
    submission = tmp_path / "submission.jsonl"
    submission.write_text('{"id":"Q1","relevant_docs":["DOC_A"],"relevant_chunks":["C1"]}\n', encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"id":"Q1","query":"a"}\n{"id":"Q1","query":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate query ID: Q1"):
        validate_submission_file(submission, setup_test_doc_map(), queries)


@pytest.mark.parametrize(
    ("field", "duplicated_id"),
    [("relevant_docs", "DOC_A"), ("relevant_chunks", "C1")],
)
def test_submission_rejects_internal_duplicates(field, duplicated_id):
    record = {"id": "Q1", "relevant_docs": ["DOC_A"], "relevant_chunks": ["C1"]}
    record[field] = [duplicated_id, duplicated_id]
    report = validate_submission_records([record], setup_test_doc_map(), expected_query_ids={"Q1"})
    assert not report.is_valid
    assert duplicated_id in report.internal_duplicates[0]


def test_validation_cli_exits_nonzero_on_invalid_submission(tmp_path):
    submission = tmp_path / "submission.jsonl"
    submission.write_text('{"id":"Q2","relevant_docs":[],"relevant_chunks":[]}\n', encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"id":"Q1","query":"question"}\n', encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "src.submission.validate", str(submission), "--test-queries", str(queries)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert completed.returncode != 0
    assert "Q1" in completed.stdout and "Q2" in completed.stdout


def test_validation_cli_rejects_requested_empty_corpus(tmp_path):
    submission = tmp_path / "submission.jsonl"
    submission.write_text('{"id":"Q1","relevant_docs":["DOC_A"],"relevant_chunks":["C1"]}\n', encoding="utf-8")
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "src.submission.validate", str(submission), "--corpus-chunks", str(corpus)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert completed.returncode != 0
    assert "Corpus map contains no chunks" in completed.stderr
