"""Tests for Pydantic data schemas."""

import pytest
from pydantic import ValidationError

from src.data.schema import ChunkRecord, GroundTruthRecord, PredictionRecord, QueryRecord


def test_query_record_valid():
    q = QueryRecord(id="Q001", query="suy tim")
    assert q.id == "Q001"
    assert q.query == "suy tim"


def test_query_record_whitespace_strip():
    q = QueryRecord(id="  Q002  ", query="  đau thắt ngực  ")
    assert q.id == "Q002"
    assert q.query == "đau thắt ngực"


def test_query_record_invalid_empty():
    with pytest.raises(ValidationError):
        QueryRecord(id="", query="valid query")

    with pytest.raises(ValidationError):
        QueryRecord(id="Q003", query="   ")


def test_query_record_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        QueryRecord.model_validate({"id": "Q001", "query": "text", "extra_field": "disallowed"})


def test_chunk_record_valid():
    c = ChunkRecord(
        chunk_id="C001_0",
        doc_id="DOC_001",
        chunk_index=0,
        language="vi",
        text="Sample clinical text",
    )
    assert c.chunk_id == "C001_0"
    assert c.doc_id == "DOC_001"
    assert c.chunk_index == 0
    assert c.language == "vi"
    assert c.text == "Sample clinical text"


def test_chunk_record_invalid():
    with pytest.raises(ValidationError):
        ChunkRecord(
            chunk_id="",
            doc_id="DOC_001",
            chunk_index=0,
            language="vi",
            text="text",
        )

    with pytest.raises(ValidationError):
        ChunkRecord(
            chunk_id="C1",
            doc_id="DOC_001",
            chunk_index=-1,  # Negative index not allowed
            language="vi",
            text="text",
        )


def test_prediction_record_valid():
    p = PredictionRecord(
        id="Q001",
        relevant_docs=["DOC_001", "DOC_002"],
        relevant_chunks=["C001_0", "C002_0"],
    )
    assert p.id == "Q001"
    assert len(p.relevant_docs) == 2
    assert len(p.relevant_chunks) == 2


def test_prediction_record_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        PredictionRecord(
            id="Q001",
            relevant_docs=["DOC_001", "DOC_001"],  # duplicate
            relevant_chunks=["C001_0"],
        )

    with pytest.raises(ValidationError):
        PredictionRecord(
            id="Q001",
            relevant_docs=["DOC_001"],
            relevant_chunks=["C001_0", "C001_0"],  # duplicate
        )


def test_ground_truth_record_valid():
    gt = GroundTruthRecord(
        id="Q001",
        relevant_docs=["DOC_001"],
        relevant_chunks=["C001_0", "C001_1"],
    )
    assert gt.id == "Q001"
    assert gt.relevant_docs == ["DOC_001"]
    assert len(gt.relevant_chunks) == 2
