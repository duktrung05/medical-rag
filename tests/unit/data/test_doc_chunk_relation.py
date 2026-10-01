"""Tests for Document-Chunk mappings and parent consistency enforcement."""

from pathlib import Path
import tempfile
from src.data.loader import DocumentChunkMap
from src.data.schema import ChunkRecord, PredictionRecord
from src.submission.build import enforce_parent_consistency


def test_document_chunk_map_indexing():
    chunks = [
        ChunkRecord(chunk_id="C1", doc_id="DOC_A", chunk_index=0, language="vi", text="A0"),
        ChunkRecord(chunk_id="C2", doc_id="DOC_A", chunk_index=1, language="vi", text="A1"),
        ChunkRecord(chunk_id="C3", doc_id="DOC_B", chunk_index=0, language="en", text="B0"),
    ]
    doc_map = DocumentChunkMap.from_chunks(chunks)

    assert doc_map.get_parent_doc("C1") == "DOC_A"
    assert doc_map.get_parent_doc("C2") == "DOC_A"
    assert doc_map.get_parent_doc("C3") == "DOC_B"
    assert doc_map.get_parent_doc("UNKNOWN") is None

    assert doc_map.get_child_chunks("DOC_A") == ["C1", "C2"]
    assert doc_map.get_child_chunks("DOC_B") == ["C3"]


def test_parent_consistency_enforcement():
    chunks = [
        ChunkRecord(chunk_id="C1", doc_id="DOC_A", chunk_index=0, language="vi", text="A0"),
        ChunkRecord(chunk_id="C2", doc_id="DOC_B", chunk_index=0, language="vi", text="B0"),
    ]
    doc_map = DocumentChunkMap.from_chunks(chunks)

    # In this prediction, C2 is retrieved, but its parent DOC_B is missing from relevant_docs
    bad_prediction = PredictionRecord(
        id="Q001",
        relevant_docs=["DOC_A"],
        relevant_chunks=["C1", "C2"],
    )

    fixed = enforce_parent_consistency(bad_prediction, doc_map)

    assert "DOC_B" in fixed.relevant_docs
    assert fixed.relevant_docs == ["DOC_A", "DOC_B"]
    assert fixed.relevant_chunks == ["C1", "C2"]


def test_parent_consistency_no_change_when_valid():
    chunks = [
        ChunkRecord(chunk_id="C1", doc_id="DOC_A", chunk_index=0, language="vi", text="A0"),
    ]
    doc_map = DocumentChunkMap.from_chunks(chunks)

    valid_prediction = PredictionRecord(
        id="Q001",
        relevant_docs=["DOC_A"],
        relevant_chunks=["C1"],
    )

    fixed = enforce_parent_consistency(valid_prediction, doc_map)
    assert fixed.relevant_docs == ["DOC_A"]
    assert fixed.relevant_chunks == ["C1"]


def test_parquet_roundtrip():
    chunks = [
        ChunkRecord(chunk_id="C1", doc_id="DOC_A", chunk_index=0, language="vi", text="A0"),
        ChunkRecord(chunk_id="C2", doc_id="DOC_B", chunk_index=0, language="en", text="B0"),
    ]
    doc_map = DocumentChunkMap.from_chunks(chunks)

    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = Path(tmpdir) / "doc_map.parquet"
        doc_map.save_parquet(pq_path)
        assert pq_path.is_file()

        loaded_map = DocumentChunkMap.load_parquet(pq_path)
        assert loaded_map.get_parent_doc("C1") == "DOC_A"
        assert loaded_map.get_parent_doc("C2") == "DOC_B"
