import pytest

from src.data.loader import DocumentChunkMap
from src.selection.prediction import select_prediction
from src.selection.threshold import select_candidates


@pytest.mark.parametrize(
    ("scores", "options", "expected"),
    [
        ([('a', 0.9), ('b', 0.2)], dict(threshold=0.5, relative_delta=1, min_k=0, max_k=5), ['a']),
        ([('a', 0.9), ('b', 0.8), ('c', 0.1)], dict(threshold=0.5, relative_delta=0.2, min_k=0, max_k=5), ['a', 'b']),
        ([('a', 0.2), ('b', 0.1)], dict(threshold=0.5, relative_delta=0.1, min_k=0, max_k=5), []),
        ([('a', 0.2), ('b', 0.1)], dict(threshold=0.5, relative_delta=0.1, min_k=1, max_k=5), ['a']),
        ([('c', 0.9), ('b', 0.9), ('a', 0.9)], dict(threshold=0, relative_delta=1, min_k=0, max_k=2), ['c', 'b']),
    ],
)
def test_selector_threshold_fallback_cap_and_tie(scores, options, expected):
    assert select_candidates(scores, **options) == expected


def test_selector_rejects_invalid_bounds_and_nonfinite_scores():
    with pytest.raises(ValueError, match="selection bounds"):
        select_candidates([], min_k=2, max_k=1)
    with pytest.raises(ValueError, match="finite"):
        select_candidates([("a", float("nan"))])


def test_parent_consistency_uses_hard_doc_cap_and_records_added_parent():
    mapping = DocumentChunkMap({"c1": "d1", "c1b": "d1", "c2": "d2"},
                               {"d1": ["c1", "c1b"], "d2": ["c2"]})
    prediction, details = select_prediction(
        "q", [("c1", 10.0), ("c2", 9.0), ("c1b", 0.0)], mapping,
        doc_aggregation="mean", chunk_threshold=9.5, chunk_delta=1, chunk_min_k=0, chunk_max_k=2,
        doc_threshold=0, doc_delta=100, doc_min_k=1, doc_max_k=1,
    )
    assert prediction.relevant_chunks == ["c1"]
    assert prediction.relevant_docs == ["d1"]
    assert details["parent_docs_added"] == ["d1"]
    assert len(prediction.relevant_docs) <= 1


def test_parent_cap_drops_chunk_when_no_doc_capacity_remains():
    mapping = DocumentChunkMap({"c1": "d1", "c2": "d2"}, {"d1": ["c1"], "d2": ["c2"]})
    prediction, details = select_prediction(
        "q", [("c1", 2.0), ("c2", 1.0)], mapping,
        chunk_threshold=0, chunk_delta=10, chunk_min_k=0, chunk_max_k=2,
        doc_threshold=0, doc_delta=10, doc_min_k=1, doc_max_k=1,
    )
    assert prediction.relevant_chunks == ["c1"]
    assert prediction.relevant_docs == ["d1"]
    assert details["chunks_dropped_for_parent_cap"] == [{"chunk_id": "c2", "reason": "parent_document_cap"}]
