import numpy as np

from src.data.loader import DocumentChunkMap
from src.retrieval.exact_dense import exact_search
from scripts.run_dense_smoke import predictions_at_k


def test_exact_search_is_score_sorted_and_breaks_ties_by_id():
    passages = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    result = exact_search(queries, passages, ["c2", "c1", "c3"], top_k=3)[0]
    assert [item.chunk_id for item in result] == ["c1", "c2", "c3"]
    assert [item.rank for item in result] == [1, 2, 3]


def test_predictions_include_each_selected_chunks_parent():
    rankings = exact_search(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        ["c1", "c2"],
        top_k=2,
    )
    doc_map = DocumentChunkMap(
        chunk_to_doc={"c1": "d1", "c2": "d1"},
        doc_to_chunks={"d1": ["c1", "c2"]},
    )
    prediction = predictions_at_k(["q1"], rankings, doc_map, k=2)[0]
    assert prediction.relevant_chunks == ["c1", "c2"]
    assert prediction.relevant_docs == ["d1"]

