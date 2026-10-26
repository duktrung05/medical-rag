from scripts.v4.run_r7_parent_floor_score_gate import (
    choose_anchors,
    passage_bm25_fallbacks,
    positive_status,
)
from src.medrag.retrieval.disk_bm25 import DiskBM25


def test_batched_parent_fallback_matches_individual_search(tmp_path):
    index = DiskBM25(tmp_path / "passages.sqlite", "passage_bm25", "fixture", create=True)
    index.append_batch([
        (1, "p1", "alpha beta beta"),
        (1, "p2", "alpha"),
        (2, "p3", "beta gamma"),
        (2, "p4", "unrelated"),
        (3, "p5", "no matching term"),
    ])
    index.complete()

    actual = passage_bm25_fallbacks(index, "alpha beta", {1, 2, 3})
    expected = {}
    for parent in (1, 2, 3):
        rows = index.search_parent("alpha beta", parent, 1)
        if rows:
            expected[parent] = rows[0].passage_id
    assert actual == expected


def test_anchor_choice_is_bounded_unique_and_deterministic():
    rows = [{
        "anchor_id": f"a{index:03d}",
        "distance_neg4": abs(index - 5),
        "distance_neg2": abs(index - 45),
        "distance_chunk_cut": abs(index - 85),
        "distance_document_cut": abs(index - 125),
    } for index in range(200)]

    first = choose_anchors(rows)
    second = choose_anchors(list(reversed(rows)))
    assert first == second
    assert len(first) == 128
    assert len({row["anchor_id"] for row in first}) == 128


def test_positive_status_separates_docs_chunk_parent_and_visible_text():
    prediction = {
        "id": 7,
        "relevant_docs": [10, 11],
        "relevant_chunks": [{"doc_id": 10, "chunk_text": "Alpha   BETA evidence"}],
    }
    assert positive_status(7, 10, "alpha beta", prediction) == {
        "document": True, "chunk_parent": True, "evidence_visible": True,
    }
    assert positive_status(7, 11, "alpha beta", prediction) == {
        "document": True, "chunk_parent": False, "evidence_visible": False,
    }
