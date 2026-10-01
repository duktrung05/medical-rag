"""Tests for scoring, RRF fusion, and dynamic threshold selection."""

import pytest
from src.retrieval.fusion import reciprocal_rank_fusion
from src.scoring.document_score import aggregate_doc_scores_max, aggregate_doc_scores_hybrid
from src.selection.threshold import select_candidates


def test_rrf_fusion():
    # List 1: C1 (rank 1), C2 (rank 2), C3 (rank 3)
    # List 2: C4 (rank 1), C1 (rank 2), C5 (rank 3)
    list1 = [("C1", 10.0), ("C2", 8.0), ("C3", 6.0)]
    list2 = [("C4", 0.9), ("C1", 0.8), ("C5", 0.7)]

    k = 60
    fused = reciprocal_rank_fusion([list1, list2], k=k, top_k=5)

    # C1 score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639 + 0.01613 = 0.03252
    # C4 score = 1/(60+1) = 1/61 ≈ 0.01639
    # C1 should be rank 1
    assert fused[0][0] == "C1"
    assert pytest.approx(fused[0][1], 1e-5) == (1 / 61.0 + 1 / 62.0)


def test_document_score_max():
    chunk_scores = {
        "C1": 0.13,
        "C2": 0.19,
        "C3": 0.94,
        "C4": 0.20,
    }
    chunk_to_doc = {
        "C1": "DOC_A",
        "C2": "DOC_A",
        "C3": "DOC_A",
        "C4": "DOC_A",
    }
    doc_scores = aggregate_doc_scores_max(chunk_scores, chunk_to_doc)
    assert pytest.approx(doc_scores["DOC_A"], 1e-5) == 0.94


def test_dynamic_selection_threshold_and_delta():
    # Candidates with scores
    # Top score = 0.95
    # threshold = 0.5, delta = 0.3 -> min score to pass delta is 0.95 - 0.3 = 0.65
    candidates = [
        ("C1", 0.95),
        ("C2", 0.90),
        ("C3", 0.70),
        ("C4", 0.60),  # passes abs (0.60 >= 0.50), but fails rel (0.60 < 0.65)
        ("C5", 0.30),  # fails both
    ]
    selected = select_candidates(
        candidates,
        threshold=0.50,
        relative_delta=0.30,
        min_k=1,
        max_k=10,
    )
    assert selected == ["C1", "C2", "C3"]


def test_selection_respects_min_k():
    # If all candidates fail threshold, min_k ensures at least top min_k are returned
    candidates = [
        ("C1", 0.20),
        ("C2", 0.15),
    ]
    selected = select_candidates(
        candidates,
        threshold=0.50,
        relative_delta=0.10,
        min_k=1,
        max_k=10,
    )
    assert selected == ["C1"]
