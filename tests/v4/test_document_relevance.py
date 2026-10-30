import pytest

from src.medrag.ranking.document_relevance import (
    FEATURE_NAMES,
    conservative_document_swaps,
    document_features,
)


def candidate(doc_id, scores, *, maximum=None, ranks=None, rrf=0.1):
    return {
        "official_id": doc_id,
        "rerank_score": scores[0] if maximum is None else maximum,
        "score": rrf,
        "ranks": ranks or {},
        "evidence": [{"rerank_score": score} for score in scores],
    }


def test_document_features_use_multiple_passages_and_branch_ranks():
    values = document_features(candidate(
        1, [3.0, 1.0, -3.0],
        ranks={"document_bm25": 30, "passage_bm25": 60, "dense": 140},
    ))
    features = dict(zip(FEATURE_NAMES, values, strict=True))
    assert features["max_rerank"] == 3.0
    assert features["second_rerank"] == 1.0
    assert features["top2_mean"] == 2.0
    assert features["branch_count"] == 3.0
    assert features["document_bm25_rr"] == pytest.approx(2 / 3)


def test_conservative_swaps_preserve_chunk_parents_count_and_positions():
    candidates = [candidate(doc_id, [score]) for doc_id, score in (
        (1, 3.0), (2, 2.0), (3, 1.0), (4, 0.0), (5, -1.0),
    )]
    scores = {1: 4.0, 2: -2.0, 3: 0.0, 4: 3.0, 5: 2.0}
    selected, changes = conservative_document_swaps(
        [1, 2, 3], {1}, candidates, scores, max_swaps=2, minimum_margin=1.0,
    )
    assert selected == [1, 4, 5]
    assert [(row["victim"], row["entrant"]) for row in changes] == [(2, 4), (3, 5)]


def test_conservative_swaps_reject_removing_unknown_chunk_parent():
    with pytest.raises(ValueError, match="chunk parent"):
        conservative_document_swaps(
            [1], {2}, [candidate(1, [1.0])], {1: 1.0},
        )
