from scripts.ablate_fusion import ordered_union, weighted_rrf
from scripts.analyze_failures import analyze_query, classify_candidate_positive


def test_error_taxonomy_tracks_candidate_fusion_rerank_and_selection():
    sample = {"id": "q", "query": "question", "topic": 1}
    truth = {"relevant_docs": ["d1"], "relevant_chunks": ["gold"]}
    rows = {
        "bm25": {"results": [("gold", 1.0), ("other", 0.5)]},
        "dense": {"results": [("other", 1.0)]},
        "hybrid": {"results": [("other", 1.0)]},
        "rerank": {"results": [("other", 2.0)], "prediction": {"relevant_docs": [], "relevant_chunks": []}},
    }
    result = analyze_query(sample, truth, rows, ranking_k=1)
    assert result["error_types"] == ["BM25_ONLY_HIT", "FUSION_DROPPED", "CANDIDATE_MISS", "DOC_AND_CHUNK_MISS"]
    assert result["primary_failure_stage"] == "retrieval_or_fusion"


def test_ranked_union_is_stable_and_weighted_rrf_respects_weights():
    sparse = [("a", 10), ("b", 9)]
    dense = [("b", 10), ("c", 9)]
    assert [cid for cid, _ in ordered_union(sparse, dense)] == ["a", "b", "c"]
    assert [cid for cid, _ in weighted_rrf(sparse, dense, k=60, sparse_weight=1, top_k=3)] == ["a", "b"]
    assert [cid for cid, _ in weighted_rrf(sparse, dense, k=60, sparse_weight=0, top_k=3)] == ["b", "c"]


def test_candidate_miss_marks_ranks_beyond_cache_unknown():
    result = classify_candidate_positive("gold", {"bm25": {}, "dense": {}, "hybrid": {}})
    assert result["error_class"] == "BOTH_MISS"
    assert result["bm25_rank"] is None and result["dense_rank"] is None and result["rrf_rank"] is None
    assert result["unknown_beyond_cache_depth"] is True
