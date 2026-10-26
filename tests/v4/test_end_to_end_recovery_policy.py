"""Guard the single candidate policy agreed for end-to-end evidence recovery."""
from scripts.v4.prepare_end_to_end_evidence_recovery import build_lock


def test_locked_recovery_policy_has_one_fixed_1200_query_treatment():
    lock = build_lock()
    policy = lock["policy"]

    assert policy["query_ids"] == list(range(1, 1201))
    assert policy["retrieval"] == {
        "official_query_only": True,
        "depth_per_branch": 800,
        "branches": ["document_bm25", "passage_bm25", "dense"],
        "parent_fusion": "RRF after grouping unique parents, k=60",
        "retrieved_unique_parent_cap": 800,
        "keep_all_cap10_documents": True,
        "new_parent_rerank_limit": 200,
    }
    assert policy["evidence"]["window_tokens"] == 192
    assert policy["evidence"]["window_stride_tokens"] == 96
    assert policy["allocation"]["max_replacements_per_query"] == 10
    assert policy["allocation"]["max_new_document_entrants_per_query"] == 3
    assert policy["allocation"]["baseline_chunks"] == 60
    assert policy["submission"]["btc_calls"] == 1
