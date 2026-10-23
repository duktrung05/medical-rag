"""P5.3B parity distinguishes ordering/numerics from evidence semantics."""
from scripts.v4.audit_p5_3b_parity import inversion_rows, without_rerank_scores


def test_semantic_normalization_removes_only_rerank_score():
    evidence = {"passage_id": "p1", "text": "medical text", "start": 3, "end": 15,
                "source_sha256": "abc", "context_passage_ids": ["p0", "p1"],
                "rerank_score": 1.25,
                "seed_evidence": {"passage_id": "p1", "rerank_score": 1.25}}
    normalized = without_rerank_scores(evidence)
    assert "rerank_score" not in normalized
    assert "rerank_score" not in normalized["seed_evidence"]
    assert normalized["text"] == evidence["text"]
    assert normalized["start"] == evidence["start"]
    assert normalized["end"] == evidence["end"]
    assert normalized["source_sha256"] == evidence["source_sha256"]
    assert normalized["context_passage_ids"] == evidence["context_passage_ids"]


def test_pairwise_rank_inversion_uses_deterministic_identity_ties():
    pair = {"pairs": [
        {"official_id": 1, "passage_id": "a", "input_sha256": "a"},
        {"official_id": 2, "passage_id": "b", "input_sha256": "b"},
    ]}
    rows = inversion_rows(53, pair, [2.0, 1.0], [0.5, 1.5])
    assert len(rows) == 1
    assert rows[0]["baseline_left_rank"] == 1
    assert rows[0]["gpu_left_rank"] == 2
