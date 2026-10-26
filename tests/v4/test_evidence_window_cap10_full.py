from scripts.v4.run_evidence_window_cap10_full import compact_pair, worker_query_ids


def test_compact_pair_removes_duplicate_window_text_but_preserves_scoring_identity():
    pair = {
        "format": "old", "id": 9999, "query": "q", "query_sha256": "qh",
        "windows": [{"window_id": "w", "official_id": 1, "context_id": "c",
                     "kind": "head", "start": 0, "end": 2, "text_sha256": "th",
                     "text": "duplicate", "payload": "title\ntext", "input_tokens": 3,
                     "body_token_start": 0, "body_token_end": 1,
                     "requested_body_token_start": 0, "requested_body_token_end": 1,
                     "query_id": 9999, "query_sha256": "qh", "input_sha256": "ih"}],
    }
    compact = compact_pair(pair)
    assert "text" not in compact["windows"][0]
    assert compact["windows"][0]["payload"] == "title\ntext"
    assert compact["windows"][0]["input_sha256"] == "ih"


def test_worker_partition_is_complete_and_disjoint():
    lock = {"workers": 2, "new_query_ids": list(range(1, 12))}
    first = worker_query_ids(lock, 0, 2)
    second = worker_query_ids(lock, 1, 2)
    assert not set(first) & set(second)
    assert sorted(first + second) == lock["new_query_ids"]
