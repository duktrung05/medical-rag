from scripts.v4.run_evidence_window_allocation import (
    MAX_SWAPS,
    WINDOW_BODY_TOKENS,
    allocate,
    stage_decision,
    window_ranges,
)


def test_window_ranges_are_bounded_deterministic_anchors():
    assert window_ranges(100) == [("full", 0, 100)]
    assert window_ranges(WINDOW_BODY_TOKENS + 1) == [
        ("head", 0, WINDOW_BODY_TOKENS),
        ("tail", 1, WINDOW_BODY_TOKENS + 1),
    ]
    assert window_ranges(500) == [
        ("head", 0, WINDOW_BODY_TOKENS),
        ("middle", 154, 346),
        ("tail", 308, 500),
    ]


def test_allocate_preserves_count_and_caps_swaps():
    documents = list(range(1, 71))
    chunks = [{"doc_id": doc, "chunk_text": f"base-{doc}"} for doc in range(1, 61)]
    contexts = []
    windows = []
    scores = []
    for rank, doc in enumerate(documents, 1):
        window_id = f"window-{doc}"
        contexts.append({"official_id": doc, "document_rank": rank,
                         "window_ids": [window_id]})
        windows.append({"window_id": window_id})
        # All ten outsiders beat all baseline parents, but the policy may replace only six.
        score = float(doc if doc > 60 else -doc)
        scores.append({"window_id": window_id, "score": score})
    pair = {"id": 1, "s2_documents": documents, "s2_chunks": chunks,
            "contexts": contexts, "windows": windows}
    selected, _, swaps = allocate(pair, {"scores": scores})

    assert len(swaps) == MAX_SWAPS
    assert len(selected) == 60
    assert len(set(selected)) == 60
    assert selected == sorted(selected)
    assert {row["entrant"] for row in swaps} == {65, 66, 67, 68, 69, 70}


def test_development_gate_requires_real_manual_gains_and_bounded_losses():
    decision, checks = stage_decision(
        "development", {"1:10", "2:20", "3:30"}, {"4:40"},
        {"5:50", "6:60"}, {"7:70"}, True,
    )
    assert decision == "OPEN_VALIDATION"
    assert all(checks.values())

    stopped, failed = stage_decision(
        "development", {"1:10", "2:20"}, set(), {"5:50"}, set(), True,
    )
    assert stopped == "STOP_DEVELOPMENT"
    assert not failed["manual_gains_at_least_3"]
    assert not failed["manual_gain_queries_at_least_3"]


def test_validation_gate_is_fail_closed_on_visible_manual_loss():
    decision, checks = stage_decision(
        "validation", {"1:10"}, {"2:20"}, {"3:30"}, set(), True,
    )
    assert decision == "STOP_VALIDATION"
    assert not checks["manual_visible_evidence_preserved"]
