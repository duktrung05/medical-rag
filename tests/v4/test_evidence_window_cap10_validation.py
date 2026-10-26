from scripts.v4.run_evidence_window_cap10_validation import MAX_SWAPS, allocate


def test_cap10_allocator_preserves_60_chunks_and_limits_changes():
    documents = list(range(1, 81))
    pair = {
        "id": 1,
        "s2_documents": documents,
        "s2_chunks": [{"doc_id": doc, "chunk_text": str(doc)} for doc in range(1, 61)],
        "contexts": [{"official_id": doc, "document_rank": doc,
                      "window_ids": [f"w{doc}"]} for doc in documents],
        "windows": [{"window_id": f"w{doc}"} for doc in documents],
    }
    scores = {"scores": [{"window_id": f"w{doc}",
                           "score": float(doc if doc > 60 else -doc)}
                          for doc in documents]}
    selected, _, swaps = allocate(pair, scores)
    assert len(swaps) == MAX_SWAPS == 10
    assert len(selected) == len(set(selected)) == 60
    assert {row["entrant"] for row in swaps} == set(range(71, 81))
