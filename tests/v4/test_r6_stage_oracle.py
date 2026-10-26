from scripts.v4.audit_r6_stage_oracle import (
    containment,
    first_loss,
    normalize,
    oracle_allocate,
    validate_s2_identity,
)


def stage_row(**updates):
    row = {
        "raw_union": True,
        "fused_top400": True,
        "rerank_200_parent": True,
        "refined_scored_parent": True,
        "selected_top200_document": True,
        "s2_selected_document": True,
        "allocated_top60_chunk": True,
        "final_evidence_exact": True,
        "available_context_exact": True,
        "source_contains_evidence": True,
    }
    row.update(updates)
    return row


def test_containment_normalizes_unicode_html_and_whitespace():
    text = "A &amp; B\n  Điều trị VIÊM dạ dày."
    matched, start, end = containment(text, "a & b điều trị viêm")
    assert matched
    assert text[start:end].startswith("A &amp; B")


def test_containment_rejects_empty_and_nonmatch():
    assert containment("abc", "")[0] is False
    assert containment("abc", "abd")[0] is False
    assert normalize("  A\nB  ") == "a b"


def test_first_loss_keeps_unknown_raw_separate_from_retrieval_miss():
    assert first_loss(stage_row(raw_union=False, fused_top400=False)) == "RAW_PARENT_MISS"
    assert first_loss(stage_row(raw_union=None, fused_top400=False)) == "ARTIFACT_UNAVAILABLE"
    assert first_loss(stage_row(raw_union=True, fused_top400=False)) == "FUSION_DROP"


def test_first_loss_stage_order_and_evidence_categories():
    assert first_loss(stage_row(rerank_200_parent=False,
                                refined_scored_parent=False)) == "PAIR_ALLOCATION_DROP"
    assert first_loss(stage_row(rerank_200_parent=False,
                                refined_scored_parent=True)) == "RETAINED_FINAL_EVIDENCE"
    assert first_loss(stage_row(selected_top200_document=False)) == "DOC_SELECTION_DROP"
    assert first_loss(stage_row(allocated_top60_chunk=False)) == "CHUNK_SLOT_DROP"
    assert first_loss(stage_row(final_evidence_exact=False,
                                available_context_exact=True)) == "PASSAGE_MISMATCH"
    assert first_loss(stage_row(final_evidence_exact=False,
                                available_context_exact=False,
                                source_contains_evidence=True)) == "CONTEXT_SPAN_MISS"


def test_oracle_respects_one_parent_one_slot_and_k():
    rows = [
        {"doc_id": 1, "hit": True},
        {"doc_id": 1, "hit": True},
        {"doc_id": 2, "hit": True},
        {"doc_id": 3, "hit": False},
    ]
    labels, query_hit = oracle_allocate(rows, "hit", 1)
    assert labels == 2
    assert query_hit == 1
    labels, query_hit = oracle_allocate(rows, "hit", 2)
    assert labels == 3
    assert query_hit == 1


def test_s2_identity_requires_parent_and_exact_linked_text():
    docs = list(range(1, 61))
    chunks = {doc: f"chunk-{doc}" for doc in docs}
    selected = {doc: {"official_id": doc} for doc in docs}
    expanded = {str(doc): {"text": f"chunk-{doc}"} for doc in docs}
    validate_s2_identity(docs, chunks, selected, expanded)


def test_s2_identity_rejects_wrong_parent_or_text():
    docs = list(range(1, 61))
    chunks = {doc: f"chunk-{doc}" for doc in docs}
    selected = {doc: {"official_id": doc} for doc in docs}
    expanded = {str(doc): {"text": f"chunk-{doc}"} for doc in docs}
    bad_chunks = dict(chunks)
    bad_chunks[61] = bad_chunks.pop(60)
    try:
        validate_s2_identity(docs, bad_chunks, selected, expanded)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned chunk parent must fail")
    expanded["1"] = {"text": "different"}
    try:
        validate_s2_identity(docs, chunks, selected, expanded)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned chunk text must fail")
