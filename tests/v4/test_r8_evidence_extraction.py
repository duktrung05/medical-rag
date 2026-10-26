from scripts.v4.run_r8_evidence_extraction import (
    choose_a1,
    choose_a2,
    exact_ids,
    lexical_features,
    split,
    token_jaccard,
)


def row(doc, score, *, baseline=False, parent=False, rank=None, text=None, tokens=100):
    text = text or f"document {doc} evidence"
    return {"doc_id": doc, "lexical_score": score, "is_baseline_text": baseline,
            "is_baseline_chunk_parent": parent, "fits_512": tokens <= 512,
            "input_tokens": tokens, "start": 0, "text_sha256": f"{doc}-{score}-{baseline}",
            "text": text, "document_rank": rank or doc}


def test_split_is_deterministic_and_pilot_development():
    assert all(split(qid) == "development" for qid in (53, 130, 380, 754, 771, 1144))
    assert split(1) == split(1)
    assert split(2) in {"development", "validation"}


def test_lexical_score_prefers_query_terms():
    query = "điều trị viêm dạ dày"
    relevant = lexical_features(query, "Điều trị viêm dạ dày bằng thuốc phù hợp")
    irrelevant = lexical_features(query, "Thời tiết hôm nay nhiều mây")
    assert relevant["lexical_score"] > irrelevant["lexical_score"]


def test_a1_preserves_exact_parent_set_and_respects_gate():
    grouped = {doc: [row(doc, 0.1, baseline=True, parent=True), row(doc, 0.3)]
               for doc in range(1, 61)}
    selected = choose_a1(grouped, {"mode": "margin", "margin": 0.1})
    assert set(selected) == set(range(1, 61))
    assert all(not value["is_baseline_text"] for value in selected.values())
    guarded = choose_a1(grouped, {"mode": "margin", "margin": 0.3})
    assert all(value["is_baseline_text"] for value in guarded.values())


def test_a2_keeps_60_unique_parents_and_swap_cap():
    grouped = {doc: [row(doc, 0.1, baseline=True, parent=True), row(doc, 0.2)]
               for doc in range(1, 61)}
    grouped.update({doc: [row(doc, 0.8, rank=doc, text=f"unique outsider {doc}")]
                    for doc in range(61, 66)})
    a1 = choose_a1(grouped, {"mode": "margin", "margin": 0.2})
    a2 = choose_a2(grouped, a1, {"swap_cap": 2, "margin": 0.1, "max_jaccard": 0.99})
    assert len(a2) == len(set(a2)) == 60
    assert len(set(a2) - set(a1)) == 2


def test_exact_matching_is_parent_scoped():
    labels = [{"registry_id": "a", "doc_id": 1, "evidence_text": "đúng evidence"},
              {"registry_id": "b", "doc_id": 2, "evidence_text": "đúng evidence"}]
    prediction = {1: {"text": "Có ĐÚNG   evidence ở đây"}}
    assert exact_ids(labels, prediction) == {"a"}


def test_jaccard_detects_duplicate_content():
    assert token_jaccard("alpha beta gamma", "alpha beta gamma") == 1.0
    assert token_jaccard("alpha beta", "delta epsilon") == 0.0
