from r2ai.evidence_selector import EvidenceCandidate, QueryRanking
from scripts.workflows.vibiomir.audit_task0 import (
    query_features,
    stratified_sample,
    summarize,
)


def candidate(doc_id, index, score, text="evidence"):
    return EvidenceCandidate(doc_id=doc_id, chunk_index=index, chunk_text=text,
                             title="", retrieval_score=0.5, rerank_score=score)


def diagnostics(rows, selected=1, max_chunks=2):
    return {
        row.query_id: {
            "id": row.query_id,
            "config": {"max_chunks": max_chunks},
            "selected_chunk_count": selected,
            "selected_doc_count": selected,
            "reasons": {"selected": selected},
        }
        for row in rows
    }


def test_stratified_sample_is_deterministic_unique_and_bounded():
    rows = [
        QueryRanking(i, [i], [candidate(i, 0, float(i), "中文" if i % 2 else "tiếng Việt")], f"q{i}")
        for i in range(1, 13)
    ]
    features, thresholds = query_features(rows, diagnostics(rows))
    first = stratified_sample(rows, features, 8, 42)
    second = stratified_sample(rows, features, 8, 42)
    assert [row.query_id for row in first] == [row.query_id for row in second]
    assert len({row.query_id for row in first}) == 8
    assert thresholds["top_score_tertiles"][0] < thresholds["top_score_tertiles"][1]


def test_summary_counts_caps_scores_and_document_diversity():
    rows = [
        QueryRanking(1, [1, 2], [candidate(1, 0, 2), candidate(1, 1, 1), candidate(2, 0, None)]),
        QueryRanking(2, [3], [candidate(3, 0, 3, "中文")]),
    ]
    diag = diagnostics(rows, selected=2, max_chunks=2)
    features, _ = query_features(rows, diag)
    report = summarize(rows, diag, features)
    assert report["candidate_pairs"] == 4
    assert report["unscored_pairs"] == 1
    assert report["queries_at_chunk_cap"] == 2
    assert report["unique_docs_per_query"]["mean"] == 1.5
    assert report["decision_reasons"] == {"selected": 4}
