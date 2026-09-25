import numpy as np
import pytest

from src.data.loader import DocumentChunkMap
from src.pipeline import RetrievalPipeline
from src.reranking.bge_reranker import BGEReranker
from src.retrieval.fusion import FusedCandidate


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return np.array([0.2, 0.9])


def test_bge_reranker_batches_query_passage_pairs_and_supports_optional_context():
    model = FakeCrossEncoder()
    reranker = BGEReranker(model=model, batch_size=8, max_length=128)
    ranked = reranker.rerank(
        "query",
        [("c1", "plain text"), ("c2", "body", "title", "nearby context")],
        top_k=2,
    )
    assert ranked == [("c2", 0.9), ("c1", 0.2)]
    pairs, kwargs = model.calls[0]
    assert pairs == [
        ("query", "plain text"),
        ("query", "Title: title\nContext: nearby context\nbody"),
    ]
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is False


def test_bge_reranker_tie_is_deterministic():
    class Tied(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            return np.ones(len(pairs))

    assert BGEReranker(model=Tied()).rerank("q", [("z", "x"), ("a", "y")]) == [
        ("a", 1.0), ("z", 1.0)
    ]


def test_pipeline_keeps_unreranked_candidates_for_diagnostics_only():
    class Retriever:
        def search(self, query, top_k=200):
            return [("c1", 0.8), ("c2", 0.7), ("c3", 0.6)][:top_k]

    class Reranker:
        def rerank(self, query, candidates, top_k=100):
            assert [cid for cid, _ in candidates] == ["c1"]
            return [("c1", 0.95)]

    pipeline = RetrievalPipeline(
        Retriever(), DocumentChunkMap({"c1": "d", "c2": "d", "c3": "d"}),
        reranker=Reranker(), chunk_text_lookup={"c1": "one", "c2": "two", "c3": "three"},
        retrieval_top_k=3, reranker_top_k=1, chunk_threshold=0, chunk_delta=2,
        doc_threshold=0, doc_delta=2, min_k=0, max_chunk_k=1, max_doc_k=5,
    )
    result = pipeline.run_query("q1", "query")
    assert result.relevant_chunks == ["c1"]
    assert list(pipeline.last_candidate_scores) == ["c1", "c2", "c3"]
    assert pipeline.last_candidate_scores["c1"] == {
        "retrieval_score": 0.8, "raw_retrieval_score": 0.8,
        "rerank_score": 0.95, "selection_score": 0.95,
        "selection_eligible": True, "selection_exclusion_reason": None,
        "retrieval_rank": 1, "provenance": {"sources": ()},
    }
    assert pipeline.last_candidate_scores["c2"]["selection_score"] is None
    assert pipeline.last_candidate_scores["c2"]["selection_eligible"] is False
    assert pipeline.last_candidate_scores["c2"]["selection_exclusion_reason"] == "outside_reranker_top_k"
    assert pipeline.last_candidate_scores["c3"]["rerank_score"] is None


def test_pipeline_does_not_compare_rerank_logits_with_raw_retrieval_scores():
    class Retriever:
        def search(self, query, top_k=200):
            return [("c1", 0.9), ("c2", 0.8)]

    class Reranker:
        def rerank(self, query, candidates, top_k=100):
            return [("c1", -2.0)]

    pipeline = RetrievalPipeline(
        Retriever(), DocumentChunkMap({"c1": "d1", "c2": "d2"}),
        reranker=Reranker(), chunk_text_lookup={"c1": "one", "c2": "two"},
        retrieval_top_k=2, reranker_top_k=1,
        chunk_threshold=0, chunk_delta=1, doc_threshold=0, doc_delta=1,
        min_k=1, max_chunk_k=1, max_doc_k=2,
    )
    prediction = pipeline.run_query("q1", "query")
    assert prediction.relevant_chunks == ["c1"]
    assert prediction.relevant_docs == ["d1"]
    assert pipeline.last_candidate_scores["c2"]["raw_retrieval_score"] == 0.8


def test_rerank_reverses_chunk_and_document_order_and_preserves_provenance():
    class Retriever:
        def search_with_provenance(self, query, top_k=200):
            return [
                FusedCandidate("c1", 100.0, 1, 2),
                FusedCandidate("c2", 0.001, None, 1),
            ]

    class Reranker:
        def rerank(self, query, candidates, top_k=100):
            assert [cid for cid, _ in candidates] == ["c1", "c2"]
            return [("c1", -2.0), ("c2", 0.8)]

    pipeline = RetrievalPipeline(
        Retriever(), DocumentChunkMap({"c1": "d1", "c2": "d2"}),
        reranker=Reranker(), chunk_text_lookup={"c1": "one", "c2": "two"},
        retrieval_top_k=2, reranker_top_k=2,
        chunk_threshold=0, chunk_delta=0.1, doc_threshold=0, doc_delta=0.1,
        min_k=0, max_chunk_k=2, max_doc_k=2,
    )
    prediction = pipeline.run_query("q1", "query")
    assert prediction.relevant_chunks == ["c2"]
    assert prediction.relevant_docs == ["d2"]
    assert pipeline.last_candidate_scores["c1"]["retrieval_score"] == 100.0
    assert pipeline.last_candidate_scores["c1"]["selection_score"] == -2.0
    assert pipeline.last_candidate_scores["c1"]["provenance"] == {
        "bm25_rank": 1, "dense_rank": 2, "sources": ("bm25", "dense")}
    assert pipeline.last_candidate_scores["c2"]["provenance"]["sources"] == ("dense",)


def test_document_aggregation_uses_rerank_selection_scores():
    class Retriever:
        def search(self, query, top_k=200):
            return [("a1", 9.0), ("a2", 8.0), ("b1", 0.1)]

    class Reranker:
        def rerank(self, query, candidates, top_k=100):
            return [("b1", 0.9), ("a1", 0.2), ("a2", 0.1)]

    pipeline = RetrievalPipeline(
        Retriever(), DocumentChunkMap({"a1": "a", "a2": "a", "b1": "b"}),
        reranker=Reranker(), chunk_text_lookup={"a1": "a", "a2": "a", "b1": "b"},
        reranker_top_k=3, doc_aggregation="mean", chunk_threshold=0, chunk_delta=2,
        doc_threshold=0, doc_delta=2, min_k=0, max_chunk_k=3, max_doc_k=2,
    )
    prediction = pipeline.run_query("q1", "query")
    assert prediction.relevant_docs == ["b", "a"]  # 0.9 > mean(0.2, 0.1)
    assert prediction.relevant_chunks == ["b1", "a1", "a2"]


def test_pipeline_without_reranker_uses_retrieval_score_and_deterministic_doc_ties():
    class Retriever:
        def search(self, query, top_k=200):
            return [("cb", 0.8), ("ca", 0.8)]

    pipeline = RetrievalPipeline(
        Retriever(), DocumentChunkMap({"cb": "db", "ca": "da"}),
        chunk_threshold=0, chunk_delta=1, doc_threshold=0, doc_delta=1,
        min_k=0, max_chunk_k=2, max_doc_k=2,
    )
    prediction = pipeline.run_query("q1", "query")
    assert prediction.relevant_chunks == ["cb", "ca"]
    assert prediction.relevant_docs == ["da", "db"]
    assert all(item["selection_score"] == item["retrieval_score"]
               and item["rerank_score"] is None and item["selection_eligible"]
               for item in pipeline.last_candidate_scores.values())


def test_pipeline_rejects_invalid_reranker_depth_and_duplicate_retrieval_ids():
    class Retriever:
        def search(self, query, top_k=200):
            return [("c1", 0.9), ("c1", 0.8)]

    class Reranker:
        def rerank(self, query, candidates, top_k=100):
            return [("c1", 0.1)]

    kwargs = {"reranker": Reranker(), "chunk_text_lookup": {"c1": "one"}}
    with pytest.raises(ValueError, match="reranker_top_k must be >= max_chunk_k"):
        RetrievalPipeline(Retriever(), DocumentChunkMap({"c1": "d1"}),
                          reranker_top_k=1, max_chunk_k=2, **kwargs)
    pipeline = RetrievalPipeline(Retriever(), DocumentChunkMap({"c1": "d1"}),
                                 reranker_top_k=1, max_chunk_k=1, **kwargs)
    with pytest.raises(ValueError, match="duplicate chunk IDs.*c1"):
        pipeline.run_query("q1", "query")
    pipeline.reranker_top_k = 0
    with pytest.raises(ValueError, match="reranker_top_k must be >= max_chunk_k"):
        pipeline.run_query("q1", "query")


def test_reranker_requires_text_for_every_selected_candidate():
    class Retriever:
        def search(self, query, top_k=200):
            return [("c1", 0.9), ("c2", 0.8)]

    class Reranker:
        def rerank(self, query, candidates, top_k=100):
            raise AssertionError("Missing passage must be rejected before reranking")

    pipeline = RetrievalPipeline(
        Retriever(), DocumentChunkMap({"c1": "d1", "c2": "d2"}),
        reranker=Reranker(), chunk_text_lookup={"c1": "one"},
        reranker_top_k=2, max_chunk_k=2,
    )
    with pytest.raises(ValueError, match="chunk ID: c2"):
        pipeline.run_query("q1", "query")
