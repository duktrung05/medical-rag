import numpy as np

from src.reranking.bge_reranker import BGEReranker
from src.pipeline import RetrievalPipeline
from src.data.loader import DocumentChunkMap


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


def test_pipeline_preserves_unreranked_candidates_and_raw_scores():
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
        doc_threshold=0, doc_delta=2, min_k=0, max_chunk_k=5, max_doc_k=5,
    )
    result = pipeline.run_query("q1", "query")
    assert result.relevant_chunks == ["c1", "c2", "c3"]
    assert pipeline.last_candidate_scores == {
        "c1": {"raw_retrieval_score": 0.8, "rerank_score": 0.95, "provenance": {"sources": ()}},
        "c2": {"raw_retrieval_score": 0.7, "rerank_score": None, "provenance": {"sources": ()}},
        "c3": {"raw_retrieval_score": 0.6, "rerank_score": None, "provenance": {"sources": ()}},
    }
