from src.retrieval.hybrid import HybridRetriever


class FixtureRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_depths = []

    def search(self, query, top_k=200):
        self.requested_depths.append(top_k)
        return self.results[:top_k]


def test_hybrid_uses_independent_depths_fusion_depth_and_reports_positive_sources():
    sparse = FixtureRetriever([("s", 4), ("both", 3)])
    dense = FixtureRetriever([("d", 0.9), ("both", 0.8)])
    hybrid = HybridRetriever(sparse, dense, rrf_k=1, sparse_top_k=2, dense_top_k=2)

    ranked = hybrid.search_with_provenance("q", top_k=2)
    assert [item.chunk_id for item in ranked] == ["both", "d"]
    assert sparse.requested_depths == [2]
    assert dense.requested_depths == [2]
    assert [(item.sparse_rank, item.dense_rank) for item in ranked] == [(2, 2), (None, 1)]

    assert hybrid.positive_coverage("q", {"s", "d", "both", "missing"}) == {
        "bm25_only": ["s"],
        "dense_only": ["d"],
        "both": ["both"],
        "missed": ["missing"],
    }
