from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.medrag.core.contracts import Hit, Passage, Source
from src.medrag.core.passage_first import run_b0, shared_pair_plan
from src.medrag.evidence.selector import SelectionPolicy


class StaticRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, limit):
        return self.hits[:limit]


class StaticReranker:
    def score(self, query, payload):
        return [3.0 if "strong" in text else 1.0 for text in payload]


class PassageFirstTest(unittest.TestCase):
    def test_scored_passage_dedup_and_parent_export(self):
        source1 = Source("v", 1, "u1", "Doc1", "weak strong", "x", "raw")
        source2 = Source("v", 2, "u2", "Doc2", "other", "x", "raw")
        passages = {
            "p1": Passage("p1", "v", 1, "u1", "weak", 0, 4, source1.text_sha256, "en", "x"),
            "p2": Passage("p2", "v", 1, "u1", "strong", 5, 11, source1.text_sha256, "en", "x"),
            "p3": Passage("p3", "v", 2, "u2", "other", 0, 5, source2.text_sha256, "en", "x"),
        }
        bm25 = [Hit(1, "p1", 2.0, "passage_bm25", 1),
                Hit(2, "p3", 1.0, "passage_bm25", 2)]
        dense = [Hit(1, "p2", 2.0, "dense", 1),
                 Hit(1, "p1", 1.0, "dense", 2)]
        pipeline = SimpleNamespace(
            passage_bm25=StaticRetriever(bm25), dense=StaticRetriever(dense),
            passage_candidate_limit=3, rrf_k=60, fused_candidate_limit=3,
            reranker=StaticReranker(), score_cache=None, sources={1: source1, 2: source2},
            passages=passages, selection_policy=SelectionPolicy(
                max_docs=200, max_chunks=60, chunks_per_doc=1,
                document_selection="independent"))
        selected, diagnostics = run_b0(pipeline, "test", pair_budget=3)
        self.assertEqual(diagnostics["rerank_pairs"], 3)
        self.assertEqual(diagnostics["scored_parents"], 2)
        self.assertEqual(diagnostics["selected_chunks"], 2)
        self.assertEqual([row["official_id"] for row in selected], [1, 2])
        self.assertEqual(selected[0]["evidence"][0]["passage_id"], "p2")
        self.assertEqual(selected[0]["document_evidence"]["passage_id"], "p2")

    def test_shared_budget_counts_unique_pairs(self):
        a = [(1, "p1"), (2, "p2"), (3, "p3")]
        b = [(2, "p2"), (4, "p4"), (5, "p5")]
        plan = shared_pair_plan(a, b, budget=4, a_slots=2)
        self.assertEqual(plan["total_inference_pairs"], 4)
        self.assertEqual(len(set(plan["unique_pairs"])), 4)
        self.assertEqual(plan["shared_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
