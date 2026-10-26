from __future__ import annotations

import math

from src.medrag.core.contracts import Passage, Source
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.core import Dense, Encoder, build_lexical, fuse, fuse_with_parent_floor


class Pipeline:
    def __init__(self, sources: list[Source], passages: list[Passage], encoder: Encoder | None = None,
                 reranker=None, *, document_candidate_limit: int = 400,
                 passage_candidate_limit: int = 200, fused_candidate_limit: int = 400,
                 fused_parent_limit: int | None = None,
                 rrf_k: int = 60, rerank_candidate_limit: int = 200,
                 rerank_passages_per_document: int = 3,
                 rerank_allocation: str = "document_order",
                 selection_policy: SelectionPolicy | None = None, score_cache=None):
        versions = {s.corpus_version for s in sources}
        if len(versions) != 1 or any(p.corpus_version not in versions for p in passages):
            raise ValueError("A V4 index must use exactly one corpus version")
        self.corpus_version = next(iter(versions))
        self.sources = {s.official_id: s for s in sources}
        self.passages = {p.passage_id: p for p in passages}
        if len(self.sources) != len(sources) or len(self.passages) != len(passages):
            raise ValueError("Duplicate source or passage identity")
        if any(not verify(p, self.sources[p.official_id]) for p in passages):
            raise ValueError("Passage is not exact source text")
        if min(document_candidate_limit, passage_candidate_limit, fused_candidate_limit,
               rerank_candidate_limit, rerank_passages_per_document, rrf_k) <= 0:
            raise ValueError("V4 retrieval and reranking budgets must be positive")
        if rerank_allocation not in {"document_order", "parent_first"}:
            raise ValueError("Unsupported rerank allocation")
        self.doc_bm25, self.passage_bm25 = build_lexical(sources, passages)
        self.dense = Dense(passages, encoder, {s.official_id: s.title for s in sources}) if encoder else None
        self.reranker = reranker
        self.document_candidate_limit = document_candidate_limit
        self.passage_candidate_limit = passage_candidate_limit
        self.fused_candidate_limit = fused_candidate_limit
        self.fused_parent_limit = fused_parent_limit
        self.rrf_k = rrf_k
        self.rerank_candidate_limit = rerank_candidate_limit
        self.rerank_passages_per_document = rerank_passages_per_document
        self.rerank_allocation = rerank_allocation
        self.selection_policy = selection_policy or SelectionPolicy()
        self.score_cache = score_cache

    def retrieve(self, query: str, limit: int = 10) -> list[dict]:
        return self.retrieve_with_diagnostics(query, limit)[0]

    def retrieve_with_diagnostics(self, query: str, limit: int = 10, *,
                                  include_candidates: bool = False) -> tuple[list[dict], dict]:
        results, pairs, diagnostics = self.prepare_rerank(query, limit)
        if pairs:
            if self.score_cache is not None:
                values = self.score_cache.score(query, pairs, self.reranker)
            else:
                values = self.reranker.score(query, [row[2] for row in pairs])
        else:
            values = []
        return self.finish_rerank(results, pairs, values, limit, diagnostics,
                                  include_candidates=include_candidates)

    def prepare_rerank(self, query: str, limit: int = 10) -> tuple[list[dict], list[tuple[int, str, str]], dict]:
        """Run the frozen retrieval/fusion path and construct exact reranker inputs."""
        if limit <= 0:
            raise ValueError("Result limit must be positive")
        rankings = [self.doc_bm25.search(query, self.document_candidate_limit),
                    self.passage_bm25.search(query, self.passage_candidate_limit)]
        if self.dense:
            rankings.append(self.dense.search(query, self.passage_candidate_limit))
        parent_limit = getattr(self, "fused_parent_limit", None)
        fused = (fuse(rankings, k=self.rrf_k, limit=self.fused_candidate_limit)
                 if parent_limit is None else
                 fuse_with_parent_floor(rankings, k=self.rrf_k,
                                        item_limit=self.fused_candidate_limit,
                                        parent_limit=parent_limit))
        by_doc: dict[int, dict] = {}
        for item in fused:
            doc = by_doc.setdefault(item.official_id, {"official_id": item.official_id, "score": 0.0,
                                                      "ranks": {}, "evidence": [], "_doc_score": 0.0,
                                                      "_passage_score": 0.0})
            if item.passage_id is None:
                doc["_doc_score"] = max(doc["_doc_score"], item.score)
            else:
                doc["_passage_score"] = max(doc["_passage_score"], item.score)
            for channel, rank in item.ranks.items():
                doc["ranks"][channel] = min(rank, doc["ranks"].get(channel, rank))
            if item.passage_id:
                passage = self.passages[item.passage_id]
                if verify(passage, self.sources[item.official_id]):
                    doc["evidence"].append({"passage_id": passage.passage_id, "text": passage.text,
                                            "start": passage.source_start, "end": passage.source_end,
                                            "score_status": "retrieval_only"})
        # A document hit must obtain its evidence from its own passage lookup.
        for doc in by_doc.values():
            doc["score"] = doc.pop("_doc_score") + doc.pop("_passage_score")
        if any(not doc["evidence"] for doc in by_doc.values()):
            for doc in by_doc.values():
                if doc["evidence"]:
                    continue
                candidates = self.passage_bm25.search_parent(query, doc["official_id"], 1)
                if not candidates:
                    continue
                candidate = candidates[0]
                passage = self.passages[candidate.passage_id]
                doc["evidence"].append({"passage_id": passage.passage_id, "text": passage.text,
                                        "start": passage.source_start, "end": passage.source_end,
                                        "score_status": "retrieval_only"})
        results = sorted(by_doc.values(), key=lambda d: (-d["score"], d["official_id"]))
        pairs: list[tuple[int, str, str]] = []
        token_diagnostics = None
        if self.reranker:
            allocated = []
            if self.rerank_allocation == "parent_first":
                ordered = ((doc, doc["evidence"][depth]) for depth in
                           range(self.rerank_passages_per_document) for doc in results
                           if len(doc["evidence"]) > depth)
            else:
                ordered = ((doc, evidence) for doc in results for evidence in
                           doc["evidence"][:self.rerank_passages_per_document])
            for doc, evidence in ordered:
                if len(allocated) >= self.rerank_candidate_limit:
                    break
                allocated.append((doc, evidence))
            titles = {doc["official_id"]: self.sources[doc["official_id"]].title for doc in results}
            payload = [f"{titles[doc['official_id']]}\n{evidence['text']}"
                       if titles.get(doc["official_id"]) else evidence["text"]
                       for doc, evidence in allocated]
            pairs = [(doc["official_id"], evidence["passage_id"], value)
                     for (doc, evidence), value in zip(allocated, payload, strict=True)]
            if hasattr(self.reranker, "pair_token_lengths"):
                lengths = self.reranker.pair_token_lengths(query, payload)
                if len(lengths) != len(allocated):
                    raise ValueError("Reranker tokenizer returned a different number of pairs")
                maximum = self.reranker.max_length
                token_diagnostics = {
                    "max_length": maximum,
                    "pair_lengths": lengths,
                    "truncated_pairs": sum(length > maximum for length in lengths),
                    "maximum_observed_length": max(lengths, default=0),
                }
        diagnostics = {
            "document_hits": len(rankings[0]),
            "passage_bm25_hits": len(rankings[1]),
            "dense_passage_hits": len(rankings[2]) if self.dense else 0,
            "fused_items": len(fused),
            "fused_unique_parents": len(by_doc),
            "rerank_pairs": len(pairs),
            "rerank_allocation": self.rerank_allocation,
            "reranker_tokens": token_diagnostics,
        }
        return results, pairs, diagnostics

    def finish_rerank(self, results: list[dict], pairs: list[tuple[int, str, str]],
                      values: list[float], limit: int, diagnostics: dict, *,
                      include_candidates: bool = False) -> tuple[list[dict], dict]:
        """Apply imported scores with the same ordering, selection and context path."""
        if len(values) != len(pairs):
            raise ValueError("Reranker returned a different number of scores")
        if self.reranker:
            by_id = {doc["official_id"]: doc for doc in results}
            for (doc_id, passage_id, _), value in zip(pairs, values, strict=True):
                doc = by_id[doc_id]
                evidence = next(item for item in doc["evidence"] if item["passage_id"] == passage_id)
                score = float(value)
                if not math.isfinite(score):
                    raise ValueError("Reranker returned a non-finite score")
                evidence["rerank_score"] = score
                evidence["score_status"] = "reranked"
                doc["rerank_score"] = max(doc.get("rerank_score", float("-inf")), score)
            for doc in results:
                for evidence in doc["evidence"]:
                    if evidence["score_status"] == "retrieval_only":
                        evidence["score_status"] = "not_reranked_budget"
                doc["evidence"].sort(
                    key=lambda evidence: (-evidence.get("rerank_score", float("-inf")),
                                          evidence["passage_id"])
                )
            results.sort(key=lambda d: (-d.get("rerank_score", float("-inf")), -d["score"], d["official_id"]))
        selected, selection_diagnostics = select_results(
            results, self.selection_policy, reranker_enabled=self.reranker is not None, limit=limit)
        selected, context_diagnostics = apply_chunk_context(
            selected, self.selection_policy, self.sources, self.passages)
        diagnostics = {**diagnostics, "selection": selection_diagnostics,
                       "context": context_diagnostics}
        if include_candidates:
            diagnostics["ranked_candidates"] = results
        return selected, diagnostics

