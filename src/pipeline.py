"""End-to-end retrieval and selection pipeline."""

from typing import Dict, List, Optional
from src.data.loader import DocumentChunkMap
from src.data.schema import PredictionRecord
from src.retrieval.bm25 import BaseRetriever
from src.reranking.base import BaseReranker
from src.scoring.document_score import aggregate_doc_scores
from src.selection.threshold import select_candidates
from src.submission.build import enforce_parent_consistency


class RetrievalPipeline:
    """Orchestrates retrieval, reranking, scoring, selection, and consistency validation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        doc_map: DocumentChunkMap,
        reranker: Optional[BaseReranker] = None,
        chunk_text_lookup: Optional[Dict[str, str]] = None,
        chunk_context_lookup: Optional[Dict[str, tuple[str | None, str | None]]] = None,
        doc_aggregation: str = "max",
        chunk_threshold: float = 0.50,
        chunk_delta: float = 0.30,
        doc_threshold: float = 0.50,
        doc_delta: float = 0.30,
        min_k: int = 1,
        max_chunk_k: int = 20,
        max_doc_k: int = 20,
        retrieval_top_k: Optional[int] = None,
        reranker_top_k: int = 100,
        chunk_min_k: Optional[int] = None,
        doc_min_k: Optional[int] = None,
        doc_top_n_mean: int = 3,
    ):
        self.retriever = retriever
        self.doc_map = doc_map
        self.reranker = reranker
        self.chunk_text_lookup = chunk_text_lookup or {}
        self.chunk_context_lookup = chunk_context_lookup or {}
        self.doc_aggregation = doc_aggregation
        self.chunk_threshold = chunk_threshold
        self.chunk_delta = chunk_delta
        self.doc_threshold = doc_threshold
        self.doc_delta = doc_delta
        self.min_k = min_k
        self.max_chunk_k = max_chunk_k
        self.max_doc_k = max_doc_k
        self.retrieval_top_k = retrieval_top_k or max_chunk_k * 5
        self.reranker_top_k = reranker_top_k
        self.chunk_min_k = min_k if chunk_min_k is None else chunk_min_k
        self.doc_min_k = min_k if doc_min_k is None else doc_min_k
        self.doc_top_n_mean = doc_top_n_mean

    def run_query(self, query_id: str, normalized_query: str) -> PredictionRecord:
        """Execute the retrieval core for a normalized query and external ID."""
        # 2. Retrieve candidates
        if hasattr(self.retriever, "search_with_provenance"):
            fused = self.retriever.search_with_provenance(normalized_query, top_k=self.retrieval_top_k)
            chunk_candidates = [(item.chunk_id, item.score) for item in fused]
            provenance = {item.chunk_id: {"bm25_rank": item.sparse_rank, "dense_rank": item.dense_rank,
                                          "sources": item.sources} for item in fused}
        else:
            chunk_candidates = self.retriever.search(normalized_query, top_k=self.retrieval_top_k)
            provenance = {cid: {"sources": ()} for cid, _ in chunk_candidates}

        # 3. Rerank if enabled
        if self.reranker and self.chunk_text_lookup:
            raw_scores = dict(chunk_candidates)
            head, tail = chunk_candidates[:self.reranker_top_k], chunk_candidates[self.reranker_top_k:]
            cand_pairs = []
            for cid, _ in head:
                title, context = self.chunk_context_lookup.get(cid, (None, None))
                text = self.chunk_text_lookup.get(cid, "")
                cand_pairs.append((cid, text, title, context) if title or context else (cid, text))
            reranked = self.reranker.rerank(normalized_query, cand_pairs, top_k=len(head))
            rerank_scores = dict(reranked)
            reranked_ids = set(rerank_scores)
            untouched = [(cid, score) for cid, score in chunk_candidates if cid not in reranked_ids]
            chunk_candidates = reranked + untouched
            self.last_candidate_scores = {
                cid: {"raw_retrieval_score": score, "rerank_score": rerank_scores.get(cid),
                      "provenance": provenance.get(cid, {"sources": ()})}
                for cid, score in raw_scores.items()
            }
        else:
            self.last_candidate_scores = {
                cid: {"raw_retrieval_score": score, "rerank_score": None,
                      "provenance": provenance.get(cid, {"sources": ()})}
                for cid, score in chunk_candidates
            }

        # 4. Chunk selection
        selected_chunks = select_candidates(
            chunk_candidates,
            threshold=self.chunk_threshold,
            relative_delta=self.chunk_delta,
            min_k=self.chunk_min_k,
            max_k=self.max_chunk_k,
        )

        # 5. Document aggregation & selection
        chunk_scores_dict = dict(chunk_candidates)
        doc_scores = aggregate_doc_scores(
            chunk_scores=chunk_scores_dict,
            chunk_to_doc=self.doc_map.chunk_to_doc,
            method=self.doc_aggregation,
            top_n=self.doc_top_n_mean,
        )
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        selected_docs = select_candidates(
            ranked_docs,
            threshold=self.doc_threshold,
            relative_delta=self.doc_delta,
            min_k=self.doc_min_k,
            max_k=self.max_doc_k,
        )

        # 6. Consistency check
        prediction = PredictionRecord(
            id=query_id,
            relevant_docs=selected_docs,
            relevant_chunks=selected_chunks,
        )
        return enforce_parent_consistency(prediction, self.doc_map)

    def run_batch(self, queries: List[tuple[str, str]]) -> List[PredictionRecord]:
        """Run the core over (query_id, normalized_text) pairs."""
        return [self.run_query(query_id, text) for query_id, text in queries]
