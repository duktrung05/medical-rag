"""End-to-end retrieval and selection pipeline."""

from typing import Dict, List, Optional
from src.data.loader import DocumentChunkMap
from src.data.preprocess import normalize_text
from src.data.schema import PredictionRecord, QueryRecord
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
        doc_aggregation: str = "max",
        chunk_threshold: float = 0.50,
        chunk_delta: float = 0.30,
        doc_threshold: float = 0.50,
        doc_delta: float = 0.30,
        min_k: int = 1,
        max_chunk_k: int = 20,
        max_doc_k: int = 20,
    ):
        self.retriever = retriever
        self.doc_map = doc_map
        self.reranker = reranker
        self.chunk_text_lookup = chunk_text_lookup or {}
        self.doc_aggregation = doc_aggregation
        self.chunk_threshold = chunk_threshold
        self.chunk_delta = chunk_delta
        self.doc_threshold = doc_threshold
        self.doc_delta = doc_delta
        self.min_k = min_k
        self.max_chunk_k = max_chunk_k
        self.max_doc_k = max_doc_k

    def run_query(self, query_record: QueryRecord) -> PredictionRecord:
        """Executes full pipeline for a single query."""
        # 1. Normalize
        norm_query = normalize_text(query_record.query, lowercase=True)

        # 2. Retrieve candidates
        chunk_candidates = self.retriever.search(norm_query, top_k=self.max_chunk_k * 5)

        # 3. Rerank if enabled
        if self.reranker and self.chunk_text_lookup:
            cand_pairs = [
                (cid, self.chunk_text_lookup.get(cid, ""))
                for cid, _ in chunk_candidates
            ]
            chunk_candidates = self.reranker.rerank(norm_query, cand_pairs)

        # 4. Chunk selection
        selected_chunks = select_candidates(
            chunk_candidates,
            threshold=self.chunk_threshold,
            relative_delta=self.chunk_delta,
            min_k=self.min_k,
            max_k=self.max_chunk_k,
        )

        # 5. Document aggregation & selection
        chunk_scores_dict = dict(chunk_candidates)
        doc_scores = aggregate_doc_scores(
            chunk_scores=chunk_scores_dict,
            chunk_to_doc=self.doc_map.chunk_to_doc,
            method="max" if self.doc_aggregation == "max" else "hybrid_mean",
        )
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        selected_docs = select_candidates(
            ranked_docs,
            threshold=self.doc_threshold,
            relative_delta=self.doc_delta,
            min_k=self.min_k,
            max_k=self.max_doc_k,
        )

        # 6. Consistency check
        prediction = PredictionRecord(
            id=query_record.id,
            relevant_docs=selected_docs,
            relevant_chunks=selected_chunks,
        )
        return enforce_parent_consistency(prediction, self.doc_map)

    def run_batch(self, queries: List[QueryRecord]) -> List[PredictionRecord]:
        """Runs pipeline across all queries in batch."""
        return [self.run_query(q) for q in queries]
