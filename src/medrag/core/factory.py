"""Build one V4 workflow from the active, versioned configuration."""
from __future__ import annotations

from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.evidence.selector import SelectionPolicy


def build_pipeline(sources: list[Source], passages: list[Passage], config: dict) -> Pipeline:
    dense_config = config.get("dense", {})
    reranker_config = config.get("reranker", {})
    retrieval = config.get("retrieval", {})
    selection = config.get("selection", {})
    encoder = None
    reranker = None
    if dense_config.get("enabled", False):
        from src.medrag.retrieval.bge_encoder import BGEM3Encoder

        encoder = BGEM3Encoder(
            dense_config.get("model", "BAAI/bge-m3"),
            revision=str(dense_config["revision"]),
            document_max_length=int(dense_config.get("document_max_length", 512)),
            query_max_length=int(dense_config.get("query_max_length", 512)),
            device=dense_config.get("device"),
        )
    if reranker_config.get("enabled", False):
        from src.medrag.ranking.reranker import BGEReranker

        reranker = BGEReranker(
            reranker_config.get("model", "BAAI/bge-reranker-v2-m3"),
            revision=str(reranker_config["revision"]),
            max_length=int(reranker_config.get("max_length", 512)),
            device=reranker_config.get("device"),
        )
    return Pipeline(
        sources,
        passages,
        encoder,
        reranker,
        document_candidate_limit=int(retrieval.get("document_candidate_limit", 400)),
        passage_candidate_limit=int(retrieval.get("passage_candidate_limit", 200)),
        fused_candidate_limit=int(retrieval.get("fused_candidate_limit", 400)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
        rerank_candidate_limit=int(retrieval.get("rerank_candidate_limit", 200)),
        rerank_passages_per_document=int(retrieval.get("rerank_passages_per_document", 3)),
        rerank_allocation=str(retrieval.get("rerank_allocation", "document_order")),
        selection_policy=SelectionPolicy(
            max_docs=int(selection.get("max_docs", 200)),
            max_chunks=int(selection.get("max_chunks", 60)),
            chunks_per_doc=int(selection.get("chunks_per_doc", 1)),
            document_selection=str(selection.get("document_selection", "with_chunks")),
            chunk_context=str(selection.get("chunk_context", "passage")),
            context_window=int(selection.get("context_window", 1)),
            context_max_characters=int(selection.get("context_max_characters", 4096)),
        ),
    )
