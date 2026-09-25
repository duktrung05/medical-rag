"""Shared composition root for batch commands and HTTP serving."""

from src.config import PipelineConfig, RuntimeConfig
from src.data.loader import DataLoader, DocumentChunkMap
from src.data.validator import validate_chunks_integrity
from src.pipeline import RetrievalPipeline
from src.retrieval.demo import DemoRetriever
from src.retrieval.factory import create_retriever
from src.reranking.factory import create_reranker
from src.indexing.sparse_index import corpus_sha256


def build_pipeline(config: RuntimeConfig | PipelineConfig) -> RetrievalPipeline:
    chunks = DataLoader.load_chunks(config.corpus)
    errors = validate_chunks_integrity(chunks)
    if not chunks:
        errors.append("Corpus must contain at least one chunk.")
    if errors:
        raise ValueError("Invalid corpus: " + "; ".join(errors))
    if isinstance(config, RuntimeConfig):
        return RetrievalPipeline(
            retriever=DemoRetriever(chunks),
            doc_map=DocumentChunkMap.from_chunks(chunks),
            chunk_text_lookup={chunk.chunk_id: chunk.text for chunk in chunks},
            chunk_threshold=0,
            doc_threshold=0,
            chunk_delta=1,
            doc_delta=1,
            min_k=0,
            max_chunk_k=config.max_chunks,
            max_doc_k=config.max_chunks,
        )

    retriever, retrieval_top_k = create_retriever(
        config.backend,
        config.retrieval,
        config.fusion,
        expected_corpus_hash=corpus_sha256(chunks),
    )
    reranker = create_reranker(config.reranker)
    chunk_selection = config.selection.chunk
    doc_selection = config.selection.document
    return RetrievalPipeline(
        retriever=retriever,
        doc_map=DocumentChunkMap.from_chunks(chunks),
        reranker=reranker,
        chunk_text_lookup={chunk.chunk_id: chunk.text for chunk in chunks},
        chunk_context_lookup={chunk.chunk_id: (chunk.title, chunk.context) for chunk in chunks},
        doc_aggregation=config.scoring.document.aggregation,
        chunk_threshold=chunk_selection.threshold,
        chunk_delta=chunk_selection.relative_delta,
        doc_threshold=doc_selection.threshold,
        doc_delta=doc_selection.relative_delta,
        chunk_min_k=chunk_selection.min_k,
        doc_min_k=doc_selection.min_k,
        max_chunk_k=chunk_selection.max_k,
        max_doc_k=doc_selection.max_k,
        retrieval_top_k=retrieval_top_k,
        reranker_top_k=config.reranker.top_k,
        doc_top_n_mean=config.scoring.document.top_n_mean,
    )
