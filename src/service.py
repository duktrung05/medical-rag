"""Shared composition root for batch commands and HTTP serving."""

from src.config import RuntimeConfig
from src.data.loader import DataLoader, DocumentChunkMap
from src.data.validator import validate_chunks_integrity
from src.pipeline import RetrievalPipeline
from src.retrieval.demo import DemoRetriever


def build_pipeline(config: RuntimeConfig) -> RetrievalPipeline:
    chunks = DataLoader.load_chunks(config.corpus)
    errors = validate_chunks_integrity(chunks)
    if not chunks:
        errors.append("Corpus must contain at least one chunk.")
    if errors:
        raise ValueError("Invalid corpus: " + "; ".join(errors))
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
