"""Shared final chunk/document selection used by runtime and offline replay."""

from src.data.loader import DocumentChunkMap
from src.data.schema import PredictionRecord
from src.scoring.document_score import aggregate_doc_scores
from src.selection.threshold import select_candidates


def select_prediction(
    query_id: str,
    ranked_chunks: list[tuple[str, float]],
    doc_map: DocumentChunkMap,
    *,
    doc_aggregation: str = "max",
    chunk_threshold: float = 0.5,
    chunk_delta: float = 0.3,
    doc_threshold: float = 0.5,
    doc_delta: float = 0.3,
    chunk_min_k: int = 1,
    chunk_max_k: int = 20,
    doc_min_k: int = 1,
    doc_max_k: int = 20,
    doc_top_n_mean: int = 3,
) -> tuple[PredictionRecord, dict]:
    """Select candidates with a hard document cap and parent consistency.

    Chunk order controls which parents receive capacity first. Parents of selected
    chunks displace lower ranked standalone document selections when the cap fills.
    """
    selected_chunks = select_candidates(
        ranked_chunks, threshold=chunk_threshold, relative_delta=chunk_delta,
        min_k=chunk_min_k, max_k=chunk_max_k,
    )
    doc_scores = aggregate_doc_scores(
        dict(ranked_chunks), doc_map.chunk_to_doc,
        method=doc_aggregation, top_n=doc_top_n_mean,
    )
    ranked_docs = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
    selected_docs = select_candidates(
        ranked_docs, threshold=doc_threshold, relative_delta=doc_delta,
        min_k=doc_min_k, max_k=doc_max_k,
    )

    output_docs = list(selected_docs[:doc_max_k]) if doc_max_k > 0 else []
    added_parents: list[str] = []
    dropped_chunks: list[dict[str, str]] = []
    kept_chunks: list[str] = []
    selected_chunk_parents: set[str] = set()
    for chunk_id in selected_chunks:
        parent = doc_map.get_parent_doc(chunk_id)
        if parent and parent not in output_docs:
            if len(output_docs) < doc_max_k:
                output_docs.append(parent)
                added_parents.append(parent)
            else:
                replace = next((doc_id for doc_id in reversed(output_docs)
                                if doc_id not in selected_chunk_parents), None)
                if replace is None:
                    dropped_chunks.append({"chunk_id": chunk_id, "reason": "parent_document_cap"})
                    continue
                output_docs.remove(replace)
                output_docs.append(parent)
                added_parents.append(parent)
        kept_chunks.append(chunk_id)
        if parent:
            selected_chunk_parents.add(parent)

    # Enforce the hard cap even when document selection has unusual inputs.
    output_docs = output_docs[:doc_max_k]
    top_score = ranked_chunks[0][1] if ranked_chunks else None
    chunk_cutoff = top_score - chunk_delta if top_score is not None else None
    eligible_chunk_count = sum(
        score >= chunk_threshold and score >= chunk_cutoff
        for _, score in ranked_chunks
    ) if chunk_cutoff is not None else 0
    required_chunk_count = min(chunk_min_k, chunk_max_k, len(ranked_chunks))
    diagnostic = {
        "top_score": top_score,
        "chunk_threshold": chunk_threshold,
        "chunk_relative_cutoff": chunk_cutoff,
        "doc_threshold": doc_threshold,
        "doc_relative_cutoff": (ranked_docs[0][1] - doc_delta) if ranked_docs else None,
        "candidate_count": len(ranked_chunks),
        "chunk_candidates_passing_thresholds": eligible_chunk_count,
        "selected_chunk_count_before_parent_cap": len(selected_chunks),
        "selected_chunk_count": len(kept_chunks),
        "selected_doc_count_before_parent_consistency": len(selected_docs),
        "selected_doc_count": len(output_docs),
        "parent_docs_added": added_parents,
        "chunks_dropped_for_parent_cap": dropped_chunks,
        "chunk_min_k_fallback_used": eligible_chunk_count < required_chunk_count,
    }
    return PredictionRecord(id=query_id, relevant_docs=output_docs, relevant_chunks=kept_chunks), diagnostic
