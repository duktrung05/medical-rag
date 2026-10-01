"""Submission builder with automated parent consistency enforcement."""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from src.data.loader import DocumentChunkMap, save_jsonl_records
from src.data.schema import PredictionRecord


def enforce_parent_consistency(
    prediction: PredictionRecord,
    doc_map: DocumentChunkMap,
    *,
    max_doc_k: int | None = None,
    diagnostics: dict | None = None,
) -> PredictionRecord:
    """Ensures that every chunk in relevant_chunks has its parent doc_id in relevant_docs.

    If missing, appends the parent doc_id while preserving order and uniqueness.
    """
    existing_docs = list(prediction.relevant_docs)
    existing_docs_set = set(existing_docs)
    required_parents = list(dict.fromkeys(
        parent for chunk_id in prediction.relevant_chunks
        if (parent := doc_map.get_parent_doc(chunk_id)) is not None
    ))
    if max_doc_k is not None and max_doc_k < 0:
        raise ValueError("max_doc_k must be nonnegative")
    if max_doc_k is not None and len(required_parents) > max_doc_k:
        raise ValueError("Selected chunks have more unique parents than max_doc_k")

    added_parents = [parent for parent in required_parents if parent not in existing_docs_set]
    for parent_doc in added_parents:
        existing_docs.append(parent_doc)
        existing_docs_set.add(parent_doc)
    dropped_docs = []
    if max_doc_k is not None:
        required = set(required_parents)
        while len(existing_docs) > max_doc_k:
            removable = next((doc for doc in reversed(existing_docs) if doc not in required), None)
            if removable is None:
                raise ValueError("Cannot preserve parent consistency within max_doc_k")
            existing_docs.remove(removable)
            dropped_docs.append(removable)
    if diagnostics is not None:
        diagnostics.update(parent_docs_added=added_parents, documents_dropped_for_cap=dropped_docs,
                           doc_count_before=len(prediction.relevant_docs), doc_count_after=len(existing_docs))

    return PredictionRecord(
        id=prediction.id,
        relevant_docs=existing_docs,
        relevant_chunks=list(prediction.relevant_chunks),
    )


def build_submission_file(
    predictions: Iterable[PredictionRecord],
    output_path: Union[str, Path],
    doc_map: Optional[DocumentChunkMap] = None,
    auto_fix_parent: bool = True,
    max_doc_k: int | None = None,
    diagnostics: list[dict] | None = None,
) -> Path:
    """Builds and writes a competition submission JSONL file.

    Optionally runs parent consistency enforcement if doc_map is provided.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    final_records: List[PredictionRecord] = []
    for pred in predictions:
        if auto_fix_parent and doc_map is not None:
            detail = {}
            fixed_pred = enforce_parent_consistency(pred, doc_map, max_doc_k=max_doc_k, diagnostics=detail)
            if diagnostics is not None:
                diagnostics.append({"query_id": pred.id, **detail})
            final_records.append(fixed_pred)
        else:
            final_records.append(pred)

    save_jsonl_records(out_p, final_records)
    return out_p
