"""Submission builder with automated parent consistency enforcement."""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from src.data.loader import DocumentChunkMap, save_jsonl_records
from src.data.schema import PredictionRecord


def enforce_parent_consistency(
    prediction: PredictionRecord,
    doc_map: DocumentChunkMap,
) -> PredictionRecord:
    """Ensures that every chunk in relevant_chunks has its parent doc_id in relevant_docs.

    If missing, appends the parent doc_id while preserving order and uniqueness.
    """
    existing_docs = list(prediction.relevant_docs)
    existing_docs_set = set(existing_docs)

    for chunk_id in prediction.relevant_chunks:
        parent_doc = doc_map.get_parent_doc(chunk_id)
        if parent_doc and parent_doc not in existing_docs_set:
            existing_docs.append(parent_doc)
            existing_docs_set.add(parent_doc)

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
) -> Path:
    """Builds and writes a competition submission JSONL file.

    Optionally runs parent consistency enforcement if doc_map is provided.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    final_records: List[PredictionRecord] = []
    for pred in predictions:
        if auto_fix_parent and doc_map is not None:
            fixed_pred = enforce_parent_consistency(pred, doc_map)
            final_records.append(fixed_pred)
        else:
            final_records.append(pred)

    save_jsonl_records(out_p, final_records)
    return out_p
