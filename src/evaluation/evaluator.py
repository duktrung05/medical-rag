"""Macro evaluator computing document-level and chunk-level Precision, Recall, and F2."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Union

import numpy as np

from src.data.loader import DocumentChunkMap
from src.data.schema import GroundTruthRecord, PredictionRecord
from src.evaluation.fbeta import calculate_query_metrics


@dataclass
class QueryMetricDetail:
    id: str
    doc_precision: float
    doc_recall: float
    doc_f2: float
    chunk_precision: float
    chunk_recall: float
    chunk_f2: float


@dataclass
class EvaluationSummary:
    num_queries: int
    doc_precision: float
    doc_recall: float
    doc_f2: float
    chunk_precision: float
    chunk_recall: float
    chunk_f2: float
    macro_f2: float  # (doc_f2 + chunk_f2) / 2

    def to_dict(self) -> dict:
        return asdict(self)

    def format_table(self, title: str = "EVALUATION RESULTS") -> str:
        lines = [
            f"=== {title} ===",
            f"Total Queries Evaluated: {self.num_queries}",
            "----------------------------------------------------------------",
            f"{'Metric':<18} | {'Document-Level':<18} | {'Chunk-Level':<18}",
            "----------------------------------------------------------------",
            f"{'Precision':<18} | {self.doc_precision:<18.4f} | {self.chunk_precision:<18.4f}",
            f"{'Recall':<18} | {self.doc_recall:<18.4f} | {self.chunk_recall:<18.4f}",
            f"{'Macro F2':<18} | {self.doc_f2:<18.4f} | {self.chunk_f2:<18.4f}",
            "----------------------------------------------------------------",
            f"Composite Macro F2 (Doc & Chunk Mean): {self.macro_f2:.4f}",
            "================================================================",
        ]
        return "\n".join(lines)


class Evaluator:
    """Evaluates IR predictions against ground truth using competition Macro P, R, F2 rules."""

    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def evaluate(
        self,
        ground_truth: Union[List[GroundTruthRecord], Dict[str, GroundTruthRecord]],
        predictions: Union[List[PredictionRecord], Dict[str, PredictionRecord]],
        *,
        doc_map: DocumentChunkMap | None = None,
    ) -> EvaluationSummary:
        """Validate query sets before scoring; check corpus references when a map is supplied.

        Submission files should also pass the submission validator, which checks
        their raw records and expected query file before this metric gate.
        """
        gt_map = _index_records(ground_truth, "Ground truth")
        pred_map = _index_records(predictions, "Prediction")

        if not gt_map:
            raise ValueError("Ground truth dataset cannot be empty.")
        missing = sorted(gt_map.keys() - pred_map.keys())
        extra = sorted(pred_map.keys() - gt_map.keys())
        if missing or extra:
            raise ValueError(f"Prediction query IDs do not match ground truth: missing={missing}, extra={extra}")

        for record in pred_map.values():
            for label, ids in (("document", record.relevant_docs), ("chunk", record.relevant_chunks)):
                seen = set()
                for item in ids:
                    if item in seen:
                        raise ValueError(f"Prediction {record.id} has duplicate {label} IDs: {item}")
                    seen.add(item)
        if doc_map is not None:
            known_docs = doc_map.all_doc_ids | set(doc_map.chunk_to_doc.values())
            known_chunks = doc_map.all_chunk_ids
            for record in (*gt_map.values(), *pred_map.values()):
                _validate_references(record, doc_map, known_docs, known_chunks)

        doc_precisions: List[float] = []
        doc_recalls: List[float] = []
        doc_f2s: List[float] = []

        chunk_precisions: List[float] = []
        chunk_recalls: List[float] = []
        chunk_f2s: List[float] = []

        for qid, gt in gt_map.items():
            pred = pred_map[qid]
            pred_docs = pred.relevant_docs
            pred_chunks = pred.relevant_chunks

            # Document metrics
            doc_metrics = calculate_query_metrics(
                actual=gt.relevant_docs,
                predicted=pred_docs,
                beta=self.beta,
            )
            doc_precisions.append(doc_metrics["precision"])
            doc_recalls.append(doc_metrics["recall"])
            doc_f2s.append(doc_metrics["fbeta"])

            # Chunk metrics
            chunk_metrics = calculate_query_metrics(
                actual=gt.relevant_chunks,
                predicted=pred_chunks,
                beta=self.beta,
            )
            chunk_precisions.append(chunk_metrics["precision"])
            chunk_recalls.append(chunk_metrics["recall"])
            chunk_f2s.append(chunk_metrics["fbeta"])

        num_q = len(gt_map)
        mean_doc_p = float(np.mean(doc_precisions)) if doc_precisions else 0.0
        mean_doc_r = float(np.mean(doc_recalls)) if doc_recalls else 0.0
        mean_doc_f2 = float(np.mean(doc_f2s)) if doc_f2s else 0.0

        mean_chunk_p = float(np.mean(chunk_precisions)) if chunk_precisions else 0.0
        mean_chunk_r = float(np.mean(chunk_recalls)) if chunk_recalls else 0.0
        mean_chunk_f2 = float(np.mean(chunk_f2s)) if chunk_f2s else 0.0

        composite_f2 = float((mean_doc_f2 + mean_chunk_f2) / 2.0)

        return EvaluationSummary(
            num_queries=num_q,
            doc_precision=mean_doc_p,
            doc_recall=mean_doc_r,
            doc_f2=mean_doc_f2,
            chunk_precision=mean_chunk_p,
            chunk_recall=mean_chunk_r,
            chunk_f2=mean_chunk_f2,
            macro_f2=composite_f2,
        )


def _index_records(records: list | dict, label: str) -> dict:
    if isinstance(records, dict):
        for key, record in records.items():
            if key != record.id:
                raise ValueError(f"{label} mapping key {key!r} does not match record ID {record.id!r}")
        return records
    indexed = {}
    for record in records:
        if record.id in indexed:
            raise ValueError(f"{label} has duplicate query ID: {record.id}")
        indexed[record.id] = record
    return indexed


def _validate_references(
    record: GroundTruthRecord | PredictionRecord,
    doc_map: DocumentChunkMap,
    known_docs: set[str],
    known_chunks: set[str],
) -> None:
    document_ids = set(record.relevant_docs)
    for doc_id in sorted(document_ids):
        if doc_id not in known_docs:
            raise ValueError(f"Query {record.id} references unknown document ID: {doc_id}")
    for chunk_id in record.relevant_chunks:
        if chunk_id not in known_chunks:
            raise ValueError(f"Query {record.id} references unknown chunk ID: {chunk_id}")
        parent = doc_map.get_parent_doc(chunk_id)
        if parent not in document_ids:
            raise ValueError(
                f"Query {record.id} has parent inconsistency: chunk {chunk_id} belongs to {parent}"
            )
