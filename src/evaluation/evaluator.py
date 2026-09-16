"""Macro evaluator computing document-level and chunk-level Precision, Recall, and F2."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
import numpy as np

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
    ) -> EvaluationSummary:
        """Evaluates a batch of predictions against ground truth annotations."""
        if isinstance(ground_truth, list):
            gt_map = {gt.id: gt for gt in ground_truth}
        else:
            gt_map = ground_truth

        if isinstance(predictions, list):
            pred_map = {pred.id: pred for pred in predictions}
        else:
            pred_map = predictions

        if not gt_map:
            raise ValueError("Ground truth dataset cannot be empty.")

        doc_precisions: List[float] = []
        doc_recalls: List[float] = []
        doc_f2s: List[float] = []

        chunk_precisions: List[float] = []
        chunk_recalls: List[float] = []
        chunk_f2s: List[float] = []

        for qid, gt in gt_map.items():
            pred = pred_map.get(qid)
            pred_docs = pred.relevant_docs if pred else []
            pred_chunks = pred.relevant_chunks if pred else []

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
