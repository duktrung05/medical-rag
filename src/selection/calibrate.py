"""Grid search calibration engine for threshold and relative delta parameters."""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

from src.data.schema import GroundTruthRecord, PredictionRecord
from src.evaluation.evaluator import Evaluator
from src.selection.threshold import select_candidates


@dataclass
class CalibrationResult:
    chunk_threshold: float
    chunk_delta: float
    doc_threshold: float
    doc_delta: float
    doc_precision: float
    doc_recall: float
    doc_f2: float
    chunk_precision: float
    chunk_recall: float
    chunk_f2: float
    composite_f2: float


def grid_search_calibration(
    query_chunk_candidates: Dict[str, List[Tuple[str, float]]],
    query_doc_candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, GroundTruthRecord],
    chunk_thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6),
    chunk_deltas: Sequence[float] = (0.1, 0.2, 0.3),
    doc_thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6),
    doc_deltas: Sequence[float] = (0.1, 0.2, 0.3),
    min_k: int = 1,
    max_chunk_k: int = 20,
    max_doc_k: int = 10,
) -> List[CalibrationResult]:
    """Performs grid search to find optimal threshold and delta parameters maximizing Macro F2."""
    evaluator = Evaluator(beta=2.0)
    results: List[CalibrationResult] = []

    combos = list(product(chunk_thresholds, chunk_deltas, doc_thresholds, doc_deltas))

    for c_thresh, c_delta, d_thresh, d_delta in combos:
        predictions: List[PredictionRecord] = []
        for qid in ground_truth:
            chunk_cands = query_chunk_candidates.get(qid, [])
            doc_cands = query_doc_candidates.get(qid, [])

            selected_chunks = select_candidates(
                chunk_cands,
                threshold=c_thresh,
                relative_delta=c_delta,
                min_k=min_k,
                max_k=max_chunk_k,
            )
            selected_docs = select_candidates(
                doc_cands,
                threshold=d_thresh,
                relative_delta=d_delta,
                min_k=min_k,
                max_k=max_doc_k,
            )

            predictions.append(
                PredictionRecord(
                    id=qid,
                    relevant_docs=selected_docs,
                    relevant_chunks=selected_chunks,
                )
            )

        summary = evaluator.evaluate(ground_truth=ground_truth, predictions=predictions)

        results.append(
            CalibrationResult(
                chunk_threshold=c_thresh,
                chunk_delta=c_delta,
                doc_threshold=d_thresh,
                doc_delta=d_delta,
                doc_precision=summary.doc_precision,
                doc_recall=summary.doc_recall,
                doc_f2=summary.doc_f2,
                chunk_precision=summary.chunk_precision,
                chunk_recall=summary.chunk_recall,
                chunk_f2=summary.chunk_f2,
                composite_f2=summary.macro_f2,
            )
        )

    # Sort results by composite macro F2 descending
    results.sort(key=lambda r: r.composite_f2, reverse=True)
    return results
