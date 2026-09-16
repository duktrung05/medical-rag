"""Evaluation package exporting metric calculations and Macro evaluator."""

from src.evaluation.precision import calculate_precision
from src.evaluation.recall import calculate_recall
from src.evaluation.fbeta import calculate_fbeta, calculate_query_metrics
from src.evaluation.evaluator import Evaluator, EvaluationSummary, QueryMetricDetail

__all__ = [
    "calculate_precision",
    "calculate_recall",
    "calculate_fbeta",
    "calculate_query_metrics",
    "Evaluator",
    "EvaluationSummary",
    "QueryMetricDetail",
]
