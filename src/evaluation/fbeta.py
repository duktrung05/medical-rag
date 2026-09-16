"""F-beta calculation for retrieval evaluation, focusing on F2."""

from typing import Collection
from src.evaluation.precision import calculate_precision
from src.evaluation.recall import calculate_recall


def calculate_fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    """Computes F-beta from precision and recall:

    F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    For beta=2:
    F_2 = 5 * P * R / (4 * P + R)
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    beta_sq = beta * beta
    denominator = (beta_sq * precision) + recall
    if denominator <= 0.0:
        return 0.0

    return float((1.0 + beta_sq) * (precision * recall) / denominator)


def calculate_query_metrics(
    actual: Collection[str],
    predicted: Collection[str],
    beta: float = 2.0,
) -> dict:
    """Calculates precision, recall, and F-beta for a single query."""
    p = calculate_precision(actual, predicted)
    r = calculate_recall(actual, predicted)
    f = calculate_fbeta(p, r, beta=beta)
    return {"precision": p, "recall": r, "fbeta": f}
