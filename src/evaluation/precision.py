"""Precision calculation for retrieval evaluation."""

from typing import Collection


def calculate_precision(actual: Collection[str], predicted: Collection[str]) -> float:
    """Computes Precision = TP / (TP + FP).

    If predicted is empty, precision is defined as 0.0.
    """
    if not predicted:
        return 0.0

    actual_set = set(actual)
    tp = sum(1 for item in predicted if item in actual_set)
    return float(tp / len(predicted))
