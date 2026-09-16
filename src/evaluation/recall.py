"""Recall calculation for retrieval evaluation."""

from typing import Collection


def calculate_recall(actual: Collection[str], predicted: Collection[str]) -> float:
    """Computes Recall = TP / (TP + FN).

    If actual is empty, recall is defined as 0.0.
    """
    if not actual:
        return 0.0

    actual_set = set(actual)
    tp = sum(1 for item in predicted if item in actual_set)
    return float(tp / len(actual_set))
