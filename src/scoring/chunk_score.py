"""Chunk score normalization and calibration utilities."""

from typing import Dict, List, Tuple


def normalize_min_max(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalizes scores to [0, 1] range."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if min_v == max_v:
        return {k: 1.0 for k in scores}
    return {k: float((v - min_v) / (max_v - min_v)) for k, v in scores.items()}


def sort_candidates(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    """Sorts candidate IDs by descending score."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
