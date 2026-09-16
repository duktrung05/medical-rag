"""Dynamic thresholding and selection rules (absolute threshold + relative delta)."""

from typing import List, Tuple


def select_candidates(
    ranked_candidates: List[Tuple[str, float]],
    threshold: float = 0.50,
    relative_delta: float = 0.30,
    min_k: int = 1,
    max_k: int = 20,
) -> List[str]:
    """Selects candidates based on dual criteria:

    1. score >= threshold (absolute threshold)
    2. score >= top_score - relative_delta (relative margin from highest-scoring item)
    Bounded by [min_k, max_k].
    """
    if not ranked_candidates:
        return []

    top_score = ranked_candidates[0][1]
    selected: List[str] = []

    for item_id, score in ranked_candidates:
        if len(selected) >= max_k:
            break

        passes_abs = score >= threshold
        passes_rel = score >= (top_score - relative_delta)

        if passes_abs and passes_rel:
            selected.append(item_id)
        elif len(selected) < min_k:
            selected.append(item_id)

    # Ensure min_k constraint is satisfied if enough candidates exist
    if len(selected) < min_k and len(ranked_candidates) >= min_k:
        selected = [c[0] for c in ranked_candidates[:min_k]]

    return selected
