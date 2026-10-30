"""Document-level features and conservative fixed-budget replacement."""
from __future__ import annotations

import math
from typing import Mapping


FEATURE_NAMES = (
    "max_rerank",
    "second_rerank",
    "top2_mean",
    "top3_mean",
    "log_passage_count",
    "log_count_ge_neg2",
    "log_count_ge_zero",
    "rrf_score",
    "branch_count",
    "document_bm25_rr",
    "passage_bm25_rr",
    "dense_rr",
)


def _reciprocal_rank(ranks: Mapping[str, object], branch: str) -> float:
    value = ranks.get(branch)
    if type(value) is not int or value <= 0:
        return 0.0
    return 60.0 / (60.0 + value)


def document_features(document: dict) -> tuple[float, ...]:
    """Represent a parent with its passage-score distribution and retrieval ranks."""
    scores = []
    for evidence in document.get("evidence", []):
        value = evidence.get("rerank_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Every candidate passage requires a finite rerank score")
        scores.append(float(value))
    if not scores:
        raise ValueError("Document candidate requires at least one scored passage")
    scores.sort(reverse=True)
    maximum = scores[0]
    second = scores[1] if len(scores) > 1 else maximum
    top2 = scores[:2]
    top3 = scores[:3]
    rrf = document.get("score", 0.0)
    if isinstance(rrf, bool) or not isinstance(rrf, (int, float)) or not math.isfinite(rrf):
        raise ValueError("Document candidate requires a finite retrieval score")
    ranks = document.get("ranks", {})
    if not isinstance(ranks, dict):
        raise ValueError("Document candidate ranks must be an object")
    branch_names = ("document_bm25", "passage_bm25", "dense")
    return (
        maximum,
        second,
        sum(top2) / len(top2),
        sum(top3) / len(top3),
        math.log1p(len(scores)),
        math.log1p(sum(value >= -2.0 for value in scores)),
        math.log1p(sum(value >= 0.0 for value in scores)),
        float(rrf),
        float(sum(type(ranks.get(name)) is int and ranks[name] > 0 for name in branch_names)),
        _reciprocal_rank(ranks, "document_bm25"),
        _reciprocal_rank(ranks, "passage_bm25"),
        _reciprocal_rank(ranks, "dense"),
    )


def conservative_document_swaps(
    baseline_ids: list[int],
    chunk_parent_ids: set[int],
    candidates: list[dict],
    document_scores: Mapping[int, float],
    *,
    max_swaps: int = 3,
    minimum_margin: float = 1.0,
    minimum_passage_score: float = -4.0,
) -> tuple[list[int], list[dict]]:
    """Replace weak non-chunk documents while preserving count, order, and chunks."""
    if max_swaps < 0 or minimum_margin < 0:
        raise ValueError("Swap cap and margin must be nonnegative")
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("Baseline document IDs must be unique")
    baseline = set(baseline_ids)
    if not chunk_parent_ids <= baseline:
        raise ValueError("Every chunk parent must be a baseline document")
    by_id = {}
    for document in candidates:
        doc_id = document.get("official_id")
        if type(doc_id) is not int or doc_id in by_id:
            raise ValueError("Candidate document IDs must be unique integers")
        by_id[doc_id] = document
    if not baseline <= set(by_id) or not set(by_id) <= set(document_scores):
        raise ValueError("Candidate documents and document scores are incomplete")

    victims = sorted(
        (doc_id for doc_id in baseline_ids if doc_id not in chunk_parent_ids),
        key=lambda doc_id: (document_scores[doc_id], doc_id),
    )
    entrants = []
    for doc_id, document in by_id.items():
        if doc_id in baseline:
            continue
        passage_score = document.get("rerank_score")
        if (isinstance(passage_score, bool)
                or not isinstance(passage_score, (int, float))
                or not math.isfinite(passage_score)):
            raise ValueError("Candidate document requires a finite max passage score")
        if passage_score >= minimum_passage_score:
            entrants.append(doc_id)
    entrants.sort(key=lambda doc_id: (-document_scores[doc_id], doc_id))

    replacements = []
    for entrant, victim in zip(entrants, victims):
        if len(replacements) >= max_swaps:
            break
        margin = document_scores[entrant] - document_scores[victim]
        if margin < minimum_margin:
            break
        replacements.append({"victim": victim, "entrant": entrant, "margin": margin})

    result = list(baseline_ids)
    for replacement in replacements:
        position = result.index(replacement["victim"])
        result[position] = replacement["entrant"]
    if len(result) != len(baseline_ids) or len(result) != len(set(result)):
        raise AssertionError("Document replacement changed count or introduced duplicates")
    if not chunk_parent_ids <= set(result):
        raise AssertionError("Document replacement removed a chunk parent")
    return result, replacements
