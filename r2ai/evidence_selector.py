"""Final evidence selection. Scores gate relevance; input order controls priority."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidenceCandidate:
    doc_id: int
    chunk_index: int
    chunk_text: str
    rerank_score: float | None = None
    retrieval_score: float | None = None
    title: str = ""

    def __post_init__(self):
        if type(self.doc_id) is not int or type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("Candidate requires integer doc_id and nonnegative chunk_index")
        if not isinstance(self.chunk_text, str) or not self.chunk_text.strip():
            raise ValueError("Candidate chunk_text must be nonempty")
        if not isinstance(self.title, str):
            raise TypeError("Candidate title must be a string")
        for score in (self.rerank_score, self.retrieval_score):
            if score is not None and (isinstance(score, bool) or not math.isfinite(score)):
                raise ValueError("Candidate scores must be finite or None")

    @property
    def key(self) -> tuple[int, int]:
        return self.doc_id, self.chunk_index


@dataclass(frozen=True)
class QueryRanking:
    query_id: int
    document_order: list[int]
    candidates: list[EvidenceCandidate]
    query_text: str = ""

    def __post_init__(self):
        if type(self.query_id) is not int:
            raise ValueError("query_id must be an integer")
        if not isinstance(self.query_text, str):
            raise TypeError("query_text must be a string")
        if any(type(doc) is not int for doc in self.document_order):
            raise ValueError("Document IDs must be integers")
        if len(set(self.document_order)) != len(self.document_order):
            raise ValueError("Duplicate documents in ranking")
        if len({c.key for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate chunk identities in ranking")
        if not {c.doc_id for c in self.candidates} <= set(self.document_order):
            raise ValueError("document_order must contain all candidate parents")


@dataclass(frozen=True)
class SelectionConfig:
    mode: str = "topk"
    max_docs: int = 10
    max_chunks: int = 10
    chunks_per_doc: int = 1
    chunk_threshold: float | None = None
    chunk_delta: float | None = None
    document_mode: str = "parents"
    doc_threshold: float | None = None
    doc_delta: float | None = None

    def __post_init__(self):
        if self.mode not in {"topk", "threshold"}:
            raise ValueError("mode must be topk or threshold")
        if self.document_mode not in {"parents", "baseline", "threshold"}:
            raise ValueError("document_mode must be parents, baseline or threshold")
        for name in ("max_docs", "max_chunks", "chunks_per_doc"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("chunk_threshold", "chunk_delta", "doc_threshold", "doc_delta"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("chunk_delta", "doc_delta"):
            if getattr(self, name) is not None and getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.mode == "threshold" and self.chunk_threshold is None:
            raise ValueError("threshold selection requires chunk_threshold")
        if self.mode == "topk" and any(getattr(self, n) is not None for n in
                                      ("chunk_threshold", "chunk_delta", "doc_threshold", "doc_delta")):
            raise ValueError("Score thresholds require mode=threshold")
        if self.document_mode != "threshold" and (self.doc_threshold is not None or self.doc_delta is not None):
            raise ValueError("Document score gates require document_mode=threshold")


def select_evidence(ranking: QueryRanking, config: SelectionConfig) -> tuple[dict, dict]:
    """Return competition predictions and diagnostics; never backfill failed gates.

    Candidate priority follows the cached document/chunk ordering (including
    optional taxonomy). Only authentic cross-encoder scores enter score gates.
    """
    scores = [c.rerank_score for c in ranking.candidates if c.rerank_score is not None]
    top_score = max(scores, default=None)
    relative_cutoff = (top_score - config.chunk_delta
                       if top_score is not None and config.chunk_delta is not None else None)
    doc_scores: dict[int, float] = {}
    for candidate in ranking.candidates:
        if candidate.rerank_score is not None:
            doc_scores[candidate.doc_id] = max(doc_scores.get(candidate.doc_id, -math.inf),
                                               candidate.rerank_score)

    if config.mode == "topk" or config.document_mode == "baseline":
        output_docs = ranking.document_order[:config.max_docs]
    elif config.document_mode == "threshold":
        threshold = config.doc_threshold if config.doc_threshold is not None else config.chunk_threshold
        best_doc = max(doc_scores.values(), default=None)
        cutoff = (best_doc - config.doc_delta
                  if best_doc is not None and config.doc_delta is not None else None)
        output_docs = [doc for doc in ranking.document_order if doc in doc_scores
                       and doc_scores[doc] >= threshold
                       and (cutoff is None or doc_scores[doc] >= cutoff)][:config.max_docs]
    else:
        output_docs = []

    allowed_docs = set(output_docs)
    counts: Counter = Counter()
    seen_text: set[str] = set()
    selected: list[EvidenceCandidate] = []
    decisions = []
    reasons: Counter = Counter()
    for candidate in ranking.candidates:
        reason = None
        if config.mode == "threshold":
            if candidate.rerank_score is None:
                reason = "not_reranked"
            elif candidate.rerank_score < config.chunk_threshold:
                reason = "below_chunk_threshold"
            elif relative_cutoff is not None and candidate.rerank_score < relative_cutoff:
                reason = "below_chunk_relative_cutoff"
            elif candidate.chunk_text in seen_text:
                reason = "duplicate_text"
        if (reason is None and (config.mode == "topk" or config.document_mode != "parents")
                and candidate.doc_id not in allowed_docs):
            reason = "document_not_selected"
        if reason is None and counts[candidate.doc_id] >= config.chunks_per_doc:
            reason = "chunks_per_doc_cap"
        if reason is None and len(selected) >= config.max_chunks:
            reason = "chunk_cap"
        if reason is None and candidate.doc_id not in allowed_docs:
            if len(allowed_docs) >= config.max_docs:
                reason = "document_cap"
            else:
                allowed_docs.add(candidate.doc_id)
        if reason is None:
            selected.append(candidate)
            counts[candidate.doc_id] += 1
            seen_text.add(candidate.chunk_text)
        reasons[reason or "selected"] += 1
        decisions.append({"doc_id": candidate.doc_id, "chunk_index": candidate.chunk_index,
                          "rerank_score": candidate.rerank_score, "reason": reason or "selected"})
    if config.mode == "threshold" and config.document_mode == "parents":
        output_docs = [doc for doc in ranking.document_order if doc in allowed_docs]
    prediction = {"id": ranking.query_id, "relevant_docs": output_docs,
                  "relevant_chunks": [{"doc_id": c.doc_id, "chunk_text": c.chunk_text} for c in selected]}
    diagnostics = {"id": ranking.query_id, "config": asdict(config), "top_score": top_score,
                   "chunk_relative_cutoff": relative_cutoff, "candidate_count": len(ranking.candidates),
                   "reranked_count": len(scores), "selected_doc_count": len(output_docs),
                   "selected_chunk_count": len(selected), "reasons": dict(reasons), "decisions": decisions}
    return prediction, diagnostics
