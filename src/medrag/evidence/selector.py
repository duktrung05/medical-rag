"""Explicit, bounded V4 evidence selection after retrieval and reranking."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionPolicy:
    max_docs: int = 200
    max_chunks: int = 60
    chunks_per_doc: int = 1
    document_selection: str = "with_chunks"
    chunk_context: str = "passage"
    context_window: int = 1
    context_max_characters: int = 4096

    def __post_init__(self) -> None:
        for name in ("max_docs", "max_chunks", "chunks_per_doc"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.document_selection not in {"with_chunks", "independent"}:
            raise ValueError("Unsupported document_selection")
        if self.chunk_context not in {"passage", "adjacent"}:
            raise ValueError("Unsupported chunk_context")
        if type(self.context_window) is not int or self.context_window < 0:
            raise ValueError("context_window must be a non-negative integer")
        if type(self.context_max_characters) is not int or self.context_max_characters <= 0:
            raise ValueError("context_max_characters must be a positive integer")


def select_results(documents: list[dict], policy: SelectionPolicy, *, reranker_enabled: bool,
                   limit: int) -> tuple[list[dict], dict]:
    """Select exact candidates; never invent scores or silently slice in the serializer."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("Result limit must be a positive integer")
    if policy.document_selection == "independent":
        return _select_independent(documents, policy, reranker_enabled=reranker_enabled,
                                   limit=limit)
    return _select_with_chunks(documents, policy, reranker_enabled=reranker_enabled, limit=limit)


def _select_with_chunks(documents: list[dict], policy: SelectionPolicy, *,
                        reranker_enabled: bool, limit: int) -> tuple[list[dict], dict]:
    """Historical A0 policy, retained explicitly for regression and comparison."""
    selected = []
    seen_ids: set[tuple[int, str]] = set()
    seen_parent_text: set[tuple[int, str]] = set()
    counts = {"candidate_documents": len(documents), "candidate_evidence": 0,
              "reranked_evidence": 0, "not_reranked_budget": 0,
              "duplicate_identity": 0, "duplicate_parent_text": 0,
              "doc_cap": 0, "chunk_cap": 0,
              "per_doc_cap": 0, "selected_documents": 0, "selected_evidence": 0}
    document_cap = min(limit, policy.max_docs)
    for document in documents:
        kept = []
        for candidate in document["evidence"]:
            counts["candidate_evidence"] += 1
            identity = (int(document["official_id"]), str(candidate["passage_id"]))
            if reranker_enabled:
                if "rerank_score" not in candidate:
                    counts["not_reranked_budget"] += 1
                    continue
                score = candidate["rerank_score"]
                if isinstance(score, bool) or not math.isfinite(float(score)):
                    raise ValueError(f"Non-finite reranker score for passage {identity}")
                counts["reranked_evidence"] += 1
            if identity in seen_ids:
                counts["duplicate_identity"] += 1
                continue
            parent_text = (identity[0], candidate["text"])
            if parent_text in seen_parent_text:
                counts["duplicate_parent_text"] += 1
                continue
            if len(selected) >= document_cap:
                counts["doc_cap"] += 1
                continue
            if counts["selected_evidence"] >= policy.max_chunks:
                counts["chunk_cap"] += 1
                continue
            if len(kept) >= policy.chunks_per_doc:
                counts["per_doc_cap"] += 1
                continue
            kept.append(candidate)
            seen_ids.add(identity)
            seen_parent_text.add(parent_text)
            counts["selected_evidence"] += 1
        if kept:
            selected.append({**document, "evidence": kept})
    counts["selected_documents"] = len(selected)
    return selected, counts


def _select_independent(documents: list[dict], policy: SelectionPolicy, *,
                        reranker_enabled: bool, limit: int) -> tuple[list[dict], dict]:
    """Retain ranked, supported parents even when their export chunk budget is exhausted."""
    selected = []
    seen_parents = set()
    counts = {"candidate_documents": len(documents), "candidate_evidence": 0,
              "reranked_evidence": 0, "not_reranked_budget": 0,
              "duplicate_identity": 0, "duplicate_parent_text": 0,
              "doc_cap": 0, "chunk_cap": 0, "per_doc_cap": 0,
              "documents_without_eligible_evidence": 0,
              "documents_without_exported_chunks": 0,
              "selected_documents": 0, "selected_evidence": 0,
              "document_selection": "independent"}
    for document in documents:
        parent = document["official_id"]
        if type(parent) is not int or parent in seen_parents:
            raise ValueError("Invalid or duplicate candidate document ID")
        seen_parents.add(parent)
        eligible = []
        identities, texts = set(), set()
        for candidate in document["evidence"]:
            counts["candidate_evidence"] += 1
            if reranker_enabled:
                if "rerank_score" not in candidate:
                    counts["not_reranked_budget"] += 1
                    continue
                score = candidate["rerank_score"]
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score)):
                    raise ValueError("Non-finite or invalid reranker score")
                if candidate.get("score_status") != "reranked":
                    raise ValueError("Invalid reranker score status")
                counts["reranked_evidence"] += 1
            identity = str(candidate["passage_id"])
            if identity in identities:
                counts["duplicate_identity"] += 1
                continue
            if candidate["text"] in texts:
                counts["duplicate_parent_text"] += 1
                continue
            identities.add(identity)
            texts.add(candidate["text"])
            eligible.append(candidate)
        if not eligible:
            counts["documents_without_eligible_evidence"] += 1
            continue
        if len(selected) >= min(limit, policy.max_docs):
            counts["doc_cap"] += 1
            continue
        anchor = (max(eligible, key=lambda item: item["rerank_score"])
                  if reranker_enabled else eligible[0])
        kept = []
        for candidate in eligible:
            if len(kept) >= policy.chunks_per_doc:
                counts["per_doc_cap"] += 1
            elif counts["selected_evidence"] >= policy.max_chunks:
                counts["chunk_cap"] += 1
            else:
                kept.append(candidate)
                counts["selected_evidence"] += 1
        selected.append({**document, "document_evidence": dict(anchor), "evidence": kept})
        counts["documents_without_exported_chunks"] += not bool(kept)
    counts["selected_documents"] = len(selected)
    return selected, counts
