"""Derive deterministic, source-grounded export spans from selected passages."""
from __future__ import annotations

import hashlib
import json

from src.medrag.evidence.selector import SelectionPolicy
from src.medrag.evidence.verifier import verify


def passages_for_parent(passages, official_id: int) -> list:
    """Return registered passages in stable source order for one parent."""
    if hasattr(passages, "for_parent"):
        rows = passages.for_parent(official_id)
    elif hasattr(passages, "passages_for_parent"):
        rows = passages.passages_for_parent(official_id)
    elif isinstance(passages, dict):
        rows = [row for row in passages.values() if row.official_id == official_id]
    elif hasattr(passages, "passages") and isinstance(passages.passages, dict):
        rows = [row for row in passages.passages.values() if row.official_id == official_id]
    else:
        raise TypeError("Passage provider cannot enumerate a parent")
    return sorted(rows, key=lambda row: (row.source_start, row.source_end, row.passage_id))


def _registered_seed(item: dict) -> dict:
    keys = ("passage_id", "text", "start", "end", "score_status")
    seed = {key: item[key] for key in keys}
    if "rerank_score" in item:
        seed["rerank_score"] = item["rerank_score"]
    return seed


def derive_adjacent_evidence(seed: dict, source, parent_passages: list,
                             policy: SelectionPolicy) -> dict:
    """Build the widest bounded adjacent source span that contains the scored seed."""
    if policy.chunk_context != "adjacent":
        raise ValueError("Adjacent evidence requires chunk_context=adjacent")
    ordered = sorted(parent_passages,
                     key=lambda row: (row.source_start, row.source_end, row.passage_id))
    matches = [index for index, row in enumerate(ordered)
               if row.passage_id == seed.get("passage_id")]
    if len(matches) != 1:
        raise ValueError("Seed passage is missing or duplicated within its parent")
    seed_index = matches[0]
    passage = ordered[seed_index]
    if (passage.official_id != source.official_id or not verify(passage, source)
            or seed.get("text") != passage.text
            or seed.get("start") != passage.source_start
            or seed.get("end") != passage.source_end):
        raise ValueError("Adjacent context seed is not an exact registered passage")

    lower = max(0, seed_index - policy.context_window)
    upper = min(len(ordered) - 1, seed_index + policy.context_window)
    options = []
    for left in range(lower, seed_index + 1):
        for right in range(seed_index, upper + 1):
            members = ordered[left:right + 1]
            if any(row.official_id != source.official_id or not verify(row, source)
                   for row in members):
                raise ValueError("Adjacent context includes an ungrounded passage")
            start = min(row.source_start for row in members)
            end = max(row.source_end for row in members)
            length = end - start
            if length <= policy.context_max_characters:
                options.append((length, len(members), -start, -end, start, end, members))
    if not options:
        raise ValueError("Seed passage exceeds context_max_characters")
    _, _, _, _, start, end, members = max(options, key=lambda row: row[:4])
    member_ids = [row.passage_id for row in members]
    contract = {
        "version": "adjacent_source_span_v1",
        "official_id": source.official_id,
        "source_sha256": source.text_sha256,
        "start": start,
        "end": end,
        "seed_passage_id": passage.passage_id,
        "context_passage_ids": member_ids,
    }
    digest = hashlib.sha256(json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    seed_copy = _registered_seed(seed)
    status = ("derived_from_reranked_seed" if seed_copy["score_status"] == "reranked"
              else "derived_from_retrieval_seed")
    result = {
        "evidence_kind": "adjacent_source_span_v1",
        "passage_id": f"context:{digest}",
        "seed_passage_id": passage.passage_id,
        "context_passage_ids": member_ids,
        "seed_evidence": seed_copy,
        "text": source.text[start:end],
        "start": start,
        "end": end,
        "source_sha256": source.text_sha256,
        "score_status": status,
    }
    if "rerank_score" in seed_copy:
        result["rerank_score"] = seed_copy["rerank_score"]
    return result


def apply_chunk_context(selected: list[dict], policy: SelectionPolicy, sources,
                        passages) -> tuple[list[dict], dict]:
    """Apply the configured export context without changing parent selection or anchors."""
    diagnostics = {
        "mode": policy.chunk_context,
        "context_window": policy.context_window,
        "context_max_characters": policy.context_max_characters,
        "input_evidence": sum(len(document["evidence"]) for document in selected),
        "derived_evidence": 0,
        "expanded_spans": 0,
        "unchanged_spans": 0,
    }
    if policy.chunk_context == "passage":
        return selected, diagnostics
    output = []
    for document in selected:
        doc_id = document["official_id"]
        source = sources[doc_id] if hasattr(sources, "__getitem__") else sources.source(doc_id)
        registered = passages_for_parent(passages, doc_id)
        evidence = []
        for seed in document["evidence"]:
            derived = derive_adjacent_evidence(seed, source, registered, policy)
            evidence.append(derived)
            diagnostics["derived_evidence"] += 1
            if derived["start"] == seed["start"] and derived["end"] == seed["end"]:
                diagnostics["unchanged_spans"] += 1
            else:
                diagnostics["expanded_spans"] += 1
        output.append({**document, "evidence": evidence})
    return output, diagnostics
