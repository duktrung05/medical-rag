"""Experimental B0 passage-first challenger; does not change the A0 pipeline."""
from __future__ import annotations

import math

from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.core import fuse


def passage_candidates(rankings, *, rrf_k: int = 60, limit: int = 400):
    """Fuse passage identities directly, before any parent aggregation."""
    if any(hit.passage_id is None for branch in rankings for hit in branch):
        raise ValueError("B0 accepts only passage-level retrieval branches")
    return fuse(rankings, k=rrf_k, limit=limit)


def shared_pair_plan(a_pairs: list[tuple[int, str]], b_pairs: list[tuple[int, str]],
                     *, budget: int = 200, a_slots: int = 100) -> dict:
    """Allocate one inference budget across paths; overlapping pairs score once."""
    if not 0 <= a_slots <= budget or budget <= 0:
        raise ValueError("Invalid shared pair budget")
    paths = {"A0": list(dict.fromkeys(a_pairs)), "B0": list(dict.fromkeys(b_pairs))}
    chosen = []
    seen = set()

    def add(path: str, count: int) -> None:
        for pair in paths[path][:count]:
            if pair not in seen and len(chosen) < budget:
                seen.add(pair)
                chosen.append(pair)

    add("A0", a_slots)
    add("B0", budget - a_slots)
    for path in ("A0", "B0"):
        add(path, len(paths[path]))
    return {"unique_pairs": chosen, "total_inference_pairs": len(chosen),
            "a0_pairs_available": sum(pair in seen for pair in paths["A0"]),
            "b0_pairs_available": sum(pair in seen for pair in paths["B0"]),
            "shared_pairs": sum(pair in paths["B0"] for pair in paths["A0"] if pair in seen)}


def run_b0(pipeline, query: str, *, limit: int = 200, pair_budget: int = 200,
           chunk_budget: int = 60) -> tuple[list[dict], dict]:
    """Search existing passage indexes, score <=200 passages, export <=60 grounded chunks."""
    if pipeline.reranker is None:
        raise ValueError("B0 requires a reranker or an explicitly injected scoring stub")
    if pair_budget <= 0 or pair_budget > 200 or chunk_budget <= 0 or chunk_budget > 60:
        raise ValueError("B0 exceeds the locked comparison budget")
    rankings = [pipeline.passage_bm25.search(query, pipeline.passage_candidate_limit)]
    if pipeline.dense is not None:
        rankings.append(pipeline.dense.search(query, pipeline.passage_candidate_limit))
    fused = passage_candidates(rankings, rrf_k=pipeline.rrf_k,
                               limit=pipeline.fused_candidate_limit)
    pairs = fused[:pair_budget]
    inputs, identities, registered = [], [], []
    for hit in pairs:
        passage = pipeline.passages[hit.passage_id]
        source = pipeline.sources[hit.official_id]
        if passage.official_id != hit.official_id or not verify(passage, source):
            raise ValueError("B0 candidate is not a registered source passage")
        identities.append((hit.official_id, hit.passage_id))
        registered.append((hit, passage, source))
        inputs.append(f"{source.title}\n{passage.text}" if source.title else passage.text)
    if pipeline.score_cache is not None:
        scores = pipeline.score_cache.score(
            query, [(doc, passage_id, value) for (doc, passage_id), value in
                    zip(identities, inputs, strict=True)], pipeline.reranker)
    else:
        scores = pipeline.reranker.score(query, inputs)
    if len(scores) != len(pairs):
        raise ValueError("B0 score count differs from passage pairs")
    by_doc = {}
    ranked_passages = []
    for (hit, passage, _), raw_score in zip(registered, scores, strict=True):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("B0 received non-finite rerank score")
        evidence = {"passage_id": passage.passage_id, "text": passage.text,
                    "start": passage.source_start, "end": passage.source_end,
                    "score_status": "reranked", "rerank_score": score}
        ranked_passages.append((score, hit, evidence))
        doc = by_doc.setdefault(hit.official_id, {"official_id": hit.official_id,
                                                   "score": hit.score, "evidence": []})
        doc["score"] = max(doc["score"], hit.score)
        doc["evidence"].append(evidence)
    for doc in by_doc.values():
        doc["evidence"].sort(key=lambda row: (-row["rerank_score"], row["passage_id"]))
        doc["rerank_score"] = doc["evidence"][0]["rerank_score"]
    documents = sorted(by_doc.values(),
                       key=lambda row: (-row["rerank_score"], -row["score"], row["official_id"]))
    base = pipeline.selection_policy
    policy = SelectionPolicy(max_docs=min(base.max_docs, limit), max_chunks=chunk_budget,
                             chunks_per_doc=1, document_selection="independent",
                             chunk_context=base.chunk_context,
                             context_window=base.context_window,
                             context_max_characters=base.context_max_characters)
    selected, selection_diag = select_results(documents, policy, reranker_enabled=True,
                                              limit=limit)
    selected, context_diag = apply_chunk_context(selected, policy, pipeline.sources,
                                                 pipeline.passages)
    return selected, {"architecture": "B0_passage_first", "branches": [
        len(branch) for branch in rankings], "passage_candidates": len(fused),
        "rerank_pairs": len(pairs), "scored_parents": len(by_doc),
        "selected_documents": len(selected),
        "selected_chunks": sum(len(row["evidence"]) for row in selected),
        "pair_identities": identities, "selection": selection_diag, "context": context_diag}
