"""Offline P1.5 diagnostics for five exact-recoverable parents; no model/retrieval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "outputs/v4/p1_chunk_allocation_20261021_v2"
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
TARGETS = [(53, 756661), (130, 553119), (130, 2968260),
           (754, 2713777), (771, 484565)]
CHUNK_CAP = 60

ASSESSMENTS = {
    (53, 756661): {
        "semantic_relevance": "Relevant symptom-level evidence: passage discusses chest tightness and possible cardiac, pulmonary, mediastinal, and musculoskeletal causes; it gives general referral advice rather than a specific diagnosis.",
        "assessment": "Plausibly under-ranked, but broad answer and weak retrieval support make this a borderline replacement candidate.",
    },
    (130, 553119): {
        "semantic_relevance": "Directly relevant: passage explains nasolacrimal duct obstruction, tearing, drainage to the nose, and that it can occur in infants.",
        "assessment": "Strongest under-ranking case. Semantic match is direct, but fused rank is very poor and there is no BM25 hit.",
    },
    (130, 2968260): {
        "semantic_relevance": "Topically related Chinese passage about eye discharge and neonatal dacryocystitis, but the source is a different 65-year-old patient and the response says the described case is not dacryocystitis.",
        "assessment": "Do not treat anchor containment or dense similarity as sufficient relevance. It is a weak/contradictory match for the infant query; reranker may be appropriately conservative. Input also exceeds the 512-token limit.",
    },
    (754, 2713777): {
        "semantic_relevance": "Related test topic, but the passage reports CA125 40.41 while the query asks about 64.80; it provides no interpretation or next-step advice.",
        "assessment": "Topical but incomplete evidence; not a safe replacement based on current signals.",
    },
    (771, 484565): {
        "semantic_relevance": "Directly relevant: passage discusses premature ejaculation and includes psychological intervention and treatment guidance.",
        "assessment": "Plausible under-ranking; rerank score is relatively high but still 0.766 below the top-60 boundary, and document BM25 rank is 77.",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_summary(rows: list[dict], query_id: int, checkpoint: dict) -> dict:
    top = sorted((r for r in rows if r["query_id"] == query_id),
                 key=lambda r: r["final_doc_rank"])[:60]
    dense = {int(r["official_id"]): r for r in checkpoint["branch_trace"][
        "branch_hits"]["dense"]}
    dense_scores = [float(dense[r["doc_id"]]["score"])
                    for r in top if r["doc_id"] in dense]
    dense_ranks = [int(r["dense_rank"]) for r in top if r["dense_rank"] is not None]
    bm25_ranks = [int(r["bm25_rank"]) for r in top if r["bm25_rank"] is not None]
    return {
        "query_id": query_id,
        "baseline_top60": 60,
        "rerank_score_median": statistics.median(r["rerank_score"] for r in top),
        "dense_rank_median": statistics.median(dense_ranks) if dense_ranks else None,
        "dense_similarity_median": statistics.median(dense_scores) if dense_scores else None,
        "dense_similarity_min": min(dense_scores) if dense_scores else None,
        "dense_similarity_max": max(dense_scores) if dense_scores else None,
        "query_coverage_median": statistics.median(r["query_coverage"] for r in top),
        "fused_rank_median": statistics.median(r["fused_rank"] for r in top),
        "bm25_parent_count": len(bm25_ranks),
        "bm25_rank_median": statistics.median(bm25_ranks) if bm25_ranks else None,
    }


def main(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    pool = [json.loads(line) for line in (P1 / "scored_parent_pool.jsonl").open(
        encoding="utf-8")]
    rows_by_query: dict[int, list[dict]] = {}
    for row in pool:
        rows_by_query.setdefault(int(row["query_id"]), []).append(row)
    anchor = read_json(P1 / "per_query_metrics.json")["A0"]
    anchor_by_id = {(int(q["query_id"]), int(a["doc_id"])): a
                    for q in anchor for a in q["anchor_rows"]}

    details = []
    summaries = []
    input_paths = [P1 / "scored_parent_pool.jsonl", P1 / "per_query_metrics.json"]
    for qid in sorted({q for q, _ in TARGETS}):
        checkpoint_path = PILOT / "queries" / f"{qid}.json"
        checkpoint = read_json(checkpoint_path)
        input_paths.append(checkpoint_path)
        rows = sorted(rows_by_query[qid], key=lambda r: r["final_doc_rank"])
        by_id = {int(r["doc_id"]): r for r in rows}
        cutoff = rows[CHUNK_CAP - 1]
        cut_row = {
            "query_id": qid,
            "doc_id": int(cutoff["doc_id"]),
            "rerank_rank": int(cutoff["final_doc_rank"]),
            "rerank_score": float(cutoff["rerank_score"]),
        }
        summaries.append(quantile_summary(pool, qid, checkpoint))
        dense = {int(r["official_id"]): r for r in checkpoint["branch_trace"][
            "branch_hits"]["dense"]}
        dense_by_passage = {(int(r["official_id"]), r["passage_id"]): r
                            for r in checkpoint["branch_trace"]["branch_hits"]["dense"]}
        pair_index = {int(pair["official_id"]): position
                      for position, pair in enumerate(checkpoint["pair_trace"], 1)}
        token_lengths = checkpoint["diagnostics"]["reranker_tokens"]["pair_lengths"]
        candidates = {int(r["official_id"]): r for r in
                      checkpoint["diagnostics"]["ranked_candidates"]}
        target_ids = [doc_id for target_q, doc_id in TARGETS if target_q == qid]
        baseline_top = rows[:CHUNK_CAP]
        baseline_dense_scores = [float(dense[int(r["doc_id"])]["score"])
                                 for r in baseline_top if int(r["doc_id"]) in dense]
        baseline_coverage_median = statistics.median(
            float(r["query_coverage"]) for r in baseline_top)
        baseline_fused_median = statistics.median(int(r["fused_rank"])
                                                   for r in baseline_top)
        for doc_id in target_ids:
            row = by_id[doc_id]
            candidate = candidates[doc_id]
            evidence = next(e for e in candidate["evidence"]
                            if e["passage_id"] == row["passage_id"])
            dense_hit = dense_by_passage.get((doc_id, row["passage_id"])) or dense.get(doc_id)
            token_length = token_lengths[pair_index[doc_id] - 1]
            bm25_hits = []
            for channel in ("document_bm25", "passage_bm25"):
                bm25_hits.extend(hit for hit in checkpoint["branch_trace"][
                    "branch_hits"][channel] if int(hit["official_id"]) == doc_id)
            best_bm25 = min((int(hit["rank"]) for hit in bm25_hits), default=None)
            anchor_row = anchor_by_id[(qid, doc_id)]
            details.append({
                "query_id": qid,
                "doc_id": doc_id,
                "passage_id": row["passage_id"],
                "rerank_rank": int(row["final_doc_rank"]),
                "slots_below_top60": max(0, int(row["final_doc_rank"]) - 60),
                "rerank_score": float(row["rerank_score"]),
                "rank60_doc_id": cut_row["doc_id"],
                "rank60_rerank_score": cut_row["rerank_score"],
                "score_gap_to_60": float(cutoff["rerank_score"] - row["rerank_score"]),
                "fused_rank": int(row["fused_rank"]),
                "dense_rank": row["dense_rank"],
                "dense_similarity": float(dense_hit["score"]) if dense_hit else None,
                "dense_similarity_percentile_vs_top60": (
                    sum(score <= float(dense_hit["score"]) for score in baseline_dense_scores)
                    / len(baseline_dense_scores) if dense_hit and baseline_dense_scores else None),
                "bm25_rank": best_bm25,
                "query_coverage": float(row["query_coverage"]),
                "query_coverage_delta_vs_top60_median": (
                    float(row["query_coverage"]) - baseline_coverage_median),
                "fused_rank_delta_vs_top60_median": (
                    int(row["fused_rank"]) - baseline_fused_median),
                "reranker_pair_tokens": int(token_length),
                "reranker_truncated_at_512": token_length > 512,
                "passage_char_length": len(evidence["text"]),
                "contains_cjk": any("\u4e00" <= char <= "\u9fff"
                                     for char in evidence["text"]),
                "context_contains_anchor_if_selected": True,
                "a0_exact_anchor": bool(anchor_row["exact_anchor"]),
                "passage_snippet": " ".join(evidence["text"].split())[:360],
                **ASSESSMENTS[(qid, doc_id)],
            })
    output.mkdir(parents=True)
    with (output / "target_parent_diagnostics.csv").open("w", encoding="utf-8",
                                                         newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (output / "top60_feature_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "diagnostics.json").write_text(
        json.dumps({"status": "P1_5_CPU_DIAGNOSTICS_COMPLETE",
                    "scope": "five exact-recoverable parents in the six-query P0 pilot",
                    "model_calls": 0, "retrieval_calls": 0,
                    "label_usage": "offline diagnostics only; no feature/policy fitting",
                    "targets": details, "baseline_top60_summaries": summaries,
                    "decision": "No shared reliable threshold/feature found; do not run or promote conservative replacement from this six-query pilot."},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    manifest = {
        "format": "v4-p1.5-score-diagnostics-v1",
        "status": "complete",
        "targets": [{"query_id": q, "doc_id": d} for q, d in TARGETS],
        "model_calls": 0,
        "retrieval_calls": 0,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path)
                         for path in input_paths},
        "artifact_sha256": {name: sha(output / name) for name in
                            ("target_parent_diagnostics.csv",
                             "top60_feature_summaries.json", "diagnostics.json")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "targets": len(details), "model_calls": 0, "retrieval_calls": 0},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
