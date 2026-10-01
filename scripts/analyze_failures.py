"""Explain ViMed validation misses using cached stage rankings and predictions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.benchmark_vimed_validation import read_rows
from scripts.run_metadata import write_run_manifest
from src.data.loader import DataLoader, DocumentChunkMap


STAGES = ("bm25", "dense", "hybrid", "rerank")


def rank_map(row):
    return {chunk_id: rank for rank, (chunk_id, _) in enumerate(row["results"], 1)}


def analyze_query(sample, truth, stage_rows, *, ranking_k=10):
    qid = sample["id"]
    maps = {stage: rank_map(stage_rows[stage]) for stage in STAGES}
    gold_chunks, gold_docs = set(truth["relevant_chunks"]), set(truth["relevant_docs"])
    bm_hit = bool(gold_chunks & maps["bm25"].keys())
    dense_hit = bool(gold_chunks & maps["dense"].keys())
    union_hit = bool(gold_chunks & (maps["bm25"].keys() | maps["dense"].keys()))
    candidate_hit = bool(gold_chunks & maps["hybrid"].keys())
    rerank_top_hit = any(maps["rerank"].get(cid, ranking_k + 1) <= ranking_k for cid in gold_chunks)
    prediction = stage_rows["rerank"]["prediction"]
    selected_chunks, selected_docs = set(prediction["relevant_chunks"]), set(prediction["relevant_docs"])
    final_chunk_hit, final_doc_hit = bool(gold_chunks & selected_chunks), bool(gold_docs & selected_docs)

    types = []
    types.append("BOTH_HIT" if bm_hit and dense_hit else "BM25_ONLY_HIT" if bm_hit else
                 "DENSE_ONLY_HIT" if dense_hit else "BOTH_MISS")
    if union_hit and not candidate_hit:
        types.append("FUSION_DROPPED")
    if not candidate_hit:
        types.append("CANDIDATE_MISS")
    if candidate_hit and not rerank_top_hit:
        types.append("RERANKER_MISS")
    if any(cid in maps["rerank"] for cid in gold_chunks) and not final_chunk_hit:
        types.append("SELECTION_MISS")
    if final_doc_hit and not final_chunk_hit:
        types.append("DOC_HIT_CHUNK_MISS")
    if not final_doc_hit and not final_chunk_hit:
        types.append("DOC_AND_CHUNK_MISS")

    first_ranks = {stage: min((maps[stage].get(cid) for cid in gold_chunks if cid in maps[stage]), default=None)
                   for stage in STAGES}
    return {
        "query_id": qid, "query": sample["query"], "topic": sample["topic"],
        "relevant_docs": sorted(gold_docs), "relevant_chunks": sorted(gold_chunks),
        "bm25_first_relevant_rank": first_ranks["bm25"],
        "dense_first_relevant_rank": first_ranks["dense"],
        "hybrid_first_relevant_rank": first_ranks["hybrid"],
        "reranker_first_relevant_rank": first_ranks["rerank"],
        "final_chunk_selected": final_chunk_hit, "final_doc_selected": final_doc_hit,
        "error_types": types, "primary_failure_stage": primary_stage(types),
        "notes": "; ".join(types),
        "diagnostics": {
            "bm25_hit": bm_hit, "dense_hit": dense_hit, "union_hit": union_hit,
            "candidate_hit": candidate_hit, "reranker_top_k_hit": rerank_top_hit,
            "candidate_depth": len(stage_rows["hybrid"]["results"]), "ranking_k": ranking_k,
        },
    }


def primary_stage(types):
    for kind, stage in (("CANDIDATE_MISS", "retrieval_or_fusion"), ("RERANKER_MISS", "reranker"),
                        ("SELECTION_MISS", "selection"), ("DOC_HIT_CHUNK_MISS", "selection"),
                        ("DOC_AND_CHUNK_MISS", "retrieval_or_fusion")):
        if kind in types:
            return stage
    return "hit_at_evaluated_cutoff"


def classify_candidate_positive(chunk_id, rank_maps):
    bm_rank = rank_maps["bm25"].get(chunk_id)
    dense_rank = rank_maps["dense"].get(chunk_id)
    rrf_rank = rank_maps["hybrid"].get(chunk_id)
    if bm_rank is not None and dense_rank is not None:
        error_class = "FUSION_DROPPED" if rrf_rank is None else "BOTH_HIT"
        coverage = "both"
    elif bm_rank is not None:
        error_class, coverage = "BM25_ONLY_HIT", "bm25_only"
    elif dense_rank is not None:
        error_class, coverage = "DENSE_ONLY_HIT", "dense_only"
    else:
        error_class, coverage = "BOTH_MISS", "neither_in_cache"
    return {
        "bm25_rank": bm_rank, "dense_rank": dense_rank, "rrf_rank": rrf_rank,
        "error_class": error_class, "source_coverage": coverage,
        "fusion_dropped": bool((bm_rank or dense_rank) and rrf_rank is None),
        "unknown_beyond_cache_depth": bm_rank is None or dense_rank is None,
    }


def write_csv(path, rows):
    keys = ["query_id", "query", "topic", "relevant_docs", "relevant_chunks",
            "bm25_first_relevant_rank", "dense_first_relevant_rank", "hybrid_first_relevant_rank",
            "reranker_first_relevant_rank", "final_chunk_selected", "final_doc_selected",
            "error_types", "primary_failure_stage", "notes", "diagnostics"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "relevant_docs": json.dumps(row["relevant_docs"]),
                             "relevant_chunks": json.dumps(row["relevant_chunks"]),
                             "error_types": ";".join(row["error_types"]),
                             "diagnostics": json.dumps(row["diagnostics"], ensure_ascii=False)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/vimed/validation"))
    parser.add_argument("--rankings-root", type=Path, default=Path("outputs/vimed_validation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/error_analysis/run_20260930"))
    parser.add_argument("--ranking-k", type=int, default=10)
    args = parser.parse_args()
    if args.ranking_k < 1 or args.output.exists():
        raise ValueError("ranking-k must be positive and output directory must be new")
    samples = read_rows(args.data / "samples.jsonl")
    truths = {r["id"]: r for r in read_rows(args.data / "ground_truth.jsonl")}
    stages = {name: read_rows(args.rankings_root / name / "rankings.jsonl") for name in STAGES}
    if any(len(stages[s]) != len(samples) for s in STAGES):
        raise ValueError("Cached stages do not cover every query")
    result_rows = []
    for i, sample in enumerate(samples):
        per_stage = {name: stages[name][i] for name in STAGES}
        if any(per_stage[name]["id"] != sample["id"] for name in STAGES):
            raise ValueError(f"Stage query order mismatch at {sample['id']}")
        if sample["id"] not in truths:
            raise ValueError(f"Missing ground truth for {sample['id']}")
        result_rows.append(analyze_query(sample, truths[sample["id"]], per_stage, ranking_k=args.ranking_k))

    chunk_map = DocumentChunkMap.from_chunks(DataLoader.load_chunks("data/vimed/chunks.jsonl"))
    candidate_details = []
    for i, sample in enumerate(samples):
        per_stage = {name: stages[name][i] for name in STAGES}
        gold_chunks = truths[sample["id"]]["relevant_chunks"]
        rank_maps = {stage: rank_map(per_stage[stage]) for stage in STAGES}
        if any(cid in rank_maps["hybrid"] for cid in gold_chunks):
            continue
        for cid in gold_chunks:
            classified = classify_candidate_positive(cid, rank_maps)
            candidate_details.append({
                "query_id": sample["id"], "query": sample["query"], "gold_chunk": cid,
                "gold_doc": chunk_map.get_parent_doc(cid), **classified, "topic": sample["topic"],
                "cache_depth": {s: len(per_stage[s]["results"]) for s in ("bm25", "dense", "hybrid")},
            })

    args.output.mkdir(parents=True)
    write_run_manifest(args.output, task="validation_error_analysis", root=Path.cwd(), inputs={
        **{f"{stage}_rankings": args.rankings_root / stage / "rankings.jsonl" for stage in STAGES},
        "samples": args.data / "samples.jsonl", "ground_truth": args.data / "ground_truth.jsonl",
        "chunks": Path("data/vimed/chunks.jsonl"),
    }, parameters={"ranking_k": args.ranking_k, "candidate_miss_cache_depth": 100})
    write_csv(args.output / "all_queries.csv", result_rows)
    failures = [row for row in result_rows if row["primary_failure_stage"] != "hit_at_evaluated_cutoff"]
    write_csv(args.output / "failures.csv", failures)
    selections = {
        "candidate_misses.jsonl": [r for r in result_rows if "CANDIDATE_MISS" in r["error_types"]],
        "reranker_misses.jsonl": [r for r in result_rows if "RERANKER_MISS" in r["error_types"]],
        "selection_misses.jsonl": [r for r in result_rows if "SELECTION_MISS" in r["error_types"]],
    }
    for filename, subset in selections.items():
        (args.output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset))
    candidate_path = args.output / "candidate_miss_analysis.csv"
    candidate_fields = ["query_id", "query", "gold_chunk", "gold_doc", "bm25_rank", "dense_rank",
                        "rrf_rank", "source_coverage", "error_class", "topic",
                        "unknown_beyond_cache_depth", "fusion_dropped", "cache_depth"]
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        for row in candidate_details:
            writer.writerow({**row, "cache_depth": json.dumps(row["cache_depth"])})
    candidate_counts = Counter(row["error_class"] for row in candidate_details)
    unknown_count = sum(row["unknown_beyond_cache_depth"] for row in candidate_details)
    candidate_summary = {
        "candidate_miss_queries": sum("CANDIDATE_MISS" in r["error_types"] for r in result_rows),
        "gold_positive_rows": len(candidate_details),
        "BM25_only": candidate_counts["BM25_ONLY_HIT"], "Dense_only": candidate_counts["DENSE_ONLY_HIT"],
        "Fusion_dropped": candidate_counts["FUSION_DROPPED"], "Both_miss_within_cache": candidate_counts["BOTH_MISS"],
        "source_hits_truncated_by_fusion": sum(row["fusion_dropped"] for row in candidate_details),
        "unknown_beyond_cache_depth_positive_rows": unknown_count,
        "note": "A BOTH_MISS row means absent from the available top-depth caches; ranks beyond cache are unknown.",
    }
    (args.output / "candidate_miss_summary.json").write_text(json.dumps(candidate_summary, indent=2) + "\n")
    counts = Counter(kind for row in result_rows for kind in row["error_types"])
    summary = {"total_validation_queries": len(samples), "ranking_k": args.ranking_k,
               "candidate_depth": {s: max(len(r["results"]) for r in stages[s]) for s in STAGES},
               "query_counts_by_error_type": {key: sum(key in row["error_types"] for row in result_rows)
                                               for key in sorted(counts)},
               "primary_failure_stage_counts": dict(Counter(r["primary_failure_stage"] for r in result_rows)),
               "overlapping_error_types": True,
               "interpretation": "Ranks outside cached top depth are unknown; candidate misses mean absent from cached pool.",
               "candidate_miss_analysis": candidate_summary,
               "input_files": {s: str(args.rankings_root / s / "rankings.jsonl") for s in STAGES}}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# ViMed validation error analysis", "", f"Total validation queries: {len(samples)}.",
             f"Reranker cutoff: top {args.ranking_k}; candidate depth comes from cached hybrid results.",
             "Error types overlap; their percentages do not sum to 100%.", "",
             "| Error type | Queries | Percent |", "|---|---:|---:|"]
    for kind, count in summary["query_counts_by_error_type"].items():
        lines.append(f"| {kind} | {count} | {100*count/len(samples):.2f}% |")
    lines += ["", "## Primary failure stage", "", "| Stage | Queries |", "|---|---:|"]
    for stage, count in summary["primary_failure_stage_counts"].items():
        lines.append(f"| {stage} | {count} |")
    lines += ["", "Cached rankings are capped at top 100. An absent item has unknown rank beyond that depth.",
              "Ground truth contains one inferred source context per query and is not exhaustive.", ""]
    (args.output / "error_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
