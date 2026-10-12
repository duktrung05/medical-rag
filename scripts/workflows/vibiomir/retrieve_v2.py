"""Run three V2 branches and save per-query provenance for a bounded pilot."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

import faiss
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from src.retrieval_v2.core import dense_search, fuse, sparse_search
from src.retrieval_v2.settings import DEFAULT_CONFIG, cached_model_path, load_config

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--pilot", type=Path, default=ROOT / "outputs/retrieval_v2/pilot")
    ap.add_argument("--queries", type=int)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N; GPU requires prior resource approval")
    args = ap.parse_args()
    config = load_config(args.config)
    args.queries = args.queries or config["corpus"]["pilot_query_count"]
    args.budget = args.budget or config["retrieval"]["candidate_budget"]
    if not 1 <= args.queries <= 50 or not 1 <= args.budget <= 400:
        raise ValueError("bounded pilot requires queries <= 50 and budget <= 400")
    rows = pq.read_table(ROOT / "data/raw/vibiomir/query.parquet").slice(0, args.queries).to_pylist()
    con = sqlite3.connect(args.pilot / "v2_sparse.sqlite")
    index = faiss.read_index(str(args.pilot / "v2_dense.faiss"))
    model = SentenceTransformer(str(cached_model_path(config["dense"]["model"],
                                                       config["dense"]["revision"])),
                                device=args.device, local_files_only=True)
    vectors = model.encode([r["query"] for r in rows], batch_size=4,
                           normalize_embeddings=True, show_progress_bar=False)
    output = args.pilot / "V2_CANDIDATE_TRACE.jsonl"
    if output.exists():
        raise FileExistsError(output)
    benchmark = []
    flat_trace = []
    result_rows = []
    with output.open("w", encoding="utf-8") as out:
        for row, vector in zip(rows, vectors, strict=True):
            start = time.perf_counter()
            retrieval = config["retrieval"]
            doc_hits = sparse_search(con, row["query"], "document", retrieval["document_bm25_top_k"])
            passage_hits = sparse_search(con, row["query"], "passage", retrieval["passage_bm25_top_k"])
            dense_hits = dense_search(args.pilot / "v2_dense.faiss", vector,
                                      retrieval["passage_dense_top_k"], index)
            fusion_args = {"doc_k": retrieval["document_bm25_top_k"],
                           "passage_k": retrieval["passage_bm25_top_k"],
                           "budget": args.budget, "rrf_k": retrieval["rrf_k"]}
            global_pool = fuse(con, row["query"], dense_hits, diverse=False, **fusion_args)
            pool = fuse(con, row["query"], dense_hits, diverse=True, **fusion_args)
            elapsed = time.perf_counter() - start
            trace = {"id": row["id"], "query": row["query"], "candidates": pool,
                     "branch_counts": {"document_bm25": len(doc_hits),
                                       "passage_bm25": len(passage_hits),
                                       "passage_dense": len(dense_hits)},
                     "global_unique_docs": len({x["doc_id"] for x in global_pool}),
                     "diverse_unique_docs": len({x["doc_id"] for x in pool}),
                     "latency_seconds": elapsed}
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            result_rows.append({"id": row["id"], "passages": [
                {"passage_id": x["passage_id"], "doc_id": x["doc_id"],
                 "fusion_rank": x["fusion_rank"], "retrieval_sources": list(x["sources"])}
                for x in pool]})
            benchmark.append({k: trace[k] for k in ("id", "branch_counts", "global_unique_docs",
                                                   "diverse_unique_docs", "latency_seconds")})
            for item in pool:
                flat_trace.append({"query_id": row["id"], "passage_id": item["passage_id"],
                                   "doc_id": item["doc_id"], "source_chunk_index": item["source_chunk_index"],
                                   "start_char": item["start_char"], "end_char": item["end_char"],
                                   "fusion_rank": item["fusion_rank"], "fusion_score": item["fusion_score"],
                                   "document_bm25_rank": item["sources"].get("document_bm25"),
                                   "passage_bm25_rank": item["sources"].get("passage_bm25"),
                                   "passage_dense_rank": item["sources"].get("passage_dense"),
                                   "retrieval_sources": ",".join(item["sources"])})
    con.close()
    pq.write_table(pa.Table.from_pylist(flat_trace), args.pilot / "V2_CANDIDATE_TRACE.parquet")
    (args.pilot / "V2_RETRIEVAL_RESULTS.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in result_rows), encoding="utf-8")
    (args.pilot / "V2_RETRIEVAL_BENCHMARK.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    with (args.pilot / "V2_RETRIEVAL_ABLATION.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "global_top_unique_docs", "diverse_top_unique_docs",
                                                 "latency_seconds"])
        writer.writeheader()
        for item in benchmark:
            writer.writerow({"query_id": item["id"], "global_top_unique_docs": item["global_unique_docs"],
                             "diverse_top_unique_docs": item["diverse_unique_docs"],
                             "latency_seconds": item["latency_seconds"]})
    with (args.pilot / "V2_RETRIEVAL_BENCHMARK.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "document_bm25_hits", "passage_bm25_hits",
                                                 "passage_dense_hits", "latency_seconds"])
        writer.writeheader()
        for item in benchmark:
            writer.writerow({"query_id": item["id"],
                             "document_bm25_hits": item["branch_counts"]["document_bm25"],
                             "passage_bm25_hits": item["branch_counts"]["passage_bm25"],
                             "passage_dense_hits": item["branch_counts"]["passage_dense"],
                             "latency_seconds": item["latency_seconds"]})
    print(f"Retrieved {len(rows)} queries to {output}")


if __name__ == "__main__":
    main()
