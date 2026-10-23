"""Measure pinned A0 retrieval and reranking on the two-part GPU pilot index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.phase2_batch import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(index: Path, config_path: Path, queries_path: Path,
        sparse_report_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    baseline = json.loads(sparse_report_path.read_text(encoding="utf-8"))
    if (baseline.get("query_count") != 50 or baseline.get("query_ids") != list(range(1, 51))
            or baseline.get("official_queries_sha256") != sha256_file(queries_path)):
        raise ValueError("Pilot query list/hash differs from locked sparse benchmark")
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("parts") != 2
            or manifest.get("source_count") != 10_000
            or manifest.get("passage_count") != 12_347):
        raise ValueError("Expected complete two-part pilot index")
    config = load_config(config_path)
    if (config["dense"].get("device") != "cuda:0"
            or config["reranker"].get("device") != "cuda:0"
            or config["batch"].get("result_limit") != 200):
        raise ValueError("Pilot requires the locked single-GPU A0 configuration")
    query_rows = {int(row["id"]): str(row["query"]) for row in
                  pq.read_table(queries_path, columns=["id", "query"]).to_pylist()}
    if any(qid not in query_rows for qid in baseline["query_ids"]):
        raise ValueError("Missing locked pilot query")

    started = time.perf_counter()
    pipeline = DiskPipeline(index, config)
    load_seconds = time.perf_counter() - started
    dense_device = str(next(pipeline.dense.encoder.model.parameters()).device)
    reranker_device = str(next(pipeline.reranker.model.model.parameters()).device)
    if dense_device != "cuda:0" or reranker_device != "cuda:0":
        raise ValueError(f"Models loaded on {dense_device} and {reranker_device}")
    torch.cuda.reset_peak_memory_stats(0)

    rows = []
    for position, qid in enumerate(baseline["query_ids"], 1):
        query = query_rows[qid]
        begin = time.perf_counter()
        selected, diagnostics = pipeline.retrieve_with_diagnostics(query, 200)
        elapsed = time.perf_counter() - begin
        pair_count = int(diagnostics["rerank_pairs"])
        if pair_count > 200 or len(selected) > 200:
            raise ValueError(f"A0 budget exceeded at query {qid}")
        selected_sha = hashlib.sha256(json.dumps(selected, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        token_info = diagnostics.get("reranker_tokens") or {}
        rows.append({"query_id": qid, "latency_seconds": elapsed,
                     "selected_count": len(selected), "selected_sha256": selected_sha,
                     "document_hits": diagnostics["document_hits"],
                     "passage_bm25_hits": diagnostics["passage_bm25_hits"],
                     "dense_passage_hits": diagnostics["dense_passage_hits"],
                     "fused_items": diagnostics["fused_items"],
                     "rerank_pairs": pair_count,
                     "truncated_pairs": token_info.get("truncated_pairs"),
                     "maximum_pair_tokens": token_info.get("maximum_observed_length")})
        print(f"pilot A0 {position}/50: {elapsed:.3f}s, {pair_count} pairs", flush=True)

    durations = sorted(row["latency_seconds"] for row in rows)
    report = {"format": "medrag-v4-task4-gpu-pilot-a0-v1", "status": "complete",
              "scope": "two_release_parts_50_locked_queries",
              "index_manifest_sha256": sha256_file(index / "manifest.json"),
              "config_sha256": sha256_file(config_path),
              "official_queries_sha256": baseline["official_queries_sha256"],
              "query_ids": baseline["query_ids"], "query_count": len(rows),
              "result_limit": 200, "dense_device": dense_device,
              "reranker_device": reranker_device, "load_seconds": load_seconds,
              "latency_seconds": {"mean": statistics.mean(durations),
                                  "p50": durations[24], "p95": durations[46],
                                  "max": durations[-1]},
              "rerank_pairs_total": sum(row["rerank_pairs"] for row in rows),
              "truncated_pairs_total": sum(row["truncated_pairs"] or 0 for row in rows),
              "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
              "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
              "rss_peak_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "queries": rows}
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--sparse-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.index.resolve(), args.config.resolve(), args.queries.resolve(),
                 args.sparse_report.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
