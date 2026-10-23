"""Measure lexical query latency and candidate coverage on a release-part pilot."""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.phase2_batch import sha256_file


def run(index: Path, query_path: Path, output: Path, count: int = 50) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = {"selection": {"max_docs": 200, "max_chunks": 60, "chunks_per_doc": 1}}
    pipeline = DiskPipeline(index, config)
    rows = sorted(pq.read_table(query_path, columns=["id", "query"]).to_pylist(),
                  key=lambda row: int(row["id"]))[:count]
    if len(rows) != count:
        raise ValueError("Pilot query count is unavailable")
    durations = []
    diagnostics = []
    for row in rows:
        start = time.perf_counter()
        selected, trace = pipeline.retrieve_with_diagnostics(str(row["query"]), 10)
        durations.append(time.perf_counter() - start)
        diagnostics.append({"id": int(row["id"]), "selected": len(selected),
                            "document_hits": trace["document_hits"],
                            "passage_hits": trace["passage_bm25_hits"]})
    ordered = sorted(durations)
    report = {"status": "complete", "scope": "lexical_release_parts_pilot",
              "query_count": count, "query_ids": [int(row["id"]) for row in rows],
              "index_manifest_sha256": sha256_file(index / "manifest.json"),
              "official_queries_sha256": sha256_file(query_path),
              "latency_seconds": {"mean": statistics.mean(durations),
                                  "p50": ordered[int(.5 * (len(ordered) - 1))],
                                  "p95": ordered[int(.95 * (len(ordered) - 1))],
                                  "max": max(durations)},
              "rss_peak_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "diagnostics": diagnostics}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.rename(temporary, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    report = run(args.index.resolve(), args.queries.resolve(), args.output.resolve(), args.count)
    print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}, indent=2))
