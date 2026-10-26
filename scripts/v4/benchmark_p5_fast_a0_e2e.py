"""Bounded synthetic end-to-end canary for old/new A0 fallback backends."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sqlite3
import statistics
import time
from pathlib import Path

import numpy as np

from scripts.v4.benchmark_parent_first_bm25 import OLD_SQL
from scripts.v4.run_p5_2_split import ROOT
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.data.pair_lineage import canonical_bytes, write_once
from src.medrag.evidence.selector import SelectionPolicy
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.disk_dense import DiskDense, input_sha256
from src.medrag.retrieval.instrumented_dense import InstrumentedDiskDense
from src.medrag.retrieval.parent_first_bm25 import PARENT_SQL, ParentFirstDiskBM25
from src.medrag.retrieval.stage_timing import prepare_rerank_with_stage_timing


class FixtureEncoder:
    def encode_query(self, texts):
        return [[1.0, 0.0] for _ in texts]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(np.ceil(len(ordered) * fraction)) - 1)]


def resource_caps() -> dict:
    os.nice(19)
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {max(allowed)})
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    return {"nice": os.getpriority(os.PRIO_PROCESS, 0),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cpu_seconds_limit": 45, "file_mib_limit": 128}


def build_fixture(root: Path, rows: int):
    body = "common fever treatment dose mg điều trị 高血压"
    sources, passages = [], []
    for qid in range(1, rows + 1):
        # IDs 1..1000 remain query-relevant but receive a lower passage score;
        # IDs 1001+ fill the passage+dense branches. The document branch is
        # deliberately independent so its first 200 fused parents need fallback.
        text = body + (" " + "padding " * 40 if qid <= 1000 else "")
        source = Source("canary", qid, f"https://canary/{qid}", f"Title {qid}",
                        text, "synthetic", "raw")
        sources.append(source)
        passages.append(Passage(f"{qid}:p", "canary", qid, source.url, text, 0,
                                len(text), source.text_sha256, "mixed", "synthetic"))
    doc = DiskBM25(root / "documents.sqlite", "document_bm25", "canary", create=True)
    doc.append_batch((s.official_id, None, body if s.official_id <= 400 else "unrelated")
                     for s in sources)
    doc.complete()
    passage = DiskBM25(root / "passages.sqlite", "passage_bm25", "canary", create=True)
    passage.append_batch((p.official_id, p.passage_id, p.text) for p in passages)
    passage.complete()
    dense = DiskDense(root / "dense", FixtureEncoder(), "canary", create=True,
                      expected_rows=rows, dimension=2, block_rows=512)
    batch = 500
    for start in range(0, rows, batch):
        selected = passages[start:start + batch]
        records = [(p.official_id, p.passage_id, input_sha256(p.text, f"Title {p.official_id}"))
                   for p in selected]
        vectors = np.zeros((len(selected), 2), dtype=np.float32)
        for offset, passage_row in enumerate(selected):
            vectors[offset, 0 if passage_row.official_id > 1000 else 1] = 1.0
        dense.append_batch(records, vectors)
    dense.complete()
    return sources, passages, doc, passage


def pipeline(sources, passages, doc, passage, dense_path: Path, optimized: bool):
    value = Pipeline.__new__(Pipeline)
    value.corpus_version = "canary"
    value.sources = {s.official_id: s for s in sources}
    value.passages = {p.passage_id: p for p in passages}
    value.doc_bm25 = doc
    value.passage_bm25 = (ParentFirstDiskBM25(passage.path, passage.channel, passage.fingerprint)
                          if optimized else passage)
    value.dense = InstrumentedDiskDense(dense_path, FixtureEncoder(), "canary", block_rows=512)
    value.reranker = object()
    value.score_cache = None
    value.document_candidate_limit = 400
    value.passage_candidate_limit = 200
    value.fused_candidate_limit = 400
    value.rrf_k = 60
    value.rerank_candidate_limit = 200
    value.rerank_passages_per_document = 3
    value.rerank_allocation = "parent_first"
    value.selection_policy = SelectionPolicy(max_docs=200, max_chunks=60, chunks_per_doc=1,
                                              document_selection="independent",
                                              chunk_context="adjacent")
    return value


def measured_call(active, query: str, serialization_path: Path):
    candidates, pairs, diagnostics, timing = prepare_rerank_with_stage_timing(active, query, 200)
    started = time.perf_counter()
    payload = canonical_bytes({"candidates": candidates, "pairs": pairs,
                               "diagnostics": diagnostics})
    serialization_path.write_bytes(payload)
    timing["serialization_seconds"] = time.perf_counter() - started
    timing["actual_query_latency_seconds"] = (
        timing["prepare_total_seconds"] + timing["serialization_seconds"]
    )
    return (candidates, pairs, diagnostics), timing


def summarize(records: list[dict]) -> dict:
    names = sorted(records[0])
    result = {}
    for name in names:
        values = [float(row[name]) for row in records]
        suffix = "count" if name.endswith("_calls") else "seconds"
        result[name] = {f"median_{suffix}": statistics.median(values),
                        f"p95_{suffix}": percentile(values, .95)}
    return result


def benchmark(output: Path, rows: int, repeats: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 1000 <= rows <= 5000 or not 2 <= repeats <= 5:
        raise ValueError("Canary bounds are rows=1000..5000 and repeats=2..5")
    output.mkdir(parents=True)
    caps = resource_caps()
    started = time.perf_counter()
    sources, passages, doc, old_passage = build_fixture(output / "fixture", rows)
    old = pipeline(sources, passages, doc, old_passage, output / "fixture/dense", False)
    new = pipeline(sources, passages, doc, old_passage, output / "fixture/dense", True)
    query = "common fever treatment dose mg điều trị 高血压"

    old_result, _ = measured_call(old, query, output / "old_warmup.json")
    new_result, _ = measured_call(new, query, output / "new_warmup.json")
    if canonical_bytes(old_result) != canonical_bytes(new_result):
        raise ValueError("STOP: bounded end-to-end candidate/pair parity failed")
    exact_raw = []
    for parent in range(201, 401):
        before = old_passage.search_parent(query, parent, 1)
        after = new.passage_bm25.search_parent(query, parent, 1)
        row = [(h.official_id, h.passage_id, h.score.hex(), h.rank) for h in before]
        if row != [(h.official_id, h.passage_id, h.score.hex(), h.rank) for h in after]:
            raise ValueError(f"STOP: raw fallback parity failed for parent {parent}")
        exact_raw.append(row)

    old_timings, new_timings = [], []
    for attempt in range(repeats):
        old_output, old_timing = measured_call(old, query, output / f"old_{attempt}.json")
        new_output, new_timing = measured_call(new, query, output / f"new_{attempt}.json")
        if canonical_bytes(old_output) != canonical_bytes(new_output):
            raise ValueError(f"STOP: repeated end-to-end parity failed: {attempt}")
        old_timings.append(old_timing)
        new_timings.append(new_timing)
    old_summary, new_summary = summarize(old_timings), summarize(new_timings)
    old_latency = old_summary["actual_query_latency_seconds"]["median_seconds"]
    new_latency = new_summary["actual_query_latency_seconds"]["median_seconds"]
    with sqlite3.connect(old_passage.path) as db:
        plans = {
            "old": [list(row) for row in db.execute("EXPLAIN QUERY PLAN " + OLD_SQL,
                                                     ("common", 201))],
            "new": [list(row) for row in db.execute("EXPLAIN QUERY PLAN " + PARENT_SQL,
                                                     (201, "common"))],
        }
    report = {
        "format": "medrag-v4-p5-fast-a0-bounded-e2e-canary-v1",
        "scope": "synthetic bounded disk retrieval; no production full-index retrieval",
        "rows": rows, "repeats": repeats, "query_count": 1,
        "fallback_calls_per_query": int(old_timings[0]["parent_fallback_calls"]),
        "gates": {
            "same_environment_raw_score_parity": "PASS",
            "ordered_candidate_parity": "PASS",
            "ordered_pair_and_input_hash_parity": "PASS",
            "historical_raw_score_reproducibility": "FAIL",
            "production_end_to_end_speedup": "UNMEASURED",
        },
        "old": old_summary, "optimized": new_summary,
        "measured_synthetic_end_to_end_speedup": old_latency / new_latency,
        "measured_synthetic_seconds_saved_per_query": old_latency - new_latency,
        "ordered_output_sha256": hashlib.sha256(canonical_bytes(old_result)).hexdigest(),
        "raw_fallback_hex_sha256": hashlib.sha256(canonical_bytes(exact_raw)).hexdigest(),
        "sql_query_plan": plans,
        "resource_limits": caps,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_seconds": time.perf_counter() - started,
    }
    if report["fallback_calls_per_query"] < 100:
        raise ValueError("STOP: canary did not exercise enough parent fallbacks")
    write_once(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = benchmark(args.output.resolve(), args.rows, args.repeats)
    print(json.dumps({"report": str(args.output / "report.json"),
                      "speedup": report["measured_synthetic_end_to_end_speedup"],
                      "fallback_calls": report["fallback_calls_per_query"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
