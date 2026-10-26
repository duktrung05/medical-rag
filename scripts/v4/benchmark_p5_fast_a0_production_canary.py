"""Post-stop, bounded production-index canary; never run beside the old A0 runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

from scripts.v4.run_p5_2_split import (CONFIG, ROOT, ScoreCache, digest, inputs,
                                       read)
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.pair_lineage import (canonical_bytes, require_process_stopped,
                                          write_once)
from src.medrag.retrieval.instrumented_dense import InstrumentedDiskDense
from src.medrag.retrieval.parent_first_bm25 import ParentFirstDiskBM25
from src.medrag.retrieval.stage_timing import prepare_rerank_with_stage_timing


SOURCE = ROOT / "outputs/v4/a0_t4_v1_full_20261022"


def hit_rows(hits):
    return [(hit.official_id, hit.passage_id, hit.score.hex(), hit.channel, hit.rank)
            for hit in hits]


class RecordingSearch:
    def __init__(self, backend):
        self.backend = backend
        self.global_hits = []
        self.parent_calls = []

    @property
    def last_timing(self):
        return self.backend.last_timing

    def search(self, query: str, limit: int, **kwargs):
        hits = self.backend.search(query, limit, **kwargs)
        self.global_hits.append(hit_rows(hits))
        return hits

    def search_parent(self, query: str, parent_id: int, limit: int = 1):
        hits = self.backend.search_parent(query, parent_id, limit)
        self.parent_calls.append({"parent_id": parent_id, "limit": limit,
                                  "hits": hit_rows(hits)})
        return hits


def timed(active, qid: int, query: str, lock_sha: str, cache: ScoreCache):
    candidates, pairs, diagnostics, timing = prepare_rerank_with_stage_timing(active, query, 200)
    started = time.perf_counter()
    pair_rows = [{"official_id": doc, "passage_id": pid, "payload": payload,
                  "input_sha256": digest(payload),
                  "cache_key": cache._key(query, doc, pid, payload)}
                 for doc, pid, payload in pairs]
    shard = {"format": "medrag-v4-p5-2-pairs-v1", "id": qid,
             "lock_sha256": lock_sha, "query": query, "query_sha256": digest(query),
             "candidate_count": len(candidates), "candidates": candidates,
             "diagnostics": diagnostics, "pairs": pair_rows}
    encoded = canonical_bytes(shard)
    timing["serialization_seconds"] = time.perf_counter() - started
    timing["actual_query_latency_seconds"] = (timing["prepare_total_seconds"]
                                               + timing["serialization_seconds"])
    return (candidates, pairs, diagnostics), shard, timing, hashlib.sha256(encoded).hexdigest()


def run(output: Path, source_pid: int, query_ids: list[int], handoff_token: str) -> dict:
    if handoff_token != "GO_STOPPED_P5_CPU":
        raise ValueError("Production canary requires explicit GO_STOPPED_P5_CPU handoff token")
    require_process_stopped(source_pid)
    if output.exists():
        raise FileExistsError(output)
    if not query_ids or len(query_ids) > 3 or len(set(query_ids)) != len(query_ids):
        raise ValueError("Production canary accepts 1..3 distinct query IDs")
    run_started = time.perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    config, _, queries, index, _ = inputs(CONFIG, "full")
    query_by_id = {int(row["id"]): str(row["query"]) for row in queries}
    if any(qid not in query_by_id for qid in query_ids):
        raise ValueError("Production canary query ID outside 1..1200")
    output.mkdir(parents=True)

    source_lock = read(SOURCE / "lock.json")
    lock_sha = hashlib.sha256((SOURCE / "lock.json").read_bytes()).hexdigest()
    cache = ScoreCache(SOURCE / "key_contract.sqlite", source_lock["score_contract"])
    runtime = copy.deepcopy(config)
    runtime["dense"]["device"] = "cpu"
    initialization_started = time.perf_counter()
    old = DiskPipeline(index, runtime, reranker=object())
    if old.dense is not None:
        dense = old.dense
        old.dense = InstrumentedDiskDense(index / "dense", dense.encoder, dense.fingerprint,
                                          block_rows=dense.block_rows)
    initialization_seconds = time.perf_counter() - initialization_started
    new = copy.copy(old)
    new.passage_bm25 = ParentFirstDiskBM25(
        index / "passages.sqlite", old.passage_bm25.channel, old.passage_bm25.fingerprint
    )

    base_doc, base_old_passage, base_dense = old.doc_bm25, old.passage_bm25, old.dense
    base_new_passage = new.passage_bm25
    checks = []
    for qid in query_ids:
        query = query_by_id[qid]
        old_doc, new_doc = RecordingSearch(base_doc), RecordingSearch(base_doc)
        old_passage, new_passage = (RecordingSearch(base_old_passage),
                                    RecordingSearch(base_new_passage))
        old_dense = RecordingSearch(base_dense) if base_dense is not None else None
        new_dense = RecordingSearch(base_dense) if base_dense is not None else None
        old.doc_bm25, old.passage_bm25, old.dense = old_doc, old_passage, old_dense
        new.doc_bm25, new.passage_bm25, new.dense = new_doc, new_passage, new_dense
        before, old_shard, old_timing, old_sha = timed(old, qid, query, lock_sha, cache)
        after, new_shard, new_timing, new_sha = timed(new, qid, query, lock_sha, cache)
        if canonical_bytes(before) != canonical_bytes(after):
            raise ValueError(f"STOP: production candidate/pair/numerical parity failed: {qid}")
        branch_parity = {
            "ordered_document_bm25_hits": old_doc.global_hits == new_doc.global_hits,
            "ordered_passage_bm25_hits": old_passage.global_hits == new_passage.global_hits,
            "ordered_dense_hits": (old_dense.global_hits == new_dense.global_hits
                                    if old_dense is not None else True),
            "parent_fallback_ids_and_raw_scores": (old_passage.parent_calls
                                                    == new_passage.parent_calls),
            "fused_candidates": before[0] == after[0],
            "ordered_pairs": before[1] == after[1],
            "final_pair_shard_payload": canonical_bytes(old_shard)
                                        == canonical_bytes(new_shard),
            "query_hash": old_shard["query_sha256"] == new_shard["query_sha256"],
            "pair_input_hashes": ([row["input_sha256"] for row in old_shard["pairs"]]
                                  == [row["input_sha256"] for row in new_shard["pairs"]]),
            "cache_keys": ([row["cache_key"] for row in old_shard["pairs"]]
                           == [row["cache_key"] for row in new_shard["pairs"]]),
        }
        if not all(branch_parity.values()):
            raise ValueError(f"STOP: production branch/final-shard parity failed: {qid}")
        historical_path = SOURCE / "pairs" / f"{qid}.json"
        historical_sha = (hashlib.sha256(historical_path.read_bytes()).hexdigest()
                          if historical_path.exists() else None)
        checks.append({"id": qid, "query_sha256": old_shard["query_sha256"],
                       "ordered_output_sha256": old_sha,
                       "optimized_output_sha256": new_sha,
                       "historical_pair_shard_sha256": historical_sha,
                       "historical_final_pair_payload_exact": (historical_sha == old_sha
                                                                if historical_sha else None),
                       "branch_parity": branch_parity,
                       "fallback_calls": len(old_passage.parent_calls),
                       "fallback_trace_sha256": hashlib.sha256(
                           canonical_bytes(old_passage.parent_calls)).hexdigest(),
                       "same_environment_exact_parity": "PASS",
                       "old": old_timing, "optimized": new_timing,
                       "speedup": (old_timing["actual_query_latency_seconds"]
                                   / new_timing["actual_query_latency_seconds"])})
        write_once(output / f"query_{qid}.json", checks[-1])
    usage_finished = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "format": "medrag-v4-p5-fast-a0-production-canary-v1",
        "query_ids": query_ids,
        "gates": {"same_environment_algorithm_and_downstream_parity": "PASS",
                  "historical_raw_score_reproducibility": "FAIL",
                  "historical_labels_used_for_selection": False},
        "median_end_to_end_speedup": statistics.median(row["speedup"] for row in checks),
        "model_index_initialization_seconds": initialization_seconds,
        "process_usage": {"user_cpu_seconds": usage_finished.ru_utime-usage_started.ru_utime,
                          "system_cpu_seconds": usage_finished.ru_stime-usage_started.ru_stime,
                          "peak_rss_kib": usage_finished.ru_maxrss,
                          "wall_seconds": time.perf_counter()-run_started},
        "checks": checks,
        "source_pid_confirmed_stopped": source_pid,
    }
    write_once(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pid", type=int, required=True)
    parser.add_argument("--query-id", type=int, action="append", required=True)
    parser.add_argument("--handoff-token", required=True)
    args = parser.parse_args()
    report = run(args.output.resolve(), args.source_pid, args.query_id, args.handoff_token)
    print(json.dumps({"report": str(args.output / "report.json"),
                      "speedup": report["median_end_to_end_speedup"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
