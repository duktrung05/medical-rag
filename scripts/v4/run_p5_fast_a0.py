"""Versioned A0 CPU pair runner using the exact ParentFirstDiskBM25 fallback."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import CONFIG, ROOT, digest, inputs
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.pair_lineage import (PairLineageResolver, canonical_bytes,
                                          write_atomic_replace, write_once)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.instrumented_dense import InstrumentedDiskDense
from src.medrag.retrieval.parent_first_bm25 import ParentFirstDiskBM25
from src.medrag.retrieval.stage_timing import prepare_rerank_with_stage_timing


def verify_runtime(target: Path):
    config, frozen, queries, index, _ = inputs(CONFIG, "full")
    resolver = PairLineageResolver(target)
    lock = resolver.lock
    if lock.get("output_namespace") != str(target.resolve()):
        raise ValueError("Optimized output namespace differs from lock")
    for field in ("config_sha256", "index_manifest_sha256", "official_queries_sha256",
                  "official_corpus_sha256", "score_contract", "query_ids"):
        if lock.get(field) != frozen.get(field):
            raise ValueError(f"Optimized lock differs from frozen A0 contract: {field}")
    for name, expected in lock.get("implementation_sha256", {}).items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Optimized implementation lock differs: {name}")
    return config, queries, index, resolver


def build_pipeline(config: dict, index: Path):
    runtime = copy.deepcopy(config)
    runtime["dense"]["device"] = "cpu"
    pipeline = DiskPipeline(index, runtime, reranker=object())
    fingerprint = pipeline.passage_bm25.fingerprint
    pipeline.passage_bm25 = ParentFirstDiskBM25(
        index / "passages.sqlite", "passage_bm25", fingerprint
    )
    if pipeline.dense is not None:
        old_dense = pipeline.dense
        pipeline.dense = InstrumentedDiskDense(
            index / "dense", old_dense.encoder, old_dense.fingerprint,
            block_rows=old_dense.block_rows,
        )
    return pipeline


def _pair_shard(qid: int, query: str, candidates: list[dict], pairs: list[tuple],
                diagnostics: dict, resolver: PairLineageResolver) -> dict:
    cache = resolver.local_cache
    pair_rows = [{
        "official_id": doc,
        "passage_id": pid,
        "payload": payload,
        "input_sha256": digest(payload),
        "cache_key": cache._key(query, doc, pid, payload),
    } for doc, pid, payload in pairs]
    return {
        "format": "medrag-v4-p5-2-pairs-v1",
        "id": qid,
        "lock_sha256": sha256_file(resolver.lock_path),
        "query": query,
        "query_sha256": digest(query),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "diagnostics": diagnostics,
        "pairs": pair_rows,
    }


def prepare(target: Path, *, max_queries: int | None = None) -> dict:
    config, queries, index, resolver = verify_runtime(target)
    query_by_id = {int(row["id"]): str(row["query"]) for row in queries}
    pending = []
    for qid in resolver.lock["query_ids"]:
        path, _ = resolver.resolve(qid)
        if path.exists():
            resolver.validate(qid, query_by_id[qid])
        else:
            pending.append(qid)
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        pending = pending[:max_queries]
    # Fresh migration namespaces contain lock/lineage, but no local pairs yet.
    # Create the publication directory before expensive model/retrieval work.
    if pending:
        (target / "pairs").mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(config, index) if pending else None
    prepared = 0
    for qid in pending:
        query = query_by_id[qid]
        candidates, pairs, diagnostics, timing = prepare_rerank_with_stage_timing(
            pipeline, query, 200
        )
        if len(pairs) != 200:
            raise ValueError(f"Frozen pair budget differs for {qid}: {len(pairs)}")
        serialization_started = time.perf_counter()
        shard = _pair_shard(qid, query, candidates, pairs, diagnostics, resolver)
        payload = canonical_bytes(shard)
        _write_bytes_checked(target / "pairs" / f"{qid}.json", payload)
        timing["serialization_seconds"] = time.perf_counter() - serialization_started
        timing_record = {
            "format": "medrag-v4-p5-fast-a0-stage-timing-v1",
            "id": qid,
            "pair_shard_sha256": sha256_file(target / "pairs" / f"{qid}.json"),
            "stages": timing,
        }
        write_once(target / "timings" / f"{qid}.json", timing_record)
        resolver.validate(qid, query)
        prepared += 1
        print(f"prepared {qid}: 200 immutable pairs; parent fallbacks "
              f"{int(timing['parent_fallback_calls'])}", flush=True)
    merged = resolver.build_merged_manifest(query_by_id, require_complete=False)
    write_atomic_replace(target / "merged_pair_manifest.json", merged)
    return {
        "prepared_now": prepared,
        "inherited_shards": merged["inherited_shards"],
        "local_shards": merged["local_shards"],
        "missing_queries": len(merged["missing_query_ids"]),
        "complete": merged["complete"],
    }


def validate(target: Path, *, require_complete: bool) -> dict:
    _, queries, _, resolver = verify_runtime(target)
    query_by_id = {int(row["id"]): str(row["query"]) for row in queries}
    merged = resolver.build_merged_manifest(query_by_id, require_complete=require_complete)
    write_atomic_replace(target / "merged_pair_manifest.json", merged)
    return {
        "complete": merged["complete"],
        "inherited_shards": merged["inherited_shards"],
        "local_shards": merged["local_shards"],
        "missing_queries": len(merged["missing_query_ids"]),
        "pair_count": sum(item["pair_count"] for item in merged["pair_shards"].values()),
    }


def rollback_check(target: Path) -> dict:
    _, queries, _, resolver = verify_runtime(target)
    query_by_id = {int(row["id"]): str(row["query"]) for row in queries}
    inherited = resolver.lineage["inherited_query_ids"]
    for qid in inherited:
        resolver.validate(qid, query_by_id[qid])
    return {
        "rollback_ready": True,
        "source_namespace": str(resolver.source),
        "validated_source_shards": len(inherited),
        "resume_from_query_id": max(inherited) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "rollback-check"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--max-queries", type=int)
        if name == "validate":
            command.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, max_queries=args.max_queries)
    elif args.command == "validate":
        result = validate(args.output, require_complete=not args.allow_incomplete)
    else:
        result = rollback_check(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
