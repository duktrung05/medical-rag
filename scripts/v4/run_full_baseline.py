"""Run and strictly export the pinned 1,200-query V4 full-corpus A0 baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.medrag.adapters.validation import (canonical_payload, package_zip,
                                            validate_selected, validate_zip)
from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.ranking.score_cache import ScoreCache


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def _write_bytes_checked(path: Path, payload: bytes) -> None:
    """Publish one artifact atomically, refusing a divergent resumed result."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing artifact differs from replay: {path}")
        return
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def _all_official_ids(path: Path) -> set[int]:
    ids = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["id"], batch_size=100_000):
        for value in batch.column(0).to_pylist():
            ident = int(value)
            if ident in ids:
                raise ValueError(f"Duplicate official corpus ID: {ident}")
            ids.add(ident)
    return ids


def run(root: Path, index: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    batch_config = config["batch"]
    result_limit = int(batch_config["result_limit"])
    if result_limit != 200 or int(batch_config["query_count"]) != 1200:
        raise ValueError("A0 full baseline requires explicit 200 result limit and 1200 queries")
    index_manifest_path = index / "manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text())
    if index_manifest.get("scope") != "full_release" or index_manifest.get("parts") != 922:
        raise ValueError("Full baseline requires a complete 922-part V4 index")
    if (index_manifest.get("source_count") != 1_073_525
            or index_manifest.get("passage_count") != 2_951_224):
        raise ValueError("Full index eligible coverage differs from release lock")
    if not isinstance(index_manifest.get("dense"), dict):
        raise ValueError("Full index has no completed dense stage")
    migration = json.loads((root / "configs/vibiomir/phase2_batch_migration_v1.json").read_text())
    query_path, corpus_path = root / migration["official_queries"], root / migration["official_corpus"]
    if (sha256_file(query_path) != migration["official_queries_sha256"]
            or sha256_file(corpus_path) != migration["official_corpus_sha256"]):
        raise ValueError("Official corpus/query fingerprints differ")
    queries = sorted(pq.read_table(query_path, columns=["id", "query"]).to_pylist(),
                     key=lambda row: int(row["id"]))
    query_ids = [int(row["id"]) for row in queries]
    if len(query_ids) != 1200 or len(set(query_ids)) != 1200:
        raise ValueError("Official query coverage is not 1200 unique IDs")
    contract = {"format": "medrag-v4-a0-pair-score-v1",
                "index_manifest_sha256": sha256_file(index_manifest_path),
                "corpus_version": config["corpus_version"],
                "model": config["reranker"]["model"],
                "revision": config["reranker"]["revision"],
                "max_length": int(config["reranker"]["max_length"]),
                "precision": "fp16", "pair_format": "query_title_newline_passage-v1"}
    lock = {"format": "medrag-v4-full-run-lock-v1", "config_sha256": sha256_file(config_path),
            "index_manifest_sha256": sha256_file(index_manifest_path),
            "official_queries_sha256": migration["official_queries_sha256"],
            "official_corpus_sha256": migration["official_corpus_sha256"],
            "result_limit": result_limit, "query_ids": query_ids,
            "pair_cache_contract": contract,
            "implementation_sha256": {name: sha256_file(root / name) for name in (
                "scripts/v4/run_full_baseline.py", "src/medrag/core/pipeline.py",
                "src/medrag/core/disk_pipeline.py", "src/medrag/retrieval/disk_dense.py",
                "src/medrag/retrieval/disk_bm25.py", "src/medrag/evidence/selector.py",
                "src/medrag/evidence/context.py",
                "src/medrag/adapters/validation.py", "src/medrag/adapters/vibiomir_submission.py",
                "src/medrag/ranking/score_cache.py")}}
    staging = output.with_name(output.name + ".building")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        if json.loads((staging / "lock.json").read_text()) != lock:
            raise ValueError("Full run checkpoint lock mismatch")
        cache = ScoreCache(staging / "scores.sqlite", contract)
    else:
        staging.mkdir()
        (staging / "queries").mkdir()
        _write_json(staging / "lock.json", lock)
        cache = ScoreCache(staging / "scores.sqlite", contract, create=True)
    pipeline = DiskPipeline(index, config, score_cache=cache)
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    if pipeline.dense is None or pipeline.reranker is None:
        raise ValueError("Full inference stages are not ready")
    policy = SelectionPolicy(**config["selection"])
    failures = []
    for position, row in enumerate(queries, 1):
        qid, query = int(row["id"]), str(row["query"])
        query_file = staging / "queries" / f"{qid}.json"
        query_sha = hashlib.sha256(query.encode()).hexdigest()
        if query_file.exists():
            prior = json.loads(query_file.read_text())
            if prior.get("id") != qid or prior.get("query_sha256") != query_sha:
                raise ValueError(f"Query checkpoint identity mismatch: {qid}")
            if prior.get("status") == "complete":
                continue
            raise ValueError(f"Query checkpoint is not complete: {qid}")
        started = time.perf_counter()
        try:
            selected, trace = pipeline.retrieve_with_diagnostics(
                query, result_limit, include_candidates=True)
            result = {"id": qid, "query_sha256": query_sha, "status": "complete",
                      "selected": selected, "trace": trace,
                      "elapsed_seconds": time.perf_counter() - started}
            _write_json(query_file, result)
            print(f"query {position}/{len(queries)} id={qid}: complete", flush=True)
        except Exception as exc:
            failures.append({"id": qid, "error": type(exc).__name__ + ": " + str(exc)})
            print(f"query {position}/{len(queries)} id={qid}: failed {exc}", flush=True)
    if failures:
        _write_json(staging / "last_failed_attempt.json", {"status": "failed", "errors": failures})
        raise RuntimeError(f"{len(failures)} queries failed; full export remains incomplete")
    selected_by_id = {}
    query_durations = []
    replay_mismatches = []
    for row in queries:
        qid = int(row["id"])
        checkpoint = json.loads((staging / "queries" / f"{qid}.json").read_text())
        if checkpoint.get("status") != "complete":
            raise ValueError(f"Query is not complete: {qid}")
        replay, _ = select_results(checkpoint["trace"]["ranked_candidates"], policy,
                                   reranker_enabled=True, limit=result_limit)
        replay, _ = apply_chunk_context(replay, policy, pipeline.sources, pipeline.passages)
        if replay != checkpoint["selected"]:
            replay_mismatches.append(qid)
        selected_by_id[qid] = checkpoint["selected"]
        query_durations.append(float(checkpoint["elapsed_seconds"]))
    if replay_mismatches:
        raise ValueError(f"CPU selection replay differs for query IDs: {replay_mismatches[:10]}")
    selected_dir = staging / "selected"
    selected_dir.mkdir(exist_ok=True)
    selected_path = selected_dir / "selected_results.jsonl"
    selected_payload = b"".join(
        (json.dumps({"id": qid, "selected": selected_by_id[qid]}, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        for qid in query_ids)
    _write_bytes_checked(selected_path, selected_payload)
    official_ids = _all_official_ids(corpus_path)
    predictions = validate_selected(query_ids, selected_by_id, store, policy,
                                    result_limit=result_limit, reranker_enabled=True,
                                    official_ids=official_ids)
    payload = canonical_payload(predictions)
    validation = {"status": "pass", "query_count": len(query_ids),
                  "selected_documents": sum(len(value) for value in selected_by_id.values()),
                  "selected_evidence": sum(len(doc["evidence"]) for selected in selected_by_id.values()
                                           for doc in selected),
                  "documents_without_exported_chunks": sum(not doc["evidence"]
                      for selected in selected_by_id.values() for doc in selected),
                  "document_selection": policy.document_selection,
                  "canonical_payload_sha256": hashlib.sha256(payload).hexdigest(),
                  "replay_mismatch_count": 0,
                  "grounding": "all_exported_chunks_and_document_support_verified"}
    export = staging / "export"
    export.mkdir(exist_ok=True)
    json_path, zip_path = export / "predictions.json", export / "submission.zip"
    _write_bytes_checked(json_path, payload)
    if not zip_path.exists():
        package_zip(zip_path, payload)
    zip_report = validate_zip(zip_path, query_ids=set(query_ids), official_ids=official_ids,
                              policy=policy, result_limit=result_limit,
                              max_bytes=int(batch_config["zip_max_bytes"]))
    if zip_report["payload_sha256"] != validation["canonical_payload_sha256"]:
        raise ValueError("Validated ZIP payload differs from selected output")
    checksums = (f"{sha256_file(json_path)}  predictions.json\n"
                 f"{sha256_file(zip_path)}  submission.zip\n").encode("ascii")
    _write_bytes_checked(export / "SHA256SUMS", checksums)
    _write_json(staging / "validation_report.json", {**validation, "zip": zip_report})
    query_status = pa.table({"id": query_ids, "status": ["complete"] * len(query_ids),
                             "elapsed_seconds": query_durations})
    pq.write_table(query_status, staging / "query_status.parquet")
    manifest = {"format": "medrag-v4-full-baseline-v1", "status": "FULL_V4_EXPORT_PASS",
                "score_status": "SCORE_PENDING", "query_count": 1200,
                "config_sha256": lock["config_sha256"],
                "index_manifest_sha256": lock["index_manifest_sha256"],
                "score_cache_sha256": sha256_file(staging / "scores.sqlite"),
                "query_status_sha256": sha256_file(staging / "query_status.parquet"),
                "selected_results_sha256": sha256_file(selected_path),
                "predictions_sha256": sha256_file(json_path), "submission_sha256": sha256_file(zip_path),
                "sha256sums_sha256": sha256_file(export / "SHA256SUMS"),
                "validation_report_sha256": sha256_file(staging / "validation_report.json"),
                "rss_peak_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "mean_query_seconds": sum(query_durations) / len(query_durations)}
    _write_json(staging / "run_manifest.json", manifest)
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.index.resolve(), args.config.resolve(),
                         args.output.resolve()), indent=2))
