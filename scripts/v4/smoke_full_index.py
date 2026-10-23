"""Run 20 locked A0 queries on a full V4 index without opening the 1,200-query runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.selector import select_results
from src.medrag.evidence.verifier import verify as verify_passage
from src.medrag.ranking.score_cache import ScoreCache
from scripts.v4.verify_release_dense import write_json


QUERY_IDS = [46, 53, 81, 92, 130, 207, 242, 380, 494, 656,
             754, 771, 789, 815, 927, 934, 959, 1144, 1185, 1200]
REPEAT_IDS = [92, 927]
GPU_UUID = "GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score_contract(index_manifest_path: Path, index_manifest: dict, config: dict) -> dict:
    return {"format": "medrag-v4-a0-pair-score-v1",
            "index_manifest_sha256": sha256_file(index_manifest_path),
            "corpus_version": config["corpus_version"],
            "model": config["reranker"]["model"],
            "revision": config["reranker"]["revision"],
            "max_length": int(config["reranker"]["max_length"]),
            "precision": "fp16", "pair_format": "query_title_newline_passage-v1"}


def audit_scores(query: str, trace: dict, cache: ScoreCache, store: DiskCorpusStore) -> dict:
    ranked = trace["ranked_candidates"]
    scored = 0
    with sqlite3.connect(f"file:{cache.path}?mode=ro", uri=True) as db:
        for doc in ranked:
            if not math.isfinite(float(doc["score"])):
                raise ValueError("Non-finite fused candidate score")
            doc_id = int(doc["official_id"])
            source = store.source(doc_id)
            for evidence in doc["evidence"]:
                status = evidence["score_status"]
                if status != "reranked":
                    if status not in {"not_reranked_budget", "retrieval_only"}:
                        raise ValueError("Unknown candidate score status")
                    continue
                score = float(evidence["rerank_score"])
                if not math.isfinite(score):
                    raise ValueError("Non-finite rerank score")
                passage_id = str(evidence["passage_id"])
                passage = store.passage(passage_id)
                if (passage.official_id != doc_id or evidence["text"] != passage.text
                        or evidence["start"] != passage.source_start
                        or evidence["end"] != passage.source_end
                        or not verify_passage(passage, source)):
                    raise ValueError(f"Ungrounded ranked passage: {passage_id}")
                payload = f"{source.title}\n{passage.text}" if source.title else passage.text
                key = cache._key(query, doc_id, passage_id, payload)
                row = db.execute("SELECT score,query_sha256,official_id,passage_id,input_sha256 "
                                 "FROM scores WHERE cache_key=?", (key,)).fetchone()
                expected = (score, hashlib.sha256(query.encode()).hexdigest(), doc_id,
                            passage_id, hashlib.sha256(payload.encode()).hexdigest())
                if row != expected:
                    raise ValueError(f"Score cache differs for {doc_id}/{passage_id}")
                scored += 1
    if scored != int(trace["rerank_pairs"]):
        raise ValueError(f"Reranked pair coverage differs: {scored} != {trace['rerank_pairs']}")
    return {"reranked_pairs_audited": scored, "score_cache_match": True}


def check_result(query: str, selected: list[dict], trace: dict,
                 cache: ScoreCache, store: DiskCorpusStore, pipeline: DiskPipeline) -> dict:
    if (int(trace["rerank_pairs"]) > 200 or int(trace["rerank_pairs"]) <= 0
            or len(selected) > 200 or len(selected) > pipeline.selection_policy.max_chunks
            or not selected):
        raise ValueError("Smoke retrieval returned invalid budget or empty selection")
    for stage in ("document_hits", "passage_bm25_hits", "dense_passage_hits"):
        if int(trace[stage]) <= 0:
            raise ValueError(f"Smoke retrieval branch returned no hits: {stage}")
    audit = audit_scores(query, trace, cache, store)
    replay, _ = select_results(trace["ranked_candidates"], pipeline.selection_policy,
                               reranker_enabled=True, limit=200)
    if replay != selected:
        raise ValueError("Smoke selection replay differs")
    seen = set()
    selected_evidence = 0
    for doc in selected:
        doc_id = doc["official_id"]
        if doc_id in seen:
            raise ValueError("Duplicate selected parent")
        seen.add(doc_id)
        if not doc["evidence"] or len(doc["evidence"]) > pipeline.selection_policy.chunks_per_doc:
            raise ValueError("Invalid selected evidence count")
        source = store.source(doc_id)
        for item in doc["evidence"]:
            passage = store.passage(item["passage_id"])
            if (item["score_status"] != "reranked"
                    or not math.isfinite(float(item["rerank_score"]))
                    or passage.official_id != doc_id
                    or item["text"] != passage.text
                    or item["start"] != passage.source_start
                    or item["end"] != passage.source_end
                    or not verify_passage(passage, source)):
                raise ValueError("Smoke selected evidence is invalid or ungrounded")
            selected_evidence += 1
    if selected_evidence > pipeline.selection_policy.max_chunks:
        raise ValueError("Smoke selection chunk cap exceeded")
    return {**audit, "selected_documents": len(selected),
            "selected_evidence": selected_evidence,
            "selected_sha256": _sha_json(selected),
            "ranked_candidates_sha256": _sha_json(trace["ranked_candidates"])}


def run(root: Path, index: Path, config_path: Path, reports: Path) -> dict:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID or torch.cuda.device_count() != 1:
        raise ValueError("Smoke must see only the pinned GPU 3 UUID")
    actual_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    normalize = lambda value: value.removeprefix("GPU-").replace("-", "").lower()
    if normalize(actual_uuid) != normalize(GPU_UUID):
        raise ValueError("Smoke CUDA UUID differs from pinned GPU 3")
    index_manifest_path = index / "manifest.json"
    cpu_report_path = reports / "cpu_acceptance.json"
    cpu_report = _json(cpu_report_path)
    config = load_config(config_path)
    index_manifest = _json(index_manifest_path)
    if (cpu_report.get("status") != "FULL_INDEX_ARTIFACTS_PASS"
            or cpu_report.get("index_manifest_sha256") != sha256_file(index_manifest_path)
            or cpu_report.get("config_sha256") != sha256_file(config_path)
            or index_manifest.get("scope") != "full_release"
            or index_manifest.get("passage_count") != 2_951_224):
        raise ValueError("Full index CPU acceptance is missing or stale")
    if (config["dense"]["device"] != "cuda:0"
            or config["reranker"]["device"] != "cuda:0"
            or config["batch"]["result_limit"] != 200
            or config["selection"].get("document_selection", "with_chunks") != "with_chunks"):
        raise ValueError("Smoke must use pinned A0 GPU 3 config")
    reference_path = root / "outputs/v4/reference_20q_20261016_selection_fix/manifest.json"
    reference = _json(reference_path)
    if reference.get("query_ids") != QUERY_IDS:
        raise ValueError("Smoke query IDs differ from locked canary reference")
    migration = _json(root / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = root / migration["official_queries"]
    if sha256_file(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query file checksum mismatch")
    rows = pq.read_table(query_path, columns=["id", "query"]).to_pylist()
    queries = {int(row["id"]): str(row["query"]) for row in rows}
    if len(queries) != len(rows) or any(qid not in queries for qid in QUERY_IDS):
        raise ValueError("Official query identity/coverage mismatch")
    lock = {"format": "medrag-v4-full-index-smoke-lock-v1",
            "index_manifest_sha256": cpu_report["index_manifest_sha256"],
            "cpu_report_sha256": sha256_file(cpu_report_path),
            "config_sha256": sha256_file(config_path),
            "official_queries_sha256": migration["official_queries_sha256"],
            "reference_manifest_sha256": sha256_file(reference_path),
            "query_ids": QUERY_IDS, "repeat_query_ids": REPEAT_IDS,
            "query_sha256": {str(qid): hashlib.sha256(queries[qid].encode()).hexdigest()
                             for qid in QUERY_IDS},
            "code_sha256": {name: sha256_file(root / name) for name in (
                "scripts/v4/smoke_full_index.py", "src/medrag/core/pipeline.py",
                "src/medrag/core/disk_pipeline.py", "src/medrag/ranking/score_cache.py",
                "src/medrag/evidence/selector.py", "src/medrag/evidence/verifier.py")}}
    reports.mkdir(parents=True, exist_ok=True)
    lock_path = reports / "smoke_lock.json"
    if lock_path.exists():
        if _json(lock_path) != lock:
            raise ValueError("Smoke resume lock mismatch")
    else:
        write_json(lock_path, lock)
    score_path = reports / "smoke_scores.sqlite"
    contract = score_contract(index_manifest_path, index_manifest, config)
    cache = ScoreCache(score_path, contract, create=not score_path.exists())
    started = time.perf_counter()
    pipeline = DiskPipeline(index, config, score_cache=cache)
    load_seconds = time.perf_counter() - started
    dense_device = str(next(pipeline.dense.encoder.model.parameters()).device)
    reranker_device = str(next(pipeline.reranker.model.model.parameters()).device)
    if dense_device != "cuda:0" or reranker_device != "cuda:0":
        raise ValueError("Smoke models are not on the selected GPU")
    torch.cuda.reset_peak_memory_stats(0)
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    query_dir = reports / "smoke_queries"
    query_dir.mkdir(exist_ok=True)
    results = []
    for position, qid in enumerate(QUERY_IDS, 1):
        query = queries[qid]
        checkpoint_path = query_dir / f"{qid}.json"
        if checkpoint_path.exists():
            checkpoint = _json(checkpoint_path)
            if checkpoint.get("status") != "complete" or checkpoint.get("query_sha256") != lock["query_sha256"][str(qid)]:
                raise ValueError(f"Smoke checkpoint differs for query {qid}")
            checks = check_result(query, checkpoint["selected"], checkpoint["trace"],
                                  cache, store, pipeline)
            if checks != checkpoint["checks"]:
                raise ValueError(f"Smoke checkpoint audit differs for query {qid}")
        else:
            begin = time.perf_counter()
            selected, trace = pipeline.retrieve_with_diagnostics(
                query, 200, include_candidates=True)
            elapsed = time.perf_counter() - begin
            checks = check_result(query, selected, trace, cache, store, pipeline)
            checkpoint = {"id": qid, "status": "complete",
                          "query_sha256": lock["query_sha256"][str(qid)],
                          "selected": selected, "trace": trace,
                          "checks": checks, "elapsed_seconds": elapsed}
            write_json(checkpoint_path, checkpoint)
        results.append({"id": qid, "elapsed_seconds": checkpoint["elapsed_seconds"],
                        "rerank_pairs": checkpoint["checks"]["reranked_pairs_audited"],
                        "selected_documents": checkpoint["checks"]["selected_documents"],
                        "selected_sha256": checkpoint["checks"]["selected_sha256"],
                        "ranked_candidates_sha256": checkpoint["checks"]["ranked_candidates_sha256"],
                        "truncated_pairs": (checkpoint["trace"].get("reranker_tokens") or {}).get("truncated_pairs")})
        print(f"full smoke query {position}/20 id={qid} complete", flush=True)
    repeats = []
    for qid in REPEAT_IDS:
        selected, trace = pipeline.retrieve_with_diagnostics(
            queries[qid], 200, include_candidates=True)
        checks = check_result(queries[qid], selected, trace, cache, store, pipeline)
        original = _json(query_dir / f"{qid}.json")["checks"]
        if (checks["selected_sha256"] != original["selected_sha256"]
                or checks["ranked_candidates_sha256"] != original["ranked_candidates_sha256"]):
            raise ValueError(f"Smoke repeat differs for query {qid}")
        repeats.append({"id": qid, "checks": checks})
    durations = sorted(row["elapsed_seconds"] for row in results)
    report = {"format": "medrag-v4-full-index-smoke-v1",
              "status": "FULL_QUERY_SMOKE_PASS", "query_count": len(results),
              "repeat_count": len(repeats), "query_results": results,
              "repeat_results": repeats, "smoke_lock_sha256": sha256_file(lock_path),
              "index_manifest_sha256": sha256_file(index_manifest_path),
              "cpu_report_sha256": sha256_file(cpu_report_path),
              "score_cache_sha256": sha256_file(score_path),
              "dense_device": dense_device, "reranker_device": reranker_device,
              "load_seconds": load_seconds,
              "latency_seconds": {"mean": statistics.mean(durations),
                                  "p50": statistics.median(durations),
                                  "p95_nearest_rank": durations[math.ceil(.95 * len(durations)) - 1],
                                  "max": max(durations)},
              "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
              "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
              "rss_peak_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(reports / "full_query_smoke.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    reports = args.reports.resolve()
    try:
        report = run(args.root.resolve(), args.index.resolve(), args.config.resolve(), reports)
    except Exception as exc:
        write_json(reports / "full_query_smoke_error.json",
                   {"status": "FULL_QUERY_SMOKE_FAIL",
                    "error": f"{type(exc).__name__}: {exc}"})
        print(f"FULL_QUERY_SMOKE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "query_count": report["query_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
