"""Lock the single end-to-end recovery policy and run CPU/GPU readiness gates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import pyarrow.parquet as pq

from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.run_r11_high_recall_coverage import OUT as R11
from src.medrag.core.config import load_config
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.sql_aggregate_bm25 import SQLAggregateDiskBM25

OUTPUT = ROOT / "outputs/v4/end_to_end_recovery_20261026"
CANARY_IDS = (53, 130, 380)
BRANCHES = ("document_bm25", "passage_bm25", "dense")


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace locked recovery artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def top_ids(rows: list[dict], limit: int) -> list[tuple[int, str | None]]:
    return [(int(row["official_id"]), row.get("passage_id")) for row in rows[:limit]]


def build_lock() -> dict:
    config_path = ROOT / "configs/v4_submission_p2_a0.yaml"
    config = load_config(config_path)
    index = ROOT / config["index_namespace"]
    index_manifest = read(index / "manifest.json")
    migration_path = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"
    migration = read(migration_path)
    query_path = ROOT / migration["official_queries"]
    query_rows = pq.read_table(query_path, columns=["id", "query"]).to_pylist()
    queries = {int(row["id"]): str(row["query"]) for row in query_rows}
    if len(queries) != 1200 or set(queries) != set(range(1, 1201)):
        raise ValueError("Official query identity differs from the locked 1,200-query task")
    cap10 = ROOT / "outputs/v4/evidence_window_cap10_full_20261025/export/submission.zip"
    with zipfile.ZipFile(cap10) as archive:
        payload = archive.read("predictions.json")
    if len(json.loads(payload)) != 1200:
        raise ValueError("Cap-10 baseline does not cover all official queries")
    run_policy = {
        "query_ids": list(range(1, 1201)),
        "retrieval": {
            "official_query_only": True,
            "depth_per_branch": 800,
            "branches": list(BRANCHES),
            "parent_fusion": "RRF after grouping unique parents, k=60",
            "retrieved_unique_parent_cap": 800,
            "keep_all_cap10_documents": True,
            "new_parent_rerank_limit": 200,
        },
        "evidence": {
            "passages": "all indexed passages from each parent in the evidence pool",
            "window_tokens": 192,
            "window_stride_tokens": 96,
            "source_offsets_required": True,
            "reranker": "pinned BGE reranker v2 M3, CUDA FP16, raw logits",
            "context": "source-contiguous context around best scored window, max 5,500 chars",
            "baseline_context_fallback": "cap-10 context retained as candidate and tie fallback",
        },
        "allocation": {
            "baseline_chunks": 60,
            "max_replacements_per_query": 10,
            "max_new_document_entrants_per_query": 3,
            "max_chunks_per_parent": 1,
            "new_document_requires_chunk_slot": True,
        },
        "submission": {
            "queries": 1200,
            "max_documents_per_query": 200,
            "max_chunks_per_query": 60,
            "zip_max_bytes": 100_000_000,
            "btc_calls": 1,
        },
    }
    source_paths = [
        config_path, migration_path, query_path,
        index / "manifest.json", index / "lock.json",
        cap10, ROOT / "outputs/v4/evidence_window_cap10_full_20261025/export/predictions.json",
        ROOT / "outputs/v4/evidence_window_cap10_full_20261025/score_manifest.json",
        ROOT / "outputs/v4/p5_gpu_incremental_a6000_20261023/runtime_lock.json",
        R11 / "runtime_canary_fast.json",
    ]
    return {
        "format": "v4-end-to-end-evidence-recovery-lock-v1",
        "status": "LOCKED_AWAITING_TECHNICAL_PREFLIGHT",
        "policy": run_policy,
        "index_fingerprint": index_manifest["fingerprint"],
        "queries_sha256": sha256_file(query_path),
        "cap10_zip_sha256": sha256_file(cap10),
        "cap10_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "input_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "implementation_sha256": {str(Path(__file__).resolve().relative_to(ROOT)):
                                   sha256_file(Path(__file__).resolve())},
    }


def branch_preflight(lock: dict) -> dict:
    config = load_config(ROOT / "configs/v4_submission_p2_a0.yaml")
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    depths = {}
    for name, filename in (("document_bm25", "documents.sqlite"),
                            ("passage_bm25", "passages.sqlite")):
        reference = DiskBM25(index / filename, name, manifest["fingerprint"])
        sql = SQLAggregateDiskBM25(index / filename, name, manifest["fingerprint"])
        depths[name] = {}
        for qid in CANARY_IDS:
            query = next(row["query"] for row in pq.read_table(
                ROOT / read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")["official_queries"],
                columns=["id", "query"]).to_pylist() if int(row["id"]) == qid)
            a, b = reference.search(query, 800), sql.search(query, 800)
            reference_ids = [(row.official_id, row.passage_id) for row in a]
            sql_ids = [(row.official_id, row.passage_id) for row in b]
            same_ids = reference_ids == sql_ids
            max_score_delta = (max((abs(left.score - right.score)
                                    for left, right in zip(a, b, strict=True)), default=0.0)
                               if same_ids else None)
            depths[name][str(qid)] = {
                "count_reference": len(a), "count_sql": len(b),
                "ordered_ids_exact": same_ids,
                "max_absolute_score_delta": max_score_delta,
            }
    regular_dir, fast_dir = R11 / "traces", R11 / "traces_fast"
    dense_parity = {}
    for qid in CANARY_IDS:
        regular = read(regular_dir / f"{qid}.json")["variants"]["raw"]["dense"][:800]
        fast = read(fast_dir / f"{qid}.json")["variants"]["raw"]["dense"][:800]
        regular_ids, fast_ids = top_ids(regular, 800), top_ids(fast, 800)
        dense_parity[str(qid)] = {
            "count_fp64": len(regular_ids), "count_fp32": len(fast_ids),
            "ordered_ids_exact": regular_ids == fast_ids,
        }
    canary = read(R11 / "runtime_canary_fast.json")
    s2_fusion = {
        str(row["query_id"]): row["s2_mixed_fusion_exact"]
        for row in canary["results"] if int(row["query_id"]) in CANARY_IDS
    }
    return {
        "format": "v4-end-to-end-evidence-recovery-preflight-v1",
        "canary_query_ids": list(CANARY_IDS),
        "sql_aggregate_bm25_top800": depths,
        "fp32_dense_vs_fp64_dense_top800": dense_parity,
        "existing_frozen_s2_mixed_fusion_parity": s2_fusion,
        "r11_frozen_prefix_canary": {
            "parity_pass": canary["parity_pass"],
            "depths_compared_to_frozen": {"document_bm25": 400,
                                           "passage_bm25": 200, "dense": 200},
            "policy_depth": lock["policy"]["retrieval"]["depth_per_branch"],
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = build_lock()
    preflight = branch_preflight(lock)
    import torch
    gpu = {
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }
    sql_ok = all(row["ordered_ids_exact"]
                 for branch in preflight["sql_aggregate_bm25_top800"].values()
                 for row in branch.values())
    dense_ok = all(row["ordered_ids_exact"]
                   for row in preflight["fp32_dense_vs_fp64_dense_top800"].values())
    fusion_ok = all(preflight["existing_frozen_s2_mixed_fusion_parity"].values())
    frozen_ok = bool(preflight["r11_frozen_prefix_canary"]["parity_pass"])
    parity_ok = sql_ok and dense_ok and fusion_ok and frozen_ok
    status = ("READY_TO_RUN_ON_PINNED_GPU" if parity_ok and gpu["cuda_available"]
              else "WAITING_FOR_PINNED_GPU" if parity_ok else "STOP_TECHNICAL_PARITY")
    lock["status"] = "POLICY_LOCKED"
    OUTPUT.mkdir(parents=True)
    write_once(OUTPUT / "policy_lock.json", lock)
    write_once(OUTPUT / "technical_preflight.json", preflight)
    write_once(OUTPUT / "runtime_probe.json", gpu)
    write_once(OUTPUT / "status.json", {
        "format": "v4-end-to-end-evidence-recovery-status-v1",
        "status": status,
        "policy_locked": True,
        "technical_parity": "PASS" if parity_ok else "FAIL",
        "gpu_runtime": gpu,
        "retrieval_started": False,
        "reranking_started": False,
        "candidate_submission_created": False,
        "official_f2": "UNMEASURED",
        "reason": ("Host has no CUDA device; plan requires the pinned CUDA FP16 runtime before retrieval/scoring and forbids mixing CPU logits with cached FP16 scores."
                   if status == "WAITING_FOR_PINNED_GPU" else None),
    })
    print(json.dumps({"status": status, "gpu": gpu, "preflight": preflight},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
