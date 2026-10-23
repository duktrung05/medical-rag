"""Bind accepted full-index CPU and GPU reports without starting inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from src.medrag.data.phase2_batch import sha256_file
from scripts.v4.smoke_full_index import GPU_UUID, QUERY_IDS, REPEAT_IDS
from scripts.v4.verify_release_dense import write_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(root: Path, index: Path, reports: Path) -> dict:
    run_lock_path = reports / "run_lock.json"
    smoke_lock_path = reports / "smoke_lock.json"
    cpu_path = reports / "cpu_acceptance.json"
    coverage_path = reports / "full_coverage.json"
    smoke_path = reports / "full_query_smoke.json"
    launch_path = reports / "gpu3_smoke_launch.json"
    run_lock, smoke_lock, cpu, coverage, smoke, launch = (
        read_json(p) for p in (run_lock_path, smoke_lock_path, cpu_path,
                              coverage_path, smoke_path, launch_path))
    if (run_lock.get("status") != "LOCKED"
            or run_lock.get("full_inference_allowed") is not False
            or cpu.get("status") != "FULL_INDEX_ARTIFACTS_PASS"
            or coverage.get("status") != "FULL_INDEX_ARTIFACTS_PASS"
            or smoke.get("status") != "FULL_QUERY_SMOKE_PASS"
            or launch.get("status") != "LAUNCHING"
            or launch.get("gpu_uuid") != GPU_UUID
            or launch.get("cuda_visible_devices") != GPU_UUID
            or launch.get("torch_device_count") != 1
            or launch.get("cpu_affinity") != [4, 5, 6, 7]):
        raise ValueError("Full-index acceptance gates or GPU launch report are missing")
    for name, expected in run_lock["paths_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Initial run lock path changed: {name}")
    for name, expected in smoke_lock["code_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Smoke code changed: {name}")
    index_manifest_path = index / "manifest.json"
    index_manifest_hash = sha256_file(index_manifest_path)
    if (cpu.get("index_manifest_sha256") != index_manifest_hash
            or coverage.get("index_manifest_sha256") != index_manifest_hash
            or smoke.get("index_manifest_sha256") != index_manifest_hash
            or smoke_lock.get("index_manifest_sha256") != index_manifest_hash
            or smoke.get("cpu_report_sha256") != sha256_file(cpu_path)
            or smoke_lock.get("cpu_report_sha256") != sha256_file(cpu_path)
            or smoke.get("smoke_lock_sha256") != sha256_file(smoke_lock_path)
            or smoke.get("score_cache_sha256") != sha256_file(reports / "smoke_scores.sqlite")):
        raise ValueError("Full-index manifest/report/cache hash chain differs")
    if (smoke_lock.get("query_ids") != QUERY_IDS
            or smoke_lock.get("repeat_query_ids") != REPEAT_IDS
            or smoke.get("query_count") != len(QUERY_IDS)
            or smoke.get("repeat_count") != len(REPEAT_IDS)
            or [row["id"] for row in smoke["query_results"]] != QUERY_IDS
            or [row["id"] for row in smoke["repeat_results"]] != REPEAT_IDS):
        raise ValueError("Smoke query coverage/repeats differ from fixed sample")
    for qid in QUERY_IDS:
        checkpoint = read_json(reports / "smoke_queries" / f"{qid}.json")
        if (checkpoint.get("id") != qid or checkpoint.get("status") != "complete"
                or checkpoint.get("query_sha256") != smoke_lock["query_sha256"][str(qid)]):
            raise ValueError(f"Smoke query checkpoint differs: {qid}")
    if (reports / "full_query_smoke_error.json").exists():
        raise ValueError("Current smoke attempt has a failure report")
    manifest = read_json(index_manifest_path)
    if set(manifest.get("files", {})) != set(cpu["files_sha256"]):
        raise ValueError("Combined manifest file inventory differs from CPU report")
    for name, expected in cpu["files_sha256"].items():
        if (manifest["files"].get(name) != expected
                or sha256_file(index / name) != expected):
            raise ValueError(f"Combined artifact changed after CPU acceptance: {name}")
    if (root / "outputs/v4/task4/full_a0_gpu3_20261019").exists():
        raise ValueError("Full inference namespace unexpectedly exists")
    report = {"format": "medrag-v4-full-index-acceptance-v1",
              "status": "FULL_INDEX_PASS", "alias": "G5_FULL_INDEX_PASS",
              "scope": "full_index_and_20_query_smoke_only",
              "index": str(index), "index_manifest_sha256": index_manifest_hash,
              "run_lock_sha256": sha256_file(run_lock_path),
              "cpu_report_sha256": sha256_file(cpu_path),
              "smoke_report_sha256": sha256_file(smoke_path),
              "smoke_lock_sha256": sha256_file(smoke_lock_path),
              "gpu_launch_report_sha256": sha256_file(launch_path),
              "full_coverage_report_sha256": sha256_file(coverage_path),
              "gpu_uuid": GPU_UUID, "smoke_query_count": len(QUERY_IDS),
              "files_sha256": cpu["files_sha256"],
              "next_stage": "STOP_BEFORE_FULL_INFERENCE"}
    write_json(reports / "full_index_acceptance.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = finalize(args.root.resolve(), args.index.resolve(), args.reports.resolve())
    except Exception as exc:
        print(f"FULL_INDEX_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"],
                      "index_manifest_sha256": report["index_manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
