"""Report resumable full CPU pair preparation progress without scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.v4.run_p5_2_split import (ROOT, ScoreCache, read, validate_pair_shard,
                                      verify_pair_manifest)
from src.medrag.data.phase2_batch import sha256_file


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def audit(source: Path) -> dict:
    source = source.resolve()
    lock = read(source / "lock.json")
    if lock.get("format") != "medrag-v4-p5-2-split-lock-v1" or lock.get("mode") != "full":
        raise ValueError("A0_T4_V1 full CPU preparation lock mismatch")
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    completed, hashes, pair_total = [], {}, 0
    for qid in lock["query_ids"]:
        path = source / "pairs" / f"{qid}.json"
        if not path.exists():
            continue
        pair = read(path)
        validate_pair_shard(pair, lock, qid, pair["query"], cache)
        completed.append(qid)
        hashes[str(qid)] = sha256_file(path)
        pair_total += len(pair["pairs"])
    manifest_path = source / "pair_manifest.json"
    complete = len(completed) == 1200 and pair_total == 240000 and manifest_path.is_file()
    if manifest_path.exists():
        verify_pair_manifest(source, lock)
        manifest = read(manifest_path)
        if set(manifest["pair_shards_sha256"]) != {str(qid) for qid in lock["query_ids"]}:
            raise ValueError("Full immutable pair manifest coverage mismatch")
    report = {
        "format": "medrag-v4-a0-t4-v1-cpu-pair-progress-v1",
        "candidate": "A0_T4_V1",
        "preflight_status": "GO",
        "status": "GO" if complete else "INCOMPLETE_RESUMABLE",
        "query_count_expected": 1200,
        "pair_count_expected": 240000,
        "query_shards_complete": len(completed),
        "pair_shards_complete": pair_total,
        "remaining_query_shards": 1200 - len(completed),
        "query_ids_completed": completed,
        "pair_shards_sha256": hashes,
        "lock_sha256": sha256_file(source / "lock.json"),
        "pair_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "full_gpu_scoring_started": False,
        "resume_command": "bash scripts/v4/prepare_a0_t4_v1_cpu.sh",
    }
    path = ROOT / "outputs/v4/a0_t4_v1_release_candidate_20261022/cpu_pair_plan_progress_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    temporary.write_bytes(canonical(report))
    os.replace(temporary, path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "outputs/v4/a0_t4_v1_full_20261022")
    args = parser.parse_args()
    report = audit(args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
