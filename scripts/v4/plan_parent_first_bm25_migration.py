"""Create a non-deployable, read-only lineage plan for completed A0 shards.

The source may keep publishing atomic shards. This command snapshots the highest
contiguous ID visible at start, validates only that prefix, and never copies,
renames, edits, or locks source artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from scripts.v4 import benchmark_parent_first_bm25 as v1
from src.medrag.retrieval.parent_first_bm25 import IMPLEMENTATION_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    expected_source = (v1.ROOT / "outputs/v4/a0_t4_v1_full_20261022").resolve()
    if source != expected_source:
        parser.error("Only the frozen A0 source namespace is accepted")
    if (output.exists() or output.parent != (v1.ROOT / "outputs/v4").resolve()
            or not output.name.startswith("p5_parent_fallback_")):
        parser.error("Require a fresh outputs/v4/p5_parent_fallback_<name> directory")
    caps = v1.limits()
    output.mkdir(parents=True)
    started = time.monotonic()
    lock_path = source / "lock.json"
    lock = v1.read(lock_path)
    ids = sorted(int(path.stem) for path in (source / "pairs").glob("*.json"))
    contiguous = 0
    for value in ids:
        if value != contiguous + 1:
            break
        contiguous = value
    if not contiguous:
        raise ValueError("No contiguous source shards")
    lock_sha = v1.sha(lock_path)
    entries, total_bytes = [], 0
    for qid in range(1, contiguous + 1):
        if time.monotonic() > started + 25:
            raise TimeoutError("Snapshot validation deadline exceeded")
        path = source / "pairs" / f"{qid}.json"
        stat_before = path.stat()
        payload = path.read_bytes()
        stat_after = path.stat()
        if ((stat_before.st_size, stat_before.st_mtime_ns, stat_before.st_ino)
                != (stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ino)):
            raise ValueError(f"Source shard changed while reading: {qid}")
        shard = json.loads(payload)
        if (shard.get("id") != qid or shard.get("format") != "medrag-v4-p5-2-pairs-v1"
                or shard.get("lock_sha256") != lock_sha or len(shard.get("pairs", [])) != 200
                or shard.get("diagnostics", {}).get("rerank_pairs") != 200):
            raise ValueError(f"Source shard contract differs: {qid}")
        for pair in shard["pairs"]:
            if hashlib.sha256(pair["payload"].encode()).hexdigest() != pair["input_sha256"]:
                raise ValueError(f"Input hash differs: {qid}/{pair.get('passage_id')}")
        digest = hashlib.sha256(payload).hexdigest()
        total_bytes += len(payload)
        entries.append({"id": qid, "source_relative_path": f"pairs/{qid}.json",
                        "sha256": digest, "bytes": len(payload),
                        "query_sha256": shard["query_sha256"], "pairs": 200})
    elapsed = time.monotonic() - started
    plan = {"format": "p5-bm25-parent-first-migration-plan-v1",
            "status": "DRY_RUN_ONLY", "deployable": False,
            "reason": "Runner/consumer integration and end-to-end gain remain unvalidated",
            "source": {"path": str(source), "lock_sha256": lock_sha,
                       "config_sha256": lock["config_sha256"],
                       "index_manifest_sha256": lock["index_manifest_sha256"],
                       "mode": lock["mode"], "snapshot_contiguous_max_id": contiguous,
                       "shards_visible_at_start": len(ids), "pair_shards": entries,
                       "total_pairs": contiguous * 200, "total_bytes": total_bytes},
            "proposed_backend": {"version": IMPLEMENTATION_VERSION,
                "sha256": v1.sha(v1.ROOT / "src/medrag/retrieval/parent_first_bm25.py"),
                "reference_backend_sha256": v1.sha(v1.ROOT / "src/medrag/retrieval/disk_bm25.py")},
            "lineage_policy": {"reuse_completed_shards_without_retrieval": True,
                "source_files_remain_immutable": True,
                "new_queries_require_new_versioned_lock": True,
                "mixed_lock_consumption_forbidden": True,
                "rollback": "Discard future namespace and resume source runner from source shards"},
            "validation_measurement": {"wall_seconds": elapsed,
                "mib_per_second": total_bytes / (1024**2) / elapsed,
                "linear_projection_1200_shards_seconds": elapsed * 1200 / contiguous,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "resource_limits": caps},
            "source_process_untouched": True,
            "created_by_pid": os.getpid()}
    v1.write(output / "migration_plan.json", plan)
    print(v1.canonical({"plan": str(output / "migration_plan.json"), "deployable": False,
        "validated_shards": contiguous, "validated_pairs": contiguous * 200,
        "wall_seconds": elapsed, "projected_1200_seconds":
        plan["validation_measurement"]["linear_projection_1200_shards_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
