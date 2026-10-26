"""Create a stopped-source A0 carry-forward namespace without copying shards."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from scripts.v4.run_p5_2_split import CONFIG, ROOT, ScoreCache, inputs
from src.medrag.data.pair_lineage import (require_process_stopped, snapshot_source,
                                          write_once)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.parent_first_bm25 import IMPLEMENTATION_VERSION

DEFAULT_SOURCE = ROOT / "outputs/v4/a0_t4_v1_full_20261022"
IMPLEMENTATION_FILES = (
    "scripts/v4/migrate_p5_fast_a0.py",
    "scripts/v4/run_p5_fast_a0.py",
    "scripts/v4/package_p5_fast_a0_colab.py",
    "scripts/v4/benchmark_p5_fast_a0_production_canary.py",
    "scripts/v4/run_p5_2_split.py",
    "src/medrag/core/pipeline.py",
    "src/medrag/core/disk_pipeline.py",
    "src/medrag/retrieval/core.py",
    "src/medrag/retrieval/disk_dense.py",
    "src/medrag/retrieval/instrumented_dense.py",
    "src/medrag/retrieval/disk_bm25.py",
    "src/medrag/retrieval/parent_first_bm25.py",
    "src/medrag/retrieval/bge_encoder.py",
    "src/medrag/retrieval/stage_timing.py",
    "src/medrag/data/disk_store.py",
    "src/medrag/data/pair_lineage.py",
    "src/medrag/evidence/selector.py",
    "src/medrag/evidence/context.py",
    "src/medrag/evidence/verifier.py",
    "src/medrag/adapters/validation.py",
    "src/medrag/ranking/score_cache.py",
)


def create_migration(source: Path, target: Path, source_pid: int, *,
                     enforce_production_paths: bool = True,
                     operator_handoff: str | None = None) -> dict:
    source, target = source.resolve(), target.resolve()
    require_process_stopped(source_pid)
    if enforce_production_paths:
        if operator_handoff != "GO_STOPPED_P5_CPU":
            raise ValueError("Production snapshot requires explicit GO_STOPPED_P5_CPU handoff token")
        if source != DEFAULT_SOURCE.resolve():
            raise ValueError("Only the frozen A0 source namespace is accepted")
        expected_parent = (ROOT / "outputs/v4").resolve()
        if target.parent != expected_parent or not target.name.startswith("a0_fast_parent_first_v1_"):
            raise ValueError("Target must be outputs/v4/a0_fast_parent_first_v1_<version>")
    if target.exists():
        raise FileExistsError(target)

    config, source_shape, queries, index, _ = inputs(CONFIG, "full")
    query_by_id = {int(row["id"]): str(row["query"]) for row in queries}
    lineage = snapshot_source(source, query_by_id)
    source_lock = json.loads((source / "lock.json").read_text(encoding="utf-8"))
    for field in ("config_sha256", "index_manifest_sha256", "official_queries_sha256",
                  "official_corpus_sha256", "score_contract", "query_ids"):
        if source_lock.get(field) != source_shape.get(field):
            raise ValueError(f"Frozen source lock differs from current input contract: {field}")

    target.mkdir(parents=True)
    write_once(target / "lineage.json", lineage)
    lock = copy.deepcopy(source_shape)
    lock.update({
        "format": "medrag-v4-p5-fast-a0-lock-v1",
        "backend_version": IMPLEMENTATION_VERSION,
        "source_lock_sha256": lineage["source_lock_sha256"],
        "lineage_sha256": sha256_file(target / "lineage.json"),
        "inherited_query_ids": lineage["inherited_query_ids"],
        "remaining_query_ids_at_snapshot": lineage["remaining_query_ids"],
        "output_namespace": str(target),
        "index_namespace": str(index.resolve()),
        "implementation_sha256": {name: sha256_file(ROOT / name)
                                   for name in IMPLEMENTATION_FILES},
        "backend_sha256": {
            "dense": sha256_file(ROOT / "src/medrag/retrieval/disk_dense.py"),
            "bm25_reference": sha256_file(ROOT / "src/medrag/retrieval/disk_bm25.py"),
            "bm25_parent_first": sha256_file(ROOT / "src/medrag/retrieval/parent_first_bm25.py"),
            "encoder": sha256_file(ROOT / "src/medrag/retrieval/bge_encoder.py"),
            "store": sha256_file(ROOT / "src/medrag/data/disk_store.py"),
        },
    })
    write_once(target / "lock.json", lock)
    ScoreCache(target / "key_contract.sqlite", lock["score_contract"], create=True)
    state = {
        "format": "medrag-v4-p5-fast-a0-migration-state-v1",
        "status": "READY_TO_PREPARE_REMAINING",
        "source_process_confirmed_stopped": True,
        "source_pid_checked": source_pid,
        "completed_shards_inherited": lineage["snapshot_contiguous_max_id"],
        "remaining_queries": len(lineage["remaining_query_ids"]),
        "source_bytes_copied": 0,
        "rollback": {"action": "stop optimized runner and resume the original namespace",
                     "namespace": str(source)},
    }
    write_once(target / "migration_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-pid", type=int, default=197011)
    parser.add_argument("--handoff-token", required=True,
                        help="After explicit GO and stop, pass GO_STOPPED_P5_CPU")
    args = parser.parse_args()
    result = create_migration(args.source, args.target, args.source_pid,
                              operator_handoff=args.handoff_token)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
