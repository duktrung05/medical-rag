"""Validated, immutable carry-forward for versioned A0 pair namespaces."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import ScoreCache, digest, read, validate_pair_shard
from src.medrag.data.phase2_batch import sha256_file

LINEAGE_FORMAT = "medrag-v4-p5-fast-a0-lineage-v1"
MERGED_MANIFEST_FORMAT = "medrag-v4-p5-fast-a0-pair-manifest-v1"


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_checked(path, canonical_bytes(value))


def write_atomic_replace(path: Path, value) -> None:
    """Atomically refresh derived status/manifest data in a target namespace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    payload = canonical_bytes(value)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def require_process_stopped(pid: int) -> None:
    if process_alive(pid):
        raise RuntimeError(
            f"Source runner PID {pid} is still active; snapshot refused until an explicit handoff"
        )


def _stable_bytes(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError(f"Source changed while snapshotting: {path}")
    return payload


def snapshot_source(source: Path, query_by_id: dict[int, str]) -> dict:
    """Validate and reference the contiguous source prefix without copying it."""
    source = source.resolve()
    source_lock_path = source / "lock.json"
    source_lock = read(source_lock_path)
    if (source_lock.get("format") != "medrag-v4-p5-2-split-lock-v1"
            or source_lock.get("mode") != "full"
            or source_lock.get("query_ids") != list(range(1, 1201))):
        raise ValueError("Source is not the frozen 1,200-query A0 namespace")
    source_lock_sha = sha256_file(source_lock_path)
    source_cache_path = source / "key_contract.sqlite"
    source_cache = ScoreCache(source_cache_path, source_lock["score_contract"])
    visible_ids = sorted(int(path.stem) for path in (source / "pairs").glob("*.json")
                         if path.stem.isdigit())
    contiguous = 0
    for qid in visible_ids:
        if qid != contiguous + 1:
            break
        contiguous = qid
    if contiguous == 0:
        raise ValueError("Source has no contiguous completed shards")
    entries = []
    for qid in range(1, contiguous + 1):
        path = source / "pairs" / f"{qid}.json"
        payload = _stable_bytes(path)
        shard = json.loads(payload)
        query = query_by_id.get(qid)
        if query is None:
            raise ValueError(f"Official query missing: {qid}")
        validate_pair_shard(shard, source_lock, qid, query, source_cache)
        entries.append({
            "id": qid,
            "source_relative_path": f"pairs/{qid}.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "query_sha256": digest(query),
            "pair_count": len(shard["pairs"]),
        })
    return {
        "format": LINEAGE_FORMAT,
        "source_namespace": str(source),
        "source_lock_sha256": source_lock_sha,
        "source_key_contract_sha256": sha256_file(source_cache_path),
        "source_config_sha256": source_lock["config_sha256"],
        "source_index_manifest_sha256": source_lock["index_manifest_sha256"],
        "source_score_contract": source_lock["score_contract"],
        "snapshot_contiguous_max_id": contiguous,
        "inherited_query_ids": list(range(1, contiguous + 1)),
        "remaining_query_ids": list(range(contiguous + 1, 1201)),
        "pair_shards": entries,
        "source_files_are_references": True,
        "source_files_copied": False,
        "rollback_namespace": str(source),
    }


class PairLineageResolver:
    """Resolve and validate inherited or local shards under their own locks."""

    def __init__(self, target: Path):
        self.target = target.resolve()
        self.lineage_path = self.target / "lineage.json"
        self.lock_path = self.target / "lock.json"
        self.lineage = read(self.lineage_path)
        self.lock = read(self.lock_path)
        if self.lineage.get("format") != LINEAGE_FORMAT:
            raise ValueError("Unsupported A0 lineage format")
        if self.lock.get("format") != "medrag-v4-p5-fast-a0-lock-v1":
            raise ValueError("Unsupported optimized A0 lock")
        if self.lock.get("lineage_sha256") != sha256_file(self.lineage_path):
            raise ValueError("Optimized lock/lineage mismatch")
        self.source = Path(self.lineage["source_namespace"]).resolve()
        if sha256_file(self.source / "lock.json") != self.lineage["source_lock_sha256"]:
            raise ValueError("Source lock changed after snapshot")
        if sha256_file(self.source / "key_contract.sqlite") != self.lineage["source_key_contract_sha256"]:
            raise ValueError("Source key contract changed after snapshot")
        self.inherited = {int(item["id"]): item for item in self.lineage["pair_shards"]}
        self.source_lock = read(self.source / "lock.json")
        self.source_cache = ScoreCache(self.source / "key_contract.sqlite",
                                       self.source_lock["score_contract"])
        self.local_cache = ScoreCache(self.target / "key_contract.sqlite",
                                      self.lock["score_contract"])

    def resolve(self, qid: int) -> tuple[Path, str]:
        if qid in self.inherited:
            local = self.target / "pairs" / f"{qid}.json"
            if local.exists():
                raise ValueError(f"Inherited/local shard conflict: {qid}")
            return self.source / self.inherited[qid]["source_relative_path"], "inherited"
        return self.target / "pairs" / f"{qid}.json", "local"

    def validate(self, qid: int, query: str) -> tuple[dict, str, str]:
        path, origin = self.resolve(qid)
        payload = _stable_bytes(path)
        actual_sha = hashlib.sha256(payload).hexdigest()
        shard = json.loads(payload)
        if origin == "inherited":
            expected = self.inherited[qid]
            if (actual_sha != expected["sha256"]
                    or len(payload) != expected["bytes"]
                    or digest(query) != expected["query_sha256"]):
                raise ValueError(f"Inherited shard differs from snapshot: {qid}")
            validate_pair_shard(shard, self.source_lock, qid, query, self.source_cache)
        else:
            validate_pair_shard(shard, self.lock, qid, query, self.local_cache)
        return shard, origin, actual_sha

    def build_merged_manifest(self, query_by_id: dict[int, str], *, require_complete: bool) -> dict:
        entries = {}
        missing = []
        for qid in self.lock["query_ids"]:
            path, origin = self.resolve(qid)
            if not path.exists():
                missing.append(qid)
                continue
            shard, origin, shard_sha = self.validate(qid, query_by_id[qid])
            entries[str(qid)] = {
                "origin": origin,
                "sha256": shard_sha,
                "pair_count": len(shard["pairs"]),
                "path": str(path),
            }
        if require_complete and missing:
            raise ValueError(f"Merged A0 coverage incomplete; first missing ID: {missing[0]}")
        return {
            "format": MERGED_MANIFEST_FORMAT,
            "optimized_lock_sha256": sha256_file(self.lock_path),
            "lineage_sha256": sha256_file(self.lineage_path),
            "query_ids": self.lock["query_ids"],
            "pair_shards": entries,
            "missing_query_ids": missing,
            "complete": not missing,
            "inherited_shards": sum(v["origin"] == "inherited" for v in entries.values()),
            "local_shards": sum(v["origin"] == "local" for v in entries.values()),
        }
