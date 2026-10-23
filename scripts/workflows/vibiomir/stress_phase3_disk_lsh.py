"""Bounded synthetic scalability test for the Phase 3 disk-backed LSH engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase3.disk_lsh import DiskLSHIndex, ResourceMonitor, directory_size


def _signature(value: int, width: int) -> tuple[int, ...]:
    digest = hashlib.sha256(f"stress-{value}".encode()).digest()
    values = []
    while len(values) < width:
        values.extend(
            int.from_bytes(digest[offset:offset + 4], "big")
            for offset in range(0, len(digest), 4)
        )
        digest = hashlib.sha256(digest).digest()
    return tuple(values[:width])


def _record(value: int, signature: tuple[int, ...]) -> dict:
    return {
        "entity_key": f"synthetic-{value:09d}", "parent_key": str(value),
        "sequence_index": 0, "ordinal": value, "strategy": "english_word",
        "shingle_size": 5, "shingles": frozenset({value + 1, value + 2, value + 3}),
        "signature": signature,
        "normalized_hash": hashlib.sha256(str(value).encode()).hexdigest(),
        "payload": {"doc_id": value, "normalized_text": f"synthetic record {value}"},
    }


def run(output: Path, *, records: int, shards: int) -> dict:
    if records < 1000 or shards < 2:
        raise ValueError("Stress test requires at least 1000 records and two shards")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    initial_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    bands, rows_per_band = 4, 2
    monitor = ResourceMonitor(
        rss_limit_bytes=2 * 1024**3,
        disk_limit_bytes=4 * 1024**3,
        work_dir=output,
    )
    monitor.preflight()
    known_pairs = []
    hot_records = min(64, records // 10)
    per_shard = (records + shards - 1) // shards
    with DiskLSHIndex(
        output / "stress.sqlite", bands=bands, rows_per_band=rows_per_band,
        partitions=32, maximum_bucket_size=32, config_hash="stress-config-v1",
        input_fingerprint=f"synthetic-{records}-{shards}", monitor=monitor,
    ) as index:
        for shard in range(shards):
            start = shard * per_shard
            end = min(records, start + per_shard)
            batch = []
            for value in range(start, end):
                signature = list(_signature(value, bands * rows_per_band))
                if value < hot_records:
                    signature[:2] = [7, 7]
                if value % per_shard == 0:
                    signature[:2] = [1_000_000 + shard // 2, 2_000_000 + shard // 2]
                if shard % 2 == 1 and value == start:
                    partner = value - per_shard
                    known_pairs.append((
                        f"synthetic-{partner:09d}", f"synthetic-{value:09d}"
                    ))
                batch.append(_record(value, tuple(signature)))
            index.ingest_shard(f"part-{shard:05d}", batch, batch_size=256)
        generation = index.generate_candidates(commit_every_buckets=500)
        found = {
            tuple(sorted((left["entity_key"], right["entity_key"])))
            for partition in index.candidate_partitions()
            for left, right, _ in index.iter_candidates(partition)
        }
        missing = sorted(pair for pair in known_pairs if tuple(sorted(pair)) not in found)
        statistics = index.statistics()
        index.atomic_checkpoint_export(output / "checkpoint.json")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result = {
        "format": "vibiomir-phase3-disk-lsh-stress-v1",
        "status": "pass" if not missing else "fail",
        "records": records, "shards": shards, "cross_shard_known_pairs": len(known_pairs),
        "missing_known_pairs": missing, "generation": generation,
        "statistics": statistics, "initial_peak_rss_bytes": initial_rss,
        "final_peak_rss_bytes": peak_rss, "peak_rss_growth_bytes": max(0, peak_rss - initial_rss),
        "disk_bytes": directory_size(output),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "rss_budget_bytes": monitor.rss_limit_bytes,
        "disk_budget_bytes": monitor.disk_limit_bytes,
    }
    (output / "stress_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if missing:
        raise RuntimeError(f"Known cross-shard candidates missing: {missing[:5]}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--shards", type=int, default=40)
    args = parser.parse_args()
    result = run(args.output.resolve(), records=args.records, shards=args.shards)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
