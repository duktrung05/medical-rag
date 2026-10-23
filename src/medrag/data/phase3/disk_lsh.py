"""Bounded-memory, resumable, disk-backed MinHash/LSH candidate storage."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sqlite3
import struct
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path


DISK_LSH_VERSION = "vibiomir-phase3-disk-lsh-v1"
SHINGLING_VERSION = "vibiomir-phase3-language-shingling-v1"


class ResourceBudgetExceeded(RuntimeError):
    """Raised only after durable state can be resumed safely."""


def current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    # ru_maxrss is KiB on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class ResourceMonitor:
    def __init__(self, *, rss_limit_bytes: int, disk_limit_bytes: int, work_dir: Path):
        if rss_limit_bytes <= 0 or disk_limit_bytes <= 0:
            raise ValueError("Resource budgets must be positive")
        self.rss_limit_bytes = int(rss_limit_bytes)
        self.disk_limit_bytes = int(disk_limit_bytes)
        self.work_dir = work_dir

    def preflight(self, *, required_free_bytes: int = 0) -> dict[str, int]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        disk = os.statvfs(self.work_dir)
        free = disk.f_bavail * disk.f_frsize
        required = max(int(required_free_bytes), self.disk_limit_bytes)
        if free < required:
            raise ResourceBudgetExceeded(
                f"Insufficient temporary disk: free={free} required={required}"
            )
        self.check()
        return {"rss_bytes": current_rss_bytes(), "disk_free_bytes": free}

    def check(self) -> None:
        rss = current_rss_bytes()
        used = directory_size(self.work_dir)
        if rss > self.rss_limit_bytes:
            raise ResourceBudgetExceeded(
                f"RSS budget exceeded: current={rss} limit={self.rss_limit_bytes}"
            )
        if used > self.disk_limit_bytes:
            raise ResourceBudgetExceeded(
                f"Temporary disk budget exceeded: used={used} limit={self.disk_limit_bytes}"
            )


def structured_event(event: str, **values: object) -> str:
    return json.dumps(
        {"event": event, **values}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _pack_u64(values: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in values)
    return struct.pack(f">{len(values)}Q", *values)


def unpack_u64(value: bytes) -> tuple[int, ...]:
    if len(value) % 8:
        raise ValueError("Corrupt uint64 blob")
    return struct.unpack(f">{len(value) // 8}Q", value)


def deterministic_partition(left_key: str, right_key: str, partitions: int) -> int:
    if partitions < 1:
        raise ValueError("partitions must be positive")
    digest = hashlib.sha256(f"{left_key}\0{right_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % partitions


class DiskLSHIndex:
    """SQLite-backed records, bands, candidates, and durable stage checkpoints."""

    def __init__(
        self,
        path: Path,
        *,
        bands: int,
        rows_per_band: int,
        partitions: int,
        maximum_bucket_size: int,
        config_hash: str,
        input_fingerprint: str,
        monitor: ResourceMonitor,
    ):
        self.path = path
        self.bands = int(bands)
        self.rows_per_band = int(rows_per_band)
        self.partitions = int(partitions)
        self.maximum_bucket_size = int(maximum_bucket_size)
        self.config_hash = config_hash
        self.input_fingerprint = input_fingerprint
        self.monitor = monitor
        if min(self.bands, self.rows_per_band, self.partitions, self.maximum_bucket_size) < 1:
            raise ValueError("Invalid disk LSH configuration")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-65536")
        self._schema()
        self._validate_or_initialize_metadata()

    def _schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
              record_id INTEGER PRIMARY KEY,
              entity_key TEXT NOT NULL UNIQUE,
              parent_key TEXT,
              sequence_index INTEGER,
              shard TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              strategy TEXT NOT NULL,
              shingle_size INTEGER NOT NULL,
              shingles BLOB NOT NULL,
              signature BLOB NOT NULL,
              normalized_hash TEXT,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bands (
              strategy TEXT NOT NULL,
              band INTEGER NOT NULL,
              band_key BLOB NOT NULL,
              record_id INTEGER NOT NULL,
              PRIMARY KEY(strategy, band, band_key, record_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidates (
              partition_id INTEGER NOT NULL,
              left_id INTEGER NOT NULL,
              right_id INTEGER NOT NULL,
              sources TEXT NOT NULL DEFAULT 'minhash_lsh',
              PRIMARY KEY(left_id, right_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS candidates_partition
              ON candidates(partition_id, left_id, right_id);
            CREATE INDEX IF NOT EXISTS records_parent_sequence
              ON records(parent_key, sequence_index, record_id);
            CREATE TABLE IF NOT EXISTS hot_buckets (
              strategy TEXT NOT NULL,
              band INTEGER NOT NULL,
              band_key BLOB NOT NULL,
              bucket_size INTEGER NOT NULL,
              potential_pairs INTEGER NOT NULL,
              examined_pairs INTEGER NOT NULL,
              policy TEXT NOT NULL,
              PRIMARY KEY(strategy, band, band_key)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS checkpoints (
              stage TEXT NOT NULL,
              unit TEXT NOT NULL,
              status TEXT NOT NULL,
              input_fingerprint TEXT NOT NULL,
              config_hash TEXT NOT NULL,
              shingling_version TEXT NOT NULL,
              detail TEXT NOT NULL,
              PRIMARY KEY(stage, unit)
            ) WITHOUT ROWID;
            """
        )
        self.connection.commit()

    def _validate_or_initialize_metadata(self) -> None:
        expected = {
            "version": DISK_LSH_VERSION,
            "bands": str(self.bands),
            "rows_per_band": str(self.rows_per_band),
            "partitions": str(self.partitions),
            "maximum_bucket_size": str(self.maximum_bucket_size),
            "config_hash": self.config_hash,
            "input_fingerprint": self.input_fingerprint,
            "shingling_version": SHINGLING_VERSION,
        }
        existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if existing and existing != expected:
            raise ValueError("Disk LSH resume metadata mismatch")
        if not existing:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DiskLSHIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def checkpoint_complete(self, stage: str, unit: str) -> bool:
        row = self.connection.execute(
            "SELECT status, input_fingerprint, config_hash, shingling_version "
            "FROM checkpoints WHERE stage=? AND unit=?", (stage, unit)
        ).fetchone()
        if row is None:
            return False
        if tuple(row) != (
            "complete", self.input_fingerprint, self.config_hash, SHINGLING_VERSION
        ):
            raise ValueError("Checkpoint fingerprint or version mismatch")
        return True

    def checkpoint(self, stage: str, unit: str, status: str, detail: dict) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                stage, unit, status, self.input_fingerprint, self.config_hash,
                SHINGLING_VERSION, json.dumps(detail, sort_keys=True),
            ),
        )
        self.connection.commit()

    def ingest_shard(self, shard: str, records: Iterable[dict], *, batch_size: int = 512) -> int:
        if self.checkpoint_complete("input_shard", shard):
            row = self.connection.execute(
                "SELECT detail FROM checkpoints WHERE stage='input_shard' AND unit=?", (shard,)
            ).fetchone()
            return int(json.loads(row[0])["records"])
        count = 0
        pending_records: list[tuple] = []
        pending_bands: list[tuple] = []
        for record in records:
            signature = tuple(record["signature"])
            if len(signature) != self.bands * self.rows_per_band:
                raise ValueError("Signature length differs from LSH band layout")
            self.connection.execute(
                "INSERT OR IGNORE INTO records(entity_key, parent_key, sequence_index, shard, ordinal, "
                "strategy, shingle_size, shingles, signature, normalized_hash, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["entity_key"], record.get("parent_key"),
                    record.get("sequence_index"), shard, int(record["ordinal"]), record["strategy"],
                    int(record["shingle_size"]), _pack_u64(sorted(record["shingles"])),
                    _pack_u64(signature), record.get("normalized_hash"),
                    json.dumps(record["payload"], ensure_ascii=False, sort_keys=True),
                ),
            )
            record_id = int(self.connection.execute(
                "SELECT record_id FROM records WHERE entity_key=?", (record["entity_key"],)
            ).fetchone()[0])
            for band in range(self.bands):
                start = band * self.rows_per_band
                pending_bands.append((
                    record["strategy"], band,
                    _pack_u64(signature[start:start + self.rows_per_band]), record_id,
                ))
            count += 1
            if count % batch_size == 0:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO bands VALUES (?, ?, ?, ?)", pending_bands
                )
                pending_bands.clear()
                self.connection.commit()
                self.monitor.check()
        if pending_bands:
            self.connection.executemany(
                "INSERT OR IGNORE INTO bands VALUES (?, ?, ?, ?)", pending_bands
            )
        self.connection.commit()
        self.checkpoint("input_shard", shard, "complete", {"records": count})
        return count

    def finalize_bands(self) -> None:
        if self.checkpoint_complete("lsh_bands", "all"):
            return
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS bands_lookup ON bands(strategy, band, band_key, record_id)"
        )
        self.connection.commit()
        self.checkpoint("lsh_bands", "all", "complete", {
            "band_rows": self.connection.execute("SELECT COUNT(*) FROM bands").fetchone()[0]
        })

    def generate_candidates(self, *, commit_every_buckets: int = 1000) -> dict[str, int]:
        """Enumerate safe buckets globally; account for and filter every hot bucket."""
        if self.checkpoint_complete("candidate_generation", "all"):
            return self.statistics()
        self.finalize_bands()
        completed = [
            (int(unit), json.loads(detail))
            for unit, detail in self.connection.execute(
                "SELECT unit, detail FROM checkpoints WHERE stage='lsh_band_partition' "
                "AND status='complete' ORDER BY CAST(unit AS INTEGER)"
            )
        ]
        resume_after = completed[-1][0] if completed else 0
        previous = completed[-1][1] if completed else {}
        buckets = self.connection.execute(
            "SELECT strategy, band, band_key, COUNT(*) AS n FROM bands "
            "GROUP BY strategy, band, band_key ORDER BY strategy, band, band_key"
        )
        regular = int(previous.get("regular_buckets", 0))
        hot = int(previous.get("hot_buckets", 0))
        potential = int(previous.get("potential_pairs", 0))
        examined = int(previous.get("examined_pair_occurrences", 0))
        for bucket_number, (strategy, band, band_key, size) in enumerate(buckets, 1):
            if bucket_number <= resume_after:
                continue
            size = int(size)
            possible = size * (size - 1) // 2
            potential += possible
            if size > self.maximum_bucket_size:
                hot += 1
                self.connection.execute(
                    "INSERT OR REPLACE INTO hot_buckets VALUES (?, ?, ?, ?, ?, 0, ?)",
                    (strategy, band, band_key, size, possible, "high_frequency_template_filtered"),
                )
            else:
                regular += 1
                ids = [row[0] for row in self.connection.execute(
                    "SELECT record_id FROM bands WHERE strategy=? AND band=? AND band_key=? "
                    "ORDER BY record_id", (strategy, band, band_key)
                )]
                keys = dict(self.connection.execute(
                    f"SELECT record_id, entity_key FROM records WHERE record_id IN "
                    f"({','.join('?' for _ in ids)})", ids
                )) if ids else {}
                rows = []
                ordered_ids = sorted(ids, key=lambda value: keys[value])
                for offset, left in enumerate(ordered_ids):
                    for right in ordered_ids[offset + 1:]:
                        examined += 1
                        rows.append((
                            deterministic_partition(keys[left], keys[right], self.partitions),
                            left, right, "minhash_lsh",
                        ))
                self.connection.executemany(
                    "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?)", rows
                )
            if bucket_number % commit_every_buckets == 0:
                self.connection.commit()
                self.checkpoint("lsh_band_partition", str(bucket_number), "complete", {
                    "buckets": bucket_number,
                    "regular_buckets": regular,
                    "hot_buckets": hot,
                    "potential_pairs": potential,
                    "examined_pair_occurrences": examined,
                })
                self.monitor.check()
        self.connection.commit()
        detail = {
            "regular_buckets": regular,
            "hot_buckets": hot,
            "potential_pairs": potential,
            "examined_pair_occurrences": examined,
            "deduplicated_candidates": self.connection.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0],
        }
        self.checkpoint("candidate_generation", "all", "complete", detail)
        return detail

    def add_candidate(self, left_id: int, right_id: int, source: str) -> None:
        left, right = int(left_id), int(right_id)
        if left == right:
            return
        keys = dict(self.connection.execute(
            "SELECT record_id, entity_key FROM records WHERE record_id IN (?, ?)", (left, right)
        ))
        if len(keys) != 2:
            raise ValueError("Candidate references an unknown record")
        if keys[right] < keys[left]:
            left, right = right, left
        partition = deterministic_partition(keys[left], keys[right], self.partitions)
        previous = self.connection.execute(
            "SELECT sources FROM candidates WHERE left_id=? AND right_id=?", (left, right)
        ).fetchone()
        sources = sorted(set((previous[0].split(",") if previous else []) + [source]))
        self.connection.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?, ?, ?, ?)",
            (partition, left, right, ",".join(sources)),
        )

    def candidate_partitions(self) -> Iterator[int]:
        for value, in self.connection.execute(
            "SELECT DISTINCT partition_id FROM candidates ORDER BY partition_id"
        ):
            yield int(value)

    def iter_candidates(self, partition: int) -> Iterator[tuple[dict, dict, list[str]]]:
        query = (
            "SELECT c.sources, l.entity_key, l.parent_key, l.sequence_index, "
            "l.strategy, l.shingle_size, l.shingles, "
            "l.signature, l.normalized_hash, l.payload, r.entity_key, r.strategy, "
            "r.parent_key, r.sequence_index, r.shingle_size, r.shingles, r.signature, "
            "r.normalized_hash, r.payload "
            "FROM candidates c JOIN records l ON l.record_id=c.left_id "
            "JOIN records r ON r.record_id=c.right_id WHERE c.partition_id=? "
            "ORDER BY c.left_id, c.right_id"
        )
        for row in self.connection.execute(query, (int(partition),)):
            sources = row[0].split(",")
            left = {
                "entity_key": row[1], "parent_key": row[2], "sequence_index": row[3],
                "strategy": row[4], "shingle_size": row[5],
                "shingles": frozenset(unpack_u64(row[6])), "signature": unpack_u64(row[7]),
                "normalized_hash": row[8], **json.loads(row[9]),
            }
            right = {
                "entity_key": row[10], "strategy": row[11], "parent_key": row[12],
                "sequence_index": row[13], "shingle_size": row[14],
                "shingles": frozenset(unpack_u64(row[15])), "signature": unpack_u64(row[16]),
                "normalized_hash": row[17], **json.loads(row[18]),
            }
            yield left, right, sources

    def statistics(self) -> dict[str, int]:
        values = {
            "records": "SELECT COUNT(*) FROM records",
            "band_rows": "SELECT COUNT(*) FROM bands",
            "deduplicated_candidates": "SELECT COUNT(*) FROM candidates",
            "hot_buckets": "SELECT COUNT(*) FROM hot_buckets",
            "hot_bucket_potential_pairs": "SELECT COALESCE(SUM(potential_pairs), 0) FROM hot_buckets",
            "hot_bucket_examined_pairs": "SELECT COALESCE(SUM(examined_pairs), 0) FROM hot_buckets",
        }
        return {key: int(self.connection.execute(query).fetchone()[0]) for key, query in values.items()}

    def atomic_checkpoint_export(self, path: Path) -> None:
        payload = {
            "format": "vibiomir-phase3-disk-lsh-checkpoint-v1",
            "input_fingerprint": self.input_fingerprint,
            "config_hash": self.config_hash,
            "shingling_version": SHINGLING_VERSION,
            "statistics": self.statistics(),
            "checkpoints": [
                {"stage": row[0], "unit": row[1], "status": row[2], "detail": json.loads(row[3])}
                for row in self.connection.execute(
                    "SELECT stage, unit, status, detail FROM checkpoints ORDER BY stage, unit"
                )
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".partial"
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
