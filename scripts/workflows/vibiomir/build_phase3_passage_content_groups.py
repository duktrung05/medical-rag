"""Build deterministic passage content identities without merging occurrences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.normalization import (  # noqa: E402
    NORMALIZATION_VERSION,
    normalize_medical_text,
    text_sha256,
)
from src.medrag.data.phase1.selection import sha256_file  # noqa: E402
from src.medrag.data.phase3.passage import (  # noqa: E402
    PASSAGE_CONTENT_SCHEMA,
    PASSAGE_CONTENT_SCHEMA_VERSION,
    PASSAGE_GROUPING_POLICY_VERSION,
    passage_content_group_id,
    passage_group_row,
    verify_group_content,
)


EXPECTED_PHASE2_RELEASE_SHA256 = (
    "e1ecd3302dde6b705e2368daaf6c864a18a2831c08acfdbde8cac3c211e141f8"
)
EXPECTED_PHASE2_VALIDATION_SHA256 = (
    "39d6fc6bcad775652fb5bc14ec7fcfebe0c8704c3690874387ca1b24030a67d4"
)
EXPECTED_TASK2_OUTPUT_SHA256 = (
    "cebc2568f80085518c82bacfed22f10cfbd909faa77471fc255434d0f08148ef"
)
EXPECTED_TASK3_OUTPUT_SHA256 = (
    "192820cbfdeb65b7e5dad564d545664fbcb17c4fdbe65e59882f1185cc99e6ec"
)
EXPECTED_PASSAGES = 2_951_273
EXPECTED_DOCUMENTS = 1_073_531
FORMAT = "vibiomir-phase3-passage-content-build-v1"


INPUT_COLUMNS = [
    "passage_id",
    "doc_id",
    "official_url",
    "passage_index",
    "passage_text",
    "char_start",
    "char_end",
    "source_text_sha256",
    "text_representation_sha256",
    "chunker_version",
]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output(path: Path, protected: list[Path]) -> Path:
    resolved = path.resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if resolved.parent != outputs_root or not resolved.name.startswith("vibiomir_phase3_"):
        raise ValueError("Output must be a direct outputs/vibiomir_phase3_<run_id> namespace")
    for source in protected:
        source = source.resolve()
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError(f"Phase 3 output overlaps immutable input: {source}")
    return resolved


def _select_parts(entries: list[dict], max_parts: int | None, full: bool) -> list[dict]:
    ordered = sorted(entries, key=lambda item: item["part"])
    if full:
        if max_parts is not None:
            raise ValueError("--full and --max-parts are mutually exclusive")
        return ordered
    if max_parts is None or max_parts < 1:
        raise ValueError("Dry-run requires --max-parts >= 1")
    if max_parts >= len(ordered):
        raise ValueError("Dry-run must select fewer than all release parts")
    return ordered[:max_parts]


def _verify_full_gate(path: Path | None) -> dict:
    if path is None or not path.is_file():
        raise ValueError("Full build requires --pilot-acceptance pointing to an accepted manifest")
    acceptance = _load_json(path)
    if acceptance.get("status") != "accepted" or acceptance.get("phase") != "phase3_pilot":
        raise ValueError("Phase 3 pilot has not been accepted")
    return acceptance


def _quick_schema_check(path: Path) -> None:
    schema = pq.read_schema(path)
    if not set(INPUT_COLUMNS).issubset(schema.names):
        missing = sorted(set(INPUT_COLUMNS) - set(schema.names))
        raise ValueError(f"Phase 2 passage schema missing fields {missing}: {path}")
    expected = {
        "passage_id": pa.string(),
        "doc_id": pa.int64(),
        "passage_index": pa.int32(),
        "char_start": pa.int64(),
        "char_end": pa.int64(),
    }
    for name, data_type in expected.items():
        if schema.field(name).type != data_type:
            raise ValueError(f"Phase 2 passage schema type mismatch for {name}: {path}")


def _open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS occurrences (
            passage_id TEXT PRIMARY KEY,
            raw_hash BLOB NOT NULL,
            normalized_hash BLOB NOT NULL,
            doc_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS raw_groups (
            hash BLOB PRIMARY KEY,
            occurrence_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS normalized_groups (
            hash BLOB PRIMARY KEY,
            occurrence_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS normalized_group_documents (
            hash BLOB NOT NULL,
            doc_id INTEGER NOT NULL,
            PRIMARY KEY (hash, doc_id)
        );
        CREATE TABLE IF NOT EXISTS processed_parts (
            part TEXT PRIMARY KEY,
            rows INTEGER NOT NULL,
            sha256 TEXT NOT NULL
        );
        """
    )
    return connection


def _passage_hashes(text: str) -> tuple[str, str, str]:
    if not isinstance(text, str) or not text:
        raise ValueError("passage_text must be a non-empty string")
    normalized = normalize_medical_text(text).normalized_text
    return text_sha256(text), text_sha256(normalized), normalized


def _count_parts(
    connection: sqlite3.Connection,
    passages_root: Path,
    entries: list[dict],
    checkpoint_path: Path,
    *,
    scope: str,
) -> int:
    completed = {
        row[0] for row in connection.execute("SELECT part FROM processed_parts")
    }
    for number, entry in enumerate(entries, 1):
        part = entry["part"]
        if part in completed:
            print(f"Reused counted shard {number}/{len(entries)} {part}", flush=True)
            continue
        path = passages_root / f"{part}.parquet"
        _quick_schema_check(path)
        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(f"Phase 2 passage fingerprint changed: {path}")
        rows = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=4096, columns=INPUT_COLUMNS
            ):
                occurrence_rows = []
                raw_hashes = []
                normalized_hashes = []
                normalized_documents = []
                for row in batch.to_pylist():
                    raw_hash, normalized_hash, _ = _passage_hashes(row["passage_text"])
                    raw_bytes = bytes.fromhex(raw_hash)
                    normalized_bytes = bytes.fromhex(normalized_hash)
                    occurrence_rows.append(
                        (row["passage_id"], raw_bytes, normalized_bytes, int(row["doc_id"]))
                    )
                    raw_hashes.append((raw_bytes,))
                    normalized_hashes.append((normalized_bytes,))
                    normalized_documents.append((normalized_bytes, int(row["doc_id"])))
                connection.executemany(
                    "INSERT INTO occurrences(passage_id, raw_hash, normalized_hash, doc_id) "
                    "VALUES (?, ?, ?, ?)",
                    occurrence_rows,
                )
                connection.executemany(
                    "INSERT INTO raw_groups(hash, occurrence_count) VALUES (?, 1) "
                    "ON CONFLICT(hash) DO UPDATE SET occurrence_count=occurrence_count+1",
                    raw_hashes,
                )
                connection.executemany(
                    "INSERT INTO normalized_groups(hash, occurrence_count) VALUES (?, 1) "
                    "ON CONFLICT(hash) DO UPDATE SET occurrence_count=occurrence_count+1",
                    normalized_hashes,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO normalized_group_documents(hash, doc_id) VALUES (?, ?)",
                    normalized_documents,
                )
                rows += batch.num_rows
            if rows != entry["rows"]:
                raise ValueError(f"Manifest row count mismatch: {path}")
            connection.execute(
                "INSERT INTO processed_parts(part, rows, sha256) VALUES (?, ?, ?)",
                (part, rows, actual_sha256),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        _atomic_json(
            checkpoint_path,
            {
                "format": "vibiomir-phase3-task4-progress-v1",
                "status": "counting",
                "scope": scope,
                "completed_parts": number,
                "total_parts": len(entries),
                "last_completed_part": part,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            replace=checkpoint_path.exists(),
        )
        print(f"Counted passage hashes {number}/{len(entries)} rows={rows}", flush=True)
    return int(connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0])


def _lookup_counts(
    connection: sqlite3.Connection,
    table: str,
    hashes: set[bytes],
) -> dict[bytes, int]:
    result: dict[bytes, int] = {}
    values = list(hashes)
    for start in range(0, len(values), 800):
        chunk = values[start:start + 800]
        placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT hash, occurrence_count FROM {table} WHERE hash IN ({placeholders})"
        result.update((bytes(row[0]), int(row[1])) for row in connection.execute(query, chunk))
    return result


def _lookup_distinct_documents(
    connection: sqlite3.Connection, hashes: set[bytes]
) -> dict[bytes, int]:
    result: dict[bytes, int] = {}
    values = list(hashes)
    for start in range(0, len(values), 800):
        chunk = values[start:start + 800]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT hash, COUNT(*) FROM normalized_group_documents "
            f"WHERE hash IN ({placeholders}) GROUP BY hash"
        )
        result.update((bytes(row[0]), int(row[1])) for row in connection.execute(query, chunk))
    return result


def _update_identity_digest(digest: "hashlib._Hash", values: tuple[object, ...]) -> None:
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def _write_output(
    connection: sqlite3.Connection,
    passages_root: Path,
    entries: list[dict],
    destination: Path,
    *,
    phase2_release_id: str,
    scope: str,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    writer = None
    raw_baselines: dict[bytes, tuple[str, str]] = {}
    normalized_baselines: dict[bytes, tuple[str, str]] = {}
    input_identity = hashlib.sha256()
    output_identity = hashlib.sha256()
    deterministic_identity = hashlib.sha256()
    stats: Counter[str] = Counter()
    written = 0
    try:
        writer = pq.ParquetWriter(
            temporary,
            PASSAGE_CONTENT_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        for number, entry in enumerate(entries, 1):
            path = passages_root / f"{entry['part']}.parquet"
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=4096, columns=INPUT_COLUMNS
            ):
                prepared = []
                raw_hashes: set[bytes] = set()
                normalized_hashes: set[bytes] = set()
                for passage in batch.to_pylist():
                    raw_hash, normalized_hash, normalized_text = _passage_hashes(
                        passage["passage_text"]
                    )
                    raw_bytes = bytes.fromhex(raw_hash)
                    normalized_bytes = bytes.fromhex(normalized_hash)
                    raw_hashes.add(raw_bytes)
                    normalized_hashes.add(normalized_bytes)
                    prepared.append(
                        (passage, raw_hash, normalized_hash, normalized_text, raw_bytes, normalized_bytes)
                    )
                raw_counts = _lookup_counts(connection, "raw_groups", raw_hashes)
                normalized_counts = _lookup_counts(
                    connection, "normalized_groups", normalized_hashes
                )
                distinct_documents = _lookup_distinct_documents(
                    connection, normalized_hashes
                )
                output_rows = []
                for passage, raw_hash, normalized_hash, normalized_text, raw_bytes, normalized_bytes in prepared:
                    raw_size = raw_counts[raw_bytes]
                    normalized_size = normalized_counts[normalized_bytes]
                    document_count = distinct_documents[normalized_bytes]
                    if raw_size > 1:
                        verify_group_content(
                            raw_baselines,
                            raw_bytes,
                            passage_id=passage["passage_id"],
                            text=passage["passage_text"],
                            label="passage_text",
                        )
                    if normalized_size > 1:
                        verify_group_content(
                            normalized_baselines,
                            normalized_bytes,
                            passage_id=passage["passage_id"],
                            text=normalized_text,
                            label="normalized_passage_text",
                        )
                    row = passage_group_row(
                        passage,
                        content_sha256=raw_hash,
                        normalized_content_hash=normalized_hash,
                        raw_group_size=raw_size,
                        content_group_size=normalized_size,
                        distinct_document_count=document_count,
                        normalization_version=NORMALIZATION_VERSION,
                        phase2_release_id=phase2_release_id,
                        grouping_scope=scope,
                    )
                    identity = (
                        passage["passage_id"],
                        int(passage["doc_id"]),
                        passage["official_url"],
                        int(passage["char_start"]),
                        int(passage["char_end"]),
                        passage["source_text_sha256"],
                    )
                    output_row_identity = (
                        row["passage_id"],
                        row["doc_id"],
                        row["official_url"],
                        row["char_start"],
                        row["char_end"],
                        row["source_text_sha256"],
                    )
                    _update_identity_digest(input_identity, identity)
                    _update_identity_digest(output_identity, output_row_identity)
                    _update_identity_digest(
                        deterministic_identity,
                        (
                            row["passage_id"],
                            row["normalized_content_hash"],
                            row["passage_content_group_id"],
                        ),
                    )
                    if raw_size > 1:
                        stats["raw_exact_duplicate_occurrences"] += 1
                    if normalized_size > 1:
                        stats["normalized_exact_duplicate_occurrences"] += 1
                    if document_count > 1:
                        stats["cross_document_duplicate_occurrences"] += 1
                    output_rows.append(row)
                writer.write_table(
                    pa.Table.from_pylist(output_rows, schema=PASSAGE_CONTENT_SCHEMA)
                )
                written += len(output_rows)
            print(f"Wrote passage groups {number}/{len(entries)} passages={written}", flush=True)
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    if input_identity.digest() != output_identity.digest():
        raise ValueError("Passage source occurrence identity changed during grouping")
    aggregate = connection.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM raw_groups),
          (SELECT COUNT(*) FROM raw_groups WHERE occurrence_count > 1),
          (SELECT MAX(occurrence_count) FROM raw_groups),
          (SELECT COUNT(*) FROM normalized_groups),
          (SELECT COUNT(*) FROM normalized_groups WHERE occurrence_count > 1),
          (SELECT MAX(occurrence_count) FROM normalized_groups),
          (SELECT COUNT(*) FROM (
             SELECT hash FROM normalized_group_documents GROUP BY hash HAVING COUNT(*) > 1
          ))
        """
    ).fetchone()
    return {
        "passages": written,
        "unique_raw_content_groups": int(aggregate[0]),
        "raw_duplicate_groups": int(aggregate[1]),
        "maximum_raw_group_size": int(aggregate[2] or 0),
        "unique_normalized_content_groups": int(aggregate[3]),
        "normalized_duplicate_groups": int(aggregate[4]),
        "maximum_normalized_group_size": int(aggregate[5] or 0),
        "cross_document_content_groups": int(aggregate[6]),
        "raw_exact_duplicate_occurrences": stats["raw_exact_duplicate_occurrences"],
        "normalized_exact_duplicate_occurrences": stats["normalized_exact_duplicate_occurrences"],
        "cross_document_duplicate_occurrences": stats["cross_document_duplicate_occurrences"],
        "verified_raw_duplicate_groups": len(raw_baselines),
        "verified_normalized_duplicate_groups": len(normalized_baselines),
        "source_occurrence_identity_sha256": input_identity.hexdigest(),
        "deterministic_grouping_sha256": deterministic_identity.hexdigest(),
    }


def _verify_output_determinism(path: Path, expected_digest: str) -> None:
    digest = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=8192,
        columns=["passage_id", "normalized_content_hash", "passage_content_group_id"],
    ):
        for row in batch.to_pylist():
            expected_group = passage_content_group_id(row["normalized_content_hash"])
            if row["passage_content_group_id"] != expected_group:
                raise ValueError("Non-deterministic passage content group ID")
            _update_identity_digest(
                digest,
                (row["passage_id"], row["normalized_content_hash"], expected_group),
            )
    if digest.hexdigest() != expected_digest:
        raise ValueError("Deterministic grouping digest mismatch")


def _rerun_grouping_digest(passages_root: Path, entries: list[dict]) -> str:
    """Independently rerun identity derivation over the same ordered input."""
    digest = hashlib.sha256()
    for entry in entries:
        path = passages_root / f"{entry['part']}.parquet"
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=["passage_id", "passage_text"]
        ):
            for row in batch.to_pylist():
                _, normalized_hash, _ = _passage_hashes(row["passage_text"])
                _update_identity_digest(
                    digest,
                    (
                        row["passage_id"],
                        normalized_hash,
                        passage_content_group_id(normalized_hash),
                    ),
                )
    return digest.hexdigest()


def _verify_pilot_alignment(task2_groups: Path, selected_doc_ids: set[int]) -> str:
    fingerprint = sha256_file(task2_groups)
    if fingerprint != EXPECTED_TASK2_OUTPUT_SHA256:
        raise ValueError("Task 2 pilot fingerprint changed")
    task2_ids = set(
        int(value)
        for value in pq.read_table(task2_groups, columns=["doc_id"])["doc_id"].to_pylist()
    )
    if task2_ids != selected_doc_ids:
        raise ValueError("Task 4 pilot documents do not match the Task 2 pilot scope")
    return fingerprint


def _verify_full_upstreams(task2_groups: Path, task3_pairs: Path) -> tuple[str, str]:
    task2_sha = sha256_file(task2_groups)
    task2_validation_path = task2_groups.parents[1] / (
        "validation/task2_exact_document_validation.json"
    )
    task2_validation = _load_json(task2_validation_path)
    if (
        task2_validation.get("status") != "pass"
        or not task2_validation.get("groups_are_corpus_complete")
        or task2_validation.get("output", {}).get("sha256") != task2_sha
        or task2_validation.get("output", {}).get("rows") != EXPECTED_DOCUMENTS
    ):
        raise ValueError("Task 4 full requires a validated corpus-complete Task 2")
    task3_sha = sha256_file(task3_pairs)
    task3_validation = _load_json(
        task3_pairs.parents[1] / "validation/task3_near_document_validation.json"
    )
    task3_recorded_sha = (
        task3_validation.get("output_sha256")
        or task3_validation.get("output", {}).get("sha256")
    )
    if (
        task3_validation.get("status") != "pass"
        or task3_validation.get("scope") != "full_corpus"
        or task3_recorded_sha != task3_sha
        or task3_validation.get("auto_merge_allowed") is not False
    ):
        raise ValueError("Task 4 full requires a validated review-only full Task 3")
    return task2_sha, task3_sha


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    phase2_release_path = args.phase2_release.resolve()
    phase2_root = phase2_release_path.parent.parent
    passages_root = phase2_root / "passages"
    output = _safe_output(args.output, [phase2_root])
    destination = output / "passage_dedup/passage_content_groups.parquet"
    validation_path = output / "validation/task4_passage_content_validation.json"
    snapshot_path = output / "manifests/input_snapshot.json"
    config_path = output / "manifests/dedup_config.json"
    checkpoint_path = output / "checkpoints/task4_passage_content_groups.json"
    report_path = output / "reports/TASK4_PASSAGE_CONTENT_GROUPING_REPORT.md"
    index_path = output / "work/task4_passage_groups.sqlite"

    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if destination.exists() or validation_path.exists():
        raise FileExistsError("Task 4 destination is already complete; use a new run namespace")
    output.mkdir(parents=True, exist_ok=True)

    phase2_release = _load_json(phase2_release_path)
    phase2_release_sha256 = sha256_file(phase2_release_path)
    if phase2_release_sha256 != args.expected_phase2_release_sha256:
        raise ValueError("Phase 2 release fingerprint changed; stop and rerun verification")
    if phase2_release.get("status") != "complete_validated" or phase2_release.get("scope") != "full":
        raise ValueError("Phase 2 release is not a validated full release")
    if phase2_release.get("passages") != EXPECTED_PASSAGES:
        raise ValueError("Unexpected Phase 2 passage count")
    if phase2_release.get("documents") != EXPECTED_DOCUMENTS:
        raise ValueError("Unexpected Phase 2 document count")
    if phase2_release.get("passage_schema_version") != "vibiomir-phase2-passage-v1":
        raise ValueError("Unexpected Phase 2 passage schema version")
    validation_ref = phase2_release.get("validation_report", {})
    phase2_validation_path = phase2_root / validation_ref.get("path", "")
    phase2_validation_sha256 = sha256_file(phase2_validation_path)
    if (
        validation_ref.get("status") != "pass"
        or phase2_validation_sha256 != validation_ref.get("sha256")
        or phase2_validation_sha256 != EXPECTED_PHASE2_VALIDATION_SHA256
    ):
        raise ValueError("Phase 2 validation fingerprint changed")

    entries = _select_parts(phase2_release.get("passage_parts", []), args.max_parts, args.full)
    if len(phase2_release.get("passage_parts", [])) != 922:
        raise ValueError("Unexpected Phase 2 shard inventory")
    acceptance = _verify_full_gate(args.pilot_acceptance) if args.full else None
    scope = "full" if args.full else f"dry_run_first_{len(entries)}_passage_parts"
    _quick_schema_check(passages_root / f"{entries[0]['part']}.parquet")

    selected_doc_ids: set[int] = set()
    selected_rows = 0
    for entry in entries:
        table = pq.read_table(
            passages_root / f"{entry['part']}.parquet", columns=["doc_id"]
        )
        selected_doc_ids.update(int(value) for value in table["doc_id"].to_pylist())
        selected_rows += table.num_rows
    task2_sha256 = None
    if not args.full:
        task2_sha256 = _verify_pilot_alignment(args.task2_groups.resolve(), selected_doc_ids)
        task3_sha256 = sha256_file(args.task3_pairs.resolve())
        if task3_sha256 != EXPECTED_TASK3_OUTPUT_SHA256:
            raise ValueError("Task 3 near-duplicate fingerprint changed")
    else:
        task2_sha256, task3_sha256 = _verify_full_upstreams(
            args.task2_groups.resolve(), args.task3_pairs.resolve()
        )

    selected_stats = {
        entry["part"]: (
            (passages_root / f"{entry['part']}.parquet").stat().st_size,
            (passages_root / f"{entry['part']}.parquet").stat().st_mtime_ns,
        )
        for entry in entries
    }
    connection = _open_index(index_path)
    try:
        indexed_rows = _count_parts(
            connection,
            passages_root,
            entries,
            checkpoint_path,
            scope=scope,
        )
        if indexed_rows != selected_rows:
            raise ValueError("Indexed passage count differs from selected input rows")
        if args.full and indexed_rows != EXPECTED_PASSAGES:
            raise ValueError("Full Task 4 build did not include every Phase 2 passage")
        statistics = _write_output(
            connection,
            passages_root,
            entries,
            destination,
            phase2_release_id=phase2_release_sha256,
            scope=scope,
        )
    finally:
        connection.close()

    output_meta = pq.read_metadata(destination)
    if output_meta.num_rows != selected_rows:
        raise ValueError("One-output-row-per-input-passage invariant failed")
    _verify_output_determinism(
        destination, statistics["deterministic_grouping_sha256"]
    )
    rerun_digest = _rerun_grouping_digest(passages_root, entries)
    if rerun_digest != statistics["deterministic_grouping_sha256"]:
        raise ValueError("Grouping rerun produced different content identities")
    if sha256_file(phase2_release_path) != phase2_release_sha256:
        raise ValueError("Phase 2 release changed during Task 4")
    for entry in entries:
        path = passages_root / f"{entry['part']}.parquet"
        if (path.stat().st_size, path.stat().st_mtime_ns) != selected_stats[entry["part"]]:
            raise ValueError(f"Phase 2 passage shard changed during Task 4: {path}")

    completed = datetime.now(timezone.utc).isoformat()
    output_sha256 = sha256_file(destination)
    snapshot = {
        "format": "vibiomir-phase3-task4-input-snapshot-v1",
        "scope": scope,
        "selected_parts": [entry["part"] for entry in entries],
        "selected_passages": selected_rows,
        "selected_documents": len(selected_doc_ids),
        "groups_are_corpus_complete": args.full,
        "phase2_release_manifest": str(phase2_release_path.relative_to(ROOT)),
        "phase2_release_manifest_sha256": phase2_release_sha256,
        "phase2_validation_sha256": phase2_validation_sha256,
        "task2_output_sha256": task2_sha256,
        "task3_near_pairs_sha256": task3_sha256,
        "input_verification_mode": "incremental_verified_manifest_plus_selected_shards",
        "pilot_acceptance": acceptance,
    }
    config = {
        "format": "vibiomir-phase3-dedup-config-v1",
        "task": 4,
        "scope": scope,
        "schema_version": PASSAGE_CONTENT_SCHEMA_VERSION,
        "grouping_policy_version": PASSAGE_GROUPING_POLICY_VERSION,
        "content_sha256_basis": "SHA-256(UTF-8 passage_text)",
        "normalized_content_hash_basis": f"SHA-256({NORMALIZATION_VERSION}(passage_text))",
        "passage_content_group_basis": "normalized exact content only",
        "near_duplicate_relations_imported": False,
        "near_duplicate_auto_merge": False,
        "source_occurrences_deleted_or_merged": False,
        "processing": "streaming parquet batches with disk-backed SQLite aggregation",
        "batch_rows": 4096,
        "resume_supported": True,
    }
    validation = {
        "format": FORMAT,
        "status": "pass",
        "task": 4,
        "scope": scope,
        "groups_are_corpus_complete": args.full,
        "completed_at_utc": completed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output": {
            "path": str(destination.relative_to(ROOT)),
            "rows": output_meta.num_rows,
            "row_groups": output_meta.num_row_groups,
            "bytes": destination.stat().st_size,
            "sha256": output_sha256,
            "schema_version": PASSAGE_CONTENT_SCHEMA_VERSION,
        },
        "statistics": statistics,
        "checks": {
            "one_output_row_per_input_passage": True,
            "passage_id_unique_and_preserved": True,
            "doc_id_preserved": True,
            "official_url_preserved": True,
            "source_offsets_preserved": True,
            "source_text_sha256_preserved": True,
            "normalized_exact_content_grouped_together": True,
            "repeated_hash_content_exactly_verified": True,
            "near_duplicates_auto_merged": False,
            "deterministic_grouping_rerun": True,
            "task2_pilot_document_scope_preserved": not args.full,
            "input_fingerprints_stable": True,
            "phase0_mutated": False,
            "phase1_mutated": False,
            "phase2_mutated": False,
            "index_mutated": False,
        },
    }
    _atomic_json(snapshot_path, snapshot)
    _atomic_json(config_path, config)
    _atomic_json(validation_path, validation)
    checkpoint = {
        "format": "vibiomir-phase3-task4-checkpoint-v1",
        "status": "complete",
        "scope": scope,
        "completed_parts": len(entries),
        "output_rows": output_meta.num_rows,
        "output_sha256": output_sha256,
        "phase2_release_manifest_sha256": phase2_release_sha256,
        "completed_at_utc": completed,
    }
    _atomic_json(checkpoint_path, checkpoint, replace=True)
    report = f"""# Phase 3 Task 4 — Passage Content Identity

- Status: **PASS**
- Scope: `{scope}`
- Corpus-complete groups: `{str(args.full).lower()}`
- Passage occurrences preserved: `{statistics['passages']:,}`
- Unique normalized content groups: `{statistics['unique_normalized_content_groups']:,}`
- Normalized duplicate groups: `{statistics['normalized_duplicate_groups']:,}`
- Cross-document content groups: `{statistics['cross_document_content_groups']:,}`
- Output SHA-256: `{output_sha256}`
- Near-duplicate relations auto-merged: `false`
- Input verification: reused the verified Phase 2 release and validation fingerprints; checked only selected direct-input shards.
- Source occurrences deleted or merged: `false`

Every passage occurrence retains its passage ID, document ID, URL, source offsets,
source checksum, and chunker version. Content identity is deterministic and based
only on exact equality after conservative normalization.

{('This is a corpus-complete Task 4 component; final Phase 3 release validation is separate.' if args.full else 'This run is bounded and is not a Full Phase 3 release.')}
"""
    _atomic_text(report_path, report)
    shutil.rmtree(index_path.parent, ignore_errors=True)
    return validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-release",
        type=Path,
        default=ROOT / "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json",
    )
    parser.add_argument(
        "--task2-groups",
        type=Path,
        default=ROOT / "outputs/vibiomir_phase3_20261015_task2_dry_run_v2/document_dedup/exact_document_groups.parquet",
    )
    parser.add_argument(
        "--task3-pairs",
        type=Path,
        default=ROOT / "outputs/vibiomir_phase3_20261015_task3_dry_run_v1/document_dedup/near_duplicate_pairs.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parts", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pilot-acceptance", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--expected-phase2-release-sha256",
        default=EXPECTED_PHASE2_RELEASE_SHA256,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
