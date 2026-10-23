"""Disk-backed full-corpus canonical selection for Phase 3 Task 6."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from scripts.workflows.vibiomir.phase3_scalable_candidates import (
    EXPECTED_DOCUMENTS,
    EXPECTED_PASSAGES,
    _atomic_json,
    _atomic_text,
    _configuration_hash,
    _safe_output,
    _write_partition,
)
from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase3.canonical import (
    CANONICAL_MEMBERSHIP_SCHEMA,
    CANONICAL_POLICY_VERSION,
    CANONICAL_REPRESENTATIVE_SCHEMA,
    non_whitespace_length,
    provenance_score,
    select_representative,
)
from src.medrag.data.phase3.disk_lsh import ResourceMonitor


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _checkpoint(connection: sqlite3.Connection, stage: str, unit: str, detail: dict) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?)",
        (stage, unit, "complete", json.dumps(detail, sort_keys=True)),
    )
    connection.commit()


def _complete(connection: sqlite3.Connection, stage: str, unit: str) -> bool:
    row = connection.execute(
        "SELECT status FROM checkpoints WHERE stage=? AND unit=?", (stage, unit)
    ).fetchone()
    return bool(row and row[0] == "complete")


def _schema(connection: sqlite3.Connection, fingerprint: str, config_hash: str) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints(
          stage TEXT NOT NULL, unit TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL,
          PRIMARY KEY(stage, unit)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS needed_docs(doc_id INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS document_quality(
          doc_id INTEGER PRIMARY KEY, source_kind TEXT, normalized_text TEXT,
          quality_review_required INTEGER, quality_flags TEXT, language_confidence REAL,
          phase0_raw_path TEXT, phase0_raw_sha256 TEXT, raw_gzip_sha256 TEXT,
          raw_body_sha256 TEXT, pmid TEXT, extractor_version TEXT,
          source_document_path TEXT, source_document_sha256 TEXT
        );
        CREATE TABLE IF NOT EXISTS document_members(
          group_id TEXT NOT NULL, group_size INTEGER NOT NULL, doc_id INTEGER PRIMARY KEY,
          official_url TEXT NOT NULL, source_text_sha256 TEXT NOT NULL,
          normalized_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS document_members_group ON document_members(group_id, doc_id);
        CREATE TABLE IF NOT EXISTS passage_members(
          group_id TEXT NOT NULL, group_size INTEGER NOT NULL, passage_id TEXT PRIMARY KEY,
          doc_id INTEGER NOT NULL, official_url TEXT NOT NULL, passage_index INTEGER NOT NULL,
          char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
          source_text_sha256 TEXT NOT NULL, normalized_hash TEXT NOT NULL,
          passage_text TEXT
        );
        CREATE INDEX IF NOT EXISTS passage_members_group ON passage_members(group_id, passage_id);
        """
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    expected = {"input_fingerprint": fingerprint, "config_hash": config_hash}
    if existing and existing != expected:
        raise ValueError("Canonical resume fingerprint mismatch")
    if not existing:
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", expected.items())
        connection.commit()


def _load_group_members(
    connection: sqlite3.Connection,
    task2_path: Path,
    task4_path: Path,
    monitor: ResourceMonitor,
) -> tuple[int, int]:
    if not _complete(connection, "groups", "document"):
        count = 0
        columns = [
            "doc_id", "official_url", "source_text_sha256", "normalized_text_sha256",
            "document_content_group_id", "normalized_exact_group_size",
            "normalized_exact_duplicate",
        ]
        for batch in pq.ParquetFile(task2_path).iter_batches(batch_size=4096, columns=columns):
            rows = [row for row in batch.to_pylist() if row["normalized_exact_duplicate"]]
            connection.executemany(
                "INSERT OR IGNORE INTO document_members VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (row["document_content_group_id"], int(row["normalized_exact_group_size"]),
                     int(row["doc_id"]), row["official_url"], row["source_text_sha256"],
                     row["normalized_text_sha256"])
                    for row in rows
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO needed_docs VALUES (?)",
                [(int(row["doc_id"]),) for row in rows],
            )
            count += len(rows)
        connection.commit()
        _checkpoint(connection, "groups", "document", {"members": count})
        monitor.check()
    if not _complete(connection, "groups", "passage"):
        count = 0
        columns = [
            "passage_id", "doc_id", "official_url", "passage_index", "char_start", "char_end",
            "source_text_sha256", "normalized_content_hash", "passage_content_group_id",
            "content_group_size",
        ]
        for batch in pq.ParquetFile(task4_path).iter_batches(batch_size=4096, columns=columns):
            rows = [row for row in batch.to_pylist() if int(row["content_group_size"]) > 1]
            connection.executemany(
                "INSERT OR IGNORE INTO passage_members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                [
                    (row["passage_content_group_id"], int(row["content_group_size"]),
                     row["passage_id"], int(row["doc_id"]), row["official_url"],
                     int(row["passage_index"]), int(row["char_start"]), int(row["char_end"]),
                     row["source_text_sha256"], row["normalized_content_hash"])
                    for row in rows
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO needed_docs VALUES (?)",
                [(int(row["doc_id"]),) for row in rows],
            )
            count += len(rows)
        connection.commit()
        _checkpoint(connection, "groups", "passage", {"members": count})
        monitor.check()
    return (
        int(connection.execute("SELECT COUNT(*) FROM document_members").fetchone()[0]),
        int(connection.execute("SELECT COUNT(*) FROM passage_members").fetchone()[0]),
    )


def _load_document_quality(
    connection: sqlite3.Connection, phase1: Path, parts: list[str], monitor: ResourceMonitor
) -> None:
    columns = [
        "doc_id", "source_kind", "normalized_text", "quality_review_required", "quality_flags",
        "language_confidence", "phase0_raw_path", "phase0_raw_sha256", "raw_gzip_sha256",
        "raw_body_sha256", "pmid", "extractor_version", "source_document_path",
        "source_document_sha256",
    ]
    for part in parts:
        if _complete(connection, "phase1_quality_shard", part):
            continue
        matched = 0
        for batch in pq.ParquetFile(phase1 / f"clean_documents/{part}.parquet").iter_batches(
            batch_size=4096, columns=columns
        ):
            rows = batch.to_pylist()
            identifiers = [int(row["doc_id"]) for row in rows]
            placeholders = ",".join("?" for _ in identifiers)
            needed = {
                value[0] for value in connection.execute(
                    f"SELECT doc_id FROM needed_docs WHERE doc_id IN ({placeholders})", identifiers
                )
            }
            selected = [row for row in rows if int(row["doc_id"]) in needed]
            connection.executemany(
                "INSERT OR REPLACE INTO document_quality VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (int(row["doc_id"]), row["source_kind"], row["normalized_text"],
                     int(bool(row["quality_review_required"])), json.dumps(row["quality_flags"] or []),
                     float(row["language_confidence"] or 0.0), row["phase0_raw_path"],
                     row["phase0_raw_sha256"], row["raw_gzip_sha256"], row["raw_body_sha256"],
                     row["pmid"], row["extractor_version"], row["source_document_path"],
                     row["source_document_sha256"])
                    for row in selected
                ],
            )
            matched += len(selected)
        connection.commit()
        _checkpoint(connection, "phase1_quality_shard", part, {"matched": matched})
        monitor.check()
    needed = connection.execute("SELECT COUNT(*) FROM needed_docs").fetchone()[0]
    loaded = connection.execute("SELECT COUNT(*) FROM document_quality").fetchone()[0]
    if needed != loaded:
        raise ValueError(f"Canonical parent quality incomplete: needed={needed} loaded={loaded}")


def _load_passage_text(
    connection: sqlite3.Connection, phase2: Path, entries: list[dict], monitor: ResourceMonitor
) -> None:
    for entry in entries:
        part = entry["part"]
        if _complete(connection, "phase2_passage_shard", part):
            continue
        matched = 0
        path = phase2 / f"passages/{part}.parquet"
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=["passage_id", "passage_text"]
        ):
            rows = batch.to_pylist()
            connection.executemany(
                "UPDATE passage_members SET passage_text=? WHERE passage_id=?",
                [(row["passage_text"], row["passage_id"]) for row in rows],
            )
            matched += connection.execute("SELECT changes()").fetchone()[0]
        connection.commit()
        _checkpoint(connection, "phase2_passage_shard", part, {"matched": matched})
        monitor.check()
    missing = connection.execute(
        "SELECT COUNT(*) FROM passage_members WHERE passage_text IS NULL"
    ).fetchone()[0]
    if missing:
        raise ValueError(f"Canonical passage text incomplete: missing={missing}")


def _document_from_row(row: tuple) -> dict:
    return {
        "doc_id": row[2], "official_url": row[3], "source_text_sha256": row[4],
        "normalized_text_sha256": row[5], "source_kind": row[6], "normalized_text": row[7],
        "quality_review_required": bool(row[8]), "quality_flags": json.loads(row[9]),
        "language_confidence": row[10], "phase0_raw_path": row[11],
        "phase0_raw_sha256": row[12], "raw_gzip_sha256": row[13],
        "raw_body_sha256": row[14], "pmid": row[15], "extractor_version": row[16],
        "source_document_path": row[17], "source_document_sha256": row[18],
    }


def _iter_groups(connection: sqlite3.Connection, entity_type: str):
    if entity_type == "document":
        query = (
            "SELECT m.group_id, m.group_size, m.doc_id, m.official_url, m.source_text_sha256, "
            "m.normalized_hash, q.source_kind, q.normalized_text, q.quality_review_required, "
            "q.quality_flags, q.language_confidence, q.phase0_raw_path, q.phase0_raw_sha256, "
            "q.raw_gzip_sha256, q.raw_body_sha256, q.pmid, q.extractor_version, "
            "q.source_document_path, q.source_document_sha256 FROM document_members m "
            "JOIN document_quality q USING(doc_id) ORDER BY m.group_id, m.doc_id"
        )
    else:
        query = (
            "SELECT m.group_id, m.group_size, m.doc_id, m.official_url, m.source_text_sha256, "
            "m.normalized_hash, q.source_kind, q.normalized_text, q.quality_review_required, "
            "q.quality_flags, q.language_confidence, q.phase0_raw_path, q.phase0_raw_sha256, "
            "q.raw_gzip_sha256, q.raw_body_sha256, q.pmid, q.extractor_version, "
            "q.source_document_path, q.source_document_sha256, m.passage_id, m.passage_index, "
            "m.char_start, m.char_end, m.passage_text FROM passage_members m "
            "JOIN document_quality q USING(doc_id) ORDER BY m.group_id, m.passage_id"
        )
    previous, expected_size, members = None, None, []
    for row in connection.execute(query):
        if previous is not None and row[0] != previous:
            if len(members) != expected_size:
                raise ValueError(f"Incomplete canonical group {previous}: {len(members)} != {expected_size}")
            yield previous, members
            members = []
        document = _document_from_row(row)
        if entity_type == "document":
            member = document
        else:
            member = {
                "doc_id": row[2], "official_url": row[3], "source_text_sha256": row[4],
                "normalized_content_hash": row[5], "passage_id": row[19],
                "passage_index": row[20], "char_start": row[21], "char_end": row[22],
                "passage_text": row[23], "document": document,
            }
        members.append(member)
        previous = row[0]
        expected_size = int(row[1])
    if previous is not None:
        if len(members) != expected_size:
            raise ValueError(f"Incomplete canonical group {previous}: {len(members)} != {expected_size}")
        yield previous, members


def _metrics(member: dict, entity_type: str) -> tuple[bool, int, int, int]:
    document = member if entity_type == "document" else member["document"]
    text = document["normalized_text"] if entity_type == "document" else member["passage_text"]
    return (
        bool(document["quality_review_required"]), len(document["quality_flags"] or []),
        non_whitespace_length(text), provenance_score(document),
    )


def _representatives(connection: sqlite3.Connection):
    for entity_type in ("document", "passage"):
        for group_id, members in _iter_groups(connection, entity_type):
            canonical, ranked, reason, chain = select_representative(
                members, entity_type=entity_type
            )
            review, flags, completeness, provenance = _metrics(canonical, entity_type)
            yield {
                "entity_type": entity_type, "content_group_id": group_id,
                "group_size": len(ranked), "canonical_doc_id": int(canonical["doc_id"]),
                "canonical_passage_id": canonical.get("passage_id"),
                "canonical_official_url": canonical["official_url"],
                "canonical_char_start": canonical.get("char_start"),
                "canonical_char_end": canonical.get("char_end"),
                "text_quality_review_required": review, "quality_flag_count": flags,
                "text_completeness_chars": completeness, "provenance_score": provenance,
                "selection_reason": reason, "selection_reason_chain": chain,
                "selection_policy_version": CANONICAL_POLICY_VERSION,
            }


def _memberships(connection: sqlite3.Connection):
    for entity_type in ("document", "passage"):
        for group_id, members in _iter_groups(connection, entity_type):
            _canonical, ranked, reason, _chain = select_representative(
                members, entity_type=entity_type
            )
            for rank, member in enumerate(ranked, 1):
                review, flags, completeness, provenance = _metrics(member, entity_type)
                yield {
                    "entity_type": entity_type, "content_group_id": group_id,
                    "group_size": len(ranked), "member_rank": rank, "is_canonical": rank == 1,
                    "doc_id": int(member["doc_id"]), "passage_id": member.get("passage_id"),
                    "official_url": member["official_url"], "char_start": member.get("char_start"),
                    "char_end": member.get("char_end"),
                    "source_text_sha256": member["source_text_sha256"],
                    "normalized_content_sha256": (
                        member["normalized_text_sha256"] if entity_type == "document"
                        else member["normalized_content_hash"]
                    ),
                    "text_quality_review_required": review, "quality_flag_count": flags,
                    "text_completeness_chars": completeness, "provenance_score": provenance,
                    "selection_reason": f"selected:{reason}" if rank == 1 else f"not_selected:{reason}",
                    "selection_policy_version": CANONICAL_POLICY_VERSION,
                }


def _scan_exclusions(task3_path: Path, task5_path: Path) -> dict:
    values = Counter()
    for batch in pq.ParquetFile(task3_path).iter_batches(
        batch_size=4096, columns=["auto_merge_allowed", "guard_reasons"]
    ):
        for row in batch.to_pylist():
            if row["auto_merge_allowed"]:
                raise ValueError("Task 3 unexpectedly authorizes auto merge")
            values["near_document_pairs_excluded"] += 1
            values["medical_guard_pairs_excluded"] += bool(row["guard_reasons"])
    for batch in pq.ParquetFile(task5_path).iter_batches(
        batch_size=4096, columns=["overlap_type", "auto_merge_allowed", "guard_reasons"]
    ):
        for row in batch.to_pylist():
            if row["auto_merge_allowed"]:
                raise ValueError("Task 5 unexpectedly authorizes auto merge")
            values["passage_overlap_pairs_excluded"] += 1
            values["medical_guard_pairs_excluded"] += bool(row["guard_reasons"])
            values[f"{row['overlap_type'].lower()}_excluded"] += 1
    return dict(values)


def run_full_canonical(
    *, root: Path, phase1: Path, phase2_release: Path, task2: Path, task3: Path,
    task4: Path, task5: Path, output: Path, resume: bool, rss_limit_gib: float,
    temporary_disk_budget_gib: float,
) -> dict:
    started = time.monotonic()
    destination = _safe_output(root, output)
    if destination.exists() and not resume:
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    phase1_manifest = phase1 / "release/phase1_release_manifest.json"
    phase2 = _load_json(phase2_release)
    task2_validation = _load_json(task2 / "validation/task2_exact_document_validation.json")
    task4_validation = _load_json(task4 / "validation/task4_passage_content_validation.json")
    if not task2_validation.get("groups_are_corpus_complete"):
        raise ValueError("Task 6 full requires corpus-complete Task 2")
    if not task4_validation.get("groups_are_corpus_complete"):
        raise ValueError("Task 6 full requires corpus-complete Task 4")
    task2_path = task2 / "document_dedup/exact_document_groups.parquet"
    task3_path = task3 / "document_dedup/near_duplicate_pairs.parquet"
    task4_path = task4 / "passage_dedup/passage_content_groups.parquet"
    task5_path = task5 / "passage_dedup/passage_overlap_pairs.parquet"
    input_hashes = {
        "phase1_manifest": sha256_file(phase1_manifest),
        "phase2_manifest": sha256_file(phase2_release), "task2": sha256_file(task2_path),
        "task3": sha256_file(task3_path), "task4": sha256_file(task4_path),
        "task5": sha256_file(task5_path),
    }
    fingerprint = _configuration_hash(input_hashes)
    config = {
        "policy_version": CANONICAL_POLICY_VERSION, "transitive_closure": False,
        "eligible": "verified normalized-exact groups only",
    }
    config_hash = _configuration_hash(config)
    monitor = ResourceMonitor(
        rss_limit_bytes=int(rss_limit_gib * 1024**3),
        disk_limit_bytes=int(temporary_disk_budget_gib * 1024**3),
        work_dir=destination / "checkpoints",
    )
    preflight = monitor.preflight(required_free_bytes=int(temporary_disk_budget_gib * 1024**3))
    database = destination / "checkpoints/task6_full.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    try:
        _schema(connection, fingerprint, config_hash)
        document_members, passage_members = _load_group_members(
            connection, task2_path, task4_path, monitor
        )
        phase1_release = _load_json(phase1_manifest)
        parts = [entry["part"] for entry in phase1_release["pass_allowlist_parts"]]
        _load_document_quality(connection, phase1, parts, monitor)
        _load_passage_text(connection, phase2_release.parent.parent, phase2["passage_parts"], monitor)
        exclusions = _scan_exclusions(task3_path, task5_path)
        representative_path = destination / "canonical/canonical_representatives.parquet"
        membership_path = destination / "canonical/group_membership.parquet"
        representative_rows = _write_partition(
            representative_path, _representatives(connection), CANONICAL_REPRESENTATIVE_SCHEMA
        )
        membership_rows = _write_partition(
            membership_path, _memberships(connection), CANONICAL_MEMBERSHIP_SCHEMA
        )
        _checkpoint(connection, "output_completion", "canonical", {
            "representatives": representative_rows, "memberships": membership_rows,
            "representatives_sha256": sha256_file(representative_path),
            "memberships_sha256": sha256_file(membership_path),
        })
    finally:
        connection.close()
    validation = {
        "format": "vibiomir-phase3-task6-full-disk-backed-v1", "status": "pass",
        "task": 6, "scope": "full_corpus", "input_fingerprints": input_hashes,
        "document_duplicate_members": document_members,
        "passage_duplicate_members": passage_members,
        "canonical_representatives": representative_rows, "group_memberships": membership_rows,
        "exclusions": exclusions, "resource_preflight": preflight,
        "rss_limit_gib": rss_limit_gib, "temporary_disk_budget_gib": temporary_disk_budget_gib,
        "near_or_overlap_auto_merged": False, "transitive_pair_closure_enabled": False,
        "source_occurrences_mutated": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    snapshot = {
        "format": "vibiomir-phase3-task6-full-input-v1", "scope": "full_corpus",
        "input_fingerprints": input_hashes, "input_fingerprint": fingerprint,
        "input_verification_mode": "verified_manifests_and_direct_upstream_outputs",
    }
    report = (
        "# Phase 3 Task 6 — Full Canonical Content Selection\n\n"
        f"- Status: **PASS**\n- Canonical representatives: `{representative_rows:,}`\n"
        f"- Group membership rows: `{membership_rows:,}`\n"
        f"- Near-document pairs excluded: `{exclusions.get('near_document_pairs_excluded', 0):,}`\n"
        f"- Passage overlap pairs excluded: `{exclusions.get('passage_overlap_pairs_excluded', 0):,}`\n"
        "- Transitive pair closure: `false`\n- Source occurrences mutated: `false`\n\n"
        "Only verified normalized-exact groups are eligible for canonical selection.\n"
    )
    _atomic_json(destination / "manifests/selection_policy.json", config)
    _atomic_json(destination / "manifests/input_snapshot.json", snapshot)
    _atomic_json(destination / "validation/task6_canonical_validation.json", validation)
    _atomic_text(destination / "reports/PHASE3_TASK6_REPORT.md", report)
    return validation
