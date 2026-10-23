"""Full-corpus disk-backed execution paths for Phase 3 Tasks 3 and 5."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase3.disk_lsh import (
    DISK_LSH_VERSION,
    SHINGLING_VERSION,
    DiskLSHIndex,
    ResourceBudgetExceeded,
    ResourceMonitor,
    structured_event,
)
from src.medrag.data.phase3.near import (
    NEAR_DUPLICATE_SCHEMA,
    NEAR_POLICY_VERSION,
    hash_shingles,
    medical_guard_reasons,
    minhash_signature,
    pair_id,
    shingle_text,
    signature_similarity,
    similarity_scores,
    threshold_passes,
)
from src.medrag.data.phase3.overlap import (
    PASSAGE_OVERLAP_SCHEMA,
    boilerplate_signals,
    classify_overlap,
    is_boilerplate_only,
    overlap_pair_id,
)


EXPECTED_DOCUMENTS = 1_073_531
EXPECTED_PASSAGES = 2_951_273


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _configuration_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_output(root: Path, output: Path) -> Path:
    resolved = output.resolve()
    outputs = (root / "outputs").resolve()
    if resolved.parent != outputs or not resolved.name.startswith("vibiomir_phase3_"):
        raise ValueError("Output must be a direct outputs/vibiomir_phase3_<run_id> namespace")
    return resolved


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_partition(path: Path, rows, schema: pa.Schema, *, batch_rows: int = 2048) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    count, batch = 0, []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_rows:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            count += len(batch)
        if count == 0:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
        writer.close()
        os.replace(temporary, path)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return count


def _combine_partitions(paths: list[Path], destination: Path, schema: pa.Schema) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    count = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096):
                writer.write_batch(batch)
                count += batch.num_rows
        if count == 0:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
        writer.close()
        os.replace(temporary, destination)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return count


def _document_records(
    clean_path: Path,
    pass_path: Path,
    *,
    part: str,
    num_perm: int,
    batch_rows: int,
):
    pass_ids = set(
        int(value) for value in pq.read_table(pass_path, columns=["doc_id"])["doc_id"].to_pylist()
    )
    ordinal = 0
    columns = [
        "doc_id", "official_url", "normalized_text", "normalized_text_sha256",
        "detected_language",
    ]
    for batch in pq.ParquetFile(clean_path).iter_batches(batch_size=batch_rows, columns=columns):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id not in pass_ids:
                continue
            shingles, strategy, size = shingle_text(
                row["normalized_text"], row["detected_language"]
            )
            hashed = hash_shingles(shingles)
            if not hashed:
                raise ValueError(f"No shingles generated: doc_id={doc_id}")
            yield {
                "entity_key": str(doc_id), "parent_key": str(doc_id),
                "sequence_index": 0, "ordinal": ordinal, "strategy": strategy,
                "shingle_size": size, "shingles": hashed,
                "signature": minhash_signature(hashed, num_perm),
                "normalized_hash": row["normalized_text_sha256"],
                "payload": row,
            }
            ordinal += 1


def _document_pair_rows(index: DiskLSHIndex, partition: int, config: dict, release_id: str):
    thresholds = tuple(config["thresholds"])
    for left, right, _sources in index.iter_candidates(partition):
        if left["normalized_hash"] == right["normalized_hash"]:
            continue
        shared, jaccard, left_containment, right_containment = similarity_scores(
            left["shingles"], right["shingles"]
        )
        passed = threshold_passes(
            jaccard, left_containment, right_containment, thresholds,
            containment_threshold=config["containment_threshold"],
            containment_jaccard_floor=config["containment_jaccard_floor"],
        )
        if config["acceptance_threshold"] not in passed:
            continue
        guards = medical_guard_reasons(left["normalized_text"], right["normalized_text"])
        if guards:
            continue
        left_id, right_id = int(left["doc_id"]), int(right["doc_id"])
        if right_id < left_id:
            left, right = right, left
            left_id, right_id = right_id, left_id
            left_containment, right_containment = right_containment, left_containment
        yield {
            "pair_id": pair_id(left_id, right_id),
            "left_doc_id": left_id, "right_doc_id": right_id,
            "left_official_url": left["official_url"],
            "right_official_url": right["official_url"],
            "left_language": left["detected_language"],
            "right_language": right["detected_language"],
            "shingle_strategy": left["strategy"], "shingle_size": left["shingle_size"],
            "left_shingles": len(left["shingles"]), "right_shingles": len(right["shingles"]),
            "shared_shingles": shared,
            "minhash_similarity": signature_similarity(left["signature"], right["signature"]),
            "jaccard_similarity": jaccard, "left_containment": left_containment,
            "right_containment": right_containment, "threshold_passes": passed,
            "guard_reasons": [], "review_status": "unlabeled_review_required",
            "auto_merge_allowed": False, "phase1_release_id": release_id,
            "policy_version": NEAR_POLICY_VERSION,
        }


def run_full_document_candidates(
    *,
    root: Path,
    candidate: Path,
    task2: Path,
    output: Path,
    resume: bool,
    partitions: int,
    maximum_bucket_size: int,
    batch_rows: int,
    rss_limit_gib: float,
    temporary_disk_budget_gib: float,
    bands: int = 16,
    rows_per_band: int = 4,
) -> dict:
    """Run Task 3 with bounded memory; does not change its matching policy."""
    started = time.monotonic()
    destination_root = _safe_output(root, output)
    if destination_root.exists() and not resume:
        raise FileExistsError(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    task2_output = task2 / "document_dedup/exact_document_groups.parquet"
    task2_validation = _load_json(task2 / "validation/task2_exact_document_validation.json")
    task2_snapshot = _load_json(task2 / "manifests/input_snapshot.json")
    if task2_validation.get("status") != "pass" or not task2_validation.get("groups_are_corpus_complete"):
        raise ValueError("Full Task 3 requires corpus-complete validated Task 2")
    if int(task2_validation["output"]["rows"]) != EXPECTED_DOCUMENTS:
        raise ValueError("Task 2 full document count mismatch")
    task2_sha = sha256_file(task2_output)
    if task2_validation["output"]["sha256"] != task2_sha:
        raise ValueError("Task 2 output fingerprint mismatch")
    release_path = candidate / "release/phase1_release_manifest.json"
    release_sha = sha256_file(release_path)
    release = _load_json(release_path)
    parts = task2_snapshot["selected_parts"]
    pass_parts = {entry["part"]: entry for entry in release["pass_allowlist_parts"]}
    config = {
        "bands": bands, "rows_per_band": rows_per_band,
        "partitions": partitions, "maximum_bucket_size": maximum_bucket_size,
        "thresholds": [0.80, 0.85, 0.90, 0.95], "acceptance_threshold": 0.90,
        "containment_threshold": 0.95, "containment_jaccard_floor": 0.60,
        "batch_rows": batch_rows, "disk_lsh_version": DISK_LSH_VERSION,
    }
    config_hash = _configuration_hash(config)
    fingerprint = _configuration_hash({"phase1": release_sha, "task2": task2_sha})
    monitor = ResourceMonitor(
        rss_limit_bytes=int(rss_limit_gib * 1024**3),
        disk_limit_bytes=int(temporary_disk_budget_gib * 1024**3),
        work_dir=destination_root / "checkpoints",
    )
    preflight = monitor.preflight(required_free_bytes=int(temporary_disk_budget_gib * 1024**3))
    index_path = destination_root / "checkpoints/task3_full_lsh.sqlite"
    try:
        with DiskLSHIndex(
            index_path, bands=bands, rows_per_band=rows_per_band,
            partitions=partitions, maximum_bucket_size=maximum_bucket_size,
            config_hash=config_hash, input_fingerprint=fingerprint, monitor=monitor,
        ) as index:
            total = 0
            for number, part in enumerate(parts, 1):
                total += index.ingest_shard(
                    part,
                    _document_records(
                        candidate / f"clean_documents/{part}.parquet",
                        candidate / f"release/pass_document_ids/{part}.parquet",
                        part=part, num_perm=bands * rows_per_band, batch_rows=batch_rows,
                    ),
                    batch_size=batch_rows,
                )
                print(structured_event("task3_input_shard", part=part, number=number, total_parts=len(parts)), flush=True)
            if total != EXPECTED_DOCUMENTS:
                raise ValueError(f"Full Task 3 ingested {total}, expected {EXPECTED_DOCUMENTS}")
            candidate_stats = index.generate_candidates()
            partition_paths = []
            for partition in range(partitions):
                part_path = destination_root / f"document_dedup/partitions/part-{partition:05d}.parquet"
                partition_paths.append(part_path)
                if index.checkpoint_complete("pair_verification", str(partition)):
                    continue
                count = _write_partition(
                    part_path, _document_pair_rows(index, partition, config, release_sha),
                    NEAR_DUPLICATE_SCHEMA,
                )
                index.checkpoint("pair_verification", str(partition), "complete", {
                    "rows": count, "sha256": sha256_file(part_path),
                })
                monitor.check()
            final_path = destination_root / "document_dedup/near_duplicate_pairs.parquet"
            rows = _combine_partitions(partition_paths, final_path, NEAR_DUPLICATE_SCHEMA)
            index.checkpoint("output_completion", "near_duplicate_pairs", "complete", {
                "rows": rows, "sha256": sha256_file(final_path),
            })
            index.atomic_checkpoint_export(
                destination_root / "checkpoints/task3_full_checkpoint.json"
            )
            storage_stats = index.statistics()
    except ResourceBudgetExceeded:
        raise
    completed = datetime.now(timezone.utc).isoformat()
    validation = {
        "format": "vibiomir-phase3-task3-full-disk-backed-v1", "status": "pass",
        "task": 3, "scope": "full_corpus", "groups_are_corpus_complete": True,
        "documents": EXPECTED_DOCUMENTS, "output_rows": rows,
        "output_sha256": sha256_file(final_path), "candidate_statistics": candidate_stats,
        "storage_statistics": storage_stats, "resource_preflight": preflight,
        "rss_limit_gib": rss_limit_gib, "temporary_disk_budget_gib": temporary_disk_budget_gib,
        "candidate_recall_complete": False,
        "candidate_recall_limitation": "MinHash/LSH is probabilistic; high-frequency template buckets are explicitly filtered and accounted.",
        "auto_merge_allowed": False, "completed_at_utc": completed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    snapshot = {
        "format": "vibiomir-phase3-task3-full-input-v1", "scope": "full_corpus",
        "phase1_release_manifest_sha256": release_sha, "task2_output_sha256": task2_sha,
        "input_fingerprint": fingerprint, "selected_parts": len(parts),
        "input_verification_mode": "verified_manifests_and_direct_upstream_output",
    }
    report = (
        "# Phase 3 Task 3 — Full Near-document Candidates\n\n"
        f"- Status: **PASS**\n- Documents: `{EXPECTED_DOCUMENTS:,}`\n"
        f"- Review-only pairs: `{rows:,}`\n"
        f"- Hot buckets: `{storage_stats['hot_buckets']:,}`\n"
        f"- Hot-bucket potential pairs not examined: "
        f"`{storage_stats['hot_bucket_potential_pairs'] - storage_stats['hot_bucket_examined_pairs']:,}`\n"
        "- Auto merge: `false`\n- Candidate recall complete: `false`\n\n"
        "MinHash/LSH recall is probabilistic. High-frequency buckets are filtered, explicitly "
        "accounted, and are not claimed as covered.\n"
    )
    _atomic_json(destination_root / "manifests/dedup_config.json", config)
    _atomic_json(destination_root / "manifests/input_snapshot.json", snapshot)
    _atomic_json(destination_root / "validation/task3_near_document_validation.json", validation)
    report_path = destination_root / "reports/TASK3_NEAR_DOCUMENT_DEDUP_REPORT.md"
    _atomic_text(report_path, report)
    return validation


def _load_task4_mapping(index: DiskLSHIndex, task4_path: Path, monitor: ResourceMonitor) -> int:
    if index.checkpoint_complete("task4_mapping", "all"):
        return int(index.connection.execute("SELECT COUNT(*) FROM passage_content").fetchone()[0])
    index.connection.execute(
        "CREATE TABLE IF NOT EXISTS passage_content (passage_id TEXT PRIMARY KEY, doc_id INTEGER, "
        "char_start INTEGER, char_end INTEGER, normalized_hash TEXT NOT NULL) WITHOUT ROWID"
    )
    count = 0
    columns = ["passage_id", "doc_id", "char_start", "char_end", "normalized_content_hash"]
    for batch in pq.ParquetFile(task4_path).iter_batches(batch_size=4096, columns=columns):
        index.connection.executemany(
            "INSERT OR IGNORE INTO passage_content VALUES (?, ?, ?, ?, ?)",
            [
                (row["passage_id"], int(row["doc_id"]), int(row["char_start"]),
                 int(row["char_end"]), row["normalized_content_hash"])
                for row in batch.to_pylist()
            ],
        )
        count += batch.num_rows
        if count % 100_000 < batch.num_rows:
            index.connection.commit()
            monitor.check()
    index.connection.commit()
    index.checkpoint("task4_mapping", "all", "complete", {"records": count})
    return count


def _passage_records(index: DiskLSHIndex, path: Path, *, num_perm: int, batch_rows: int):
    columns = [
        "passage_id", "doc_id", "official_url", "passage_index", "passage_text",
        "char_start", "char_end", "source_text_sha256", "language",
    ]
    ordinal = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        rows = batch.to_pylist()
        identifiers = [row["passage_id"] for row in rows]
        placeholders = ",".join("?" for _ in identifiers)
        mapping = {
            key: (doc_id, start, end, normalized_hash)
            for key, doc_id, start, end, normalized_hash in index.connection.execute(
                f"SELECT passage_id, doc_id, char_start, char_end, normalized_hash "
                f"FROM passage_content WHERE passage_id IN ({placeholders})", identifiers
            )
        }
        if len(mapping) != len(rows):
            raise ValueError("Task 4 mapping is incomplete for a Phase 2 passage batch")
        for row in rows:
            expected = mapping[row["passage_id"]]
            if (int(row["doc_id"]), int(row["char_start"]), int(row["char_end"])) != expected[:3]:
                raise ValueError("Task 4 occurrence mapping changed")
            shingles, strategy, size = shingle_text(row["passage_text"], row["language"])
            hashed = hash_shingles(shingles)
            if not hashed:
                raise ValueError(f"No shingles generated: {row['passage_id']}")
            yield {
                "entity_key": row["passage_id"], "parent_key": str(row["doc_id"]),
                "sequence_index": int(row["passage_index"]), "ordinal": ordinal,
                "strategy": strategy, "shingle_size": size, "shingles": hashed,
                "signature": minhash_signature(hashed, num_perm),
                "normalized_hash": expected[3], "payload": row,
            }
            ordinal += 1


def _add_passage_relation_candidates(index: DiskLSHIndex, task3_path: Path, monitor: ResourceMonitor) -> dict:
    adjacent = near_expanded = 0
    if not index.checkpoint_complete("candidate_source", "adjacent"):
        previous = None
        for record_id, parent, sequence in index.connection.execute(
            "SELECT record_id, parent_key, sequence_index FROM records "
            "ORDER BY parent_key, sequence_index, entity_key"
        ):
            if previous and previous[1] == parent and int(sequence) - int(previous[2]) == 1:
                index.add_candidate(previous[0], record_id, "adjacent_source_offsets")
                adjacent += 1
            previous = (record_id, parent, sequence)
            if adjacent and adjacent % 100_000 == 0:
                index.connection.commit()
                monitor.check()
        index.connection.commit()
        index.checkpoint("candidate_source", "adjacent", "complete", {"pairs": adjacent})
    if not index.checkpoint_complete("candidate_source", "task3_near_document"):
        for pair in pq.ParquetFile(task3_path).iter_batches(
            batch_size=1024, columns=["left_doc_id", "right_doc_id"]
        ):
            for row in pair.to_pylist():
                left = [value[0] for value in index.connection.execute(
                    "SELECT record_id FROM records WHERE parent_key=? ORDER BY sequence_index, entity_key",
                    (str(row["left_doc_id"]),),
                )]
                right = [value[0] for value in index.connection.execute(
                    "SELECT record_id FROM records WHERE parent_key=? ORDER BY sequence_index, entity_key",
                    (str(row["right_doc_id"]),),
                )]
                for left_id in left:
                    for right_id in right:
                        index.add_candidate(left_id, right_id, "task3_near_document")
                        near_expanded += 1
                if near_expanded and near_expanded % 100_000 == 0:
                    index.connection.commit()
                    monitor.check()
        index.connection.commit()
        index.checkpoint(
            "candidate_source", "task3_near_document", "complete", {"pairs": near_expanded}
        )
    return {"adjacent_candidates_added": adjacent, "task3_candidates_added": near_expanded}


def _passage_pair_rows(index: DiskLSHIndex, partition: int, config: dict):
    for left, right, sources in index.iter_candidates(partition):
        if left["normalized_hash"] == right["normalized_hash"]:
            continue
        shared, jaccard, left_containment, right_containment = similarity_scores(
            left["shingles"], right["shingles"]
        )
        left_class = {**left, "shingle_count": len(left["shingles"])}
        right_class = {**right, "shingle_count": len(right["shingles"])}
        overlap_type, reasons, source_chars = classify_overlap(
            left_class, right_class, shared_shingles=shared, jaccard=jaccard,
            left_containment=left_containment, right_containment=right_containment,
            minimum_shared_shingles=config["minimum_shared_shingles"],
            jaccard_threshold=config["jaccard_threshold"],
            containment_threshold=config["containment_threshold"],
            containment_jaccard_floor=config["containment_jaccard_floor"],
        )
        if overlap_type is None:
            continue
        maximum_containment = max(left_containment, right_containment)
        boilerplate_only, boilerplate = is_boilerplate_only(
            left["passage_text"], right["passage_text"],
            shared_shingles=shared, maximum_containment=maximum_containment,
        )
        if boilerplate_only:
            continue
        guards = medical_guard_reasons(left["passage_text"], right["passage_text"])
        left_key = (int(left["doc_id"]), int(left["passage_index"]), left["passage_id"])
        right_key = (int(right["doc_id"]), int(right["passage_index"]), right["passage_id"])
        if right_key < left_key:
            left, right = right, left
            left_containment, right_containment = right_containment, left_containment
        yield {
            "pair_id": overlap_pair_id(left["passage_id"], right["passage_id"]),
            "left_passage_id": left["passage_id"], "right_passage_id": right["passage_id"],
            "left_doc_id": int(left["doc_id"]), "right_doc_id": int(right["doc_id"]),
            "left_official_url": left["official_url"], "right_official_url": right["official_url"],
            "left_source_offsets": {"char_start": int(left["char_start"]), "char_end": int(left["char_end"])},
            "right_source_offsets": {"char_start": int(right["char_start"]), "char_end": int(right["char_end"])},
            "left_language": left["language"], "right_language": right["language"],
            "candidate_sources": sorted(sources), "shingle_strategy": left["strategy"],
            "shingle_size": left["shingle_size"], "left_shingles": len(left["shingles"]),
            "right_shingles": len(right["shingles"]), "shared_shingles": shared,
            "minhash_similarity": signature_similarity(left["signature"], right["signature"]),
            "jaccard_similarity": jaccard, "left_containment": left_containment,
            "right_containment": right_containment, "source_overlap_chars": source_chars,
            "overlap_type": overlap_type, "classification_reasons": reasons,
            "guard_reasons": guards,
            "boilerplate_signals": sorted(set(boilerplate) | set(boilerplate_signals(left["passage_text"])) | set(boilerplate_signals(right["passage_text"]))),
            "review_status": "medical_guard_review" if guards else "unlabeled_review_required",
            "auto_merge_allowed": False, "policy_version": "vibiomir-phase3-passage-overlap-v1",
        }


def run_full_passage_candidates(
    *,
    root: Path,
    phase2_release: Path,
    task4: Path,
    task3_pairs: Path,
    output: Path,
    resume: bool,
    partitions: int,
    maximum_bucket_size: int,
    batch_rows: int,
    rss_limit_gib: float,
    temporary_disk_budget_gib: float,
    bands: int = 32,
    rows_per_band: int = 2,
) -> dict:
    """Run Task 5 across all Phase 2 passages with disk-bounded state."""
    started = time.monotonic()
    destination_root = _safe_output(root, output)
    if destination_root.exists() and not resume:
        raise FileExistsError(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    release = _load_json(phase2_release)
    if release.get("status") != "complete_validated" or int(release["passages"]) != EXPECTED_PASSAGES:
        raise ValueError("Phase 2 full release is not complete and validated")
    task4_validation = _load_json(task4 / "validation/task4_passage_content_validation.json")
    if task4_validation.get("status") != "pass" or not task4_validation.get("groups_are_corpus_complete"):
        raise ValueError("Full Task 5 requires corpus-complete validated Task 4")
    task4_path = task4 / "passage_dedup/passage_content_groups.parquet"
    task4_sha = sha256_file(task4_path)
    task3_sha = sha256_file(task3_pairs)
    task3_validation = _load_json(
        task3_pairs.parents[1] / "validation/task3_near_document_validation.json"
    )
    if (
        task3_validation.get("status") != "pass"
        or task3_validation.get("scope") != "full_corpus"
        or task3_validation.get("output_sha256") != task3_sha
        or task3_validation.get("auto_merge_allowed") is not False
    ):
        raise ValueError("Full Task 5 requires a validated review-only full Task 3")
    config = {
        "bands": bands, "rows_per_band": rows_per_band, "partitions": partitions,
        "maximum_bucket_size": maximum_bucket_size, "minimum_shared_shingles": 5,
        "jaccard_threshold": 0.45, "containment_threshold": 0.75,
        "containment_jaccard_floor": 0.20, "batch_rows": batch_rows,
        "disk_lsh_version": DISK_LSH_VERSION,
    }
    config_hash = _configuration_hash(config)
    fingerprint = _configuration_hash({
        "phase2": sha256_file(phase2_release), "task4": task4_sha, "task3": task3_sha,
    })
    monitor = ResourceMonitor(
        rss_limit_bytes=int(rss_limit_gib * 1024**3),
        disk_limit_bytes=int(temporary_disk_budget_gib * 1024**3),
        work_dir=destination_root / "checkpoints",
    )
    preflight = monitor.preflight(required_free_bytes=int(temporary_disk_budget_gib * 1024**3))
    index_path = destination_root / "checkpoints/task5_full_lsh.sqlite"
    with DiskLSHIndex(
        index_path, bands=bands, rows_per_band=rows_per_band, partitions=partitions,
        maximum_bucket_size=maximum_bucket_size, config_hash=config_hash,
        input_fingerprint=fingerprint, monitor=monitor,
    ) as index:
        mapped = _load_task4_mapping(index, task4_path, monitor)
        if mapped != EXPECTED_PASSAGES:
            raise ValueError("Task 4 full passage count mismatch")
        passages_root = phase2_release.parent.parent / "passages"
        total = 0
        for number, entry in enumerate(release["passage_parts"], 1):
            part = entry["part"]
            total += index.ingest_shard(
                part,
                _passage_records(
                    index, passages_root / f"{part}.parquet",
                    num_perm=bands * rows_per_band, batch_rows=batch_rows,
                ),
                batch_size=batch_rows,
            )
            print(structured_event("task5_input_shard", part=part, number=number, total_parts=len(release["passage_parts"])), flush=True)
        if total != EXPECTED_PASSAGES:
            raise ValueError(f"Full Task 5 ingested {total}, expected {EXPECTED_PASSAGES}")
        candidate_stats = index.generate_candidates()
        source_stats = _add_passage_relation_candidates(index, task3_pairs, monitor)
        partition_paths = []
        for partition in range(partitions):
            part_path = destination_root / f"passage_dedup/partitions/part-{partition:05d}.parquet"
            partition_paths.append(part_path)
            if index.checkpoint_complete("pair_verification", str(partition)):
                continue
            count = _write_partition(
                part_path, _passage_pair_rows(index, partition, config), PASSAGE_OVERLAP_SCHEMA,
            )
            index.checkpoint("pair_verification", str(partition), "complete", {
                "rows": count, "sha256": sha256_file(part_path),
            })
            monitor.check()
        final_path = destination_root / "passage_dedup/passage_overlap_pairs.parquet"
        rows = _combine_partitions(partition_paths, final_path, PASSAGE_OVERLAP_SCHEMA)
        index.checkpoint("output_completion", "passage_overlap_pairs", "complete", {
            "rows": rows, "sha256": sha256_file(final_path),
        })
        index.atomic_checkpoint_export(destination_root / "checkpoints/task5_full_checkpoint.json")
        storage_stats = index.statistics()
    overlap_types: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    guard_pairs = 0
    for batch in pq.ParquetFile(final_path).iter_batches(
        batch_size=4096, columns=["overlap_type", "review_status", "guard_reasons", "auto_merge_allowed"]
    ):
        for row in batch.to_pylist():
            if row["auto_merge_allowed"]:
                raise ValueError("Full Task 5 emitted an auto-merge relation")
            overlap_types[row["overlap_type"]] += 1
            review_statuses[row["review_status"]] += 1
            guard_pairs += bool(row["guard_reasons"])
    validation = {
        "format": "vibiomir-phase3-task5-full-disk-backed-v1", "status": "pass",
        "task": 5, "scope": "full_corpus", "passages": EXPECTED_PASSAGES,
        "output_rows": rows, "output_sha256": sha256_file(final_path),
        "candidate_statistics": {**candidate_stats, **source_stats},
        "storage_statistics": storage_stats, "resource_preflight": preflight,
        "overlap_type_counts": dict(sorted(overlap_types.items())),
        "review_status_counts": dict(sorted(review_statuses.items())),
        "medical_guard_pairs": guard_pairs,
        "rss_limit_gib": rss_limit_gib, "temporary_disk_budget_gib": temporary_disk_budget_gib,
        "candidate_recall_complete": False,
        "candidate_recall_limitation": "MinHash/LSH is probabilistic; filtered hot buckets and their potential pairs are reported.",
        "auto_merge_allowed": False, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    snapshot = {
        "format": "vibiomir-phase3-task5-full-input-v1", "scope": "full_corpus",
        "phase2_release_manifest_sha256": sha256_file(phase2_release),
        "task3_pairs_sha256": task3_sha, "task4_output_sha256": task4_sha,
        "input_fingerprint": fingerprint,
        "input_verification_mode": "verified_manifests_and_direct_upstream_outputs",
    }
    report = (
        "# Phase 3 Task 5 — Full Passage Overlap Relationships\n\n"
        f"- Status: **PASS**\n- Passages: `{EXPECTED_PASSAGES:,}`\n"
        f"- Output relationships: `{rows:,}`\n"
        f"- Overlap types: `{json.dumps(dict(sorted(overlap_types.items())), sort_keys=True)}`\n"
        f"- Medical-guard review pairs: `{guard_pairs:,}`\n"
        f"- Hot buckets: `{storage_stats['hot_buckets']:,}`\n"
        "- Auto merge: `false`\n- Candidate recall complete: `false`\n\n"
        "MinHash/LSH recall is probabilistic. Filtered hot-bucket coverage is reported in "
        "validation and is excluded from completeness claims.\n"
    )
    _atomic_json(destination_root / "manifests/overlap_config.json", config)
    _atomic_json(destination_root / "manifests/input_snapshot.json", snapshot)
    _atomic_json(destination_root / "validation/overlap_validation_report.json", validation)
    report_path = destination_root / "reports/PHASE3_TASK5_REPORT.md"
    _atomic_text(report_path, report)
    return validation
