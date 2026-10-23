"""Perform the single final validation and publish a reference-based Phase 3 release."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.selection import sha256_file


EXPECTED_DOCUMENTS = 1_073_531
EXPECTED_PASSAGES = 2_951_273


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifact(root: Path, relative: str) -> dict:
    path = root / relative
    return {
        "path": str(path.relative_to(ROOT)), "rows": pq.read_metadata(path).num_rows,
        "bytes": path.stat().st_size, "sha256": sha256_file(path),
    }


def _assert_review_only(path: Path, label: str) -> int:
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=4096, columns=["auto_merge_allowed"]
    ):
        if any(batch.column(0).to_pylist()):
            raise ValueError(f"{label} contains an auto-merge relation")
        rows += batch.num_rows
    return rows


def _validate_canonical(representatives: Path, memberships: Path, database: Path) -> dict:
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            "CREATE TABLE groups(entity_type TEXT, group_id TEXT, expected INTEGER, "
            "canonical_count INTEGER DEFAULT 0, members INTEGER DEFAULT 0, "
            "PRIMARY KEY(entity_type, group_id));"
        )
        for batch in pq.ParquetFile(representatives).iter_batches(
            batch_size=4096, columns=["entity_type", "content_group_id", "group_size"]
        ):
            connection.executemany(
                "INSERT INTO groups(entity_type, group_id, expected) VALUES (?, ?, ?)",
                [(row["entity_type"], row["content_group_id"], int(row["group_size"]))
                 for row in batch.to_pylist()],
            )
        membership_rows = 0
        for batch in pq.ParquetFile(memberships).iter_batches(
            batch_size=4096,
            columns=["entity_type", "content_group_id", "is_canonical", "official_url"],
        ):
            rows = batch.to_pylist()
            if any(not row["official_url"] for row in rows):
                raise ValueError("Canonical membership lost an official URL")
            connection.executemany(
                "UPDATE groups SET members=members+1, canonical_count=canonical_count+? "
                "WHERE entity_type=? AND group_id=?",
                [(int(row["is_canonical"]), row["entity_type"], row["content_group_id"])
                 for row in rows],
            )
            membership_rows += len(rows)
        connection.commit()
        invalid = connection.execute(
            "SELECT COUNT(*) FROM groups WHERE members != expected OR canonical_count != 1"
        ).fetchone()[0]
        matched_memberships = connection.execute(
            "SELECT COALESCE(SUM(members), 0) FROM groups"
        ).fetchone()[0]
        if matched_memberships != membership_rows:
            raise ValueError(
                f"Canonical memberships reference unknown groups: "
                f"matched={matched_memberships} rows={membership_rows}"
            )
        if invalid:
            raise ValueError(f"Invalid canonical groups: {invalid}")
        return {
            "representative_rows": connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0],
            "membership_rows": membership_rows, "invalid_groups": invalid,
        }
    finally:
        connection.close()


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.parent != (ROOT / "outputs").resolve() or not output.name.startswith("vibiomir_phase3_"):
        raise ValueError("Final release must use outputs/vibiomir_phase3_<run_id>")
    if output.exists():
        raise FileExistsError(output)
    task2, task3, task4, task5, task6 = (
        args.task2.resolve(), args.task3.resolve(), args.task4.resolve(),
        args.task5.resolve(), args.task6.resolve(),
    )
    validations = {
        "task2": _load(task2 / "validation/task2_exact_document_validation.json"),
        "task3": _load(task3 / "validation/task3_near_document_validation.json"),
        "task4": _load(task4 / "validation/task4_passage_content_validation.json"),
        "task5": _load(task5 / "validation/overlap_validation_report.json"),
        "task6": _load(task6 / "validation/task6_canonical_validation.json"),
    }
    if any(value.get("status") != "pass" for value in validations.values()):
        raise ValueError("All Task 2-6 validations must PASS")
    if not validations["task2"].get("groups_are_corpus_complete"):
        raise ValueError("Task 2 is not corpus-complete")
    if not validations["task4"].get("groups_are_corpus_complete"):
        raise ValueError("Task 4 is not corpus-complete")
    artifacts = {
        "exact_document_groups": _artifact(task2, "document_dedup/exact_document_groups.parquet"),
        "near_document_pairs": _artifact(task3, "document_dedup/near_duplicate_pairs.parquet"),
        "passage_content_groups": _artifact(task4, "passage_dedup/passage_content_groups.parquet"),
        "passage_overlap_pairs": _artifact(task5, "passage_dedup/passage_overlap_pairs.parquet"),
        "canonical_representatives": _artifact(task6, "canonical/canonical_representatives.parquet"),
        "canonical_membership": _artifact(task6, "canonical/group_membership.parquet"),
    }
    if artifacts["exact_document_groups"]["rows"] != EXPECTED_DOCUMENTS:
        raise ValueError("Final document occurrence count mismatch")
    if artifacts["passage_content_groups"]["rows"] != EXPECTED_PASSAGES:
        raise ValueError("Final passage occurrence count mismatch")
    near_rows = _assert_review_only(
        task3 / "document_dedup/near_duplicate_pairs.parquet", "Task 3"
    )
    overlap_rows = _assert_review_only(
        task5 / "passage_dedup/passage_overlap_pairs.parquet", "Task 5"
    )
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".sqlite", delete=False) as stream:
        validation_database = Path(stream.name)
    validation_database.unlink()
    try:
        canonical = _validate_canonical(
            task6 / "canonical/canonical_representatives.parquet",
            task6 / "canonical/group_membership.parquet",
            validation_database,
        )
    finally:
        validation_database.unlink(missing_ok=True)
    manifest = {
        "format": "vibiomir-phase3-final-release-v1", "status": "complete_validated",
        "scope": "full", "documents": EXPECTED_DOCUMENTS, "passages": EXPECTED_PASSAGES,
        "artifacts": artifacts, "canonical_validation": canonical,
        "near_document_review_pairs": near_rows, "passage_overlap_review_pairs": overlap_rows,
        "candidate_recall_complete": False,
        "candidate_recall_limitation": (
            "MinHash/LSH is probabilistic; hot-bucket exclusions and potential pairs are reported "
            "by Task 3/5 and preclude an exhaustive-recall claim."
        ),
        "phase0_mutated": False, "phase1_mutated": False, "phase2_mutated": False,
        "index_mutated": False, "phase4_started": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True)
    _atomic(output / "manifests/phase3_release_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    report = (
        "# Phase 3 Final Release\n\n- Status: **COMPLETE VALIDATED**\n"
        f"- Documents: `{EXPECTED_DOCUMENTS:,}`\n- Passages: `{EXPECTED_PASSAGES:,}`\n"
        f"- Near-document review pairs: `{near_rows:,}`\n"
        f"- Passage-overlap review pairs: `{overlap_rows:,}`\n"
        "- Auto merge of review-only relationships: `false`\n"
        "- Candidate recall complete: `false`\n- Phase 4 started: `false`\n"
    )
    _atomic(output / "reports/PHASE3_FINAL_RELEASE_REPORT.md", report)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for task in range(2, 7):
        parser.add_argument(f"--task{task}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
