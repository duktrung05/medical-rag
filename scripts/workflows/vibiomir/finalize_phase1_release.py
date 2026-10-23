"""Freeze a full Phase 1 candidate corpus into PASS and held-document allowlists."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.quality_policy import load_quality_policy
from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase1.validation import QualityAssessment


DECISION_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("quality_status", pa.string()),
    ("quality_decision_reason", pa.string()),
    ("quality_flags", pa.list_(pa.string())),
    ("quality_reasons", pa.list_(pa.string())),
])
PASS_SCHEMA = pa.schema([("doc_id", pa.int64())])
HELD_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("quality_status", pa.string()),
    ("quality_decision_reason", pa.string()),
    ("quality_flags", pa.list_(pa.string())),
])


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".partial",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_candidate(candidate: Path) -> dict:
    manifest_path = candidate / "phase1_manifest.json"
    snapshot_path = candidate / "source_snapshot.json"
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise ValueError("Candidate is missing manifest or source snapshot")
    manifest = _load_json(manifest_path)
    if manifest.get("build_scope") != "full" or manifest.get("status") != "complete":
        raise ValueError("Candidate is not a completed full Phase 1 build")
    if manifest.get("source_snapshot_sha256") != sha256_file(snapshot_path):
        raise ValueError("Candidate source snapshot checksum mismatch")
    parts = manifest.get("part_checkpoints_sha256")
    if not isinstance(parts, dict) or len(parts) != int(manifest.get("parts", -1)):
        raise ValueError("Candidate part checkpoint count mismatch")
    for stem, expected_hash in parts.items():
        path = candidate / "clean_documents" / f"{stem}.parquet"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Candidate part checksum mismatch: {path}")
    return manifest


def finalize(candidate: Path, policy_path: Path) -> dict:
    candidate = candidate.resolve()
    policy_path = policy_path.resolve()
    manifest = _validate_candidate(candidate)
    policy = load_quality_policy(policy_path)
    release = candidate / "release"
    if release.exists():
        raise FileExistsError(f"Release output already exists: {release}")

    decision_dir = release / "decision_ledger"
    pass_dir = release / "pass_document_ids"
    held_dir = release / "held_documents"
    seen_ids: set[int] = set()
    status_counts: Counter[str] = Counter()
    decision_parts = []
    pass_parts = []
    held_parts = []

    for stem in sorted(manifest["part_checkpoints_sha256"]):
        source = candidate / "clean_documents" / f"{stem}.parquet"
        rows = pq.read_table(source, columns=[
            "doc_id", "quality_flags", "quality_reasons", "quality_review_required",
        ]).to_pylist()
        decisions = []
        passes = []
        held = []
        for row in rows:
            doc_id = int(row["doc_id"])
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate candidate document ID: {doc_id}")
            seen_ids.add(doc_id)
            assessment = QualityAssessment(
                tuple(row["quality_flags"] or ()),
                tuple(row["quality_reasons"] or ()),
                bool(row["quality_review_required"]),
            )
            decision = policy.decide(doc_id, assessment)
            decision_row = {
                "doc_id": doc_id,
                "quality_status": decision.status,
                "quality_decision_reason": decision.reason,
                "quality_flags": list(assessment.flags),
                "quality_reasons": list(assessment.reasons),
            }
            decisions.append(decision_row)
            status_counts[decision.status] += 1
            if decision.status == "PASS":
                passes.append({"doc_id": doc_id})
            else:
                held.append({
                    "doc_id": doc_id,
                    "quality_status": decision.status,
                    "quality_decision_reason": decision.reason,
                    "quality_flags": list(assessment.flags),
                })

        decision_path = decision_dir / f"{stem}.parquet"
        pass_path = pass_dir / f"{stem}.parquet"
        held_path = held_dir / f"{stem}.parquet"
        _atomic_parquet(decision_path, pa.Table.from_pylist(decisions, schema=DECISION_SCHEMA))
        _atomic_parquet(pass_path, pa.Table.from_pylist(passes, schema=PASS_SCHEMA))
        _atomic_parquet(held_path, pa.Table.from_pylist(held, schema=HELD_SCHEMA))
        decision_parts.append({"part": stem, "rows": len(decisions),
                               "sha256": sha256_file(decision_path)})
        pass_parts.append({"part": stem, "rows": len(passes), "sha256": sha256_file(pass_path)})
        held_parts.append({"part": stem, "rows": len(held), "sha256": sha256_file(held_path)})

    total = int(manifest["documents"])
    if len(seen_ids) != total:
        raise ValueError(f"Release decisions cover {len(seen_ids)} IDs, expected {total}")
    if sum(status_counts.values()) != total:
        raise ValueError("Release status counts do not equal candidate count")

    release_manifest = {
        "status": "complete",
        "format": "vibiomir-phase1-release-v1",
        "candidate": _portable_path(candidate),
        "candidate_manifest_sha256": sha256_file(candidate / "phase1_manifest.json"),
        "candidate_documents": total,
        "quality_policy": _portable_path(policy_path),
        "quality_policy_sha256": sha256_file(policy_path),
        "quality_policy_version": policy.version,
        "status_counts": dict(sorted(status_counts.items())),
        "pass_documents": status_counts["PASS"],
        "held_documents": total - status_counts["PASS"],
        "decision_parts": decision_parts,
        "pass_allowlist_parts": pass_parts,
        "held_document_parts": held_parts,
    }
    _atomic_json(release / "phase1_release_manifest.json", release_manifest)
    return release_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--quality-policy", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(args.candidate, args.quality_policy)
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Phase 1 release finalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
