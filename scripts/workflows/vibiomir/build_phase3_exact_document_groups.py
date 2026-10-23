"""Build exact Phase 3 document groups without mutating prior releases."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase3.exact import (
    EXACT_DOCUMENT_SCHEMA,
    EXACT_DOCUMENT_SCHEMA_VERSION,
    GROUPING_POLICY_VERSION,
    exact_group_row,
    validate_sha256,
    verify_group_content,
)


EXPECTED_PASS_DOCUMENTS = 1_073_531
EXPECTED_PHASE2_RELEASE_SHA256 = (
    "e1ecd3302dde6b705e2368daaf6c864a18a2831c08acfdbde8cac3c211e141f8"
)
EXPECTED_TASK1_AUDIT_SHA256 = (
    "9bc2a9575a4171bd1cc687bd883fbac82a8beaa104aff08800aee95cd7ea791a"
)
FORMAT = "vibiomir-phase3-exact-document-build-v1"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
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


def _safe_output(output: Path, protected: list[Path]) -> Path:
    resolved = output.resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if resolved.parent != outputs_root or not resolved.name.startswith("vibiomir_phase3_"):
        raise ValueError("Output must be a direct outputs/vibiomir_phase3_<run_id> namespace")
    for source in protected:
        source = source.resolve()
        if (
            resolved == source
            or resolved.is_relative_to(source)
            or source.is_relative_to(resolved)
        ):
            raise ValueError(f"Phase 3 output overlaps immutable input: {source}")
    return resolved


def _part_map(entries: list[dict]) -> dict[str, dict]:
    result = {entry["part"]: entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError("Duplicate part names in manifest")
    return result


def _select_parts(all_parts: list[str], max_parts: int | None, full: bool) -> list[str]:
    if full:
        if max_parts is not None:
            raise ValueError("--full and --max-parts are mutually exclusive")
        return all_parts
    if max_parts is None or max_parts < 1:
        raise ValueError("Dry-run requires --max-parts >= 1")
    if max_parts >= len(all_parts):
        raise ValueError("Dry-run must select fewer than all release parts")
    return all_parts[:max_parts]


def _verify_full_gate(path: Path | None) -> dict:
    if path is None or not path.is_file():
        raise ValueError("Full build requires --pilot-acceptance pointing to an accepted manifest")
    acceptance = _load_json(path)
    if acceptance.get("status") != "accepted" or acceptance.get("phase") != "phase3_pilot":
        raise ValueError("Phase 3 pilot has not been accepted")
    return acceptance


def _quick_schema_check(candidate: Path, part: str) -> None:
    clean_path = candidate / f"clean_documents/{part}.parquet"
    pass_path = candidate / f"release/pass_document_ids/{part}.parquet"
    clean_fields = set(pq.read_schema(clean_path).names)
    required_clean = {
        "doc_id",
        "official_url",
        "source_text",
        "normalized_text",
        "source_text_sha256",
        "normalized_text_sha256",
    }
    if not required_clean.issubset(clean_fields):
        raise ValueError(f"Phase 1 clean schema mismatch: {clean_path}")
    if pq.read_schema(pass_path).names != ["doc_id"]:
        raise ValueError(f"Phase 1 PASS schema mismatch: {pass_path}")


def _read_selected_documents(
    candidate: Path,
    part: str,
    pass_entry: dict,
    *,
    include_text: bool,
) -> list[dict]:
    clean_path = candidate / f"clean_documents/{part}.parquet"
    pass_path = candidate / f"release/pass_document_ids/{part}.parquet"
    pass_ids = pq.read_table(pass_path, columns=["doc_id"])["doc_id"].to_pylist()
    if len(pass_ids) != pass_entry["rows"] or len(pass_ids) != len(set(pass_ids)):
        raise ValueError(f"Invalid PASS IDs: {pass_path}")
    pass_set = set(pass_ids)
    columns = [
        "doc_id",
        "official_url",
        "source_text_sha256",
        "normalized_text_sha256",
    ]
    if include_text:
        columns.extend(["source_text", "normalized_text"])
    rows = pq.read_table(clean_path, columns=columns).to_pylist()
    by_id = {int(row["doc_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate candidate doc_id: {clean_path}")
    if not pass_set.issubset(by_id):
        raise ValueError(f"PASS IDs missing from clean shard: {part}")
    return [by_id[int(doc_id)] for doc_id in pass_ids]


def _count_hashes(
    candidate: Path,
    parts: list[str],
    pass_parts: dict[str, dict],
    *,
    progress=print,
) -> tuple[Counter, Counter, int]:
    raw_counts: Counter[bytes] = Counter()
    normalized_counts: Counter[bytes] = Counter()
    seen_ids: set[int] = set()
    for number, part in enumerate(parts, 1):
        documents = _read_selected_documents(
            candidate, part, pass_parts[part], include_text=False
        )
        for row in documents:
            doc_id = int(row["doc_id"])
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate PASS doc_id across shards: {doc_id}")
            seen_ids.add(doc_id)
            if not isinstance(row["official_url"], str) or not row["official_url"]:
                raise ValueError(f"Missing official_url for doc_id={doc_id}")
            raw_hash = validate_sha256(
                row["source_text_sha256"], field="source_text_sha256"
            )
            normalized_hash = validate_sha256(
                row["normalized_text_sha256"], field="normalized_text_sha256"
            )
            raw_counts[bytes.fromhex(raw_hash)] += 1
            normalized_counts[bytes.fromhex(normalized_hash)] += 1
        progress(f"Counted exact hashes {number}/{len(parts)} documents={len(seen_ids)}")
    return raw_counts, normalized_counts, len(seen_ids)


def _build_output(
    *,
    candidate: Path,
    destination: Path,
    parts: list[str],
    pass_parts: dict[str, dict],
    raw_counts: Counter,
    normalized_counts: Counter,
    normalization_version: str,
    phase1_release_id: str,
    grouping_scope: str,
    progress=print,
) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)

    raw_baselines: dict[bytes, tuple[int, str]] = {}
    normalized_baselines: dict[bytes, tuple[int, str]] = {}
    normalized_raw_variants: dict[bytes, set[bytes]] = defaultdict(set)
    match_types: Counter[str] = Counter()
    written = 0
    writer = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            EXACT_DOCUMENT_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        for number, part in enumerate(parts, 1):
            documents = _read_selected_documents(
                candidate, part, pass_parts[part], include_text=True
            )
            output_rows = []
            for row in documents:
                doc_id = int(row["doc_id"])
                raw_digest = bytes.fromhex(row["source_text_sha256"])
                normalized_digest = bytes.fromhex(row["normalized_text_sha256"])
                raw_size = raw_counts[raw_digest]
                normalized_size = normalized_counts[normalized_digest]
                if raw_size > 1:
                    if not isinstance(row["source_text"], str):
                        raise ValueError(f"Missing source_text for doc_id={doc_id}")
                    verify_group_content(
                        raw_baselines,
                        raw_digest,
                        doc_id=doc_id,
                        text=row["source_text"],
                        label="source_text",
                    )
                if normalized_size > 1:
                    if not isinstance(row["normalized_text"], str):
                        raise ValueError(f"Missing normalized_text for doc_id={doc_id}")
                    verify_group_content(
                        normalized_baselines,
                        normalized_digest,
                        doc_id=doc_id,
                        text=row["normalized_text"],
                        label="normalized_text",
                    )
                    normalized_raw_variants[normalized_digest].add(raw_digest)
                output = exact_group_row(
                    row,
                    raw_size=raw_size,
                    normalized_size=normalized_size,
                    normalization_version=normalization_version,
                    phase1_release_id=phase1_release_id,
                    grouping_scope=grouping_scope,
                )
                match_types[output["exact_match_type"]] += 1
                output_rows.append(output)
            writer.write_table(pa.Table.from_pylist(output_rows, schema=EXACT_DOCUMENT_SCHEMA))
            written += len(output_rows)
            progress(f"Wrote exact groups {number}/{len(parts)} documents={written}")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    return {
        "documents": written,
        "raw_exact_groups": sum(count > 1 for count in raw_counts.values()),
        "raw_exact_documents": sum(count for count in raw_counts.values() if count > 1),
        "normalized_exact_groups": sum(
            count > 1 for count in normalized_counts.values()
        ),
        "normalized_exact_documents": sum(
            count for count in normalized_counts.values() if count > 1
        ),
        "normalized_groups_with_raw_variants": sum(
            len(values) > 1 for values in normalized_raw_variants.values()
        ),
        "unique_raw_groups": len(raw_counts),
        "unique_normalized_groups": len(normalized_counts),
        "maximum_raw_group_size": max(raw_counts.values(), default=0),
        "maximum_normalized_group_size": max(normalized_counts.values(), default=0),
        "exact_match_type_counts": dict(sorted(match_types.items())),
        "verified_raw_duplicate_groups": len(raw_baselines),
        "verified_normalized_duplicate_groups": len(normalized_baselines),
    }


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    candidate = args.candidate.resolve()
    phase2_release_path = args.phase2_release.resolve()
    output = _safe_output(args.output, [candidate, phase2_release_path.parent.parent])
    destination = output / "document_dedup/exact_document_groups.parquet"
    validation_path = output / "validation/task2_exact_document_validation.json"
    snapshot_path = output / "manifests/input_snapshot.json"
    config_path = output / "manifests/dedup_config.json"
    checkpoint_path = output / "checkpoints/task2_exact_document_groups.json"
    report_path = output / "reports/TASK2_EXACT_DOCUMENT_DEDUP_REPORT.md"
    for path in (
        destination,
        validation_path,
        snapshot_path,
        config_path,
        checkpoint_path,
        report_path,
    ):
        if path.exists():
            raise FileExistsError(path)

    manifest_path = candidate / "phase1_manifest.json"
    release_path = candidate / "release/phase1_release_manifest.json"
    manifest = _load_json(manifest_path)
    release = _load_json(release_path)
    phase2_release = _load_json(phase2_release_path)
    candidate_sha256 = sha256_file(manifest_path)
    release_sha256 = sha256_file(release_path)
    phase2_release_sha256 = sha256_file(phase2_release_path)
    if phase2_release_sha256 != args.expected_phase2_release_sha256:
        raise ValueError("Phase 2 release fingerprint changed; stop and rerun verification")
    if manifest.get("status") != "complete" or manifest.get("build_scope") != "full":
        raise ValueError("Phase 1 candidate is not a completed full build")
    if release.get("status") != "complete":
        raise ValueError("Phase 1 release is not complete")
    if release.get("candidate_manifest_sha256") != candidate_sha256:
        raise ValueError("Phase 1 candidate manifest checksum mismatch")
    if release.get("pass_documents") != EXPECTED_PASS_DOCUMENTS:
        raise ValueError("Unexpected Phase 1 PASS count")
    if phase2_release.get("status") != "complete_validated":
        raise ValueError("Phase 2 release is not complete_validated")
    if phase2_release.get("phase1_release_manifest_sha256") != release_sha256:
        raise ValueError("Phase 2 does not reference the selected Phase 1 release")

    task1_report = ROOT / (
        "outputs/vibiomir_phase3_20261015_task1/reports/PHASE3_DATA_AUDIT.md"
    )
    if not task1_report.is_file():
        raise ValueError("Verified Phase 3 Task 1 report is missing")
    task1_audit_sha256 = sha256_file(task1_report)
    if task1_audit_sha256 != EXPECTED_TASK1_AUDIT_SHA256:
        raise ValueError("Phase 3 Task 1 fingerprint changed; stop and rerun verification")

    clean_hashes = manifest.get("part_checkpoints_sha256", {})
    pass_parts = _part_map(release.get("pass_allowlist_parts", []))
    all_parts = sorted(clean_hashes)
    if len(all_parts) != 922 or set(all_parts) != set(pass_parts):
        raise ValueError("Phase 1 shard inventory mismatch")
    parts = _select_parts(all_parts, args.max_parts, args.full)
    _quick_schema_check(candidate, parts[0])
    acceptance = _verify_full_gate(args.pilot_acceptance) if args.full else None
    scope = "full" if args.full else f"dry_run_first_{len(parts)}_parts"

    raw_counts, normalized_counts, selected_documents = _count_hashes(
        candidate, parts, pass_parts
    )
    if args.full and selected_documents != EXPECTED_PASS_DOCUMENTS:
        raise ValueError("Full build did not select the complete PASS allowlist")

    statistics = _build_output(
        candidate=candidate,
        destination=destination,
        parts=parts,
        pass_parts=pass_parts,
        raw_counts=raw_counts,
        normalized_counts=normalized_counts,
        normalization_version=manifest["normalization_version"],
        phase1_release_id=release_sha256,
        grouping_scope=scope,
    )

    output_meta = pq.read_metadata(destination)
    if output_meta.num_rows != selected_documents:
        raise ValueError("Output row count differs from selected PASS documents")
    if sha256_file(release_path) != release_sha256:
        raise ValueError("Phase 1 release changed during build")
    if sha256_file(phase2_release_path) != phase2_release_sha256:
        raise ValueError("Phase 2 release changed during build")

    snapshot = {
        "format": "vibiomir-phase3-input-snapshot-v1",
        "scope": scope,
        "selected_parts": parts,
        "selected_documents": selected_documents,
        "groups_are_corpus_complete": args.full,
        "phase1_candidate": str(candidate.relative_to(ROOT)),
        "phase1_candidate_manifest_sha256": candidate_sha256,
        "phase1_release_manifest_sha256": release_sha256,
        "phase2_release_manifest": str(phase2_release_path.relative_to(ROOT)),
        "phase2_release_manifest_sha256": phase2_release_sha256,
        "task1_audit_sha256": task1_audit_sha256,
        "input_verification_mode": "incremental_verified_manifest",
        "pilot_acceptance": acceptance,
    }
    config = {
        "format": "vibiomir-phase3-dedup-config-v1",
        "task": 2,
        "scope": scope,
        "grouping_policy_version": GROUPING_POLICY_VERSION,
        "schema_version": EXACT_DOCUMENT_SCHEMA_VERSION,
        "raw_exact_basis": "SHA-256(UTF-8 source_text) plus exact string verification",
        "normalized_exact_basis": (
            "SHA-256(UTF-8 normalized_text) plus exact string verification"
        ),
        "document_content_group_basis": "normalized exact content",
        "pass_allowlist_only": True,
        "delete_or_merge_occurrences": False,
        "canonical_selection_performed": False,
        "source_hashes_recomputed": False,
        "verified_source_hashes_reused": True,
    }
    completed = datetime.now(timezone.utc).isoformat()
    validation = {
        "format": FORMAT,
        "status": "pass",
        "task": 2,
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
            "sha256": sha256_file(destination),
            "schema_version": EXACT_DOCUMENT_SCHEMA_VERSION,
        },
        "statistics": statistics,
        "checks": {
            "phase1_release_checksum_stable": True,
            "phase2_release_checksum_stable": True,
            "pass_allowlist_only": True,
            "doc_id_unique": True,
            "one_output_row_per_selected_doc_id": True,
            "verified_source_hashes_reused": True,
            "source_hashes_recomputed": False,
            "normalized_hashes_recomputed": False,
            "duplicate_hash_groups_exactly_verified": True,
            "phase0_mutated": False,
            "phase1_mutated": False,
            "phase2_mutated": False,
            "index_mutated": False,
            "canonical_selection_performed": False,
        },
    }
    _atomic_json(snapshot_path, snapshot)
    _atomic_json(config_path, config)
    _atomic_json(validation_path, validation)
    checkpoint = {
        "format": "vibiomir-phase3-task2-checkpoint-v1",
        "status": "complete",
        "scope": scope,
        "phase1_release_manifest_sha256": release_sha256,
        "phase2_release_manifest_sha256": phase2_release_sha256,
        "task1_audit_sha256": task1_audit_sha256,
        "output_sha256": validation["output"]["sha256"],
        "output_rows": output_meta.num_rows,
        "completed_at_utc": completed,
    }
    report = f"""# Phase 3 Task 2 — Exact Document Deduplication

- Status: **PASS**
- Scope: `{scope}`
- Corpus-complete groups: `{str(args.full).lower()}`
- Documents: `{statistics['documents']:,}`
- Raw-exact groups: `{statistics['raw_exact_groups']:,}`
- Raw-exact documents: `{statistics['raw_exact_documents']:,}`
- Normalized-exact groups: `{statistics['normalized_exact_groups']:,}`
- Normalized-exact documents: `{statistics['normalized_exact_documents']:,}`
- Output SHA-256: `{validation['output']['sha256']}`
- Input verification: reused verified Task 1 and release fingerprints; no full source rehash.
- Occurrences deleted or merged: `false`
- Canonical selection performed: `false`

{('This is a corpus-complete Task 2 component; final Phase 3 release validation is separate.' if args.full else 'This run is bounded and is not a Full Phase 3 release.')}
"""
    _atomic_text(report_path, report)
    _atomic_json(checkpoint_path, checkpoint)
    return validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "outputs/vibiomir_phase1_full_20261014_v1",
    )
    parser.add_argument(
        "--phase2-release",
        type=Path,
        default=ROOT / "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parts", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pilot-acceptance", type=Path)
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

