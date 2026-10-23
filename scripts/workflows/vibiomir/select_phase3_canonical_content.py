"""Select canonical representatives only for verified Phase 3 exact groups."""

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
from src.medrag.data.phase3.canonical import (
    CANONICAL_MEMBERSHIP_SCHEMA,
    CANONICAL_POLICY_VERSION,
    CANONICAL_REPRESENTATIVE_SCHEMA,
    CANONICAL_SCHEMA_VERSION,
    MEMBERSHIP_SCHEMA_VERSION,
    non_whitespace_length,
    provenance_score,
    select_representative,
)


EXPECTED = {
    "task2": "cebc2568f80085518c82bacfed22f10cfbd909faa77471fc255434d0f08148ef",
    "task3": "192820cbfdeb65b7e5dad564d545664fbcb17c4fdbe65e59882f1185cc99e6ec",
    "task4": "e0e8cc411aa31aac97bd2cf63d509c0a9d9068b4524295ae5788023271af9769",
    "task5": "5bd49e71ac4e09ab797dbd83c38ae36b84cf1ca0274e86992239b1d433cab234",
}

DEFAULTS = {
    "task2": ROOT / "outputs/vibiomir_phase3_20261015_task2_dry_run_v2/document_dedup/exact_document_groups.parquet",
    "task3": ROOT / "outputs/vibiomir_phase3_20261015_task3_dry_run_v1/document_dedup/near_duplicate_pairs.parquet",
    "task4": ROOT / "outputs/vibiomir_phase3_20261015_task4_dry_run_v2/passage_dedup/passage_content_groups.parquet",
    "task5": ROOT / "outputs/vibiomir_phase3_20261015_task5_pilot_v6/passage_dedup/passage_overlap_pairs.parquet",
    "documents": ROOT / "outputs/vibiomir_phase1_full_20261014_v1/clean_documents/part-00000.parquet",
    "passages": ROOT / "outputs/vibiomir_phase2_full_v2/passages/part-00000.parquet",
}


def _safe_output(path: Path) -> Path:
    resolved = path.resolve()
    output_root = (ROOT / "outputs").resolve()
    if resolved.parent != output_root or not resolved.name.startswith("vibiomir_phase3_"):
        raise ValueError("Output must be a direct outputs/vibiomir_phase3_<run_id> namespace")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _verify_inputs(paths: dict[str, Path]) -> dict[str, str]:
    fingerprints = {}
    for label in ("task2", "task3", "task4", "task5"):
        actual = sha256_file(paths[label])
        if actual != EXPECTED[label]:
            raise ValueError(f"{label} accepted artifact fingerprint changed")
        fingerprints[label] = actual
    # Direct pilot shards only; deliberately no corpus-wide checksum pass.
    fingerprints["documents"] = sha256_file(paths["documents"])
    fingerprints["passages"] = sha256_file(paths["passages"])
    return fingerprints


def _negative_gates(paths: dict[str, Path]) -> dict:
    near = pq.read_table(paths["task3"], columns=[
        "review_status", "auto_merge_allowed", "guard_reasons"
    ]).to_pylist()
    overlap = pq.read_table(paths["task5"], columns=[
        "overlap_type", "review_status", "auto_merge_allowed", "guard_reasons"
    ]).to_pylist()
    if any(row["auto_merge_allowed"] for row in near + overlap):
        raise ValueError("Unverified near/overlap relation attempted to authorize merging")
    if any(row["review_status"] not in {"unlabeled_review_required"} for row in near):
        raise ValueError("Task 3 contains a relation outside the accepted review-only contract")
    return {
        "near_document_pairs_excluded": len(near),
        "passage_overlap_pairs_excluded": len(overlap),
        "medical_guard_pairs_excluded": sum(bool(row["guard_reasons"]) for row in near + overlap),
        "intentional_chunk_overlap_pairs_excluded": sum(
            row.get("overlap_type") == "INTENTIONAL_CHUNK_OVERLAP" for row in overlap
        ),
        "partial_overlap_pairs_excluded": sum(
            row.get("overlap_type") in {"PARTIAL_CONTENT_REUSE", "CROSS_DOCUMENT_OVERLAP"}
            for row in overlap
        ),
    }


def _document_groups(paths: dict[str, Path]) -> tuple[dict[str, list[dict]], dict[int, dict]]:
    group_rows = pq.read_table(
        paths["task2"],
        filters=[("normalized_exact_duplicate", "=", True)],
    ).to_pylist()
    document_ids = {int(row["doc_id"]) for row in group_rows}
    columns = [
        "doc_id", "official_url", "source_kind", "title", "normalized_text",
        "source_text_sha256", "normalized_text_sha256", "quality_flags",
        "quality_review_required", "language_confidence", "phase0_raw_path",
        "phase0_raw_sha256", "raw_gzip_sha256", "raw_body_sha256", "pmid",
        "extractor_version", "source_document_path", "source_document_sha256",
    ]
    documents = {
        int(row["doc_id"]): row
        for row in pq.read_table(paths["documents"], columns=columns).to_pylist()
        if int(row["doc_id"]) in document_ids
    }
    if set(documents) != document_ids:
        raise ValueError("Verified Task 2 members are missing from the pilot document shard")
    groups: dict[str, list[dict]] = defaultdict(list)
    for group_row in group_rows:
        document = documents[int(group_row["doc_id"])]
        if document["official_url"] != group_row["official_url"]:
            raise ValueError("Task 2 document URL changed")
        if document["normalized_text_sha256"] != group_row["normalized_text_sha256"]:
            raise ValueError("Task 2 normalized identity changed")
        groups[group_row["document_content_group_id"]].append(document)
    if any(len(members) < 2 for members in groups.values()):
        raise ValueError("Task 2 duplicate group is incomplete")
    return dict(groups), documents


def _passage_groups(
    paths: dict[str, Path], documents: dict[int, dict]
) -> dict[str, list[dict]]:
    group_rows = pq.read_table(
        paths["task4"], filters=[("content_group_size", ">", 1)]
    ).to_pylist()
    passage_ids = {row["passage_id"] for row in group_rows}
    passages = {
        row["passage_id"]: row
        for row in pq.read_table(paths["passages"]).to_pylist()
        if row["passage_id"] in passage_ids
    }
    if set(passages) != passage_ids:
        raise ValueError("Verified Task 4 members are missing from the pilot passage shard")
    groups: dict[str, list[dict]] = defaultdict(list)
    for group_row in group_rows:
        passage = passages[group_row["passage_id"]]
        if int(passage["doc_id"]) not in documents:
            raise ValueError("Passage parent document is outside verified Task 2 pilot scope")
        for field in ("official_url", "char_start", "char_end", "source_text_sha256"):
            if passage[field] != group_row[field]:
                raise ValueError(f"Task 4 passage {field} changed")
        groups[group_row["passage_content_group_id"]].append({
            **passage,
            "document": documents[int(passage["doc_id"])],
            "normalized_content_hash": group_row["normalized_content_hash"],
        })
    if any(len(members) < 2 for members in groups.values()):
        raise ValueError("Task 4 duplicate group is incomplete")
    return dict(groups)


def _metric(member: dict, entity_type: str) -> tuple[bool, int, int, int]:
    document = member if entity_type == "document" else member["document"]
    text = document["normalized_text"] if entity_type == "document" else member["passage_text"]
    return (
        bool(document["quality_review_required"]),
        len(document["quality_flags"] or []),
        non_whitespace_length(text),
        provenance_score(document),
    )


def _select_all(document_groups: dict, passage_groups: dict) -> tuple[list[dict], list[dict]]:
    representatives, membership = [], []
    for entity_type, groups in (("document", document_groups), ("passage", passage_groups)):
        for group_id in sorted(groups):
            canonical, ranked, reason, chain = select_representative(
                groups[group_id], entity_type=entity_type
            )
            review, flag_count, completeness, provenance = _metric(canonical, entity_type)
            representatives.append({
                "entity_type": entity_type,
                "content_group_id": group_id,
                "group_size": len(ranked),
                "canonical_doc_id": int(canonical["doc_id"]),
                "canonical_passage_id": canonical.get("passage_id"),
                "canonical_official_url": canonical["official_url"],
                "canonical_char_start": canonical.get("char_start"),
                "canonical_char_end": canonical.get("char_end"),
                "text_quality_review_required": review,
                "quality_flag_count": flag_count,
                "text_completeness_chars": completeness,
                "provenance_score": provenance,
                "selection_reason": reason,
                "selection_reason_chain": chain,
                "selection_policy_version": CANONICAL_POLICY_VERSION,
            })
            for rank, member in enumerate(ranked, 1):
                member_review, member_flags, member_complete, member_provenance = _metric(
                    member, entity_type
                )
                membership.append({
                    "entity_type": entity_type,
                    "content_group_id": group_id,
                    "group_size": len(ranked),
                    "member_rank": rank,
                    "is_canonical": rank == 1,
                    "doc_id": int(member["doc_id"]),
                    "passage_id": member.get("passage_id"),
                    "official_url": member["official_url"],
                    "char_start": member.get("char_start"),
                    "char_end": member.get("char_end"),
                    "source_text_sha256": member["source_text_sha256"],
                    "normalized_content_sha256": (
                        member["normalized_text_sha256"] if entity_type == "document"
                        else member["normalized_content_hash"]
                    ),
                    "text_quality_review_required": member_review,
                    "quality_flag_count": member_flags,
                    "text_completeness_chars": member_complete,
                    "provenance_score": member_provenance,
                    "selection_reason": (
                        f"selected:{reason}" if rank == 1 else f"not_selected:{reason}"
                    ),
                    "selection_policy_version": CANONICAL_POLICY_VERSION,
                })
    return representatives, membership


def _atomic_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(*, paths: dict[str, Path], output: Path | None, dry_run: bool) -> dict:
    started = time.monotonic()
    fingerprints = _verify_inputs(paths)
    exclusions = _negative_gates(paths)
    document_groups, documents = _document_groups(paths)
    passage_groups = _passage_groups(paths, documents)
    representatives, membership = _select_all(document_groups, passage_groups)
    # A second in-memory run is cheap and proves order-independent determinism.
    rerun_representatives, rerun_membership = _select_all(
        {key: list(reversed(value)) for key, value in document_groups.items()},
        {key: list(reversed(value)) for key, value in passage_groups.items()},
    )
    if representatives != rerun_representatives or membership != rerun_membership:
        raise ValueError("Canonical selection is not deterministic")
    result = {
        "status": "pass",
        "mode": "dry_run" if dry_run else "pilot",
        "scope": "accepted_task2_task4_exact_groups_in_part-00000",
        "document_groups": len(document_groups),
        "passage_groups": len(passage_groups),
        "canonical_representatives": len(representatives),
        "group_memberships": len(membership),
        "exclusions": exclusions,
        "full_corpus_started": False,
    }
    if dry_run:
        return result
    if output is None:
        raise ValueError("Pilot mode requires --output")
    destination = _safe_output(output)
    canonical_path = destination / "canonical/canonical_representatives.parquet"
    membership_path = destination / "canonical/group_membership.parquet"
    _atomic_parquet(canonical_path, representatives, CANONICAL_REPRESENTATIVE_SCHEMA)
    _atomic_parquet(membership_path, membership, CANONICAL_MEMBERSHIP_SCHEMA)
    outputs = {
        "canonical_representatives": {
            "path": str(canonical_path.relative_to(ROOT)), "rows": len(representatives),
            "sha256": sha256_file(canonical_path), "schema_version": CANONICAL_SCHEMA_VERSION,
        },
        "group_membership": {
            "path": str(membership_path.relative_to(ROOT)), "rows": len(membership),
            "sha256": sha256_file(membership_path), "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        },
    }
    manifest = {
        "format": "vibiomir-phase3-task6-input-snapshot-v1",
        "input_verification_mode": "accepted_artifact_fingerprints_plus_selected_direct_shards",
        "input_sha256": fingerprints,
        "full_corpus_rehashed": False,
        "full_corpus_started": False,
    }
    policy = {
        "format": "vibiomir-phase3-canonical-selection-policy-v1",
        "policy_version": CANONICAL_POLICY_VERSION,
        "eligible_relationship": "verified normalized-exact groups from Tasks 2 and 4 only",
        "rank_order": [
            "quality_review_status", "quality_flag_count", "text_completeness",
            "source_provenance", "language_confidence", "stable_source_kind",
            "stable entity identifiers",
        ],
        "near_duplicates_eligible": False,
        "partial_or_intentional_overlap_eligible": False,
        "medical_guard_pairs_eligible": False,
        "transitive_pair_closure_enabled": False,
        "source_occurrences_mutated": False,
    }
    validation = {
        **result,
        "task": 6,
        "format": "vibiomir-phase3-task6-validation-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checks": {
            "accepted_exact_groups_only": True,
            "near_duplicates_without_equivalence_excluded": True,
            "partial_and_intentional_overlap_excluded": True,
            "medical_guard_pairs_excluded": True,
            "transitive_pair_closure_disabled": True,
            "one_canonical_per_group": True,
            "all_memberships_preserved": True,
            "source_ids_urls_offsets_preserved": True,
            "deterministic_rerun": True,
            "source_occurrences_mutated": False,
            "full_corpus_started": False,
        },
        "outputs": outputs,
    }
    report = f"""# Phase 3 Task 6 — Canonical Content Selection

- Status: **PASS**
- Scope: bounded pilot on accepted Task 2/4 exact groups in `part-00000`; full corpus was not run.
- Document groups: `{len(document_groups)}`; passage groups: `{len(passage_groups)}`.
- Canonical representatives: `{len(representatives)}`; membership rows: `{len(membership)}`.
- Near-document pairs excluded: `{exclusions['near_document_pairs_excluded']}`.
- Passage overlap/reuse pairs excluded: `{exclusions['passage_overlap_pairs_excluded']}`.
- Medical-guard pairs excluded: `{exclusions['medical_guard_pairs_excluded']}`.
- Source IDs, URLs, and passage offsets mutated: `false`.
- Full corpus started: `false`.

Only normalized-exact groups already verified by Tasks 2 and 4 were eligible. Task 3
review-only candidates and every Task 5 partial, cross-document, intentional chunk,
or medical-guard relation were excluded. Pairwise transitive closure is disabled.
"""
    _atomic_text(destination / "manifests/input_snapshot.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _atomic_text(destination / "manifests/selection_policy.json", json.dumps(policy, indent=2, sort_keys=True) + "\n")
    _atomic_text(destination / "validation/task6_canonical_validation.json", json.dumps(validation, indent=2, sort_keys=True) + "\n")
    _atomic_text(destination / "reports/PHASE3_TASK6_REPORT.md", report)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "pilot", "full"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--phase1", type=Path,
        default=ROOT / "outputs/vibiomir_phase1_full_20261014_v1",
    )
    parser.add_argument(
        "--phase2-release", type=Path,
        default=ROOT / "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json",
    )
    parser.add_argument("--task2", type=Path)
    parser.add_argument("--task3", type=Path)
    parser.add_argument("--task4", type=Path)
    parser.add_argument("--task5", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rss-limit-gib", type=float, default=16.0)
    parser.add_argument("--temporary-disk-budget-gib", type=float, default=128.0)
    args = parser.parse_args()
    if args.mode == "dry-run" and args.output is not None:
        parser.error("--output is only valid in pilot mode")
    if args.mode == "full":
        missing = [name for name in ("task2", "task3", "task4", "task5") if getattr(args, name) is None]
        if args.output is None or missing:
            parser.error(f"full mode requires --output and Task 2-5 roots; missing={missing}")
        from scripts.workflows.vibiomir.phase3_scalable_canonical import run_full_canonical
        result = run_full_canonical(
            root=ROOT, phase1=args.phase1.resolve(),
            phase2_release=args.phase2_release.resolve(), task2=args.task2.resolve(),
            task3=args.task3.resolve(), task4=args.task4.resolve(), task5=args.task5.resolve(),
            output=args.output, resume=args.resume, rss_limit_gib=args.rss_limit_gib,
            temporary_disk_budget_gib=args.temporary_disk_budget_gib,
        )
    else:
        result = run(paths=DEFAULTS, output=args.output, dry_run=args.mode == "dry-run")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
