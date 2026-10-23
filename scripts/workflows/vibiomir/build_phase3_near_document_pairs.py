"""Build bounded, review-only near-document pairs for Phase 3 Task 3."""

from __future__ import annotations

import argparse
import json
import os
import resource
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

from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase3.near import (
    NEAR_DUPLICATE_SCHEMA,
    NEAR_POLICY_VERSION,
    NEAR_SCHEMA_VERSION,
    hash_shingles,
    lsh_candidates,
    medical_guard_reasons,
    minhash_signature,
    pair_id,
    shingle_text,
    signature_similarity,
    similarity_scores,
    threshold_passes,
)


EXPECTED_PHASE1_RELEASE_SHA256 = (
    "483d7dd4e606ee1f6ed011e66d62116c97fb2db7bf6b75027e89b7be58198bae"
)
EXPECTED_PHASE2_RELEASE_SHA256 = (
    "e1ecd3302dde6b705e2368daaf6c864a18a2831c08acfdbde8cac3c211e141f8"
)
EXPECTED_TASK2_OUTPUT_SHA256 = (
    "cebc2568f80085518c82bacfed22f10cfbd909faa77471fc255434d0f08148ef"
)


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
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".partial", delete=False,
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
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".partial", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=NEAR_DUPLICATE_SCHEMA),
            temporary,
            compression="zstd",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output(path: Path) -> Path:
    resolved = path.resolve()
    outputs = (ROOT / "outputs").resolve()
    if resolved.parent != outputs or not resolved.name.startswith("vibiomir_phase3_"):
        raise ValueError("Output must be a direct outputs/vibiomir_phase3_<run_id> namespace")
    return resolved


def _load_documents(candidate: Path, parts: list[str], pass_parts: dict) -> list[dict]:
    documents = []
    seen = set()
    required = {
        "doc_id", "official_url", "normalized_text", "normalized_text_sha256",
        "detected_language",
    }
    for part in parts:
        clean_path = candidate / f"clean_documents/{part}.parquet"
        pass_path = candidate / f"release/pass_document_ids/{part}.parquet"
        if not required.issubset(pq.read_schema(clean_path).names):
            raise ValueError(f"Phase 1 schema mismatch: {clean_path}")
        pass_ids = pq.read_table(pass_path, columns=["doc_id"])["doc_id"].to_pylist()
        if len(pass_ids) != pass_parts[part]["rows"]:
            raise ValueError(f"PASS row count mismatch: {pass_path}")
        rows = pq.read_table(clean_path, columns=sorted(required)).to_pylist()
        by_id = {int(row["doc_id"]): row for row in rows}
        for doc_id in pass_ids:
            doc_id = int(doc_id)
            if doc_id in seen or doc_id not in by_id:
                raise ValueError(f"Invalid Task 3 doc_id mapping: {doc_id}")
            seen.add(doc_id)
            documents.append(by_id[doc_id])
    return documents


def _prepare_records(documents: list[dict], *, num_perm: int) -> list[dict]:
    records = []
    for number, document in enumerate(documents, 1):
        text = document["normalized_text"]
        if not isinstance(text, str) or not text:
            raise ValueError(f"Missing normalized_text: {document['doc_id']}")
        shingles, strategy, size = shingle_text(text, document["detected_language"])
        hashed = hash_shingles(shingles)
        if not hashed:
            raise ValueError(f"No shingles generated: {document['doc_id']}")
        records.append({
            **document,
            "shingles": hashed,
            "shingle_strategy": strategy,
            "shingle_size": size,
            "signature": minhash_signature(hashed, num_perm),
        })
        if number % 1000 == 0 or number == len(documents):
            print(f"Prepared MinHash {number}/{len(documents)}", flush=True)
    return records


def _build_pairs(
    records: list[dict],
    *,
    thresholds: tuple[float, ...],
    acceptance_threshold: float,
    containment_threshold: float,
    containment_jaccard_floor: float,
    bands: int,
    rows_per_band: int,
    maximum_bucket_size: int,
    phase1_release_id: str,
) -> tuple[list[dict], dict]:
    candidate_indices, overflow = lsh_candidates(
        records,
        bands=bands,
        rows_per_band=rows_per_band,
        maximum_bucket_size=maximum_bucket_size,
    )
    output = []
    stats = Counter()
    threshold_counts = Counter()
    guard_counts = Counter()
    for left_index, right_index in sorted(candidate_indices):
        left, right = records[left_index], records[right_index]
        stats["lsh_candidates"] += 1
        if left["normalized_text_sha256"] == right["normalized_text_sha256"]:
            stats["exact_normalized_excluded"] += 1
            continue
        shared, jaccard, left_containment, right_containment = similarity_scores(
            left["shingles"], right["shingles"]
        )
        estimated = signature_similarity(left["signature"], right["signature"])
        passed = threshold_passes(
            jaccard,
            left_containment,
            right_containment,
            thresholds,
            containment_threshold=containment_threshold,
            containment_jaccard_floor=containment_jaccard_floor,
        )
        for threshold in passed:
            threshold_counts[str(threshold)] += 1
        guards = medical_guard_reasons(
            left["normalized_text"], right["normalized_text"]
        )
        for guard in guards:
            guard_counts[guard] += 1
        if acceptance_threshold not in passed:
            stats["below_acceptance_threshold"] += 1
            continue
        if guards:
            stats["medical_guard_rejected"] += 1
            continue
        left_doc_id, right_doc_id = int(left["doc_id"]), int(right["doc_id"])
        if right_doc_id < left_doc_id:
            left, right = right, left
            left_doc_id, right_doc_id = right_doc_id, left_doc_id
            left_containment, right_containment = right_containment, left_containment
        output.append({
            "pair_id": pair_id(left_doc_id, right_doc_id),
            "left_doc_id": left_doc_id,
            "right_doc_id": right_doc_id,
            "left_official_url": left["official_url"],
            "right_official_url": right["official_url"],
            "left_language": left["detected_language"],
            "right_language": right["detected_language"],
            "shingle_strategy": left["shingle_strategy"],
            "shingle_size": left["shingle_size"],
            "left_shingles": len(left["shingles"]),
            "right_shingles": len(right["shingles"]),
            "shared_shingles": shared,
            "minhash_similarity": estimated,
            "jaccard_similarity": jaccard,
            "left_containment": left_containment,
            "right_containment": right_containment,
            "threshold_passes": passed,
            "guard_reasons": [],
            "review_status": "unlabeled_review_required",
            "auto_merge_allowed": False,
            "phase1_release_id": phase1_release_id,
            "policy_version": NEAR_POLICY_VERSION,
        })
    stats["accepted_unlabeled_pairs"] = len(output)
    return output, {
        **dict(stats),
        "lsh_overflow_events": overflow,
        "threshold_candidate_counts": dict(sorted(threshold_counts.items())),
        "medical_guard_counts": dict(sorted(guard_counts.items())),
    }


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    output = _safe_output(args.output)
    task2 = args.task2.resolve()
    candidate = args.candidate.resolve()
    destination = output / "document_dedup/near_duplicate_pairs.parquet"
    validation_path = output / "validation/task3_near_document_validation.json"
    checkpoint_path = output / "checkpoints/task3_near_document_pairs.json"
    report_path = output / "reports/TASK3_NEAR_DOCUMENT_DEDUP_REPORT.md"
    snapshot_path = output / "manifests/input_snapshot.json"
    config_path = output / "manifests/dedup_config.json"
    for path in (
        destination, validation_path, checkpoint_path, report_path, snapshot_path,
        config_path,
    ):
        if path.exists():
            raise FileExistsError(path)

    phase1_release_path = candidate / "release/phase1_release_manifest.json"
    phase2_release_path = ROOT / (
        "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json"
    )
    task2_output = task2 / "document_dedup/exact_document_groups.parquet"
    task2_snapshot_path = task2 / "manifests/input_snapshot.json"
    task2_validation_path = task2 / "validation/task2_exact_document_validation.json"
    phase1_release_sha = sha256_file(phase1_release_path)
    phase2_release_sha = sha256_file(phase2_release_path)
    task2_output_sha = sha256_file(task2_output)
    if phase1_release_sha != EXPECTED_PHASE1_RELEASE_SHA256:
        raise ValueError("Phase 1 release fingerprint changed")
    if phase2_release_sha != EXPECTED_PHASE2_RELEASE_SHA256:
        raise ValueError("Phase 2 release fingerprint changed")
    if task2_output_sha != args.expected_task2_output_sha256:
        raise ValueError("Task 2 output fingerprint changed")
    task2_snapshot = _load_json(task2_snapshot_path)
    task2_validation = _load_json(task2_validation_path)
    if task2_validation.get("status") != "pass":
        raise ValueError("Task 2 validation is not PASS")
    if task2_snapshot.get("phase1_release_manifest_sha256") != phase1_release_sha:
        raise ValueError("Task 2 references a different Phase 1 release")
    if task2_validation["output"]["sha256"] != task2_output_sha:
        raise ValueError("Task 2 validation fingerprint mismatch")

    parts = task2_snapshot["selected_parts"]
    release = _load_json(phase1_release_path)
    pass_parts = {entry["part"]: entry for entry in release["pass_allowlist_parts"]}
    documents = _load_documents(candidate, parts, pass_parts)
    task2_ids = pq.read_table(task2_output, columns=["doc_id"])["doc_id"].to_pylist()
    if [row["doc_id"] for row in documents] != task2_ids:
        raise ValueError("Task 3 document IDs differ from Task 2 scope")

    records = _prepare_records(documents, num_perm=args.bands * args.rows_per_band)
    thresholds = tuple(sorted(set(args.thresholds)))
    if args.acceptance_threshold not in thresholds:
        raise ValueError("Acceptance threshold must be included in --thresholds")
    pairs, statistics = _build_pairs(
        records,
        thresholds=thresholds,
        acceptance_threshold=args.acceptance_threshold,
        containment_threshold=args.containment_threshold,
        containment_jaccard_floor=args.containment_jaccard_floor,
        bands=args.bands,
        rows_per_band=args.rows_per_band,
        maximum_bucket_size=args.maximum_bucket_size,
        phase1_release_id=phase1_release_sha,
    )
    pair_ids = [row["pair_id"] for row in pairs]
    pairs_unique = {(row["left_doc_id"], row["right_doc_id"]) for row in pairs}
    if len(pair_ids) != len(set(pair_ids)) or len(pairs) != len(pairs_unique):
        raise ValueError("Duplicate near-document output pair")
    _atomic_parquet(destination, pairs)
    output_sha = sha256_file(destination)
    output_meta = pq.read_metadata(destination)
    completed = datetime.now(timezone.utc).isoformat()

    snapshot = {
        "format": "vibiomir-phase3-task3-input-snapshot-v1",
        "scope": task2_snapshot["scope"],
        "groups_are_corpus_complete": task2_snapshot["groups_are_corpus_complete"],
        "selected_parts": parts,
        "documents": len(documents),
        "phase1_release_manifest_sha256": phase1_release_sha,
        "phase2_release_manifest_sha256": phase2_release_sha,
        "task2_output_sha256": task2_output_sha,
        "input_verification_mode": "incremental_verified_manifest",
    }
    config = {
        "format": "vibiomir-phase3-near-document-config-v1",
        "task": 3,
        "policy_version": NEAR_POLICY_VERSION,
        "schema_version": NEAR_SCHEMA_VERSION,
        "bands": args.bands,
        "rows_per_band": args.rows_per_band,
        "maximum_bucket_size": args.maximum_bucket_size,
        "thresholds": thresholds,
        "acceptance_threshold": args.acceptance_threshold,
        "containment_threshold": args.containment_threshold,
        "containment_jaccard_floor": args.containment_jaccard_floor,
        "labeled_evaluation_available": False,
        "auto_merge_allowed": False,
    }
    validation = {
        "format": "vibiomir-phase3-task3-validation-v1",
        "status": "pass",
        "task": 3,
        "scope": task2_snapshot["scope"],
        "groups_are_corpus_complete": task2_snapshot["groups_are_corpus_complete"],
        "completed_at_utc": completed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output": {
            "path": str(destination.relative_to(ROOT)),
            "rows": output_meta.num_rows,
            "bytes": destination.stat().st_size,
            "sha256": output_sha,
            "schema_version": NEAR_SCHEMA_VERSION,
        },
        "statistics": statistics,
        "evaluation": {
            "labeled_set_present": False,
            "precision": None,
            "recall": None,
            "reason": "No valid labeled near-duplicate set was found; all output pairs require review.",
        },
        "checks": {
            "input_fingerprints_stable": True,
            "task2_document_scope_preserved": True,
            "pair_ids_unique": True,
            "exact_normalized_pairs_excluded": True,
            "medical_guards_applied": True,
            "auto_merge_allowed": False,
            "phase0_mutated": False,
            "phase1_mutated": False,
            "phase2_mutated": False,
            "index_mutated": False,
        },
    }
    checkpoint = {
        "format": "vibiomir-phase3-task3-checkpoint-v1",
        "status": "complete",
        "scope": task2_snapshot["scope"],
        "task2_output_sha256": task2_output_sha,
        "output_sha256": output_sha,
        "output_rows": output_meta.num_rows,
        "completed_at_utc": completed,
    }
    report = f"""# Phase 3 Task 3 — Near-document Deduplication

- Status: **PASS**
- Scope: `{task2_snapshot['scope']}`
- Documents: `{len(documents):,}`
- LSH candidates: `{statistics.get('lsh_candidates', 0):,}`
- Output review pairs: `{len(pairs):,}`
- Labeled evaluation available: `false`
- Precision/Recall: `not reported`
- Auto merge allowed: `false`
- Output SHA-256: `{output_sha}`
- Input verification: incremental verified manifests; no Task 1 rerun or full source rehash.

Every output pair is marked `unlabeled_review_required`. This run is not a Full Phase 3 release.
"""
    _atomic_json(snapshot_path, snapshot)
    _atomic_json(config_path, config)
    _atomic_json(validation_path, validation)
    _atomic_text(report_path, report)
    _atomic_json(checkpoint_path, checkpoint)
    return validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", type=Path,
        default=ROOT / "outputs/vibiomir_phase1_full_20261014_v1",
    )
    parser.add_argument("--task2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bands", type=int, default=16)
    parser.add_argument("--rows-per-band", type=int, default=4)
    parser.add_argument("--maximum-bucket-size", type=int, default=200)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95]
    )
    parser.add_argument("--acceptance-threshold", type=float, default=0.90)
    parser.add_argument("--containment-threshold", type=float, default=0.95)
    parser.add_argument("--containment-jaccard-floor", type=float, default=0.60)
    parser.add_argument(
        "--expected-task2-output-sha256", default=EXPECTED_TASK2_OUTPUT_SHA256
    )
    parser.add_argument("--full-disk-backed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-partitions", type=int, default=256)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--rss-limit-gib", type=float, default=16.0)
    parser.add_argument("--temporary-disk-budget-gib", type=float, default=256.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.full_disk_backed:
        from scripts.workflows.vibiomir.phase3_scalable_candidates import (
            run_full_document_candidates,
        )
        result = run_full_document_candidates(
            root=ROOT,
            candidate=args.candidate.resolve(),
            task2=args.task2.resolve(),
            output=args.output,
            resume=args.resume,
            partitions=args.candidate_partitions,
            maximum_bucket_size=args.maximum_bucket_size,
            batch_rows=args.batch_rows,
            rss_limit_gib=args.rss_limit_gib,
            temporary_disk_budget_gib=args.temporary_disk_budget_gib,
            bands=args.bands,
            rows_per_band=args.rows_per_band,
        )
    else:
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

