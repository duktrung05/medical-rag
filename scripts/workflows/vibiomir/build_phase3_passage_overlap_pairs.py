"""Build a bounded, review-oriented Phase 3 passage-overlap pilot."""

from __future__ import annotations

import argparse
import hashlib
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

from src.medrag.data.phase1.normalization import normalize_medical_text, text_sha256  # noqa: E402
from src.medrag.data.phase1.selection import sha256_file  # noqa: E402
from src.medrag.data.phase3.near import (  # noqa: E402
    hash_shingles,
    lsh_candidates,
    medical_guard_reasons,
    minhash_signature,
    shingle_text,
    signature_similarity,
    similarity_scores,
)
from src.medrag.data.phase3.overlap import (  # noqa: E402
    INTENTIONAL_CHUNK_OVERLAP,
    OVERLAP_POLICY_VERSION,
    OVERLAP_SCHEMA_VERSION,
    PASSAGE_OVERLAP_SCHEMA,
    classify_overlap,
    is_boilerplate_only,
    overlap_pair_id,
)


EXPECTED_PHASE2_RELEASE_SHA256 = (
    "e1ecd3302dde6b705e2368daaf6c864a18a2831c08acfdbde8cac3c211e141f8"
)
EXPECTED_TASK4_OUTPUT_SHA256 = (
    "e0e8cc411aa31aac97bd2cf63d509c0a9d9068b4524295ae5788023271af9769"
)
EXPECTED_TASK3_OUTPUT_SHA256 = (
    "192820cbfdeb65b7e5dad564d545664fbcb17c4fdbe65e59882f1185cc99e6ec"
)
FORMAT = "vibiomir-phase3-passage-overlap-build-v1"


INPUT_COLUMNS = [
    "passage_id", "doc_id", "official_url", "passage_index", "passage_text",
    "char_start", "char_end", "source_text_sha256", "language",
    "chunker_version",
]


PREPARED_SCHEMA = pa.schema([
    ("passage_id", pa.string()),
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("passage_index", pa.int32()),
    ("passage_text", pa.string()),
    ("char_start", pa.int64()),
    ("char_end", pa.int64()),
    ("source_text_sha256", pa.string()),
    ("language", pa.string()),
    ("chunker_version", pa.string()),
    ("normalized_content_hash", pa.string()),
    ("shingle_strategy", pa.string()),
    ("shingle_size", pa.int16()),
    ("shingles", pa.list_(pa.uint64())),
    ("signature", pa.list_(pa.uint64())),
])


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


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output(path: Path, protected: list[Path]) -> Path:
    resolved = path.resolve()
    outputs = (ROOT / "outputs").resolve()
    if resolved.parent != outputs or not resolved.name.startswith("vibiomir_phase3_"):
        raise ValueError("Output must be a direct outputs/vibiomir_phase3_<run_id> namespace")
    for source in protected:
        source = source.resolve()
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError(f"Task 5 output overlaps immutable input: {source}")
    return resolved


def _quick_schema_check(path: Path) -> None:
    schema = pq.read_schema(path)
    if not set(INPUT_COLUMNS).issubset(schema.names):
        raise ValueError(f"Phase 2 passage schema mismatch: {path}")


def _stable_key(seed: str, language: str, doc_id: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{language}\0{doc_id}".encode("utf-8")).digest()


def _select_pilot_rows(
    passages_root: Path,
    release_parts: dict[str, dict],
    config: dict,
) -> tuple[list[dict], dict]:
    base_part = config["base_task4_part"]
    requested_parts = {base_part}
    for spec in config["supplemental_language_sampling"].values():
        requested_parts.update(spec["parts"])
    missing = requested_parts - set(release_parts)
    if missing:
        raise ValueError(f"Pilot config references missing Phase 2 shards: {sorted(missing)}")
    rows_by_part: dict[str, list[dict]] = {}
    checksums = {}
    for number, part in enumerate(sorted(requested_parts), 1):
        path = passages_root / f"{part}.parquet"
        _quick_schema_check(path)
        fingerprint = sha256_file(path)
        if fingerprint != release_parts[part]["sha256"]:
            raise ValueError(f"Selected Phase 2 shard fingerprint changed: {path}")
        rows = pq.read_table(path, columns=INPUT_COLUMNS).to_pylist()
        if len(rows) != release_parts[part]["rows"]:
            raise ValueError(f"Selected Phase 2 shard row count changed: {path}")
        rows_by_part[part] = rows
        checksums[part] = fingerprint
        print(f"Loaded pilot source shard {number}/{len(requested_parts)} {part}", flush=True)

    selected = list(rows_by_part[base_part])
    selection = {"base": len(selected)}
    seed = config["selection_seed"]
    for language, spec in sorted(config["supplemental_language_sampling"].items()):
        by_doc: dict[int, list[dict]] = defaultdict(list)
        for part in spec["parts"]:
            for row in rows_by_part[part]:
                if (row["language"] or "unknown").casefold() == language.casefold():
                    by_doc[int(row["doc_id"])].append(row)
        ordered_docs = sorted(by_doc, key=lambda doc_id: _stable_key(seed, language, doc_id))
        language_rows = []
        for doc_id in ordered_docs:
            group = sorted(by_doc[doc_id], key=lambda row: int(row["passage_index"]))
            language_rows.extend(group)
            if len(language_rows) >= int(spec["target_passages"]):
                break
        if len(language_rows) < int(spec["target_passages"]):
            raise ValueError(f"Insufficient {language} passages for pilot")
        selected.extend(language_rows)
        selection[language] = len(language_rows)
    passage_ids = [row["passage_id"] for row in selected]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Pilot sampling produced duplicate passage IDs")
    return selected, {"selected_by_stratum": selection, "selected_shard_sha256": checksums}


def _verify_task4_base(rows: list[dict], task4_path: Path, base_part_rows: int) -> dict[str, str]:
    fingerprint = sha256_file(task4_path)
    if fingerprint != EXPECTED_TASK4_OUTPUT_SHA256:
        raise ValueError("Task 4 passage content fingerprint changed")
    table = pq.read_table(
        task4_path,
        columns=[
            "passage_id", "doc_id", "char_start", "char_end",
            "normalized_content_hash",
        ],
    )
    if table.num_rows != base_part_rows:
        raise ValueError("Task 4 coverage differs from the configured base passage shard")
    task4 = {row["passage_id"]: row for row in table.to_pylist()}
    for row in rows[:base_part_rows]:
        previous = task4.get(row["passage_id"])
        if previous is None or (
            int(previous["doc_id"]), int(previous["char_start"]), int(previous["char_end"])
        ) != (int(row["doc_id"]), int(row["char_start"]), int(row["char_end"])):
            raise ValueError("Task 4 source occurrence mapping changed")
    return {key: value["normalized_content_hash"] for key, value in task4.items()}


def _prepare_records(
    rows: list[dict],
    task4_hashes: dict[str, str],
    *,
    num_perm: int,
) -> list[dict]:
    records = []
    for number, row in enumerate(rows, 1):
        text = row["passage_text"]
        shingles, strategy, size = shingle_text(text, row["language"])
        hashed = hash_shingles(shingles)
        if not hashed:
            raise ValueError(f"No shingles generated: {row['passage_id']}")
        normalized_hash = task4_hashes.get(row["passage_id"])
        if normalized_hash is None:
            normalized_hash = text_sha256(normalize_medical_text(text).normalized_text)
        records.append({
            **row,
            "normalized_content_hash": normalized_hash,
            "shingle_strategy": strategy,
            "shingle_size": size,
            "shingles": sorted(hashed),
            "signature": list(minhash_signature(hashed, num_perm)),
        })
        if number % 1000 == 0 or number == len(rows):
            print(f"Prepared passage MinHash {number}/{len(rows)}", flush=True)
    return records


def _write_prepared_cache(path: Path, records: list[dict]) -> str:
    _atomic_parquet(path, pa.Table.from_pylist(records, schema=PREPARED_SCHEMA))
    return sha256_file(path)


def _load_prepared_cache(path: Path, expected_sha256: str) -> list[dict]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Task 5 prepared checkpoint fingerprint changed")
    records = pq.read_table(path).to_pylist()
    for row in records:
        row["shingles"] = frozenset(row["shingles"])
        row["signature"] = tuple(row["signature"])
    return records


def _candidate_sources(records: list[dict], config: dict, task3_pairs: Path) -> tuple[dict, dict]:
    normalized_records = [
        {**row, "shingles": frozenset(row["shingles"]), "signature": tuple(row["signature"])}
        for row in records
    ]
    lsh, overflow = lsh_candidates(
        normalized_records,
        bands=int(config["bands"]),
        rows_per_band=int(config["rows_per_band"]),
        maximum_bucket_size=int(config["maximum_bucket_size"]),
    )
    candidates: dict[tuple[int, int], set[str]] = defaultdict(set)
    for pair in lsh:
        candidates[pair].add("minhash_lsh")

    by_doc: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_doc[int(row["doc_id"])].append(index)
    for indices in by_doc.values():
        ordered = sorted(indices, key=lambda index: int(records[index]["passage_index"]))
        for left, right in zip(ordered, ordered[1:]):
            if abs(int(records[left]["passage_index"]) - int(records[right]["passage_index"])) == 1:
                candidates[min(left, right), max(left, right)].add("adjacent_source_offsets")

    if sha256_file(task3_pairs) != EXPECTED_TASK3_OUTPUT_SHA256:
        raise ValueError("Task 3 near-document fingerprint changed")
    for pair in pq.read_table(task3_pairs, columns=["left_doc_id", "right_doc_id"]).to_pylist():
        for left in by_doc.get(int(pair["left_doc_id"]), []):
            for right in by_doc.get(int(pair["right_doc_id"]), []):
                candidates[min(left, right), max(left, right)].add("task3_near_document")
    if len(candidates) > int(config["maximum_candidates"]):
        raise ValueError("Task 5 candidate limit exceeded")
    return candidates, {
        "lsh_candidates": len(lsh),
        "lsh_overflow_events": overflow,
        "combined_candidates": len(candidates),
    }


def _canonical_pair(left: dict, right: dict) -> tuple[dict, dict]:
    left_key = (int(left["doc_id"]), int(left["passage_index"]), left["passage_id"])
    right_key = (int(right["doc_id"]), int(right["passage_index"]), right["passage_id"])
    return (left, right) if left_key <= right_key else (right, left)


def _load_labels(path: Path | None) -> tuple[dict[str, dict], str | None]:
    if path is None:
        return {}, None
    payload = _load_json(path)
    if payload.get("format") != "vibiomir-phase3-task5-labels-v1":
        raise ValueError("Unsupported Task 5 label format")
    labels = {}
    for row in payload.get("labels", []):
        if row.get("label") not in {"true_overlap", "false_positive"}:
            raise ValueError("Invalid Task 5 overlap label")
        if row["pair_id"] in labels:
            raise ValueError("Duplicate Task 5 label pair_id")
        labels[row["pair_id"]] = row
    return labels, sha256_file(path)


def _build_pairs(
    records: list[dict],
    config: dict,
    task3_pairs: Path,
    labels: dict[str, dict],
) -> tuple[list[dict], dict, dict]:
    candidates, candidate_stats = _candidate_sources(records, config, task3_pairs)
    output = []
    stats: Counter[str] = Counter()
    guard_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for (left_index, right_index), sources in sorted(candidates.items()):
        left, right = _canonical_pair(records[left_index], records[right_index])
        left_shingles = frozenset(left["shingles"])
        right_shingles = frozenset(right["shingles"])
        if left["normalized_content_hash"] == right["normalized_content_hash"]:
            stats["exact_content_identity_excluded"] += 1
            continue
        shared, jaccard, left_containment, right_containment = similarity_scores(
            left_shingles, right_shingles
        )
        left_for_classification = {**left, "shingle_count": len(left_shingles)}
        right_for_classification = {**right, "shingle_count": len(right_shingles)}
        overlap_type, reasons, source_overlap_chars = classify_overlap(
            left_for_classification,
            right_for_classification,
            shared_shingles=shared,
            jaccard=jaccard,
            left_containment=left_containment,
            right_containment=right_containment,
            minimum_shared_shingles=int(config["minimum_shared_shingles"]),
            jaccard_threshold=float(config["jaccard_threshold"]),
            containment_threshold=float(config["containment_threshold"]),
            containment_jaccard_floor=float(config["containment_jaccard_floor"]),
        )
        if overlap_type is None:
            stats[reasons[0]] += 1
            continue
        boilerplate_only, boilerplate = is_boilerplate_only(
            left["passage_text"],
            right["passage_text"],
            shared_shingles=shared,
            maximum_containment=max(left_containment, right_containment),
        )
        if boilerplate_only and overlap_type != INTENTIONAL_CHUNK_OVERLAP:
            stats["boilerplate_only_excluded"] += 1
            continue
        guards = medical_guard_reasons(left["passage_text"], right["passage_text"])
        for guard in guards:
            guard_counts[guard] += 1
        pair_identifier = overlap_pair_id(left["passage_id"], right["passage_id"])
        label = labels.get(pair_identifier)
        if label is not None:
            review_status = f"labeled_{label['label']}"
        elif guards:
            review_status = "medical_guard_review_required"
        else:
            review_status = "unlabeled_review_required"
        output.append({
            "pair_id": pair_identifier,
            "left_passage_id": left["passage_id"],
            "right_passage_id": right["passage_id"],
            "left_doc_id": int(left["doc_id"]),
            "right_doc_id": int(right["doc_id"]),
            "left_official_url": left["official_url"],
            "right_official_url": right["official_url"],
            "left_source_offsets": {
                "char_start": int(left["char_start"]), "char_end": int(left["char_end"])
            },
            "right_source_offsets": {
                "char_start": int(right["char_start"]), "char_end": int(right["char_end"])
            },
            "left_language": left["language"],
            "right_language": right["language"],
            "candidate_sources": sorted(sources),
            "shingle_strategy": left["shingle_strategy"],
            "shingle_size": int(left["shingle_size"]),
            "left_shingles": len(left_shingles),
            "right_shingles": len(right_shingles),
            "shared_shingles": shared,
            "minhash_similarity": signature_similarity(
                tuple(left["signature"]), tuple(right["signature"])
            ),
            "jaccard_similarity": jaccard,
            "left_containment": left_containment,
            "right_containment": right_containment,
            "source_overlap_chars": source_overlap_chars,
            "overlap_type": overlap_type,
            "classification_reasons": reasons,
            "guard_reasons": guards,
            "boilerplate_signals": boilerplate,
            "review_status": review_status,
            "auto_merge_allowed": False,
            "policy_version": OVERLAP_POLICY_VERSION,
        })
        type_counts[overlap_type] += 1
    output.sort(key=lambda row: row["pair_id"])
    pair_ids = [row["pair_id"] for row in output]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate Task 5 pair IDs")

    output_by_id = {row["pair_id"]: row for row in output}
    missing_labels = sorted(set(labels) - set(output_by_id))
    if missing_labels:
        raise ValueError(f"Labels reference non-output pairs: {missing_labels[:3]}")
    labeled = len(labels)
    false_positives = sum(row["label"] == "false_positive" for row in labels.values())
    wrong_types = sum(
        bool(row.get("expected_overlap_type"))
        and output_by_id[pair_id]["overlap_type"] != row["expected_overlap_type"]
        for pair_id, row in labels.items()
    )
    evaluation = {
        "labeled_pairs": labeled,
        "true_overlap_labels": labeled - false_positives,
        "false_positive_labels": false_positives,
        "false_positive_rate": false_positives / labeled if labeled else None,
        "labeled_precision": (labeled - false_positives) / labeled if labeled else None,
        "overlap_type_disagreements": wrong_types,
        "recall": None,
        "recall_reason": "The review set labels emitted candidates, not all possible passage pairs.",
    }
    return output, {
        **candidate_stats,
        **dict(stats),
        "output_pairs": len(output),
        "overlap_type_counts": dict(sorted(type_counts.items())),
        "medical_guard_counts": dict(sorted(guard_counts.items())),
    }, evaluation


def _rows_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    phase2_release_path = args.phase2_release.resolve()
    phase2_root = phase2_release_path.parent.parent
    passages_root = phase2_root / "passages"
    task4_root = args.task4.resolve()
    output = _safe_output(args.output, [phase2_root, task4_root])
    destination = output / "passage_dedup/passage_overlap_pairs.parquet"
    validation_path = output / "validation/overlap_validation_report.json"
    report_path = output / "reports/PHASE3_TASK5_REPORT.md"
    snapshot_path = output / "manifests/input_snapshot.json"
    config_snapshot_path = output / "manifests/overlap_config.json"
    checkpoint_path = output / "checkpoints/task5_passage_overlap.json"
    prepared_path = output / "checkpoints/task5_prepared_records.parquet"

    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if destination.exists() or validation_path.exists():
        raise FileExistsError("Task 5 output is already complete; use a new namespace")
    output.mkdir(parents=True, exist_ok=True)

    config = _load_json(args.config.resolve())
    config_sha256 = sha256_file(args.config.resolve())
    if int(config["bands"]) * int(config["rows_per_band"]) != 64:
        raise ValueError("Task 5 MinHash configuration must use 64 permutations")
    phase2_release_sha256 = sha256_file(phase2_release_path)
    if phase2_release_sha256 != EXPECTED_PHASE2_RELEASE_SHA256:
        raise ValueError("Phase 2 release fingerprint changed")
    phase2_release = _load_json(phase2_release_path)
    if phase2_release.get("status") != "complete_validated":
        raise ValueError("Phase 2 release is not validated")
    release_parts = {row["part"]: row for row in phase2_release["passage_parts"]}

    task4_output = task4_root / "passage_dedup/passage_content_groups.parquet"
    task4_validation = _load_json(
        task4_root / "validation/task4_passage_content_validation.json"
    )
    if task4_validation.get("status") != "pass":
        raise ValueError("Task 4 validation is not PASS")
    task4_hashes = None
    sampling_metadata = None
    checkpoint = _load_json(checkpoint_path) if args.resume and checkpoint_path.exists() else None
    if checkpoint and checkpoint.get("status") == "prepared":
        if (
            checkpoint.get("phase2_release_sha256") != phase2_release_sha256
            or checkpoint.get("config_sha256") != config_sha256
        ):
            raise ValueError("Task 5 resume fingerprint mismatch")
        records = _load_prepared_cache(prepared_path, checkpoint["prepared_sha256"])
        sampling_metadata = checkpoint["sampling_metadata"]
        task4_output_sha256 = checkpoint["task4_output_sha256"]
        print(f"Resumed {len(records)} prepared passage records", flush=True)
    else:
        rows, sampling_metadata = _select_pilot_rows(passages_root, release_parts, config)
        base_rows = sampling_metadata["selected_by_stratum"]["base"]
        task4_hashes = _verify_task4_base(rows, task4_output, base_rows)
        task4_output_sha256 = sha256_file(task4_output)
        records = _prepare_records(
            rows,
            task4_hashes,
            num_perm=int(config["bands"]) * int(config["rows_per_band"]),
        )
        prepared_sha256 = _write_prepared_cache(prepared_path, records)
        checkpoint = {
            "format": "vibiomir-phase3-task5-checkpoint-v1",
            "status": "prepared",
            "prepared_records": len(records),
            "prepared_sha256": prepared_sha256,
            "phase2_release_sha256": phase2_release_sha256,
            "task4_output_sha256": task4_output_sha256,
            "config_sha256": config_sha256,
            "sampling_metadata": sampling_metadata,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(checkpoint_path, checkpoint, replace=checkpoint_path.exists())
    if task4_output_sha256 != EXPECTED_TASK4_OUTPUT_SHA256:
        raise ValueError("Task 4 fingerprint changed")

    task3_pairs = args.task3_pairs.resolve()
    labels, labels_sha256 = _load_labels(args.labels.resolve() if args.labels else None)
    pairs, statistics, evaluation = _build_pairs(records, config, task3_pairs, labels)
    first_digest = _rows_digest(pairs)
    rerun_pairs, rerun_statistics, rerun_evaluation = _build_pairs(
        records, config, task3_pairs, labels
    )
    if (
        _rows_digest(rerun_pairs) != first_digest
        or rerun_statistics != statistics
        or rerun_evaluation != evaluation
    ):
        raise ValueError("Task 5 deterministic rerun mismatch")

    _atomic_parquet(
        destination,
        pa.Table.from_pylist(pairs, schema=PASSAGE_OVERLAP_SCHEMA),
    )
    output_meta = pq.read_metadata(destination)
    output_sha256 = sha256_file(destination)
    languages = Counter((row["language"] or "unknown").casefold() for row in records)
    completed = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "format": "vibiomir-phase3-task5-input-snapshot-v1",
        "scope": "stratified_passage_overlap_pilot",
        "passages": len(records),
        "documents": len({int(row["doc_id"]) for row in records}),
        "languages": dict(sorted(languages.items())),
        "sampling": sampling_metadata,
        "phase2_release_manifest_sha256": phase2_release_sha256,
        "task4_output_sha256": task4_output_sha256,
        "task3_pairs_sha256": sha256_file(task3_pairs),
        "config_sha256": config_sha256,
        "labels_sha256": labels_sha256,
        "input_verification_mode": "incremental_verified_manifests_and_selected_shards",
    }
    validation = {
        "format": FORMAT,
        "status": "pass",
        "task": 5,
        "scope": "stratified_passage_overlap_pilot",
        "completed_at_utc": completed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output": {
            "path": str(destination.relative_to(ROOT)),
            "rows": output_meta.num_rows,
            "bytes": destination.stat().st_size,
            "sha256": output_sha256,
            "schema_version": OVERLAP_SCHEMA_VERSION,
            "deterministic_rows_sha256": first_digest,
        },
        "statistics": statistics,
        "evaluation": evaluation,
        "checks": {
            "input_fingerprints_stable": True,
            "task4_source_occurrence_mapping_preserved": True,
            "language_aware_vi_en_zh_shingling": all(language in languages for language in ("vi", "en", "zh")),
            "all_candidates_verified_with_jaccard_and_containment": True,
            "exact_content_identity_excluded": True,
            "boilerplate_only_not_equivalence_evidence": True,
            "medical_guards_recorded": True,
            "passage_source_metadata_preserved": True,
            "pair_ids_unique": True,
            "deterministic_rerun": True,
            "bounded_lsh_buckets": True,
            "candidate_limit_enforced": True,
            "checkpoint_resume_supported": True,
            "auto_merge_allowed": False,
            "phase0_mutated": False,
            "phase1_mutated": False,
            "phase2_mutated": False,
            "task4_mutated": False,
            "index_mutated": False,
            "task6_started": False,
        },
    }
    _atomic_json(snapshot_path, snapshot)
    _atomic_json(config_snapshot_path, config)
    _atomic_json(validation_path, validation)
    completed_checkpoint = {
        **checkpoint,
        "status": "complete",
        "output_rows": output_meta.num_rows,
        "output_sha256": output_sha256,
        "completed_at_utc": completed,
    }
    _atomic_json(checkpoint_path, completed_checkpoint, replace=True)
    report = f"""# Phase 3 Task 5 — Partial Passage Overlap Pilot

- Status: **PASS**
- Scope: stratified pilot; full Phase 3 was not run.
- Passages: `{len(records):,}` across `{len(set(int(row['doc_id']) for row in records)):,}` documents.
- Languages: `{json.dumps(dict(sorted(languages.items())), ensure_ascii=False)}`
- Verified candidates: `{statistics['combined_candidates']:,}`
- Output overlap pairs: `{len(pairs):,}`
- Overlap types: `{json.dumps(statistics['overlap_type_counts'], ensure_ascii=False, sort_keys=True)}`
- Boilerplate-only candidates excluded: `{statistics.get('boilerplate_only_excluded', 0):,}`
- Medical guard signals on review-only pairs: `{sum(statistics['medical_guard_counts'].values()):,}`
- Labeled pairs: `{evaluation['labeled_pairs']:,}`
- Labeled false-positive rate: `{evaluation['false_positive_rate']}`
- Recall: not reported; labels cover emitted candidates only.
- Maximum RSS: `{validation['maximum_rss_kib']:,}` KiB
- Elapsed: `{validation['elapsed_seconds']}` seconds
- Output SHA-256: `{output_sha256}`
- Auto merge allowed: `false`
- Task 6 started: `false`

Every pair retains both passage IDs, document IDs, URLs, and source offsets. Exact
content identities are excluded because Task 4 already owns that relationship.
Medical guard mismatches remain review-only and never trigger grouping or merging.
"""
    _atomic_text(report_path, report)
    return validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-release", type=Path,
        default=ROOT / "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json",
    )
    parser.add_argument(
        "--task4", type=Path,
        default=ROOT / "outputs/vibiomir_phase3_20261015_task4_dry_run_v2",
    )
    parser.add_argument(
        "--task3-pairs", type=Path,
        default=ROOT / "outputs/vibiomir_phase3_20261015_task3_dry_run_v1/document_dedup/near_duplicate_pairs.parquet",
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/vibiomir/phase3_task5_pilot_v1.json",
    )
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-disk-backed", action="store_true")
    parser.add_argument("--candidate-partitions", type=int, default=512)
    parser.add_argument("--maximum-bucket-size", type=int, default=200)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--rss-limit-gib", type=float, default=16.0)
    parser.add_argument("--temporary-disk-budget-gib", type=float, default=512.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.full_disk_backed:
        from scripts.workflows.vibiomir.phase3_scalable_candidates import (
            run_full_passage_candidates,
        )
        result = run_full_passage_candidates(
            root=ROOT,
            phase2_release=args.phase2_release.resolve(),
            task4=args.task4.resolve(),
            task3_pairs=args.task3_pairs.resolve(),
            output=args.output,
            resume=args.resume,
            partitions=args.candidate_partitions,
            maximum_bucket_size=args.maximum_bucket_size,
            batch_rows=args.batch_rows,
            rss_limit_gib=args.rss_limit_gib,
            temporary_disk_budget_gib=args.temporary_disk_budget_gib,
        )
    else:
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
