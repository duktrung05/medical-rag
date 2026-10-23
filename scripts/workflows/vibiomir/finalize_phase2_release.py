"""Validate and publish a completed ViBioMIR Phase 2 full passage build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflows.vibiomir.build_phase2_passages import (  # noqa: E402
    EXPECTED_HELD,
    EXPECTED_PASS,
    _load_json,
    verify_release,
)
from src.medrag.data.phase1.selection import sha256_file  # noqa: E402
from src.medrag.data.phase2.schema import (  # noqa: E402
    CHUNKER_VERSION,
    MAPPING_SCHEMA,
    PASSAGE_SCHEMA,
    PASSAGE_SCHEMA_VERSION,
    STRATEGIES,
)
from src.medrag.data.phase2.tokenizer import (  # noqa: E402
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    default_tokenizer_path,
)


PASSAGE_ID = re.compile(r"^p2_[0-9a-f]{64}$")


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


def _publish_passage(source: Path, destination: Path) -> None:
    """Publish a write-once hard link without duplicating multi-gigabyte shards."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise FileExistsError(destination)
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _strategy_c() -> dict:
    return next(value.__dict__ for value in STRATEGIES if value.strategy_id == "C")


def _check_frozen_config(path: Path, candidate: Path) -> dict:
    config = _load_json(path)
    if config.get("status") != "frozen_for_full_build":
        raise ValueError("Phase 2 config is not frozen for a full build")
    if config.get("selected_strategy") != _strategy_c():
        raise ValueError("Frozen config does not select the runtime Strategy C")
    if config.get("chunker_version") != CHUNKER_VERSION:
        raise ValueError("Frozen chunker version differs from runtime")
    if config.get("passage_schema_version") != PASSAGE_SCHEMA_VERSION:
        raise ValueError("Frozen passage schema differs from runtime")
    if config.get("tokenizer_name") != TOKENIZER_NAME:
        raise ValueError("Frozen tokenizer name differs from runtime")
    phase1 = config.get("phase1", {})
    if phase1.get("candidate") != str(candidate.resolve().relative_to(ROOT)):
        raise ValueError("Frozen Phase 1 candidate differs from requested candidate")
    if int(phase1.get("documents_to_chunk", -1)) != EXPECTED_PASS:
        raise ValueError("Frozen PASS count differs from release contract")
    if int(phase1.get("held_documents", -1)) != EXPECTED_HELD:
        raise ValueError("Frozen held count differs from release contract")
    if phase1.get("mutation_allowed") is not False:
        raise ValueError("Frozen config does not protect Phase 1 from mutation")
    return config


def _write_combined_mapping(build: Path, stems: list[str], destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.",
        suffix=".partial", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    writer = pq.ParquetWriter(temporary, MAPPING_SCHEMA, compression="zstd")
    try:
        for stem in stems:
            table = pq.read_table(build / f"metadata/strategy_C/{stem}.parquet")
            if table.schema != MAPPING_SCHEMA:
                table = table.cast(MAPPING_SCHEMA)
            writer.write_table(table)
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def finalize(candidate: Path, build: Path, frozen_config: Path, tokenizer_json: Path) -> dict:
    candidate = candidate.resolve()
    build = build.resolve()
    frozen_config = frozen_config.resolve()
    tokenizer_json = tokenizer_json.resolve()
    config = _check_frozen_config(frozen_config, candidate)
    context = verify_release(candidate, tokenizer_json)
    snapshot_path = build / "manifests/input_snapshot.json"
    build_manifest_path = build / "manifests/chunking_manifest.json"
    snapshot = _load_json(snapshot_path)
    build_manifest = _load_json(build_manifest_path)
    expected_snapshot = {
        "scope": "full",
        "candidate_manifest_sha256": context["manifest_sha256"],
        "phase1_release_manifest_sha256": context["release_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
        "pass_documents": EXPECTED_PASS,
        "held_documents": EXPECTED_HELD,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "chunker_version": CHUNKER_VERSION,
        "frozen_config_sha256": sha256_file(frozen_config),
    }
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"Input snapshot mismatch for {key}")
    if snapshot.get("strategies") != [_strategy_c()]:
        raise ValueError("Full build snapshot is not Strategy C only")
    if build_manifest.get("status") != "complete" or build_manifest.get("scope") != "full":
        raise ValueError("Full chunking build is not complete")
    if build_manifest.get("input_snapshot_sha256") != sha256_file(snapshot_path):
        raise ValueError("Build manifest input snapshot checksum mismatch")

    stems = sorted(context["manifest"]["part_checkpoints_sha256"])
    if int(build_manifest.get("parts", -1)) != len(stems):
        raise ValueError("Build manifest shard count mismatch")
    totals = build_manifest.get("strategy_totals", {}).get("C", {})
    if int(totals.get("documents", -1)) != EXPECTED_PASS:
        raise ValueError("Build manifest document count mismatch")

    mapping_path = build / "metadata/document_passage_mapping.parquet"
    _write_combined_mapping(build, stems, mapping_path)

    validation_counts = Counter()
    passage_parts: list[dict] = []
    seen_doc_ids: set[int] = set()
    token_min: int | None = None
    token_max = 0
    token_sum = 0
    completed_at = datetime.now(timezone.utc).isoformat()
    bucket_root = build / "validation"
    bucket_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".passage-id-validation-", dir=bucket_root) as temp_name:
        temp = Path(temp_name)
        bucket_streams = [(temp / f"{index:02x}.ids").open("w", encoding="ascii")
                          for index in range(256)]
        try:
            for number, stem in enumerate(stems, 1):
                source_passage = build / f"passages/strategy_C/{stem}.parquet"
                source_mapping = build / f"metadata/strategy_C/{stem}.parquet"
                source_failure = build / f"validation/strategy_C/{stem}.parquet"
                checkpoint_path = build / f"checkpoints/strategy_C/{stem}.json"
                for path in (source_passage, source_mapping, source_failure, checkpoint_path):
                    if not path.is_file():
                        raise ValueError(f"Missing full build artifact: {path}")
                checkpoint = _load_json(checkpoint_path)
                if checkpoint.get("status") != "complete":
                    raise ValueError(f"Incomplete checkpoint: {stem}")
                checks = {
                    source_passage: checkpoint.get("passage_sha256"),
                    source_mapping: checkpoint.get("mapping_sha256"),
                    source_failure: checkpoint.get("failure_sha256"),
                }
                for path, expected in checks.items():
                    if sha256_file(path) != expected:
                        raise ValueError(f"Checkpoint checksum mismatch: {path}")
                failures = pq.read_table(source_failure)
                if failures.num_rows or int(checkpoint.get("failures", -1)) != 0:
                    raise ValueError(f"Nonzero validation failures in {stem}")

                mapping_rows = pq.read_table(source_mapping, schema=MAPPING_SCHEMA).to_pylist()
                part_doc_ids = set()
                for row in mapping_rows:
                    doc_id = int(row["doc_id"])
                    if doc_id in seen_doc_ids or doc_id in part_doc_ids:
                        raise ValueError(f"Duplicate mapped doc_id: {doc_id}")
                    part_doc_ids.add(doc_id)
                    if row["strategy_id"] != "C":
                        raise ValueError(f"Non-C mapping row in {stem}")
                    if int(row["passage_count"]) <= 0:
                        raise ValueError(f"Document without passages: {doc_id}")
                    if int(row["covered_non_whitespace_chars"]) != int(row["source_non_whitespace_chars"]):
                        raise ValueError(f"Coverage mismatch for doc_id={doc_id}")
                    validation_counts["coverage_checked_documents"] += 1
                seen_doc_ids.update(part_doc_ids)

                parquet = pq.ParquetFile(source_passage)
                if parquet.schema_arrow != PASSAGE_SCHEMA:
                    raise ValueError(f"Passage schema mismatch: {stem}")
                part_rows = 0
                first_by_doc: dict[int, str] = {}
                last_by_doc: dict[int, str] = {}
                count_by_doc = Counter()
                for batch in parquet.iter_batches(batch_size=4_096):
                    for row in batch.to_pylist():
                        passage_id = row["passage_id"]
                        doc_id = int(row["doc_id"])
                        if doc_id not in part_doc_ids:
                            raise ValueError(f"Passage outside mapped shard: {passage_id}")
                        if not PASSAGE_ID.fullmatch(passage_id):
                            raise ValueError(f"Malformed passage_id: {passage_id}")
                        bucket_streams[int(passage_id[3:5], 16)].write(passage_id + "\n")
                        if (
                            row["strategy_id"] != "C"
                            or row["chunker_version"] != CHUNKER_VERSION
                            or row["tokenizer_name"] != TOKENIZER_NAME
                            or row["tokenizer_revision"] != TOKENIZER_REVISION
                            or row["phase1_release_id"] != context["release_sha256"]
                            or row["corpus_revision"] != context["manifest"]["corpus_revision"]
                            or row["processing_run_id"] != build_manifest["processing_run_id"]
                            or row["text_representation"] != "normalized_text"
                        ):
                            raise ValueError(f"Passage provenance mismatch: {passage_id}")
                        token_count = int(row["token_count"])
                        if token_count <= 0 or token_count > _strategy_c()["max_tokens"]:
                            raise ValueError(f"Invalid token count: {passage_id}")
                        if int(row["char_start"]) < 0 or int(row["char_end"]) <= int(row["char_start"]):
                            raise ValueError(f"Invalid source offsets: {passage_id}")
                        first_by_doc.setdefault(doc_id, passage_id)
                        last_by_doc[doc_id] = passage_id
                        count_by_doc[doc_id] += 1
                        part_rows += 1
                        token_min = token_count if token_min is None else min(token_min, token_count)
                        token_max = max(token_max, token_count)
                        token_sum += token_count
                if part_rows != int(checkpoint.get("passages", -1)):
                    raise ValueError(f"Checkpoint passage count mismatch: {stem}")
                for row in mapping_rows:
                    doc_id = int(row["doc_id"])
                    if (
                        count_by_doc[doc_id] != int(row["passage_count"])
                        or first_by_doc.get(doc_id) != row["first_passage_id"]
                        or last_by_doc.get(doc_id) != row["last_passage_id"]
                    ):
                        raise ValueError(f"Passage mapping mismatch for doc_id={doc_id}")

                published = build / f"passages/{stem}.parquet"
                _publish_passage(source_passage, published)
                passage_parts.append({
                    "part": stem,
                    "rows": part_rows,
                    "sha256": checkpoint["passage_sha256"],
                    "bytes": published.stat().st_size,
                })
                validation_counts["passages"] += part_rows
                validation_counts["checkpoint_validated_shards"] += 1
                if number % 25 == 0 or number == len(stems):
                    print(f"Final validation: {number}/{len(stems)} shards", flush=True)
        finally:
            for stream in bucket_streams:
                stream.close()

        unique_passage_ids = 0
        duplicate_passage_ids = 0
        for path in sorted(temp.glob("*.ids")):
            values: set[str] = set()
            with path.open(encoding="ascii") as stream:
                for line in stream:
                    ident = line.rstrip("\n")
                    if ident in values:
                        duplicate_passage_ids += 1
                    else:
                        values.add(ident)
            unique_passage_ids += len(values)

    if seen_doc_ids != context["pass_ids"]:
        raise ValueError("Final mapping doc_ids differ from the Phase 1 PASS allowlist")
    if validation_counts["passages"] != int(totals.get("passages", -1)):
        raise ValueError("Final passage count differs from build manifest")
    if duplicate_passage_ids or unique_passage_ids != validation_counts["passages"]:
        raise ValueError("Global passage_id uniqueness validation failed")
    mapping_metadata = pq.ParquetFile(mapping_path).metadata
    if mapping_metadata.num_rows != EXPECTED_PASS:
        raise ValueError("Combined mapping row count mismatch")

    validation_report = {
        "format": "vibiomir-phase2-full-validation-v1",
        "status": "pass",
        "scope": "full",
        "completed_at_utc": completed_at,
        "strategy": _strategy_c(),
        "input": expected_snapshot,
        "checks": {
            "phase1_pass_allowlist_exact": True,
            "phase1_held_excluded": True,
            "input_shard_checksums": "pass",
            "output_shard_checksums": "pass",
            "passage_id_format": "pass",
            "passage_id_global_uniqueness": "pass",
            "source_offsets_exact_streaming_validation": "pass",
            "token_counts_exact_streaming_validation": "pass",
            "max_tokens": "pass",
            "document_coverage": "pass",
            "document_passage_mapping": "pass",
            "passage_provenance": "pass",
        },
        "counts": {
            "phase1_pass_documents": EXPECTED_PASS,
            "phase1_held_documents": EXPECTED_HELD,
            "mapped_documents": len(seen_doc_ids),
            "coverage_checked_documents": validation_counts["coverage_checked_documents"],
            "passages": validation_counts["passages"],
            "unique_passage_ids": unique_passage_ids,
            "duplicate_passage_ids": duplicate_passage_ids,
            "validation_failures": 0,
            "validated_shards": validation_counts["checkpoint_validated_shards"],
        },
        "token_counts": {
            "minimum": token_min,
            "maximum": token_max,
            "mean": token_sum / validation_counts["passages"],
        },
    }
    validation_path = build / "validation/full_validation_report.json"
    if validation_path.exists():
        raise FileExistsError(validation_path)
    _atomic_json(validation_path, validation_report)

    release_manifest = {
        "format": "vibiomir-phase2-release-v1",
        "status": "complete_validated",
        "scope": "full",
        "completed_at_utc": completed_at,
        "processing_run_id": build_manifest["processing_run_id"],
        "strategy": _strategy_c(),
        "chunker_version": CHUNKER_VERSION,
        "passage_schema_version": PASSAGE_SCHEMA_VERSION,
        "documents": EXPECTED_PASS,
        "passages": validation_counts["passages"],
        "parts": len(passage_parts),
        "frozen_config": str(frozen_config.relative_to(ROOT)),
        "frozen_config_sha256": sha256_file(frozen_config),
        "input_snapshot": str(snapshot_path.relative_to(build)),
        "input_snapshot_sha256": sha256_file(snapshot_path),
        "phase1_release_manifest_sha256": context["release_sha256"],
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "passage_parts": passage_parts,
        "document_passage_mapping": {
            "path": str(mapping_path.relative_to(build)),
            "rows": mapping_metadata.num_rows,
            "sha256": sha256_file(mapping_path),
        },
        "validation_report": {
            "path": str(validation_path.relative_to(build)),
            "sha256": sha256_file(validation_path),
            "status": "pass",
        },
        "phase0_mutated": False,
        "phase1_mutated": False,
        "index_mutated": False,
        "phase3_started": False,
    }
    release_path = build / "manifests/phase2_release_manifest.json"
    if release_path.exists():
        raise FileExistsError(release_path)
    _atomic_json(release_path, release_manifest)

    report = f"""# ViBioMIR Phase 2 Full Release Report

- Status: **PASS — complete and validated**
- Strategy: **C** (`target=384`, `max=512`, `min=48`, `overlap=64` tokens)
- Phase 1 PASS documents: **{EXPECTED_PASS:,}**
- Held documents excluded: **{EXPECTED_HELD:,}**
- Passage rows: **{validation_counts['passages']:,}**
- Passage shards: **{len(passage_parts):,}**
- Duplicate passage IDs: **0**
- Coverage failures: **0**
- Source offset failures: **0**
- Token count failures: **0**

The release was produced from the frozen Phase 2 configuration and the immutable
Phase 1 PASS allowlist. Every document was validated during streaming chunking;
the finalization pass rechecked all shard checksums, global passage ID uniqueness,
mapping cardinality, coverage totals, token bounds, and provenance fields.

## Provenance

- Phase 1 release manifest SHA-256: `{context['release_sha256']}`
- Phase 2 input snapshot SHA-256: `{sha256_file(snapshot_path)}`
- Frozen config SHA-256: `{sha256_file(frozen_config)}`
- Tokenizer: `{TOKENIZER_NAME}` revision `{TOKENIZER_REVISION}`
- Tokenizer JSON SHA-256: `{sha256_file(tokenizer_json)}`
- Chunker: `{CHUNKER_VERSION}`

## Immutability and stop boundary

Phase 0, Phase 1, and the current index were not modified. Phase 3 was not
started. The release boundary is `manifests/phase2_release_manifest.json`.
"""
    report_path = build / "reports/PHASE2_FULL_RELEASE_REPORT.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    _atomic_text(report_path, report)
    return release_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, default=default_tokenizer_path(ROOT))
    args = parser.parse_args(argv)
    try:
        result = finalize(args.candidate, args.build, args.frozen_config, args.tokenizer_json)
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Phase 2 finalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
