"""Audit Phase 1 and build a deterministic Phase 2 hierarchical passage pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import re
import resource
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase2.chunker import chunk_document
from src.medrag.data.phase2.evaluation import passage_metrics
from src.medrag.data.phase2.schema import (
    CHUNKER_VERSION,
    FAILURE_SCHEMA,
    MAPPING_SCHEMA,
    PASSAGE_SCHEMA,
    PASSAGE_SCHEMA_VERSION,
    PILOT_SCHEMA,
    STRATEGIES,
)
from src.medrag.data.phase2.structure import has_reliable_heading
from src.medrag.data.phase2.tokenizer import (
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    BgeM3Tokenizer,
    default_tokenizer_path,
)
from src.medrag.data.phase2.validation import covered_non_whitespace, validate_document_passages


EXPECTED_PASS = 1_073_531
EXPECTED_HELD = 2_068
PILOT_SELECTION_VERSION = "phase2-stratified-hash-v1"
_MEASUREMENT = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|kg|mL|mmol|mmHg|%|IU|m²)\b|"
    r"HbA1c|eGFR|[<>≤≥]\s*\d)", re.I
)
_REVIEW_SIGNAL = re.compile(
    r"(?:\b(?:not|không|chống chỉ định|contraindicat|dose|dosage|liều)\b|"
    r"HbA1c|eGFR|mg(?:/day)?|mL/min|[<>≤≥]\s*\d|[\u4e00-\u9fff])", re.I
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


def _atomic_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_score(doc_id: int) -> int:
    digest = hashlib.sha256(f"{PILOT_SELECTION_VERSION}\0{doc_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _length_bin(length: int) -> str:
    if length < 1_000:
        return "short"
    if length < 5_000:
        return "medium"
    if length < 20_000:
        return "long"
    return "very_long"


def _language_group(language: str, text: str) -> tuple[str, bool]:
    latin = sum(char.isascii() and char.isalpha() for char in text)
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    mixed = latin >= 20 and han >= 20
    value = (language or "unknown").lower()
    if value.startswith("vi"):
        return "vi", mixed
    if value.startswith("en"):
        return "en", mixed
    if value.startswith("zh") or han >= 20:
        return "zh", mixed
    return "other", mixed


def _paragraph_profile(text: str) -> str:
    sizes = [len(value.strip()) for value in re.split(r"\n\s*\n", text) if value.strip()]
    if not sizes:
        return "empty"
    maximum = max(sizes)
    if maximum >= 5_000:
        return "very_long_paragraph"
    if maximum >= 1_500:
        return "long_paragraph"
    if len(sizes) >= 8:
        return "many_paragraphs"
    return "regular"


def _part_map(entries: list[dict]) -> dict[str, dict]:
    output = {entry["part"]: entry for entry in entries}
    if len(output) != len(entries):
        raise ValueError("Duplicate part names in release manifest")
    return output


def ensure_output_is_safe(output: Path, candidate: Path) -> Path:
    output = output.resolve()
    protected = [
        candidate.resolve(),
        (ROOT / "outputs/vibiomir_phase0_20261014").resolve(),
        (ROOT / "outputs/vibiomir_step2_20261013").resolve(),
    ]
    for source in protected:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Phase 2 output overlaps immutable input: {source}")
    return output


def strategies_from_frozen_config(path: Path) -> tuple:
    config = _load_json(path)
    if config.get("status") != "frozen_for_full_build":
        raise ValueError("Frozen config is not approved for a full build")
    if config.get("chunker_version") != CHUNKER_VERSION:
        raise ValueError("Frozen config chunker version differs from runtime")
    selected = config.get("selected_strategy")
    for strategy in STRATEGIES:
        if selected == strategy.__dict__:
            return (strategy,)
    raise ValueError("Frozen strategy differs from runtime strategy definitions")


def verify_release(candidate: Path, tokenizer_path: Path, *, progress=print) -> dict:
    candidate = candidate.resolve()
    manifest_path = candidate / "phase1_manifest.json"
    snapshot_path = candidate / "source_snapshot.json"
    release_path = candidate / "release/phase1_release_manifest.json"
    for path in (manifest_path, snapshot_path, release_path, tokenizer_path):
        if not path.is_file():
            raise ValueError(f"Required input is missing: {path}")
    manifest = _load_json(manifest_path)
    snapshot = _load_json(snapshot_path)
    release = _load_json(release_path)
    if manifest.get("status") != "complete" or manifest.get("build_scope") != "full":
        raise ValueError("Phase 1 candidate is not a completed full build")
    if release.get("status") != "complete" or release.get("format") != "vibiomir-phase1-release-v1":
        raise ValueError("Phase 1 release is not complete")
    if release.get("candidate_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Candidate manifest checksum mismatch")
    if manifest.get("source_snapshot_sha256") != sha256_file(snapshot_path):
        raise ValueError("Source snapshot checksum mismatch")
    if int(release.get("pass_documents", -1)) != EXPECTED_PASS:
        raise ValueError("Unexpected PASS document count")
    if int(release.get("held_documents", -1)) != EXPECTED_HELD:
        raise ValueError("Unexpected held document count")
    clean_parts = manifest.get("part_checkpoints_sha256", {})
    pass_parts = _part_map(release.get("pass_allowlist_parts", []))
    held_parts = _part_map(release.get("held_document_parts", []))
    if set(clean_parts) != set(pass_parts) or set(clean_parts) != set(held_parts):
        raise ValueError("Clean, PASS and held part names differ")

    pass_ids: set[int] = set()
    held_ids: set[int] = set()
    for number, stem in enumerate(sorted(clean_parts), 1):
        paths = (
            (candidate / "clean_documents" / f"{stem}.parquet", clean_parts[stem]),
            (candidate / "release/pass_document_ids" / f"{stem}.parquet", pass_parts[stem]["sha256"]),
            (candidate / "release/held_documents" / f"{stem}.parquet", held_parts[stem]["sha256"]),
        )
        for path, expected in paths:
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Input shard checksum mismatch: {path}")
        for ident in pq.read_table(paths[1][0], columns=["doc_id"])["doc_id"].to_pylist():
            ident = int(ident)
            if ident in pass_ids:
                raise ValueError(f"Duplicate PASS doc_id: {ident}")
            pass_ids.add(ident)
        for ident in pq.read_table(paths[2][0], columns=["doc_id"])["doc_id"].to_pylist():
            ident = int(ident)
            if ident in held_ids:
                raise ValueError(f"Duplicate held doc_id: {ident}")
            held_ids.add(ident)
        if number % 100 == 0 or number == len(clean_parts):
            progress(f"Verified release shards {number}/{len(clean_parts)}")
    if len(pass_ids) != EXPECTED_PASS or len(held_ids) != EXPECTED_HELD:
        raise ValueError("Release ID counts do not match manifest")
    if pass_ids & held_ids:
        raise ValueError("PASS and held allowlists overlap")
    if len(pass_ids | held_ids) != int(release["candidate_documents"]):
        raise ValueError("Release allowlists do not cover candidate")
    return {
        "candidate": candidate,
        "manifest": manifest,
        "snapshot": snapshot,
        "release": release,
        "pass_ids": pass_ids,
        "held_ids": held_ids,
        "manifest_sha256": sha256_file(manifest_path),
        "release_sha256": sha256_file(release_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
    }


def _official_mapping(repo_root: Path, expected_sha256: str) -> dict[int, str]:
    path = repo_root / "outputs/vibiomir_phase0_20261014/audit/document_state.parquet"
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("Phase 0 official mapping checksum mismatch")
    table = pq.read_table(path, columns=["doc_id", "official_url"])
    mapping = dict(zip(table["doc_id"].to_pylist(), table["official_url"].to_pylist()))
    if len(mapping) != table.num_rows:
        raise ValueError("Duplicate doc_id in Phase 0 mapping")
    return mapping


def select_pilot(context: dict, pilot_documents: int, *, progress=print) -> tuple[list[dict], dict[int, dict]]:
    if pilot_documents <= 0:
        raise ValueError("pilot_documents must be positive")
    candidate = context["candidate"]
    expected_phase0_hash = context["snapshot"]["fingerprints"]["phase0_state_sha256"]
    official = _official_mapping(ROOT, expected_phase0_hash)
    winners: dict[tuple, dict] = {}
    global_heap: list[tuple[int, int, dict]] = []
    clean_parts = context["manifest"]["part_checkpoints_sha256"]
    mapping_errors = 0
    seen_pass = 0
    columns = [
        "doc_id", "official_url", "source_kind", "title", "normalized_text",
        "detected_language", "source_text_sha256", "normalized_text_sha256",
    ]
    for number, stem in enumerate(sorted(clean_parts), 1):
        pass_path = candidate / "release/pass_document_ids" / f"{stem}.parquet"
        part_pass = set(map(int, pq.read_table(pass_path, columns=["doc_id"])["doc_id"].to_pylist()))
        for batch in pq.ParquetFile(candidate / "clean_documents" / f"{stem}.parquet").iter_batches(
            batch_size=1_000, columns=columns
        ):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id not in part_pass:
                    continue
                seen_pass += 1
                if official.get(doc_id) != row["official_url"]:
                    mapping_errors += 1
                    continue
                text = row["normalized_text"]
                language, mixed = _language_group(row["detected_language"], text)
                domain = (urlsplit(row["official_url"]).hostname or "unknown").lower()
                heading = has_reliable_heading(text, row.get("title"))
                length_bin = _length_bin(len(text))
                paragraph = _paragraph_profile(text)
                measurement = bool(_MEASUREMENT.search(text))
                score = _hash_score(doc_id)
                metadata = {
                    "doc_id": doc_id,
                    "official_url": row["official_url"],
                    "source_kind": row["source_kind"],
                    "domain": domain,
                    "language": language,
                    "length_chars": len(text),
                    "length_bin": length_bin,
                    "has_heading": heading,
                    "paragraph_profile": paragraph,
                    "mixed_language": mixed,
                    "medical_measurement": measurement,
                    "selection_score": score,
                    "part": stem,
                }
                stratum = (
                    language, domain, row["source_kind"], length_bin, heading,
                    paragraph, mixed, measurement,
                )
                previous = winners.get(stratum)
                if previous is None or (score, doc_id) < (previous["selection_score"], previous["doc_id"]):
                    winners[stratum] = metadata
                heapq.heappush(global_heap, (-score, -doc_id, metadata))
                if len(global_heap) > max(5_000, pilot_documents * 5):
                    heapq.heappop(global_heap)
        if number % 100 == 0 or number == len(clean_parts):
            progress(f"Scanned pilot strata {number}/{len(clean_parts)}")
    del official
    if mapping_errors:
        raise ValueError(f"Official ID-URL mapping errors: {mapping_errors}")
    if seen_pass != EXPECTED_PASS:
        raise ValueError(f"Scanned {seen_pass} PASS rows, expected {EXPECTED_PASS}")

    selected: dict[int, dict] = {}
    for metadata in sorted(winners.values(), key=lambda row: (row["selection_score"], row["doc_id"])):
        if len(selected) == pilot_documents:
            break
        selected[metadata["doc_id"]] = metadata
    if len(selected) < pilot_documents:
        candidates = sorted(
            (entry[2] for entry in global_heap),
            key=lambda row: (row["selection_score"], row["doc_id"]),
        )
        for metadata in candidates:
            selected.setdefault(metadata["doc_id"], metadata)
            if len(selected) == pilot_documents:
                break
    if len(selected) != pilot_documents:
        raise ValueError("Unable to select requested pilot size")

    rows_by_id: dict[int, dict] = {}
    selected_parts: dict[str, set[int]] = {}
    for doc_id, metadata in selected.items():
        selected_parts.setdefault(metadata["part"], set()).add(doc_id)
    for stem, ids in selected_parts.items():
        table = pq.read_table(candidate / "clean_documents" / f"{stem}.parquet", columns=columns)
        for row in table.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in ids:
                rows_by_id[doc_id] = row
    if set(rows_by_id) != set(selected):
        raise ValueError("Selected pilot rows could not be reloaded")
    pilot_rows = [{key: value for key, value in row.items() if key != "part"}
                  for row in sorted(selected.values(), key=lambda value: value["doc_id"])]
    return pilot_rows, rows_by_id


def _input_snapshot(context: dict, tokenizer: BgeM3Tokenizer, pilot_rows: list[dict]) -> dict:
    pilot_payload = "".join(f"{row['doc_id']}\n" for row in pilot_rows).encode("ascii")
    return {
        "format": "vibiomir-phase2-input-snapshot-v1",
        "candidate": str(context["candidate"].relative_to(ROOT)),
        "candidate_manifest_sha256": context["manifest_sha256"],
        "phase1_release_manifest_sha256": context["release_sha256"],
        "phase1_release_id": context["release_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
        "corpus_revision": context["manifest"]["corpus_revision"],
        "normalization_version": context["manifest"]["normalization_version"],
        "language_detection_version": context["manifest"]["language_detection_version"],
        "pass_documents": EXPECTED_PASS,
        "held_documents": EXPECTED_HELD,
        "pilot_documents": len(pilot_rows),
        "pilot_selection_version": PILOT_SELECTION_VERSION,
        "pilot_ids_sha256": hashlib.sha256(pilot_payload).hexdigest(),
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": tokenizer.sha256,
        "strategies": [strategy.__dict__ for strategy in STRATEGIES],
        "chunker_version": CHUNKER_VERSION,
    }


def _reload_pilot_documents(context: dict, pilot_path: Path) -> tuple[list[dict], dict[int, dict]]:
    pilot_rows = pq.read_table(pilot_path, schema=PILOT_SCHEMA).to_pylist()
    ids = {int(row["doc_id"]) for row in pilot_rows}
    if len(ids) != len(pilot_rows) or not ids <= context["pass_ids"]:
        raise ValueError("Existing pilot manifest is duplicate or outside PASS allowlist")
    columns = [
        "doc_id", "official_url", "source_kind", "title", "normalized_text",
        "detected_language", "source_text_sha256", "normalized_text_sha256",
    ]
    documents: dict[int, dict] = {}
    candidate = context["candidate"]
    for stem in sorted(context["manifest"]["part_checkpoints_sha256"]):
        table = pq.read_table(candidate / "clean_documents" / f"{stem}.parquet", columns=columns)
        for row in table.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in ids:
                documents[doc_id] = row
    if set(documents) != ids:
        raise ValueError("Existing pilot documents could not be reloaded")
    return pilot_rows, documents


def _checkpoint_valid(path: Path, expected_snapshot_sha256: str, strategy: dict,
                      passage_path: Path) -> bool:
    if not path.is_file() or not passage_path.is_file():
        return False
    checkpoint = _load_json(path)
    return (
        checkpoint.get("status") == "complete"
        and checkpoint.get("input_snapshot_sha256") == expected_snapshot_sha256
        and checkpoint.get("strategy") == strategy
        and checkpoint.get("passage_sha256") == sha256_file(passage_path)
    )


def build_full_corpus(context: dict, tokenizer: BgeM3Tokenizer, output: Path,
                      run_id: str, *, resume: bool = False,
                      strategies=STRATEGIES, frozen_config_sha256: str | None = None,
                      shard_modulus: int | None = None,
                      shard_remainder: int | None = None,
                      progress=print) -> dict:
    """Stream the full PASS corpus by aligned Phase 1 shard.

    This path is prepared for the approved production run but is intentionally
    not called by the Phase 2 pilot workflow.
    """
    output = ensure_output_is_safe(output, context["candidate"])
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists; use --resume only for the same run: {output}")
    if not output.exists():
        output.mkdir(parents=True)
    snapshot = {
        "format": "vibiomir-phase2-input-snapshot-v1",
        "scope": "full",
        "candidate": str(context["candidate"].relative_to(ROOT)),
        "candidate_manifest_sha256": context["manifest_sha256"],
        "phase1_release_manifest_sha256": context["release_sha256"],
        "phase1_release_id": context["release_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
        "corpus_revision": context["manifest"]["corpus_revision"],
        "normalization_version": context["manifest"]["normalization_version"],
        "pass_documents": EXPECTED_PASS,
        "held_documents": EXPECTED_HELD,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": tokenizer.sha256,
        "strategies": [strategy.__dict__ for strategy in strategies],
        "chunker_version": CHUNKER_VERSION,
        "frozen_config_sha256": frozen_config_sha256,
    }
    snapshot_path = output / "manifests/input_snapshot.json"
    if snapshot_path.exists():
        if _load_json(snapshot_path) != snapshot:
            raise ValueError("Resume input snapshot differs from existing full run")
    else:
        _atomic_json(snapshot_path, snapshot)
    snapshot_sha256 = sha256_file(snapshot_path)
    candidate = context["candidate"]
    columns = [
        "doc_id", "official_url", "source_kind", "title", "normalized_text",
        "detected_language", "source_text_sha256", "normalized_text_sha256",
    ]
    totals = {strategy.strategy_id: {"documents": 0, "passages": 0} for strategy in strategies}
    clean_parts = context["manifest"]["part_checkpoints_sha256"]
    selected_stems = sorted(clean_parts)
    if shard_modulus is not None:
        if shard_modulus <= 0 or shard_remainder is None or not 0 <= shard_remainder < shard_modulus:
            raise ValueError("Require 0 <= --shard-remainder < --shard-modulus")
        selected_stems = [
            stem for index, stem in enumerate(selected_stems)
            if index % shard_modulus == shard_remainder
        ]
    for number, stem in enumerate(selected_stems, 1):
        pass_path = candidate / "release/pass_document_ids" / f"{stem}.parquet"
        part_ids = set(map(int, pq.read_table(pass_path, columns=["doc_id"])["doc_id"].to_pylist()))
        rows = [
            row for row in pq.read_table(
                candidate / "clean_documents" / f"{stem}.parquet", columns=columns
            ).to_pylist()
            if int(row["doc_id"]) in part_ids
        ]
        if len(rows) != len(part_ids):
            raise ValueError(f"PASS rows do not align with clean shard: {stem}")
        for strategy in strategies:
            sid = strategy.strategy_id
            passage_path = output / f"passages/strategy_{sid}/{stem}.parquet"
            mapping_path = output / f"metadata/strategy_{sid}/{stem}.parquet"
            failure_path = output / f"validation/strategy_{sid}/{stem}.parquet"
            checkpoint_path = output / f"checkpoints/strategy_{sid}/{stem}.json"
            if resume and _checkpoint_valid(
                checkpoint_path, snapshot_sha256, strategy.__dict__, passage_path
            ):
                checkpoint = _load_json(checkpoint_path)
                if (
                    checkpoint.get("mapping_sha256") != sha256_file(mapping_path)
                    or checkpoint.get("failure_sha256") != sha256_file(failure_path)
                ):
                    raise ValueError(f"Resume companion checksum mismatch: {sid}/{stem}")
                totals[sid]["documents"] += int(checkpoint["documents"])
                totals[sid]["passages"] += int(checkpoint["passages"])
                continue
            if any(path.exists() for path in (passage_path, mapping_path, failure_path, checkpoint_path)):
                raise ValueError(f"Incomplete or incompatible full output: {sid}/{stem}")
            passages: list[dict] = []
            mappings: list[dict] = []
            failures: list[dict] = []
            for row in rows:
                document_passages = chunk_document(
                    row, tokenizer, strategy,
                    phase1_release_id=context["release_sha256"],
                    corpus_revision=context["manifest"]["corpus_revision"],
                    processing_run_id=run_id,
                )
                for failure in validate_document_passages(
                    row, document_passages, tokenizer, strategy
                ):
                    failures.append({
                        "strategy_id": sid, "doc_id": int(row["doc_id"]), **failure,
                    })
                covered, source = covered_non_whitespace(row["normalized_text"], document_passages)
                mappings.append({
                    "doc_id": int(row["doc_id"]), "strategy_id": sid,
                    "passage_count": len(document_passages),
                    "first_passage_id": document_passages[0]["passage_id"] if document_passages else None,
                    "last_passage_id": document_passages[-1]["passage_id"] if document_passages else None,
                    "covered_non_whitespace_chars": covered,
                    "source_non_whitespace_chars": source,
                })
                passages.extend(document_passages)
            _atomic_parquet(passage_path, passages, PASSAGE_SCHEMA)
            _atomic_parquet(mapping_path, mappings, MAPPING_SCHEMA)
            _atomic_parquet(failure_path, failures, FAILURE_SCHEMA)
            checkpoint = {
                "status": "failed" if failures else "complete",
                "input_snapshot_sha256": snapshot_sha256,
                "strategy": strategy.__dict__,
                "part": stem,
                "documents": len(rows),
                "passages": len(passages),
                "failures": len(failures),
                "passage_sha256": sha256_file(passage_path),
                "mapping_sha256": sha256_file(mapping_path),
                "failure_sha256": sha256_file(failure_path),
            }
            _atomic_json(checkpoint_path, checkpoint)
            if failures:
                raise ValueError(f"Full shard validation failed: {sid}/{stem}")
            totals[sid]["documents"] += len(rows)
            totals[sid]["passages"] += len(passages)
        if number % 10 == 0 or number == len(selected_stems):
            progress(f"Full Phase 2: {number}/{len(selected_stems)} selected input shards")
    if shard_modulus is not None:
        return {
            "status": "partition_complete",
            "format": PASSAGE_SCHEMA_VERSION,
            "scope": "full_partition",
            "processing_run_id": run_id,
            "input_snapshot_sha256": snapshot_sha256,
            "shard_modulus": shard_modulus,
            "shard_remainder": shard_remainder,
            "selected_parts": len(selected_stems),
            "strategy_totals": totals,
        }
    if any(value["documents"] != EXPECTED_PASS for value in totals.values()):
        raise ValueError("Full output document totals differ from PASS count")
    manifest = {
        "status": "complete",
        "format": PASSAGE_SCHEMA_VERSION,
        "scope": "full",
        "processing_run_id": run_id,
        "input_snapshot_sha256": snapshot_sha256,
        "parts": len(clean_parts),
        "strategy_totals": totals,
    }
    manifest_path = output / "manifests/chunking_manifest.json"
    if manifest_path.exists():
        if _load_json(manifest_path) != manifest:
            raise ValueError("Existing full manifest differs from completed run")
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


def build_pilot(context: dict, tokenizer: BgeM3Tokenizer, output: Path,
                pilot_documents: int, run_id: str, *, resume: bool = False,
                locked_pilot_manifest: Path | None = None, progress=print) -> dict:
    output = ensure_output_is_safe(output, context["candidate"])
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists; use --resume only for the same run: {output}")
    if not output.exists():
        output.mkdir(parents=True)
    started = time.perf_counter()
    pilot_path = output / "pilot/pilot_manifest.parquet"
    snapshot_path = output / "manifests/input_snapshot.json"
    if locked_pilot_manifest is not None:
        if resume or pilot_path.exists() or snapshot_path.exists():
            raise ValueError("Locked pilot manifest requires a new output directory")
        pilot_rows, documents = _reload_pilot_documents(context, locked_pilot_manifest)
        if len(pilot_rows) != pilot_documents:
            raise ValueError("Locked pilot size differs from --pilot-documents")
        progress(f"Loaded locked pilot manifest: {len(pilot_rows)} documents")
    elif resume and pilot_path.is_file() and snapshot_path.is_file():
        pilot_rows, documents = _reload_pilot_documents(context, pilot_path)
        if len(pilot_rows) != pilot_documents:
            raise ValueError("Resume pilot size differs from --pilot-documents")
        progress(f"Reloaded locked pilot manifest: {len(pilot_rows)} documents")
    else:
        pilot_rows, documents = select_pilot(context, pilot_documents, progress=progress)
    snapshot = _input_snapshot(context, tokenizer, pilot_rows)
    if snapshot_path.exists():
        if _load_json(snapshot_path) != snapshot:
            raise ValueError("Resume input snapshot differs from existing run")
    else:
        _atomic_json(snapshot_path, snapshot)
    snapshot_sha256 = sha256_file(snapshot_path)
    if not pilot_path.exists():
        _atomic_parquet(pilot_path, pilot_rows, PILOT_SCHEMA)
    elif not resume:
        raise FileExistsError(pilot_path)

    all_failures: list[dict] = []
    all_mappings: list[dict] = []
    metrics: dict[str, dict] = {}
    all_passages: dict[str, list[dict]] = {}
    release_id = context["release_sha256"]
    corpus_revision = context["manifest"]["corpus_revision"]
    ordered_documents = [documents[row["doc_id"]] for row in pilot_rows]
    for strategy in STRATEGIES:
        strategy_started = time.perf_counter()
        passage_path = output / f"pilot/strategy_{strategy.strategy_id}/part-00000.parquet"
        checkpoint_path = output / f"checkpoints/strategy-{strategy.strategy_id}.json"
        if resume and _checkpoint_valid(
            checkpoint_path, snapshot_sha256, strategy.__dict__, passage_path
        ):
            passages = pq.read_table(passage_path).to_pylist()
            progress(f"Resumed completed strategy {strategy.strategy_id}")
        else:
            if passage_path.exists() or checkpoint_path.exists():
                raise ValueError(f"Incomplete or incompatible strategy output: {strategy.strategy_id}")
            passages = []
            mappings = []
            strategy_failures = []
            for number, row in enumerate(ordered_documents, 1):
                document_passages = chunk_document(
                    row, tokenizer, strategy, phase1_release_id=release_id,
                    corpus_revision=corpus_revision, processing_run_id=run_id,
                )
                failures = validate_document_passages(row, document_passages, tokenizer, strategy)
                for failure in failures:
                    strategy_failures.append({
                        "strategy_id": strategy.strategy_id,
                        "doc_id": int(row["doc_id"]),
                        **failure,
                    })
                covered, source = covered_non_whitespace(row["normalized_text"], document_passages)
                mappings.append({
                    "doc_id": int(row["doc_id"]),
                    "strategy_id": strategy.strategy_id,
                    "passage_count": len(document_passages),
                    "first_passage_id": document_passages[0]["passage_id"] if document_passages else None,
                    "last_passage_id": document_passages[-1]["passage_id"] if document_passages else None,
                    "covered_non_whitespace_chars": covered,
                    "source_non_whitespace_chars": source,
                })
                passages.extend(document_passages)
                if number % 100 == 0 or number == len(ordered_documents):
                    progress(f"Strategy {strategy.strategy_id}: {number}/{len(ordered_documents)} docs")
            if strategy_failures:
                all_failures.extend(strategy_failures)
            _atomic_parquet(passage_path, passages, PASSAGE_SCHEMA)
            all_mappings.extend(mappings)
            _atomic_json(checkpoint_path, {
                "status": "complete",
                "input_snapshot_sha256": snapshot_sha256,
                "strategy": strategy.__dict__,
                "documents": len(ordered_documents),
                "passages": len(passages),
                "passage_sha256": sha256_file(passage_path),
            })
        all_passages[strategy.strategy_id] = passages
        result = passage_metrics(passages, len(ordered_documents), strategy.min_chunk_tokens)
        result.update({
            "strategy_id": strategy.strategy_id,
            "target_tokens": strategy.target_tokens,
            "max_tokens": strategy.max_tokens,
            "overlap_tokens": strategy.overlap_tokens,
            "processing_seconds": time.perf_counter() - strategy_started,
            "source_mapping_errors": sum(
                row["strategy_id"] == strategy.strategy_id and row["failure_type"] == "source_slice"
                for row in all_failures
            ),
            "oversized_passages": sum(
                row["strategy_id"] == strategy.strategy_id and row["failure_type"] == "max_tokens"
                for row in all_failures
            ),
            "coverage_failures": sum(
                row["strategy_id"] == strategy.strategy_id and row["failure_type"] == "structural_coverage"
                for row in all_failures
            ),
        })
        metrics[strategy.strategy_id] = result

    if len(all_mappings) != len(STRATEGIES) * len(ordered_documents):
        # Resume reloads mapping deterministically from passage files.
        all_mappings = []
        for strategy in STRATEGIES:
            grouped: dict[int, list[dict]] = {}
            for passage in all_passages[strategy.strategy_id]:
                grouped.setdefault(int(passage["doc_id"]), []).append(passage)
            for row in ordered_documents:
                doc_passages = grouped.get(int(row["doc_id"]), [])
                covered, source = covered_non_whitespace(row["normalized_text"], doc_passages)
                all_mappings.append({
                    "doc_id": int(row["doc_id"]), "strategy_id": strategy.strategy_id,
                    "passage_count": len(doc_passages),
                    "first_passage_id": doc_passages[0]["passage_id"] if doc_passages else None,
                    "last_passage_id": doc_passages[-1]["passage_id"] if doc_passages else None,
                    "covered_non_whitespace_chars": covered,
                    "source_non_whitespace_chars": source,
                })
    mapping_path = output / "metadata/document_passage_mapping.parquet"
    if not mapping_path.exists():
        _atomic_parquet(mapping_path, all_mappings, MAPPING_SCHEMA)
    failure_path = output / "validation/chunking_validation_failures.parquet"
    if not failure_path.exists():
        _atomic_parquet(failure_path, all_failures, FAILURE_SCHEMA)

    seen_passage_ids: set[str] = set()
    collisions = 0
    for passages in all_passages.values():
        for passage in passages:
            if passage["passage_id"] in seen_passage_ids:
                collisions += 1
            seen_passage_ids.add(passage["passage_id"])
    validation = {
        "status": "pass" if not all_failures and not collisions else "failed",
        "documents": len(ordered_documents),
        "strategies": len(STRATEGIES),
        "failures": len(all_failures),
        "passage_id_collisions": collisions,
        "source_mapping_errors": sum(row["failure_type"] == "source_slice" for row in all_failures),
        "token_count_errors": sum(row["failure_type"] == "token_count" for row in all_failures),
        "oversized_passages": sum(row["failure_type"] == "max_tokens" for row in all_failures),
        "coverage_failures": sum(row["failure_type"] == "structural_coverage" for row in all_failures),
    }
    validation_path = output / "validation/chunking_validation_report.json"
    if not validation_path.exists():
        _atomic_json(validation_path, validation)
    if validation["status"] != "pass":
        raise ValueError(f"Pilot validation failed: {validation}")

    review_pool = []
    for strategy_id, passages in all_passages.items():
        for row in passages:
            priority = 0 if _REVIEW_SIGNAL.search(row["passage_text"]) else 1
            score = int.from_bytes(hashlib.sha256(row["passage_id"].encode()).digest()[:8], "big")
            review_pool.append((priority, score, strategy_id, row))
    review_rows = [item[3] for item in sorted(review_pool)[:150]]
    review_path = output / "reports/manual_review_sample.parquet"
    if not review_path.exists():
        _atomic_parquet(review_path, review_rows, PASSAGE_SCHEMA)

    quality = {
        "status": "pilot_complete_unreviewed",
        "manual_review_confirmed": False,
        "input": snapshot,
        "pilot_strata": {
            "language": dict(Counter(row["language"] for row in pilot_rows)),
            "source_kind": dict(Counter(row["source_kind"] for row in pilot_rows)),
            "domain": dict(Counter(row["domain"] for row in pilot_rows)),
            "length_bin": dict(Counter(row["length_bin"] for row in pilot_rows)),
            "has_heading": dict(Counter(str(row["has_heading"]).lower() for row in pilot_rows)),
            "paragraph_profile": dict(Counter(row["paragraph_profile"] for row in pilot_rows)),
            "mixed_language": sum(row["mixed_language"] for row in pilot_rows),
            "medical_measurement": sum(row["medical_measurement"] for row in pilot_rows),
        },
        "strategy_metrics": metrics,
        "validation": validation,
        "processing_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    quality_path = output / "reports/chunking_quality_report.json"
    if not quality_path.exists():
        _atomic_json(quality_path, quality)

    comparison_path = output / "reports/chunking_strategy_comparison.csv"
    if not comparison_path.exists():
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=comparison_path.parent,
                                         prefix=f".{comparison_path.name}.", suffix=".partial",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(next(iter(metrics.values())).keys()))
            writer.writeheader()
            writer.writerows(metrics.values())
        os.replace(temporary, comparison_path)

    manifest = {
        "status": "pilot_complete_unreviewed",
        "format": PASSAGE_SCHEMA_VERSION,
        "processing_run_id": run_id,
        "input_snapshot_sha256": snapshot_sha256,
        "pilot_documents": len(ordered_documents),
        "strategies": [strategy.__dict__ for strategy in STRATEGIES],
        "strategy_passages": {key: value["passage_count"] for key, value in metrics.items()},
        "validation_status": validation["status"],
        "manual_review_confirmed": False,
        "artifact_sha256": {
            "pilot_manifest": sha256_file(pilot_path),
            "mapping": sha256_file(mapping_path),
            "validation_report": sha256_file(validation_path),
            "validation_failures": sha256_file(failure_path),
            "quality_report": sha256_file(quality_path),
            "strategy_comparison": sha256_file(comparison_path),
            "manual_review_sample": sha256_file(review_path),
        },
    }
    manifest_path = output / "manifests/chunking_manifest.json"
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot-documents", type=int, default=1_000)
    parser.add_argument("--run-id", default="phase2_pilot_v1")
    parser.add_argument("--tokenizer-json", type=Path, default=default_tokenizer_path(ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Build all PASS documents after explicit approval")
    parser.add_argument(
        "--strategy", choices=[strategy.strategy_id for strategy in STRATEGIES],
        action="append", dest="strategy_ids",
        help="Strategy to build in --full mode; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--frozen-config", type=Path,
        help="Validate and record the accepted production config in --full mode",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-modulus", type=int,
                        help="Process one deterministic shard partition in --full mode")
    parser.add_argument("--shard-remainder", type=int,
                        help="Zero-based partition selected by --shard-modulus")
    parser.add_argument("--locked-pilot-manifest", type=Path,
                        help="Reuse an immutable pilot manifest for paired regression")
    args = parser.parse_args(argv)
    try:
        tokenizer = BgeM3Tokenizer(args.tokenizer_json)
        context = verify_release(args.candidate, args.tokenizer_json)
        if args.dry_run:
            print(json.dumps({
                "status": "pass",
                "candidate": str(context["candidate"]),
                "pass_documents": len(context["pass_ids"]),
                "held_documents": len(context["held_ids"]),
                "intersection": len(context["pass_ids"] & context["held_ids"]),
                "tokenizer_name": TOKENIZER_NAME,
                "tokenizer_revision": TOKENIZER_REVISION,
                "tokenizer_json_sha256": tokenizer.sha256,
            }, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            parser.error("--output is required unless --dry-run is used")
        if args.full:
            if (args.shard_modulus is None) != (args.shard_remainder is None):
                raise ValueError("--shard-modulus and --shard-remainder must be used together")
            if args.frozen_config:
                selected = strategies_from_frozen_config(args.frozen_config)
                if args.strategy_ids and args.strategy_ids != [selected[0].strategy_id]:
                    raise ValueError("--strategy differs from --frozen-config")
            else:
                selected = tuple(
                    strategy for strategy in STRATEGIES
                    if not args.strategy_ids or strategy.strategy_id in args.strategy_ids
                )
            result = build_full_corpus(
                context, tokenizer, args.output, args.run_id,
                resume=args.resume, strategies=selected,
                frozen_config_sha256=(
                    sha256_file(args.frozen_config) if args.frozen_config else None
                ),
                shard_modulus=args.shard_modulus,
                shard_remainder=args.shard_remainder,
            )
        else:
            result = build_pilot(
                context, tokenizer, args.output, args.pilot_documents, args.run_id,
                resume=args.resume, locked_pilot_manifest=args.locked_pilot_manifest,
            )
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Phase 2 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
