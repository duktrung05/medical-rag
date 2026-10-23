"""Read a bounded set of documents from the immutable Phase 1/2 releases."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pyarrow as pa
import pyarrow.parquet as pq

from src.medrag.data.phase2.provenance import passage_id as make_passage_id
from src.medrag.data.phase2.provenance import sha256_text


class ReleaseIntegrityError(ValueError):
    """An input does not match the selected immutable release contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseIntegrityError(f"Expected a JSON object: {path}")
    return value


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ReleaseIntegrityError(f"Invalid official URL: {url!r}")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


@dataclass(frozen=True)
class PartRange:
    name: str
    path: Path
    low: int
    high: int
    rows: int
    sha256: str
    bytes: int
    doc_ids: frozenset[int]


def _part_ranges(root: Path, parts: list[dict], *, expected_hashes: dict | None = None) -> list[PartRange]:
    result = []
    for part in parts:
        name = str(part["part"])
        path = root / f"{name}.parquet"
        if not path.is_file():
            raise ReleaseIntegrityError(f"Release part is missing: {path}")
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        if expected_hashes is None and rows != int(part["rows"]):
            raise ReleaseIntegrityError(f"Row-count mismatch in {path}: {rows} != {part['rows']}")
        expected = part.get("sha256")
        if expected_hashes is not None:
            expected = expected_hashes.get(name)
        if not expected:
            raise ReleaseIntegrityError(f"No checksum recorded for {path}")
        if rows == 0:
            if sha256_file(path) != expected:
                raise ReleaseIntegrityError(f"Empty-shard checksum mismatch: {path}")
            continue
        if parquet.schema_arrow.get_field_index("doc_id") < 0:
            raise ReleaseIntegrityError(f"Release part has no doc_id column: {path}")
        doc_ids = []
        for batch in parquet.iter_batches(columns=["doc_id"], batch_size=250_000):
            doc_ids.extend(map(int, batch.column(0).to_pylist()))
        if len(doc_ids) != rows:
            raise ReleaseIntegrityError(f"doc_id scan row count differs from footer: {path}")
        if expected_hashes is not None and len(set(doc_ids)) != len(doc_ids):
            raise ReleaseIntegrityError(f"Duplicate document IDs in Phase 1 clean shard: {path}")
        low, high = min(doc_ids), max(doc_ids)
        result.append(PartRange(name, path, low, high, rows, str(expected), path.stat().st_size,
                                frozenset(doc_ids)))
    return result


def _read_ids(path: Path, *, id_column: str, ids: set[int]) -> dict[int, dict]:
    if not ids:
        return {}
    table = pq.read_table(path, filters=[(id_column, "in", sorted(ids))])
    rows = table.to_pylist()
    found = {}
    for row in rows:
        ident = int(row[id_column])
        if ident in found:
            raise ReleaseIntegrityError(f"Duplicate {id_column}={ident} in {path}")
        found[ident] = row
    return found


def _ids_in_part(path: Path, ids: set[int]) -> set[int]:
    rows = _read_ids(path, id_column="doc_id", ids=ids)
    return set(rows)


def _read_parts_for_ids(parts: list[PartRange], ids: set[int], *, columns: list[str]) -> tuple[list[dict], list[dict]]:
    if not ids:
        raise ReleaseIntegrityError("At least one document ID is required")
    assignments = {part.name: ids & part.doc_ids for part in parts if ids & part.doc_ids}
    found_ids = set().union(*assignments.values()) if assignments else set()
    missing_ids = ids - found_ids
    if missing_ids:
        raise ReleaseIntegrityError(f"Document IDs are absent from release shards: {sorted(missing_ids)[:5]}")
    rows, consumed = [], []
    by_name = {part.name: part for part in parts}
    for name in sorted(assignments):
        part = by_name[name]
        actual_hash = sha256_file(part.path)
        if actual_hash != part.sha256:
            raise ReleaseIntegrityError(f"Shard checksum mismatch: {part.path}")
        table = pq.read_table(part.path, columns=columns,
                              filters=[("doc_id", "in", sorted(assignments[name]))])
        selected_rows = table.to_pylist()
        selected_ids = {int(row["doc_id"]) for row in selected_rows}
        missing = assignments[name] - selected_ids
        if missing:
            raise ReleaseIntegrityError(f"IDs missing in {part.path}: {sorted(missing)[:5]}")
        rows.extend(selected_rows)
        consumed.append({"part": name, "path": str(part.path), "rows": part.rows,
                         "bytes": part.bytes, "sha256": actual_hash})
    if not ids <= {int(row["doc_id"]) for row in rows}:
        raise ReleaseIntegrityError("Release shard lookup did not return all requested IDs")
    return rows, consumed


@dataclass
class Phase2Batch:
    documents: list[dict]
    passages: list[dict]
    queries: list[dict]
    manifest: dict


def load_phase2_batch(config_path: Path, *, doc_ids: list[int], query_ids: list[int],
                      max_documents: int = 1000) -> Phase2Batch:
    """Load and verify explicit subsets; this API cannot request a full-corpus run."""
    if not 0 < len(set(doc_ids)) <= max_documents:
        raise ValueError(f"Provide 1..{max_documents} unique doc IDs")
    if not 0 < len(set(query_ids)) <= 1200:
        raise ValueError("Provide 1..1200 unique official query IDs")
    if len(set(doc_ids)) != len(doc_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate document or query ID")
    ids, qids = set(map(int, doc_ids)), set(map(int, query_ids))

    config = _read_json(config_path)
    root = config_path.resolve().parents[2]
    p1_root = (root / config["phase1_root"]).resolve()
    p2_root = (root / config["phase2_root"]).resolve()
    official_corpus = (root / config["official_corpus"]).resolve()
    official_queries = (root / config["official_queries"]).resolve()
    staging_path = (root / config["semantic_staging"]).resolve()

    p1_manifest_path = p1_root / "phase1_manifest.json"
    p1_release_path = p1_root / "release/phase1_release_manifest.json"
    p1_snapshot_path = p1_root / "source_snapshot.json"
    p2_manifest_path = p2_root / "manifests/phase2_release_manifest.json"
    p2_snapshot_path = p2_root / "manifests/input_snapshot.json"
    p2_validation_path = p2_root / "validation/full_validation_report.json"
    p2_mapping_path = p2_root / "metadata/document_passage_mapping.parquet"
    frozen_config_path = (root / config["phase2_frozen_config"]).resolve()
    for path in (p1_manifest_path, p1_release_path, p1_snapshot_path, p2_manifest_path, p2_snapshot_path,
                 p2_validation_path, p2_mapping_path, frozen_config_path, official_corpus, official_queries,
                 staging_path):
        if not path.is_file():
            raise ReleaseIntegrityError(f"Required input is missing: {path}")

    p1_manifest = _read_json(p1_manifest_path)
    p1_release = _read_json(p1_release_path)
    p2_manifest = _read_json(p2_manifest_path)
    p2_snapshot = _read_json(p2_snapshot_path)
    p2_validation = _read_json(p2_validation_path)
    frozen = _read_json(frozen_config_path)
    if p1_manifest.get("status") != "complete" or p1_manifest.get("build_scope") != "full":
        raise ReleaseIntegrityError("Phase 1 candidate is not a completed full build")
    if p1_release.get("status") != "complete" or p1_release.get("format") != "vibiomir-phase1-release-v1":
        raise ReleaseIntegrityError("Phase 1 release is not complete")
    if p2_manifest.get("status") != "complete_validated" or p2_manifest.get("scope") != "full":
        raise ReleaseIntegrityError("Phase 2 release is not complete and validated")
    if p2_validation.get("status") != "pass" or p2_manifest.get("validation_report", {}).get("status") != "pass":
        raise ReleaseIntegrityError("Phase 2 validation report is not PASS")

    checksums = {
        p1_manifest_path: p1_release.get("candidate_manifest_sha256"),
        p1_snapshot_path: p1_manifest.get("source_snapshot_sha256"),
        p2_snapshot_path: p2_manifest.get("input_snapshot_sha256"),
        p2_validation_path: p2_manifest.get("validation_report", {}).get("sha256"),
        p2_mapping_path: p2_manifest.get("document_passage_mapping", {}).get("sha256"),
        frozen_config_path: p2_manifest.get("frozen_config_sha256"),
        official_corpus: config["official_corpus_sha256"],
        official_queries: config["official_queries_sha256"],
    }
    verified_files = {}
    for path, expected in checksums.items():
        if not expected:
            raise ReleaseIntegrityError(f"Missing expected checksum for {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ReleaseIntegrityError(f"Checksum mismatch: {path}")
        verified_files[str(path)] = {"bytes": path.stat().st_size, "sha256": actual}
    p1_release_sha = sha256_file(p1_release_path)
    if p2_manifest.get("phase1_release_manifest_sha256") != p1_release_sha:
        raise ReleaseIntegrityError("Phase 2 is linked to a different Phase 1 release manifest")
    if p2_validation.get("input", {}).get("phase1_release_manifest_sha256") != p1_release_sha:
        raise ReleaseIntegrityError("Phase 2 validation report references a different Phase 1 release")
    if p2_validation.get("input", {}).get("candidate_manifest_sha256") != p1_release.get("candidate_manifest_sha256"):
        raise ReleaseIntegrityError("Phase 2 validation report references a different Phase 1 candidate")
    if (p2_snapshot.get("phase1_release_manifest_sha256") != p1_release_sha
            or p2_snapshot.get("candidate_manifest_sha256") != p1_release.get("candidate_manifest_sha256")):
        raise ReleaseIntegrityError("Phase 2 input snapshot references a different Phase 1 release/candidate")
    if (p2_snapshot.get("corpus_revision") != p1_manifest.get("corpus_revision")
            or p2_snapshot.get("scope") != "full"):
        raise ReleaseIntegrityError("Phase 1/2 corpus revisions differ")
    if frozen.get("selected_strategy") != p2_manifest.get("strategy"):
        raise ReleaseIntegrityError("Frozen Phase 2 strategy differs from the release")
    if p1_release.get("pass_documents") != p2_manifest.get("documents"):
        raise ReleaseIntegrityError("Phase 2 document count differs from the Phase 1 PASS release")
    if (frozen.get("tokenizer_name") != p2_snapshot.get("tokenizer_name")
            or p2_snapshot.get("tokenizer_revision") != p2_validation.get("input", {}).get("tokenizer_revision")):
        raise ReleaseIntegrityError("Frozen tokenizer name/revision differs from the validated Phase 2 release")

    p1_parts = p1_release.get("pass_allowlist_parts", [])
    p1_clean_hashes = p1_manifest.get("part_checkpoints_sha256", {})
    if {part["part"] for part in p1_parts} != set(p1_clean_hashes):
        raise ReleaseIntegrityError("Phase 1 clean and PASS allowlist shard names differ")
    phase1_clean = _part_ranges(p1_root / "clean_documents", p1_parts,
                                expected_hashes=p1_clean_hashes)
    phase1_allow = _part_ranges(p1_root / "release/pass_document_ids", p1_parts)
    phase2_parts = _part_ranges(p2_root / "passages/strategy_C", p2_manifest.get("passage_parts", []))

    staging = _read_json(staging_path)
    staging_rows = staging.get("documents", [])
    excluded = {int(row["doc_id"]) for row in staging_rows}
    if len(excluded) != len(staging_rows):
        raise ReleaseIntegrityError("Semantic staging document IDs are duplicated")
    requested_in_pass = set()
    p1_consumed = []
    clean_by_name = {part.name: part for part in phase1_clean}
    allow_by_name = {part.name: part for part in phase1_allow}
    clean_location = {}
    allow_location = {}
    for part in phase1_clean:
        location = clean_location
        for ident in part.doc_ids:
            if ident in location:
                raise ReleaseIntegrityError(f"Duplicate Phase 1 clean document ID across shards: {ident}")
            location[ident] = part.name
    for part in phase1_allow:
        for ident in part.doc_ids:
            if ident in allow_location:
                raise ReleaseIntegrityError(f"Duplicate Phase 1 PASS ID across shards: {ident}")
            allow_location[ident] = part.name
    assignments: dict[str, set[int]] = {}
    for ident in ids:
        clean_name = clean_location.get(ident)
        allow_name = allow_location.get(ident)
        if clean_name is None or allow_name is None:
            raise ReleaseIntegrityError(f"doc_id={ident} is not in the Phase 1 PASS release")
        if clean_name != allow_name:
            raise ReleaseIntegrityError(f"Phase 1 clean and PASS rows map to different shards for doc_id={ident}")
        assignments.setdefault(clean_name, set()).add(ident)
    documents = []
    for name in sorted(assignments):
        clean_part, allow_part = clean_by_name[name], allow_by_name[name]
        if sha256_file(clean_part.path) != clean_part.sha256:
            raise ReleaseIntegrityError(f"Phase 1 clean shard checksum mismatch: {clean_part.path}")
        if sha256_file(allow_part.path) != allow_part.sha256:
            raise ReleaseIntegrityError(f"Phase 1 PASS allowlist checksum mismatch: {allow_part.path}")
        allowed = _ids_in_part(allow_part.path, assignments[name])
        if allowed != assignments[name]:
            raise ReleaseIntegrityError(f"Requested documents are not all PASS in {name}")
        selected = _read_ids(clean_part.path, id_column="doc_id", ids=assignments[name])
        if set(selected) != assignments[name]:
            raise ReleaseIntegrityError(f"PASS clean rows missing from {name}")
        documents.extend(selected.values())
        requested_in_pass |= allowed
        p1_consumed.append({"part": name, "clean_sha256": clean_part.sha256,
                             "pass_allowlist_sha256": allow_part.sha256})
    if requested_in_pass != ids:
        raise ReleaseIntegrityError("Requested document IDs do not exactly match PASS release rows")

    staging_overlap = ids & excluded
    if staging_overlap:
        raise ReleaseIntegrityError(f"Semantic staging exclusions requested: {sorted(staging_overlap)}")
    if {int(row["doc_id"]) for row in documents} != ids:
        raise ReleaseIntegrityError("Phase 1 lookup did not return the exact requested documents")

    official_rows = _read_ids(official_corpus, id_column="id", ids=ids)
    if set(official_rows) != ids:
        raise ReleaseIntegrityError("Some requested document IDs are absent from official corpus")
    documents_by_id = {int(row["doc_id"]): row for row in documents}
    for ident, row in documents_by_id.items():
        official_url = official_rows[ident].get("url")
        if normalize_url(str(official_url)) != normalize_url(str(row["official_url"])):
            raise ReleaseIntegrityError(f"Official doc_id/URL mismatch: {ident}")
        normalized = row.get("normalized_text")
        normalized_sha = row.get("normalized_text_sha256")
        if not isinstance(normalized, str) or sha256_text(normalized) != normalized_sha:
            raise ReleaseIntegrityError(f"Normalized source hash mismatch: {ident}")

    query_rows = _read_ids(official_queries, id_column="id", ids=qids)
    if set(query_rows) != qids:
        raise ReleaseIntegrityError("Some requested query IDs are absent from official query corpus")
    queries = [{"id": ident, "query": query_rows[ident]["query"]} for ident in sorted(qids)]

    passage_rows, p2_consumed = _read_parts_for_ids(
        phase2_parts, ids,
        columns=["passage_id", "doc_id", "official_url", "passage_index", "passage_text",
                 "char_start", "char_end", "source_text_sha256", "text_representation",
                 "text_representation_sha256", "token_count", "language", "section_id",
                 "section_heading", "strategy_id", "chunker_version", "tokenizer_name",
                 "tokenizer_revision", "phase1_release_id", "corpus_revision",
                 "processing_run_id"],
    )
    seen_passages = set()
    for row in passage_rows:
        ident = int(row["doc_id"])
        source = documents_by_id[ident]
        text, start, end = row["passage_text"], int(row["char_start"]), int(row["char_end"])
        if row["text_representation"] != "normalized_text":
            raise ReleaseIntegrityError(f"Unexpected passage text representation: {row['passage_id']}")
        if row["text_representation_sha256"] != source["normalized_text_sha256"]:
            raise ReleaseIntegrityError(f"Passage normalized-source hash mismatch: {row['passage_id']}")
        if row["source_text_sha256"] != source["source_text_sha256"]:
            raise ReleaseIntegrityError(f"Passage raw-source hash mismatch: {row['passage_id']}")
        if not 0 <= start < end <= len(source["normalized_text"]) or source["normalized_text"][start:end] != text:
            raise ReleaseIntegrityError(f"Passage offsets/text do not match normalized source: {row['passage_id']}")
        expected_id = make_passage_id(
            chunker_version=row["chunker_version"], strategy_id=row["strategy_id"], doc_id=ident,
            passage_index=int(row["passage_index"]), source_sha256=row["text_representation_sha256"],
            char_start=start, char_end=end,
        )
        if row["passage_id"] != expected_id or expected_id in seen_passages:
            raise ReleaseIntegrityError(f"Passage identity is invalid or duplicated: {row['passage_id']}")
        seen_passages.add(expected_id)
        if row["official_url"] != source["official_url"]:
            raise ReleaseIntegrityError(f"Passage URL differs from its source document: {row['passage_id']}")
        expected_lineage = {
            "strategy_id": p2_manifest["strategy"]["strategy_id"],
            "chunker_version": p2_manifest["chunker_version"],
            "tokenizer_name": p2_manifest["tokenizer_name"],
            "tokenizer_revision": p2_manifest["tokenizer_revision"],
            "phase1_release_id": p1_release_sha,
            "corpus_revision": p2_snapshot["corpus_revision"],
        }
        for field, expected_value in expected_lineage.items():
            if row[field] != expected_value:
                raise ReleaseIntegrityError(
                    f"Passage {field} differs from the release manifest: {row['passage_id']}"
                )
    if {int(row["doc_id"]) for row in passage_rows} != ids:
        raise ReleaseIntegrityError("At least one selected document has no Phase 2 passage")
    excluded_rows, excluded_parts = _read_parts_for_ids(
        phase2_parts, excluded,
        columns=["doc_id", "passage_index"],
    ) if excluded else ([], [])
    excluded_counts = {}
    for row in excluded_rows:
        excluded_counts[int(row["doc_id"])] = excluded_counts.get(int(row["doc_id"]), 0) + 1
    mapping_ids = ids | excluded
    mapping_rows = _read_ids(p2_mapping_path, id_column="doc_id", ids=mapping_ids)
    if set(mapping_rows) != mapping_ids:
        raise ReleaseIntegrityError("Selected/excluded documents are absent from Phase 2 mapping")
    actual_passage_counts = {}
    for row in passage_rows:
        ident = int(row["doc_id"])
        actual_passage_counts[ident] = actual_passage_counts.get(ident, 0) + 1
    for ident in ids:
        mapping = mapping_rows[ident]
        if (mapping["strategy_id"] != p2_manifest["strategy"]["strategy_id"]
                or int(mapping["passage_count"]) != actual_passage_counts.get(ident, 0)):
            raise ReleaseIntegrityError(f"Phase 2 passage coverage differs from mapping for doc_id={ident}")
    for ident in excluded:
        if int(mapping_rows[ident]["passage_count"]) != excluded_counts.get(ident, 0):
            raise ReleaseIntegrityError(f"Semantic exclusion passage count differs from mapping for doc_id={ident}")
    passage_rows.sort(key=lambda row: (int(row["doc_id"]), int(row["passage_index"])))

    document_rows = [{"doc_id": ident, "official_url": documents_by_id[ident]["official_url"],
                      "title": documents_by_id[ident].get("title") or "",
                      "normalized_text": documents_by_id[ident]["normalized_text"],
                      "normalized_text_sha256": documents_by_id[ident]["normalized_text_sha256"],
                      "source_text_sha256": documents_by_id[ident]["source_text_sha256"],
                      "detected_language": documents_by_id[ident].get("detected_language")}
                     for ident in sorted(ids)]
    provenance = [{key: row[key] for key in (
        "passage_id", "doc_id", "passage_index", "official_url", "char_start", "char_end",
        "source_text_sha256", "text_representation", "text_representation_sha256",
        "language", "section_id", "section_heading", "strategy_id", "chunker_version",
        "tokenizer_name", "tokenizer_revision", "phase1_release_id", "corpus_revision",
        "processing_run_id",
    )} for row in passage_rows]
    return Phase2Batch(
        documents=document_rows, passages=passage_rows, queries=queries,
        manifest={
            "format": "medrag-phase2-batch-input-v1", "scope": "explicit_document_query_subset",
            "phase1_release_manifest_sha256": p1_release_sha,
            "phase2_release_manifest_sha256": sha256_file(p2_manifest_path),
            "phase2_input_snapshot_sha256": verified_files[str(p2_snapshot_path)]["sha256"],
            "phase2_validation_sha256": verified_files[str(p2_validation_path)]["sha256"],
            "phase2_strategy": p2_manifest["strategy"], "corpus_revision": p2_snapshot["corpus_revision"],
            "documents_requested": len(ids), "documents_loaded": len(documents),
            "passages_loaded": len(passage_rows), "queries_loaded": len(queries),
            "semantic_excluded_ids": sorted(excluded),
            "passages_in_phase2_before_exclusion": len(passage_rows) + len(excluded_rows),
            "passages_after_semantic_exclusion": len(passage_rows),
            "semantic_exclusion_passages": sum(excluded_counts.values()),
            "verified_files": verified_files,
            "phase1_shards_consumed": p1_consumed,
            "phase2_shards_consumed": p2_consumed,
            "phase2_exclusion_shards_consumed": excluded_parts,
            "semantic_exclusion_ledger": [
                {"doc_id": int(row["doc_id"]), "review_case": row["review_case"],
                 "reason": row["reason"], "phase2_passages": excluded_counts.get(int(row["doc_id"]), 0)}
                for row in staging_rows
            ],
            "preflight": {"status": "pass", "checks": [
                "phase1_phase2_release_link", "phase1_pass_allowlist", "semantic_exclusions",
                "official_doc_id_url", "normalized_text_sha256", "phase2_passage_offsets_and_ids",
                "selected_shard_sha256",
            ]},
        },
    )


def write_batch(batch: Phase2Batch, output: Path) -> dict:
    """Write a legacy-compatible batch table plus a lossless provenance table."""
    output.mkdir(parents=True, exist_ok=False)
    titles = {int(doc["doc_id"]): doc["title"] for doc in batch.documents}
    chunks = [{"doc_id": int(row["doc_id"]), "chunk_index": int(row["passage_index"]),
               "title": titles[int(row["doc_id"])],
               "text": row["passage_text"], "passage_id": row["passage_id"]}
              for row in batch.passages]
    pq.write_table(pa.Table.from_pylist(chunks), output / "chunks.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(batch.documents), output / "documents.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(batch.queries), output / "queries.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist([
        {key: row[key] for key in row if key != "passage_text"} for row in batch.passages
    ]), output / "passage_provenance.parquet", compression="zstd")
    batch.manifest.update({
        "chunks_path": "chunks.parquet", "chunks_rows": len(chunks),
        "documents_path": "documents.parquet", "queries_path": "queries.parquet",
        "provenance_path": "passage_provenance.parquet",
    })
    (output / "input_manifest.json").write_text(
        json.dumps(batch.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "preflight_report.json").write_text(
        json.dumps(batch.manifest["preflight"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pq.write_table(pa.Table.from_pylist(batch.manifest["semantic_exclusion_ledger"]),
                   output / "exclusion_ledger.parquet", compression="zstd")
    return batch.manifest
