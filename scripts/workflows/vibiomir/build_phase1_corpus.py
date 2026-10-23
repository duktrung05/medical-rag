"""Build the versioned Phase 1 clean-document corpus from audited text only.

The command performs a complete, write-free preflight before creating an output
directory. It never imports crawler, chunking, indexing, NER, or taxonomy code.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.language import (
    LANGUAGE_DETECTION_VERSION,
    detect_document_language,
    language_stratum,
)
from src.medrag.data.phase1.normalization import NORMALIZATION_VERSION, normalize_medical_text
from src.medrag.data.phase1.schema import CLEAN_DOCUMENT_SCHEMA, PHASE1_SCHEMA_VERSION
from src.medrag.data.phase1.selection import (
    ContentFile,
    Phase1IntegrityError,
    Phase1Paths,
    PreflightReport,
    ProductionExpectations,
    SelectionState,
    load_content_files,
    load_selection,
    preflight_canonical_content,
    sha256_file,
    validate_locked_inputs,
)
from src.medrag.data.phase1.validation import (
    QUALITY_VALIDATION_VERSION,
    REPEATED_TEMPLATE_THRESHOLD,
    assess_text_quality,
)


DEFAULT_OUTPUT = ROOT / "outputs/vibiomir_phase1_20261014_v1"


@dataclass
class PreflightContext:
    paths: Phase1Paths
    expectations: ProductionExpectations
    selection: SelectionState
    content_files: list[ContentFile]
    report: PreflightReport


@dataclass(frozen=True)
class PilotPlan:
    ids: frozenset[int]
    strategy: str
    ids_sha256: str
    available_by_stratum: dict[str, int]
    selected_by_stratum: dict[str, int]
    selection_source: str | None = None
    selection_source_manifest_sha256: str | None = None


PILOT_SAMPLING_VERSION = "deterministic-source-language-stratified-v1"
_PILOT_WEIGHTS = {
    "html:vi": 0.25,
    "html:en": 0.15,
    "html:zh": 0.20,
    "html:other": 0.025,
    "html:unknown": 0.025,
    "pubmed:vi": 0.01,
    "pubmed:en": 0.25,
    "pubmed:zh": 0.05,
    "pubmed:other": 0.03,
    "pubmed:unknown": 0.01,
}


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
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_output_is_safe(output: Path, paths: Phase1Paths) -> Path:
    output = output.resolve()
    protected = [
        paths.repo_root / "data",
        paths.repo_root / "outputs/vibiomir_phase0_20261014",
        paths.repo_root / "outputs/vibiomir_step2_20261013",
        paths.repo_root / "outputs/vibiomir_official_verify_20261012",
    ]
    for root in protected:
        root = root.resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise Phase1IntegrityError(f"Output overlaps protected input root: {root}")
    return output


def run_preflight(
    paths: Phase1Paths,
    expectations: ProductionExpectations | None = None,
    *,
    batch_size: int = 5_000,
    progress=None,
) -> PreflightContext:
    expectations = expectations or ProductionExpectations()
    progress = progress or (lambda _message: None)
    fingerprints, _audit, release = validate_locked_inputs(paths, expectations)
    progress("Locked input checksums verified")
    selection = load_selection(paths, expectations, fingerprints)
    progress(f"Phase 0 allowlist verified: {len(selection.documents):,} unique USABLE IDs")
    content_files = load_content_files(paths, release, expectations)
    report = preflight_canonical_content(
        selection, content_files, batch_size=batch_size, progress=progress
    )
    return PreflightContext(paths, expectations, selection, content_files, report)


def _source_columns(kind: str) -> list[str]:
    if kind == "html":
        return [
            "id", "official_url", "status", "title", "text", "encoding",
            "encoding_source", "extraction_method", "raw_path", "gzip_sha256",
            "raw_sha256", "text_sha256",
        ]
    return [
        "id", "official_url", "title", "text", "language_metadata", "pmid",
        "raw_file", "raw_sha256", "text_sha256", "extractor_version",
    ]


def _pilot_columns(kind: str) -> list[str]:
    return ["id", "title", "text"] + (["language_metadata"] if kind == "pubmed" else [])


def _pilot_score(doc_id: int) -> int:
    digest = hashlib.sha256(f"{PILOT_SAMPLING_VERSION}\0{doc_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _allocate_pilot_quotas(available: dict[str, int], total: int) -> dict[str, int]:
    if total > sum(available.values()):
        raise Phase1IntegrityError("Pilot size exceeds canonical document count")
    allocation = {key: 0 for key, count in available.items() if count}
    while sum(allocation.values()) < total:
        candidates = [key for key in allocation if allocation[key] < available[key]]
        if not candidates:
            raise Phase1IntegrityError("Unable to allocate the requested pilot size")
        # Weighted fair allocation. Unknown future strata retain a small but
        # non-zero QA presence instead of silently disappearing.
        key = max(
            candidates,
            key=lambda value: (
                _PILOT_WEIGHTS.get(value, 0.005) / (allocation[value] + 1),
                -allocation[value],
                value,
            ),
        )
        allocation[key] += 1
    return dict(sorted(allocation.items()))


def plan_stratified_pilot(
    context: PreflightContext,
    pilot_documents: int,
    *,
    batch_size: int = 5_000,
    progress=None,
) -> PilotPlan:
    """Select a deterministic QA pilot across source and language strata."""
    if pilot_documents <= 0:
        raise ValueError("pilot_documents must be positive")
    progress = progress or (lambda _message: None)
    heaps: dict[str, list[tuple[int, int, int]]] = {}
    available: Counter[str] = Counter()
    scanned = 0
    for number, spec in enumerate(context.content_files, 1):
        for batch in pq.ParquetFile(spec.documents_path).iter_batches(
            batch_size=batch_size, columns=_pilot_columns(spec.kind)
        ):
            scanned += batch.num_rows
            for row in batch.to_pylist():
                ident = int(row["id"])
                selected = context.selection.documents.get(ident)
                if selected is None or selected.content_group != spec.group:
                    continue
                detection = detect_document_language(
                    f"{row.get('title') or ''}\n{row['text']}", row.get("language_metadata")
                )
                stratum = f"{spec.kind}:{language_stratum(detection.language)}"
                available[stratum] += 1
                heap = heaps.setdefault(stratum, [])
                score = _pilot_score(ident)
                heapq.heappush(heap, (-score, -ident, ident))
                if len(heap) > pilot_documents:
                    heapq.heappop(heap)
        if number % 100 == 0 or number == len(context.content_files):
            progress(f"Planned pilot from {number}/{len(context.content_files)} files")
    if scanned != context.report.content_rows_scanned:
        raise Phase1IntegrityError("Pilot planning did not scan the preflight content row count")
    quotas = _allocate_pilot_quotas(dict(available), pilot_documents)
    ids: set[int] = set()
    selected_counts = {}
    for stratum, quota in quotas.items():
        candidates = sorted(
            ((-negative_score, ident) for negative_score, _negative_id, ident in heaps[stratum]),
            key=lambda item: (item[0], item[1]),
        )[:quota]
        ids.update(ident for _score, ident in candidates)
        selected_counts[stratum] = len(candidates)
    if len(ids) != pilot_documents:
        raise Phase1IntegrityError(
            f"Stratified pilot contains {len(ids)} unique IDs, expected {pilot_documents}"
        )
    ids_payload = "".join(f"{ident}\n" for ident in sorted(ids)).encode("ascii")
    return PilotPlan(
        ids=frozenset(ids),
        strategy=PILOT_SAMPLING_VERSION,
        ids_sha256=hashlib.sha256(ids_payload).hexdigest(),
        available_by_stratum=dict(sorted(available.items())),
        selected_by_stratum=dict(sorted(selected_counts.items())),
    )


def load_reusable_pilot_plan(
    context: PreflightContext,
    pilot: Path,
    pilot_documents: int,
) -> PilotPlan:
    """Reuse only the ID selection from a checksum-verified completed pilot."""
    pilot = pilot.resolve()
    try:
        relative = pilot.relative_to(context.paths.repo_root.resolve())
    except ValueError as exc:
        raise Phase1IntegrityError("Reusable pilot must be inside the repository") from exc
    manifest_path = pilot / "phase1_manifest.json"
    snapshot_path = pilot / "source_snapshot.json"
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise Phase1IntegrityError("Reusable pilot is missing manifest or source snapshot")
    manifest = _load_json(manifest_path)
    snapshot = _load_json(snapshot_path)
    if manifest.get("status") != "pilot_complete" or manifest.get("build_scope") != "pilot":
        raise Phase1IntegrityError("Reusable selection source is not a completed pilot")
    if int(manifest.get("documents", -1)) != pilot_documents:
        raise Phase1IntegrityError("Reusable pilot size differs from --pilot-documents")
    if snapshot.get("fingerprints") != context.report.fingerprints:
        raise Phase1IntegrityError("Reusable pilot source fingerprints differ from preflight")
    if manifest.get("source_snapshot_sha256") != sha256_file(snapshot_path):
        raise Phase1IntegrityError("Reusable pilot source snapshot checksum mismatch")
    ids = set()
    checkpoints = manifest.get("part_checkpoints_sha256", {})
    if len(checkpoints) != int(manifest.get("parts", -1)):
        raise Phase1IntegrityError("Reusable pilot checkpoint count mismatch")
    for stem, expected_hash in checkpoints.items():
        part = pilot / "clean_documents" / f"{stem}.parquet"
        if not part.is_file() or sha256_file(part) != expected_hash:
            raise Phase1IntegrityError(f"Reusable pilot part checksum mismatch: {part}")
        for ident in pq.read_table(part, columns=["doc_id"])["doc_id"].to_pylist():
            ident = int(ident)
            if ident in ids:
                raise Phase1IntegrityError(f"Duplicate reusable pilot ID: {ident}")
            if ident not in context.selection.documents:
                raise Phase1IntegrityError(f"Reusable pilot ID is outside Phase 0 allowlist: {ident}")
            ids.add(ident)
    if len(ids) != pilot_documents:
        raise Phase1IntegrityError("Reusable pilot unique ID count mismatch")
    ids_payload = "".join(f"{ident}\n" for ident in sorted(ids)).encode("ascii")
    ids_sha256 = hashlib.sha256(ids_payload).hexdigest()
    if manifest.get("pilot_ids_sha256") and manifest["pilot_ids_sha256"] != ids_sha256:
        raise Phase1IntegrityError("Reusable pilot ID checksum mismatch")
    return PilotPlan(
        ids=frozenset(ids),
        strategy=manifest.get("pilot_sampling", "reused-versioned-pilot-selection"),
        ids_sha256=ids_sha256,
        available_by_stratum=manifest.get("pilot_available_by_stratum", {}),
        selected_by_stratum=manifest.get("pilot_selected_by_stratum", {}),
        selection_source=str(relative),
        selection_source_manifest_sha256=sha256_file(manifest_path),
    )


def _clean_row(
    row: dict,
    *,
    spec: ContentFile,
    selected,
    repeated_count: int,
    repo_root: Path,
) -> dict:
    normalized = normalize_medical_text(row["text"])
    language = detect_document_language(
        f"{row.get('title') or ''}\n{row['text']}", row.get("language_metadata")
    )
    if normalized.source_text_sha256 != selected.text_sha256:
        raise Phase1IntegrityError(f"Text changed after preflight for ID {selected.doc_id}")
    quality = assess_text_quality(
        row["text"],
        title=row.get("title") or "",
        source_kind=spec.kind,
        encoding=row.get("encoding"),
        repeated_text_count=repeated_count,
    )
    if spec.kind == "html":
        raw_path = row.get("raw_path")
        raw_gzip_sha256 = row.get("gzip_sha256")
        raw_body_sha256 = row.get("raw_sha256")
    else:
        raw_path = row.get("raw_file")
        raw_gzip_sha256 = None
        raw_body_sha256 = None
    return {
        "doc_id": selected.doc_id,
        "official_url": selected.official_url,
        "source_kind": spec.kind,
        "content_group": spec.group,
        "title": row.get("title") or "",
        "source_text": normalized.source_text,
        "normalized_text": normalized.normalized_text,
        "source_text_sha256": normalized.source_text_sha256,
        "normalized_text_sha256": normalized.normalized_text_sha256,
        "normalization_version": normalized.version,
        "normalization_changed": normalized.changed,
        "quality_flags": list(quality.flags),
        "quality_reasons": list(quality.reasons),
        "quality_review_required": quality.review_required,
        "quality_validation_version": QUALITY_VALIDATION_VERSION,
        "phase0_raw_path": selected.phase0_raw_path,
        "phase0_raw_sha256": selected.phase0_raw_sha256,
        "raw_path": raw_path,
        "raw_gzip_sha256": raw_gzip_sha256,
        "raw_body_sha256": raw_body_sha256,
        "encoding": row.get("encoding"),
        "encoding_source": row.get("encoding_source"),
        "extraction_method": row.get("extraction_method"),
        "language_metadata": row.get("language_metadata"),
        "language_metadata_normalized": list(language.metadata_languages),
        "detected_language": language.language,
        "language_confidence": language.confidence,
        "language_detection_method": language.method,
        "language_detection_version": language.version,
        "language_metadata_agrees": language.metadata_agrees,
        "pmid": row.get("pmid"),
        "extractor_version": row.get("extractor_version"),
        "source_document_path": str(spec.documents_path.relative_to(repo_root)),
        "source_document_sha256": spec.documents_sha256,
    }


def _snapshot_payload(
    context: PreflightContext,
    pilot_documents: int | None = None,
    pilot_plan: PilotPlan | None = None,
) -> dict:
    payload = {
        "format": PHASE1_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "language_detection_version": LANGUAGE_DETECTION_VERSION,
        "quality_validation_version": QUALITY_VALIDATION_VERSION,
        "corpus_revision": context.expectations.revision,
        "expected_usable_documents": context.expectations.usable_rows,
        "content_files": len(context.content_files),
        "build_scope": "pilot" if pilot_documents is not None else "full",
        "pilot_documents": pilot_documents,
        "fingerprints": context.report.fingerprints,
    }
    if pilot_plan is not None:
        payload["pilot_sampling"] = pilot_plan.strategy
        payload["pilot_ids_sha256"] = pilot_plan.ids_sha256
        payload["pilot_available_by_stratum"] = pilot_plan.available_by_stratum
        payload["pilot_selected_by_stratum"] = pilot_plan.selected_by_stratum
        if pilot_plan.selection_source is not None:
            payload["pilot_selection_source"] = pilot_plan.selection_source
            payload["pilot_selection_source_manifest_sha256"] = \
                pilot_plan.selection_source_manifest_sha256
    return payload


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase1IntegrityError(f"Expected JSON object: {path}")
    return value


def build_corpus(
    context: PreflightContext,
    output: Path,
    *,
    batch_size: int = 5_000,
    resume: bool = False,
    pilot_documents: int | None = None,
    reuse_pilot_selection: Path | None = None,
    progress=None,
) -> dict:
    """Write clean parts only after the complete preflight has succeeded."""
    if pilot_documents is not None and pilot_documents <= 0:
        raise ValueError("pilot_documents must be positive")
    progress = progress or (lambda _message: None)
    output = ensure_output_is_safe(output, context.paths)
    pilot_plan = None
    if reuse_pilot_selection is not None and pilot_documents is None:
        raise ValueError("reuse_pilot_selection requires pilot_documents")
    if pilot_documents is not None:
        pilot_plan = (
            load_reusable_pilot_plan(context, reuse_pilot_selection, pilot_documents)
            if reuse_pilot_selection is not None
            else plan_stratified_pilot(
                context, pilot_documents, batch_size=batch_size, progress=progress
            )
        )
    snapshot = _snapshot_payload(context, pilot_documents, pilot_plan)
    snapshot_path = output / "source_snapshot.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Versioned Phase 1 output already exists: {output}")
        if not snapshot_path.is_file() or _load_json(snapshot_path) != snapshot:
            raise Phase1IntegrityError("Resume source snapshot differs from current preflight")
    else:
        (output / "clean_documents").mkdir(parents=True)
        (output / "checkpoints").mkdir()
        (output / "reports").mkdir()
        _atomic_json(snapshot_path, snapshot)
    snapshot_sha256 = sha256_file(snapshot_path)

    hash_counts = Counter(record.text_sha256 for record in context.selection.documents.values())
    repeated = {
        digest: count for digest, count in hash_counts.items()
        if count >= REPEATED_TEMPLATE_THRESHOLD
    }
    del hash_counts
    written_ids: set[int] = set()
    total_rows = 0
    quality_counts: Counter[str] = Counter()
    quality_review_rows = 0
    part_entries = []
    for number, spec in enumerate(context.content_files, 1):
        if pilot_plan is not None and len(written_ids) >= len(pilot_plan.ids):
            break
        stem = f"part-{spec.index:05d}"
        destination = output / "clean_documents" / f"{stem}.parquet"
        checkpoint_path = output / "checkpoints" / f"{stem}.json"
        if checkpoint_path.exists():
            if not resume:
                raise FileExistsError(checkpoint_path)
            checkpoint = _load_json(checkpoint_path)
            if (
                checkpoint.get("input_documents_sha256") != spec.documents_sha256
                or checkpoint.get("source_snapshot_sha256") != snapshot_sha256
                or checkpoint.get("output_sha256") != sha256_file(destination)
            ):
                raise Phase1IntegrityError(f"Resume checkpoint mismatch: {checkpoint_path}")
            ids = pq.read_table(destination, columns=["doc_id"])["doc_id"].to_pylist()
            duplicate = written_ids.intersection(ids)
            if duplicate:
                raise Phase1IntegrityError(f"Duplicate IDs across resumed parts: {sorted(duplicate)[:5]}")
            written_ids.update(map(int, ids))
            total_rows += int(checkpoint["rows"])
            quality_counts.update(checkpoint.get("quality_flag_counts", {}))
            quality_review_rows += int(checkpoint.get("quality_review_rows", 0))
            part_entries.append(checkpoint)
            continue
        if destination.exists():
            raise Phase1IntegrityError(f"Output part exists without checkpoint: {destination}")

        rows = []
        for batch in pq.ParquetFile(spec.documents_path).iter_batches(
            batch_size=batch_size, columns=_source_columns(spec.kind)
        ):
            for row in batch.to_pylist():
                ident = int(row["id"])
                selected = context.selection.documents.get(ident)
                if selected is None or selected.content_group != spec.group:
                    continue
                if pilot_plan is not None and ident not in pilot_plan.ids:
                    continue
                if ident in written_ids:
                    raise Phase1IntegrityError(f"Duplicate canonical ID during build: {ident}")
                if row["official_url"] != selected.official_url:
                    raise Phase1IntegrityError(f"URL changed after preflight for ID {ident}")
                clean = _clean_row(
                    row,
                    spec=spec,
                    selected=selected,
                    repeated_count=repeated.get(selected.text_sha256, 1),
                    repo_root=context.paths.repo_root,
                )
                rows.append(clean)
                written_ids.add(ident)
                quality_counts.update(clean["quality_flags"])
                quality_review_rows += int(clean["quality_review_required"])
            if pilot_plan is not None and len(written_ids) >= len(pilot_plan.ids):
                break
        table = pa.Table.from_pylist(rows, schema=CLEAN_DOCUMENT_SCHEMA)
        _atomic_parquet(destination, table)
        checkpoint = {
            "part": spec.index,
            "input_documents": str(spec.documents_path.relative_to(context.paths.repo_root)),
            "input_documents_sha256": spec.documents_sha256,
            "source_snapshot_sha256": snapshot_sha256,
            "output": str(destination.relative_to(context.paths.repo_root)),
            "output_sha256": sha256_file(destination),
            "rows": len(rows),
            "first_doc_id": min((row["doc_id"] for row in rows), default=None),
            "last_doc_id": max((row["doc_id"] for row in rows), default=None),
            "quality_review_rows": sum(row["quality_review_required"] for row in rows),
            "quality_flag_counts": dict(sorted(Counter(
                flag for row in rows for flag in row["quality_flags"]
            ).items())),
        }
        _atomic_json(checkpoint_path, checkpoint)
        total_rows += len(rows)
        part_entries.append(checkpoint)
        if number % 50 == 0 or number == len(context.content_files):
            progress(f"Built {number}/{len(context.content_files)} parts; {total_rows:,} rows")

    expected_rows = min(pilot_documents, context.expectations.usable_rows) \
        if pilot_documents is not None else context.expectations.usable_rows
    if total_rows != expected_rows:
        raise Phase1IntegrityError(
            f"Built {total_rows} rows, expected {expected_rows}"
        )
    if len(written_ids) != expected_rows:
        raise Phase1IntegrityError("Built corpus does not contain the expected number of unique IDs")
    if pilot_plan is not None and written_ids != set(pilot_plan.ids):
        raise Phase1IntegrityError("Built pilot IDs differ from the stratified pilot plan")
    if pilot_documents is None and set(context.selection.documents).difference(written_ids):
        raise Phase1IntegrityError("Built corpus is missing allowlisted IDs")

    manifest = {
        **snapshot,
        "status": "pilot_complete" if pilot_documents is not None else "complete",
        "documents": total_rows,
        "unique_document_ids": len(written_ids),
        "parts": len(part_entries),
        "quality_review_documents": quality_review_rows,
        "quality_flag_counts": dict(sorted(quality_counts.items())),
        "source_snapshot_sha256": snapshot_sha256,
        "part_checkpoints_sha256": {
            f"part-{entry['part']:05d}": entry["output_sha256"] for entry in part_entries
        },
    }
    manifest_path = output / "phase1_manifest.json"
    if manifest_path.exists():
        if not resume or _load_json(manifest_path) != manifest:
            raise Phase1IntegrityError("Existing Phase 1 manifest differs from rebuilt state")
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--dry-run", action="store_true", help="Run full preflight and write nothing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pilot-documents", type=int,
        help="After full preflight, write only this many deterministic canonical rows",
    )
    parser.add_argument(
        "--reuse-pilot-selection", type=Path,
        help="Reuse IDs from a checksum-verified completed pilot after full source preflight",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.pilot_documents is not None and args.pilot_documents <= 0:
        raise SystemExit("--pilot-documents must be positive")
    if args.dry_run and args.pilot_documents is not None:
        raise SystemExit("--pilot-documents cannot be combined with --dry-run")
    if args.reuse_pilot_selection is not None and args.pilot_documents is None:
        raise SystemExit("--reuse-pilot-selection requires --pilot-documents")
    paths = Phase1Paths.from_repo(args.root)
    try:
        context = run_preflight(
            paths,
            batch_size=args.batch_size,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        if args.dry_run:
            print(json.dumps({"status": "dry_run_complete", **context.report.to_dict()},
                             ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        result = build_corpus(
            context, args.output, batch_size=args.batch_size, resume=args.resume,
            pilot_documents=args.pilot_documents,
            reuse_pilot_selection=args.reuse_pilot_selection,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Phase 1 failed before completion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
