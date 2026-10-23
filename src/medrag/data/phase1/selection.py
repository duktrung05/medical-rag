"""Fail-closed selection of the audited ViBioMIR Phase 1 corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from .normalization import text_sha256
from .validation import assess_text_quality


REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
OFFICIAL_CORPUS_SHA256 = "87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7"
OFFICIAL_QUERY_SHA256 = "ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08"
PHASE0_STATE_SHA256 = "a8df05bda42862d643cc428c79e6237f43b5bab91c72f608b7b21aab1579462b"
RELEASE_STATUS_SHA256 = "98914ab9015399e7bc5c7336da564639e7fbcfe80a755d1d28667de6f14cb2f9"
STAGING_RECOVERY_IDS = frozenset({
    82086, 106695, 106840, 107200, 108063,
    108227, 108264, 783862, 869573, 876706,
})
PRODUCTION_STATUS_COUNTS = {
    "CRAWLED_SUCCESS": 126,
    "CRAWL_FAILED": 552,
    "DEFERRED": 134_241,
    "EXTRACTION_FAILED": 266,
    "REVIEW_REQUIRED": 9_814,
    "USABLE": 1_075_599,
}


class Phase1IntegrityError(ValueError):
    """An audited input no longer satisfies the locked Phase 1 contract."""


@dataclass(frozen=True)
class Phase1Paths:
    repo_root: Path
    source_lock: Path
    audit_manifest: Path
    phase0_state: Path
    official_corpus: Path
    official_query: Path
    release_manifest: Path
    release_status: Path

    @classmethod
    def from_repo(cls, repo_root: Path) -> "Phase1Paths":
        root = repo_root.resolve()
        phase0 = root / "outputs/vibiomir_phase0_20261014"
        official = root / "outputs/vibiomir_official_verify_20261012"
        release = root / "outputs/vibiomir_step2_20261013/release"
        return cls(
            repo_root=root,
            source_lock=phase0 / "source_lock.json",
            audit_manifest=phase0 / "audit/audit_manifest.json",
            phase0_state=phase0 / "audit/document_state.parquet",
            official_corpus=official / "links_corpus_vi_en_zh.parquet",
            official_query=official / "query.parquet",
            release_manifest=release / "release_manifest.json",
            release_status=release / "document_status.parquet",
        )


@dataclass(frozen=True)
class ProductionExpectations:
    revision: str = REVISION
    official_corpus_sha256: str = OFFICIAL_CORPUS_SHA256
    official_query_sha256: str = OFFICIAL_QUERY_SHA256
    phase0_state_sha256: str = PHASE0_STATE_SHA256
    release_status_sha256: str = RELEASE_STATUS_SHA256
    official_rows: int = 1_220_598
    query_rows: int = 1_200
    usable_rows: int = 1_075_599
    review_rows: int = 9_814
    release_usable_rows: int = 1_085_413
    content_files: int = 922
    status_counts: dict[str, int] = field(
        default_factory=lambda: dict(PRODUCTION_STATUS_COUNTS)
    )
    staging_ids: frozenset[int] = STAGING_RECOVERY_IDS


@dataclass
class SelectedDocument:
    doc_id: int
    official_url: str
    content_group: str
    text_sha256: str
    phase0_raw_path: str | None
    phase0_raw_sha256: str | None
    release_source_kind: str | None = None
    release_seen: bool = False


@dataclass(frozen=True)
class ContentFile:
    index: int
    group: str
    kind: str
    documents_path: Path
    documents_rows: int
    documents_sha256: str


@dataclass
class SelectionState:
    documents: dict[int, SelectedDocument]
    status_counts: dict[str, int]
    fingerprints: dict[str, str]


@dataclass(frozen=True)
class PreflightReport:
    official_rows: int
    phase0_rows: int
    usable_rows: int
    review_rows: int
    staging_usable_intersection: int
    content_files_verified: int
    content_rows_scanned: int
    canonical_rows_verified: int
    repeated_template_hashes: int
    repeated_template_documents: int
    quality_review_documents: int
    quality_flag_counts: dict[str, int]
    fingerprints: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Phase1IntegrityError(f"Cannot read locked JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase1IntegrityError(f"Locked JSON must contain an object: {path}")
    return value


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise Phase1IntegrityError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise Phase1IntegrityError(
            f"{label} checksum mismatch: expected {expected}, got {actual}: {path}"
        )
    return actual


def validate_locked_inputs(
    paths: Phase1Paths,
    expectations: ProductionExpectations,
) -> tuple[dict[str, str], dict, dict]:
    """Verify all small/ledger locks before loading selection state."""
    lock = _read_json(paths.source_lock)
    audit = _read_json(paths.audit_manifest)
    release = _read_json(paths.release_manifest)
    if lock.get("revision") != expectations.revision:
        raise Phase1IntegrityError("source_lock revision differs from the expected revision")
    if audit.get("revision") != expectations.revision:
        raise Phase1IntegrityError("Phase 0 audit revision differs from source_lock")
    if release.get("corpus_revision") != expectations.revision or not release.get("complete"):
        raise Phase1IntegrityError("Step 2 release is incomplete or uses another revision")
    if audit.get("official_ids") != expectations.official_rows:
        raise Phase1IntegrityError("Phase 0 audit official row count is unexpected")
    if audit.get("queries") != expectations.query_rows:
        raise Phase1IntegrityError("Phase 0 audit query row count is unexpected")
    if audit.get("status_counts") != expectations.status_counts:
        raise Phase1IntegrityError("Phase 0 audit status counts are unexpected")
    if audit.get("staged_recovery_merged") is not False:
        raise Phase1IntegrityError("Phase 0 audit reports staging recovery as merged")
    if set(audit.get("staged_recovery_id_list", [])) != set(expectations.staging_ids):
        raise Phase1IntegrityError("Phase 0 staging recovery ID set is unexpected")
    if release.get("official_ids") != expectations.official_rows:
        raise Phase1IntegrityError("Step 2 release official row count is unexpected")
    if release.get("usable_documents") != expectations.release_usable_rows:
        raise Phase1IntegrityError("Step 2 release usable count is unexpected")

    lock_files = lock.get("files", {})
    corpus_lock = lock_files.get("links_corpus_vi_en_zh.parquet", {})
    query_lock = lock_files.get("query.parquet", {})
    if corpus_lock.get("sha256") != expectations.official_corpus_sha256:
        raise Phase1IntegrityError("source_lock official corpus checksum is unexpected")
    if query_lock.get("sha256") != expectations.official_query_sha256:
        raise Phase1IntegrityError("source_lock official query checksum is unexpected")
    if audit.get("official_corpus_sha256") != expectations.official_corpus_sha256:
        raise Phase1IntegrityError("audit manifest official corpus checksum is unexpected")
    if audit.get("official_query_sha256") != expectations.official_query_sha256:
        raise Phase1IntegrityError("audit manifest official query checksum is unexpected")
    state_artifact = audit.get("artifacts", {}).get("document_state.parquet", {})
    if state_artifact.get("sha256") != expectations.phase0_state_sha256:
        raise Phase1IntegrityError("audit manifest Phase 0 state checksum is unexpected")
    release_status = release.get("document_status", {})
    if release_status.get("sha256") != expectations.release_status_sha256:
        raise Phase1IntegrityError("release document_status checksum is unexpected")

    fingerprints = {
        "source_lock_sha256": sha256_file(paths.source_lock),
        "audit_manifest_sha256": sha256_file(paths.audit_manifest),
        "release_manifest_sha256": sha256_file(paths.release_manifest),
        "official_corpus_sha256": _require_hash(
            paths.official_corpus, expectations.official_corpus_sha256, "official corpus"
        ),
        "official_query_sha256": _require_hash(
            paths.official_query, expectations.official_query_sha256, "official query"
        ),
        "phase0_state_sha256": _require_hash(
            paths.phase0_state, expectations.phase0_state_sha256, "Phase 0 state"
        ),
        "release_status_sha256": _require_hash(
            paths.release_status, expectations.release_status_sha256, "release status"
        ),
    }
    return fingerprints, audit, release


def load_selection(
    paths: Phase1Paths,
    expectations: ProductionExpectations,
    fingerprints: dict[str, str],
) -> SelectionState:
    """Build the exact Phase 0 USABLE allowlist and validate release mapping."""
    official_file = pq.ParquetFile(paths.official_corpus)
    if official_file.metadata.num_rows != expectations.official_rows:
        raise Phase1IntegrityError("Official corpus row count differs from the locked value")
    official: dict[int, str] = {}
    for batch in official_file.iter_batches(batch_size=100_000, columns=["id", "url"]):
        values = batch.to_pydict()
        for ident, url in zip(values["id"], values["url"], strict=True):
            ident = int(ident)
            if ident in official:
                raise Phase1IntegrityError(f"Duplicate official document ID: {ident}")
            if not isinstance(url, str) or not url:
                raise Phase1IntegrityError(f"Blank official URL for document {ident}")
            official[ident] = url
    if len(official) != expectations.official_rows:
        raise Phase1IntegrityError("Official corpus unique ID count is incorrect")

    query_file = pq.ParquetFile(paths.official_query)
    if query_file.metadata.num_rows != expectations.query_rows:
        raise Phase1IntegrityError("Official query row count differs from the locked value")

    phase0_file = pq.ParquetFile(paths.phase0_state)
    if phase0_file.metadata.num_rows != expectations.official_rows:
        raise Phase1IntegrityError("Phase 0 state does not cover the official corpus")
    columns = [
        "doc_id", "official_url", "phase0_status", "usable", "content_group",
        "text_sha256", "raw_path", "raw_sha256",
    ]
    selected: dict[int, SelectedDocument] = {}
    counts: Counter[str] = Counter()
    staging_usable: set[int] = set()
    for batch in phase0_file.iter_batches(batch_size=50_000, columns=columns):
        for row in batch.to_pylist():
            ident = int(row["doc_id"])
            expected_url = official.pop(ident, None)
            if expected_url is None:
                raise Phase1IntegrityError(f"Duplicate or non-official Phase 0 ID: {ident}")
            if row["official_url"] != expected_url:
                raise Phase1IntegrityError(f"Exact official URL mismatch for document {ident}")
            status = row["phase0_status"]
            counts[status] += 1
            if status != "USABLE":
                continue
            if row["usable"] is not True:
                raise Phase1IntegrityError(f"USABLE row has false usable flag: {ident}")
            if ident in expectations.staging_ids:
                staging_usable.add(ident)
            if not row["content_group"] or not row["text_sha256"]:
                raise Phase1IntegrityError(f"USABLE row lacks canonical provenance: {ident}")
            selected[ident] = SelectedDocument(
                doc_id=ident,
                official_url=row["official_url"],
                content_group=row["content_group"],
                text_sha256=row["text_sha256"],
                phase0_raw_path=row["raw_path"],
                phase0_raw_sha256=row["raw_sha256"],
            )
    if official:
        sample = sorted(official)[:5]
        raise Phase1IntegrityError(f"Phase 0 state misses official IDs, e.g. {sample}")
    if dict(sorted(counts.items())) != dict(sorted(expectations.status_counts.items())):
        raise Phase1IntegrityError(
            f"Phase 0 status counts differ: {dict(sorted(counts.items()))}"
        )
    if len(selected) != expectations.usable_rows:
        raise Phase1IntegrityError(
            f"USABLE unique ID count is {len(selected)}, expected {expectations.usable_rows}"
        )
    if counts["REVIEW_REQUIRED"] != expectations.review_rows:
        raise Phase1IntegrityError("REVIEW_REQUIRED count differs from the locked value")
    if staging_usable:
        raise Phase1IntegrityError(
            f"Staging recovery IDs entered Phase 1 allowlist: {sorted(staging_usable)}"
        )

    release_file = pq.ParquetFile(paths.release_status)
    if release_file.metadata.num_rows != expectations.official_rows:
        raise Phase1IntegrityError("Release status does not cover the official corpus")
    release_seen = 0
    for batch in release_file.iter_batches(
        batch_size=100_000,
        columns=["id", "official_url", "source_kind", "usable", "content_group"],
    ):
        for row in batch.to_pylist():
            record = selected.get(int(row["id"]))
            if record is None:
                continue
            if record.release_seen:
                raise Phase1IntegrityError(f"Duplicate release status for ID {record.doc_id}")
            if row["official_url"] != record.official_url:
                raise Phase1IntegrityError(f"Release URL mismatch for ID {record.doc_id}")
            if row["usable"] is not True:
                raise Phase1IntegrityError(f"Phase 0 USABLE ID is not usable in release: {record.doc_id}")
            if row["content_group"] != record.content_group:
                raise Phase1IntegrityError(f"Canonical content_group mismatch for ID {record.doc_id}")
            record.release_source_kind = row["source_kind"]
            record.release_seen = True
            release_seen += 1
    if release_seen != expectations.usable_rows:
        raise Phase1IntegrityError(
            f"Release mapping covers {release_seen} allowlisted IDs, expected {expectations.usable_rows}"
        )
    return SelectionState(selected, dict(sorted(counts.items())), dict(fingerprints))


def load_content_files(
    paths: Phase1Paths,
    release: dict,
    expectations: ProductionExpectations,
) -> list[ContentFile]:
    specs = release.get("content_files")
    if not isinstance(specs, list) or len(specs) != expectations.content_files:
        raise Phase1IntegrityError(
            f"Release lists {len(specs) if isinstance(specs, list) else 'invalid'} content files; "
            f"expected {expectations.content_files}"
        )
    result: list[ContentFile] = []
    seen_paths: set[Path] = set()
    for index, spec in enumerate(specs):
        try:
            path = (paths.repo_root / spec["documents"]).resolve()
            path.relative_to(paths.repo_root)
        except (KeyError, ValueError) as exc:
            raise Phase1IntegrityError(f"Unsafe content path in release entry {index}") from exc
        if path in seen_paths:
            raise Phase1IntegrityError(f"Duplicate content file in release: {path}")
        if spec.get("kind") not in {"html", "pubmed"}:
            raise Phase1IntegrityError(f"Unknown content kind in release entry {index}")
        seen_paths.add(path)
        result.append(ContentFile(
            index=index,
            group=spec["group"],
            kind=spec["kind"],
            documents_path=path,
            documents_rows=int(spec["documents_rows"]),
            documents_sha256=spec["documents_sha256"],
        ))
    return result


def _preflight_columns(kind: str) -> list[str]:
    common = ["id", "official_url", "title", "text", "text_sha256"]
    return common + (["status", "encoding"] if kind == "html" else [])


def preflight_canonical_content(
    selection: SelectionState,
    content_files: list[ContentFile],
    *,
    batch_size: int = 5_000,
    progress=None,
) -> PreflightReport:
    """Verify every canonical row and source file without writing output."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    progress = progress or (lambda _message: None)
    hash_counts = Counter(record.text_sha256 for record in selection.documents.values())
    repeated = {digest: count for digest, count in hash_counts.items() if count >= 20}
    del hash_counts

    seen: set[int] = set()
    rows_scanned = 0
    quality_counts: Counter[str] = Counter()
    quality_review = 0
    for number, spec in enumerate(content_files, 1):
        _require_hash(spec.documents_path, spec.documents_sha256, "canonical content file")
        parquet = pq.ParquetFile(spec.documents_path)
        if parquet.metadata.num_rows != spec.documents_rows:
            raise Phase1IntegrityError(f"Content row count mismatch: {spec.documents_path}")
        required = _preflight_columns(spec.kind)
        missing = set(required).difference(parquet.schema_arrow.names)
        if missing:
            raise Phase1IntegrityError(
                f"Content schema is missing {sorted(missing)}: {spec.documents_path}"
            )
        for batch in parquet.iter_batches(batch_size=batch_size, columns=required):
            rows_scanned += batch.num_rows
            for row in batch.to_pylist():
                ident = int(row["id"])
                record = selection.documents.get(ident)
                if record is None or record.content_group != spec.group:
                    continue
                if ident in seen:
                    raise Phase1IntegrityError(f"Multiple canonical content rows for ID {ident}")
                if row["official_url"] != record.official_url:
                    raise Phase1IntegrityError(f"Canonical content URL mismatch for ID {ident}")
                if spec.kind == "html" and row["status"] != "usable":
                    raise Phase1IntegrityError(f"Canonical HTML row is not usable for ID {ident}")
                text = row["text"]
                if not isinstance(text, str):
                    raise Phase1IntegrityError(f"Canonical text is not a string for ID {ident}")
                actual_text_hash = text_sha256(text)
                if row["text_sha256"] != actual_text_hash:
                    raise Phase1IntegrityError(f"Document text checksum mismatch for ID {ident}")
                if record.text_sha256 != actual_text_hash:
                    raise Phase1IntegrityError(f"Phase 0 text checksum mismatch for ID {ident}")
                assessment = assess_text_quality(
                    text,
                    title=row.get("title") or "",
                    source_kind=spec.kind,
                    encoding=row.get("encoding"),
                    repeated_text_count=repeated.get(actual_text_hash, 1),
                )
                quality_counts.update(assessment.flags)
                quality_review += int(assessment.review_required)
                seen.add(ident)
        if number % 50 == 0 or number == len(content_files):
            progress(
                f"Verified {number}/{len(content_files)} content files; "
                f"{len(seen):,}/{len(selection.documents):,} canonical rows"
            )
    if len(seen) != len(selection.documents):
        missing_ids = sorted(set(selection.documents).difference(seen))[:10]
        raise Phase1IntegrityError(
            f"Canonical content covers {len(seen)} allowlisted IDs; missing e.g. {missing_ids}"
        )
    repeated_documents = sum(repeated.values())
    return PreflightReport(
        official_rows=sum(selection.status_counts.values()),
        phase0_rows=sum(selection.status_counts.values()),
        usable_rows=len(selection.documents),
        review_rows=selection.status_counts.get("REVIEW_REQUIRED", 0),
        staging_usable_intersection=0,
        content_files_verified=len(content_files),
        content_rows_scanned=rows_scanned,
        canonical_rows_verified=len(seen),
        repeated_template_hashes=len(repeated),
        repeated_template_documents=repeated_documents,
        quality_review_documents=quality_review,
        quality_flag_counts=dict(sorted(quality_counts.items())),
        fingerprints=dict(selection.fingerprints),
    )
