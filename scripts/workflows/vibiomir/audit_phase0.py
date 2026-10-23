"""Freeze and independently audit the pinned ViBioMIR crawl snapshot.

This command reads existing source data and writes only to a new output folder.
CRAWLED_SUCCESS is a fetch/content flag; USABLE and EXTRACTION_FAILED are
subsequent outcomes. Deferred URLs are never described as attempted failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
STEP1 = ROOT / "outputs/vibiomir_step1_20261013/manifest.json"
RELEASE = ROOT / "outputs/vibiomir_step2_20261013/release/release_manifest.json"
RECOVERY = ROOT / "outputs/crawl_failure_recovery/RECOVERY_RESULTS.parquet"
DEFAULT_OUTPUT = ROOT / "outputs/vibiomir_phase0_20261014/audit"

ERROR_TITLE = re.compile(
    r"^(?:just a moment|access denied|403 forbidden|404 not found|page not found|"
    r"không tìm thấy trang|页面不存在|"
    r"captcha|security check|verify you are human|truy cập bị từ chối|"
    r"xác minh bạn là con người|访问被拒绝|安全验证)\b", re.I)
HOMEPAGE_TITLE = re.compile(r"^(?:trang chủ|home(?:page)?|首页)(?:\s*[|:\-–]\s*.+)?$", re.I)
LOGIN_TITLE = re.compile(r"^(?:đăng nhập|log[ -]?in|登录)(?:\s*[|:\-–]\s*.+)?$", re.I)
HEX_NOISE = re.compile(r"^(?:[0-9a-f]{24,}\s*){2,}", re.I)
ERROR_BODY = re.compile(
    r"(?:checking your browser before accessing|enable javascript and cookies to continue|"
    r"cf-challenge|cloudflare ray id|verify you are human|"
    r"请完成安全验证|访问被拒绝)", re.I)

SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("official_url", pa.string()),
    ("final_status", pa.string()), ("content_group", pa.string()),
    ("crawled_success", pa.bool_()), ("usable", pa.bool_()),
    ("extraction_failed", pa.bool_()), ("crawl_failed", pa.bool_()),
    ("deferred", pa.bool_()), ("review_required", pa.bool_()),
    ("phase0_status", pa.string()),
    ("reason", pa.string()), ("raw_path", pa.string()),
    ("raw_sha256", pa.string()), ("text_sha256", pa.string()),
    ("raw_hash_verified", pa.bool_()), ("source", pa.string()),
])


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_source(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Source checksum changed: {path}")


def load_official(path: Path) -> dict[int, str]:
    official = {}
    urls = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=["id", "url"]):
        for ident, url in zip(*batch.to_pydict().values(), strict=True):
            if ident in official or url in urls or not url:
                raise ValueError(f"Duplicate or blank official ID/URL: {ident}")
            official[ident] = url
            urls.add(url)
    return official


def validate_text(row: dict, kind: str) -> str | None:
    text = row.get("text") or ""
    title = row.get("title") or ""
    digest = row.get("text_sha256")
    if not text.strip() or not digest or hashlib.sha256(text.encode()).hexdigest() != digest:
        return "missing_or_mismatched_text"
    if kind == "html" and len(text.strip()) < 80:
        return "short_extracted_text"
    if kind == "html" and HOMEPAGE_TITLE.search(title.strip()):
        return "homepage_or_redirect_content"
    if kind == "html" and LOGIN_TITLE.search(title.strip()):
        return "login_or_access_gate_content"
    if kind == "html" and HEX_NOISE.search(text.strip()[:300]):
        return "hex_or_asset_noise"
    if ERROR_TITLE.search(title.strip()) or ERROR_BODY.search(text[:1000]):
        return "error_or_challenge_content"
    return None


def validate_raw(row: dict, kind: str, cache: dict[str, str], *, deep: bool) -> tuple[str | None, bool]:
    name = row.get("raw_path") if kind == "html" else row.get("raw_file")
    if not name:
        return "missing_raw_path", False
    expected = (row.get("gzip_sha256") or row.get("raw_sha256")) if kind == "html" else row.get("raw_sha256")
    if not expected:
        return "missing_raw_checksum", False
    # The complete raw tree is archived and its file membership is verified
    # separately. Random reads of ~833k small files make the audit impractical.
    if not deep:
        return None, False
    path = Path(name)
    if not path.is_file():
        return "missing_raw_file", True
    # PubMed records share raw XML batches; each unique batch is read once.
    actual = cache.get(name) if kind == "pubmed" else None
    if actual is None:
        actual = sha256(path)
        if kind == "pubmed":
            cache[name] = actual
    return (None if actual == expected else "raw_checksum_mismatch"), True


def classify(record: dict, evidence: dict | None) -> dict:
    status = record["final_status"]
    reason = evidence.get("issue") if evidence else None
    raw_ok = bool(evidence and evidence["raw_ok"])
    text_ok = bool(evidence and evidence["text_ok"])
    usable = bool(record["usable"] and raw_ok and text_ok)
    if usable:
        label = "USABLE"
    elif reason and record["usable"]:
        label = "REVIEW_REQUIRED" if raw_ok else "CRAWL_FAILED"
    elif status in {"extract_failed", "quality_review_empty", "quality_review_short"}:
        label = "EXTRACTION_FAILED" if raw_ok else "CRAWL_FAILED"
    elif status == "metadata_only":
        label = "CRAWLED_SUCCESS" if raw_ok else "CRAWL_FAILED"
    elif status.startswith("domain_deferred_"):
        label = "DEFERRED"
    elif status == "unavailable" or status.startswith("fetch_"):
        label = "CRAWL_FAILED"
    else:
        label = "REVIEW_REQUIRED"
    return {"doc_id": record["id"], "official_url": record["official_url"],
            "final_status": status, "content_group": record["content_group"],
            "crawled_success": raw_ok, "usable": usable,
            "extraction_failed": label == "EXTRACTION_FAILED",
            "crawl_failed": label == "CRAWL_FAILED", "deferred": label == "DEFERRED",
            "review_required": label == "REVIEW_REQUIRED",
            "phase0_status": label, "reason": reason or record["reason"] or status,
            "raw_path": evidence.get("raw_path") if evidence else None,
            "raw_sha256": evidence.get("raw_sha256") if evidence else None,
            "text_sha256": evidence.get("text_sha256") if evidence else None,
            "raw_hash_verified": bool(evidence and evidence["raw_hash_verified"]),
            "source": "canonical_release"}


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new audit output: {output}")
    step1 = json.loads(STEP1.read_text())
    release = json.loads(RELEASE.read_text())
    revision = step1["revision"]
    if release["corpus_revision"] != revision or not release["complete"]:
        raise ValueError("Canonical release is incomplete or has another revision")
    corpus = Path(step1["files"]["links_corpus_vi_en_zh.parquet"]["path"])
    query = Path(step1["files"]["query.parquet"]["path"])
    for path, expected in ((corpus, step1["files"]["links_corpus_vi_en_zh.parquet"]["sha256"]),
                           (query, step1["files"]["query.parquet"]["sha256"]),
                           (ROOT / release["document_status"]["path"], release["document_status"]["sha256"])):
        check_source(path, expected)
    official = load_official(corpus)
    if len(official) != 1_220_598 or pq.ParquetFile(query).metadata.num_rows != 1200:
        raise ValueError("Official corpus/query count differs from the pinned snapshot")
    states = {}
    status_file = ROOT / release["document_status"]["path"]
    for batch in pq.ParquetFile(status_file).iter_batches(batch_size=50_000):
        for row in batch.to_pylist():
            ident = row["id"]
            if ident in states or official.get(ident) != row["official_url"]:
                raise ValueError(f"Duplicate or mismatched release ID: {ident}")
            states[ident] = row
    if len(states) != len(official):
        raise ValueError("Release does not cover the official ID set")

    evidence = {}
    raw_cache = {}
    scanned = 0
    source_files = 0
    raw_verified_rows = 0
    for spec in release["content_files"]:
        path = ROOT / spec["documents"]
        check_source(path, spec["documents_sha256"])
        source_files += 1
        kind = spec["kind"]
        first_canonical = True
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                scanned += 1
                ident = row["id"]
                state = states.get(ident)
                if state is None or official[ident] != row["official_url"]:
                    raise ValueError(f"Content ID/URL outside official snapshot: {ident}")
                if state["content_group"] != spec["group"]:
                    continue
                if ident in evidence:
                    raise ValueError(f"Multiple canonical content rows: {ident}")
                deep = first_canonical or not state["usable"] or ident % 997 == 0
                first_canonical = False
                raw_issue, raw_checked = validate_raw(row, kind, raw_cache, deep=deep)
                raw_verified_rows += int(raw_checked and raw_issue is None)
                text_issue = validate_text(row, kind) if state["usable"] else None
                evidence[ident] = {"raw_ok": raw_issue is None,
                                   "text_ok": text_issue is None if state["usable"] else False,
                                   "issue": raw_issue or text_issue,
                                   "raw_path": row.get("raw_path") or row.get("raw_file"),
                                   "raw_sha256": row.get("gzip_sha256") or row.get("raw_sha256"),
                                   "text_sha256": row.get("text_sha256"),
                                   "raw_hash_verified": raw_checked}

    # Metadata-only PubMed records have no materialized content row, but their
    # source export retains the raw XML path and checksum.
    metadata_ids = {ident for ident, row in states.items() if row["final_status"] == "metadata_only"}
    if metadata_ids:
        pubmed_dir = ROOT / "data/vibiomir/pubmed" / revision / "run_20261013_2gpu/documents"
        for path in sorted(pubmed_dir.glob("part_*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=5000,
                    columns=["id", "official_url", "raw_file", "raw_sha256"]):
                for row in batch.to_pylist():
                    ident = row["id"]
                    if ident not in metadata_ids:
                        continue
                    if official[ident] != row["official_url"]:
                        raise ValueError(f"PubMed metadata URL mismatch: {ident}")
                    issue, raw_checked = validate_raw(row, "pubmed", raw_cache, deep=True)
                    raw_verified_rows += int(raw_checked and issue is None)
                    evidence[ident] = {"raw_ok": issue is None, "text_ok": False,
                                       "issue": issue, "raw_path": row["raw_file"],
                                       "raw_sha256": row["raw_sha256"], "text_sha256": None,
                                       "raw_hash_verified": raw_checked}

    # A URL-specific article cannot have the exact same extracted body as
    # dozens of unrelated URLs. These groups are set aside for review; the
    # raw files and previous extraction remain untouched.
    duplicate_text = Counter(item["text_sha256"] for ident, item in evidence.items()
                             if states[ident]["usable"] and item["text_ok"]
                             and item["text_sha256"])
    duplicate_review = 0
    for ident, item in evidence.items():
        if (states[ident]["usable"] and item["text_ok"]
                and duplicate_text[item["text_sha256"]] >= 20):
            item["text_ok"] = False
            item["issue"] = "duplicated_template_content"
            duplicate_review += 1

    output.mkdir(parents=True)
    state_path = output / "document_state.parquet"
    recovery_path = output / "recovery_candidates.parquet"
    counts = Counter()
    failures = Counter()
    source_counts = Counter()
    with pq.ParquetWriter(state_path, SCHEMA, compression="zstd") as writer, \
         pq.ParquetWriter(recovery_path, SCHEMA, compression="zstd") as retry_writer:
        buffer, retry_buffer = [], []
        for ident in sorted(states):
            row = classify(states[ident], evidence.get(ident))
            if states[ident]["usable"] and ident not in evidence:
                row.update(usable=False, extraction_failed=True, review_required=False,
                           phase0_status="EXTRACTION_FAILED", reason="missing_canonical_content")
            counts[row["phase0_status"]] += 1
            if row["reason"] != "usable":
                failures[row["reason"]] += 1
            source_counts[row["source"]] += 1
            buffer.append(row)
            if row["phase0_status"] in {"CRAWL_FAILED", "EXTRACTION_FAILED", "DEFERRED", "REVIEW_REQUIRED"}:
                retry_buffer.append(row)
            if len(buffer) >= 10_000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                buffer.clear()
            if len(retry_buffer) >= 10_000:
                retry_writer.write_table(pa.Table.from_pylist(retry_buffer, schema=SCHEMA))
                retry_buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
        if retry_buffer:
            retry_writer.write_table(pa.Table.from_pylist(retry_buffer, schema=SCHEMA))

    staged = pq.read_table(RECOVERY).to_pylist()
    staged_usable = {row["id"] for row in staged if row["usable"]}
    if any(states[ident]["usable"] for ident in staged_usable):
        raise ValueError("Recovery staged a previously usable document")
    report = {
        "status": "complete", "checked_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "repository": step1["repository"], "revision": revision,
        "official_corpus_sha256": sha256(corpus), "official_query_sha256": sha256(query),
        "official_ids": len(official), "queries": 1200,
        "status_counts": dict(sorted(counts.items())),
        "recovery_candidates": pq.ParquetFile(recovery_path).metadata.num_rows,
        "canonical_evidence_rows": len(evidence), "content_rows_scanned": scanned,
        "content_files_sha256_verified": source_files,
        "raw_rows_sha256_verified": raw_verified_rows,
        "raw_pubmed_batches_verified": len(raw_cache),
        "high_frequency_duplicate_rows_for_review": duplicate_review,
        "staged_recovery_usable_ids": len(staged_usable),
        "staged_recovery_id_list": sorted(staged_usable),
        "staged_recovery_merged": False,
        "failed_or_nonusable_reasons": dict(sorted(failures.items())),
        "classification_note": "DEFERRED means no per-URL failure is proven. CRAWLED_SUCCESS is supported by prior extraction provenance and current content; raw hashes are sampled here. Full raw file membership is checked against the backup archive separately. Automated checks do not replace manual article review.",
        "artifacts": {name: {"rows": pq.ParquetFile(path).metadata.num_rows, "sha256": sha256(path)}
                      for name, path in (("document_state.parquet", state_path),
                                         ("recovery_candidates.parquet", recovery_path))},
    }
    (output / "audit_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({key: result[key] for key in ("official_ids", "status_counts",
          "recovery_candidates", "staged_recovery_usable_ids")}, indent=2))


if __name__ == "__main__":
    main()
