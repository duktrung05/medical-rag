"""Build a canonical step-2 status ledger and immutable content manifest.

The materialized documents and passages remain sharded in their source folders.
This command selects one canonical source per official ID, records every missing
or deferred ID, and publishes checksums for the files consumed by step 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/vibiomir_step2_20261013"
DEFAULT_OUTPUT = BASE / "release"
REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"

STATUS_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("source_kind", pa.string()), ("ledger_state", pa.string()),
    ("final_status", pa.string()), ("usable", pa.bool_()),
    ("content_group", pa.string()), ("reason", pa.string()),
])
MISSING_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("source_kind", pa.string()), ("ledger_state", pa.string()),
    ("final_status", pa.string()), ("reason", pa.string()),
])


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_pairs(root: Path) -> list[tuple[Path, Path]]:
    single = root / "documents.parquet"
    if single.exists():
        passages = root / "passages.parquet"
        if not passages.exists():
            raise FileNotFoundError(passages)
        return [(single, passages)]
    result = []
    for documents in sorted((root / "documents").glob("part_*.parquet")):
        passages = root / "passages" / documents.name
        if not passages.exists():
            raise FileNotFoundError(passages)
        result.append((documents, passages))
    return result


def source_specs() -> list[tuple[str, Path, int, str]]:
    content = BASE / "content"
    return [
        ("pubmed", content / "pubmed", 20, "pubmed"),
        ("saved_raw", content / "needs_extract", 20, "html"),
        ("legacy_html", content / "legacy_chunk_candidate", 10, "html"),
        ("baotayninh_css_fallback", content / "baotayninh_css_fallback", 40, "html"),
        ("recovery_pilot", content / "recovery_pilot", 40, "html"),
        ("recovery_healthy_pilot", content / "recovery_healthy_pilot", 40, "html"),
        ("recovery_misc_probe", content / "recovery_misc_probe", 40, "html"),
        ("recovery_39subdomains_probe", content / "recovery_39subdomains_probe", 40, "html"),
        ("recovery_longchau_partial", content / "recovery_longchau_partial_00000", 30, "html"),
        ("recovery_healthy_partial", content / "recovery_healthy_partial_00000", 30, "html"),
        ("recovery_longchau", content / "recovery_longchau", 20, "html"),
        ("recovery_healthy3", content / "recovery_healthy3", 20, "html"),
        ("recovery_misc_healthy", content / "recovery_misc_healthy", 20, "html"),
        ("recovery_misc_stable", content / "recovery_misc_stable", 20, "html"),
        ("recovery_39subdomains", content / "recovery_39subdomains", 20, "html"),
    ]


def crawl_result_files() -> list[Path]:
    paths = set(BASE.glob("**/crawl_results.parquet"))
    return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def normalize_attempt(row: dict) -> str:
    status = row["status"]
    if status == "http_error":
        if row.get("error_type") == "access_challenge":
            return "access_challenge"
        if row.get("http_status") in {403, 428}:
            return "access_blocked"
    return status


def plan_states() -> tuple[set[str], dict[str, list[dict]]]:
    active = set()
    details = defaultdict(list)
    for plan_path in sorted(BASE.glob("recovery_*_batches/plan_manifest.json")):
        plan = json.loads(plan_path.read_text())
        progress_path = plan_path.parent / "runner_progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
        completed = (progress or {}).get("completed_batches", 0)
        stopped = (progress or {}).get("stopped_reason")
        status = (progress or {}).get("status")
        # Older single-domain plans use ``domain`` while grouped plans use
        # ``domains``. A prepared plan without runner progress is queued, not
        # actively crawling; treating it as active hides the real reason for
        # missing content in the final ledger.
        domains = plan.get("domains") or ([plan["domain"]] if plan.get("domain") else [])
        is_active = (progress is not None and completed < plan["batches"]
                     and stopped is None and status not in {"stopped", "complete"})
        for domain in domains:
            details[domain].append({
                "plan": str(plan_path.relative_to(ROOT)), "rows": plan["rows"],
                "batches": plan["batches"], "completed_batches": completed,
                "status": status or ("in_progress" if is_active else "prepared"),
                "stopped_reason": stopped,
            })
            if is_active:
                active.add(domain)
    return active, dict(details)


def choose_domain_reason(counter: Counter) -> str:
    for status in ("rate_limited", "access_blocked", "access_challenge",
                   "robots_denied", "robots_unavailable", "http_error",
                   "not_found", "timeout", "connection_error", "network_error"):
        if counter[status]:
            return "domain_deferred_" + status
    if counter["success"]:
        return "unresolved_after_domain_success"
    return "not_attempted"


def run(output: Path) -> dict:
    ledger_manifest = json.loads((BASE / "ledger_manifest.json").read_text())
    if ledger_manifest["corpus_revision"] != REVISION:
        raise ValueError("Step-2 ledger uses another corpus revision")
    ledger_path = BASE / "document_state.parquet"
    id_table = pq.read_table(ledger_path, columns=["id"])
    maximum_id = int(np.max(id_table["id"].to_numpy(zero_copy_only=False)))
    official = np.zeros(maximum_id + 1, dtype=np.bool_)
    official[id_table["id"].to_numpy(zero_copy_only=False)] = True

    status_names = [""]
    status_codes = {"": 0}
    source_names = [""]
    source_codes = {"": 0}

    def status_code(value: str) -> int:
        if value not in status_codes:
            status_codes[value] = len(status_names)
            status_names.append(value)
        return status_codes[value]

    def source_code(value: str) -> int:
        if value not in source_codes:
            source_codes[value] = len(source_names)
            source_names.append(value)
        return source_codes[value]

    content_status = np.zeros(maximum_id + 1, dtype=np.uint16)
    content_source = np.zeros(maximum_id + 1, dtype=np.uint16)
    content_priority = np.zeros(maximum_id + 1, dtype=np.uint8)
    content_files = []
    duplicate_content_rows = 0
    overridden_content_rows = 0
    conflicting_content_rows = 0
    for group, root, priority, kind in source_specs():
        for documents_path, passages_path in content_pairs(root):
            schema = pq.read_schema(documents_path)
            columns = ["id"] + (["status"] if "status" in schema.names else [])
            rows = pq.read_table(documents_path, columns=columns).to_pylist()
            for row in rows:
                ident = int(row["id"])
                if ident > maximum_id or not official[ident]:
                    raise ValueError(f"Content ID outside official corpus: {ident}")
                final_status = row.get("status", "usable")
                code = status_code(final_status)
                if content_status[ident]:
                    duplicate_content_rows += 1
                    old = status_names[int(content_status[ident])]
                    if priority == content_priority[ident] and old != final_status:
                        conflicting_content_rows += 1
                if priority > content_priority[ident]:
                    if content_status[ident]:
                        overridden_content_rows += 1
                    content_status[ident] = code
                    content_source[ident] = source_code(group)
                    content_priority[ident] = priority
                elif priority == content_priority[ident] and content_status[ident] == 0:
                    content_status[ident] = code
                    content_source[ident] = source_code(group)
            content_files.append({
                "group": group, "kind": kind,
                "documents": str(documents_path.relative_to(ROOT)),
                "documents_rows": pq.ParquetFile(documents_path).metadata.num_rows,
                "documents_sha256": sha(documents_path),
                "passages": str(passages_path.relative_to(ROOT)),
                "passages_rows": pq.ParquetFile(passages_path).metadata.num_rows,
                "passages_sha256": sha(passages_path),
            })

    attempt_status = np.zeros(maximum_id + 1, dtype=np.uint16)
    domain_attempts = defaultdict(Counter)
    attempt_files = []
    for path in crawl_result_files():
        columns = ["id", "domain", "status", "http_status", "error_type"]
        rows = pq.read_table(path, columns=columns).to_pylist()
        for row in rows:
            ident = int(row["id"])
            if ident > maximum_id or not official[ident]:
                raise ValueError(f"Crawl result ID outside official corpus: {ident}")
            normalized = normalize_attempt(row)
            attempt_status[ident] = status_code(normalized)
            domain_attempts[row["domain"]][normalized] += 1
        attempt_files.append({"path": str(path.relative_to(ROOT)), "rows": len(rows),
                              "sha256": sha(path)})

    active_domains, plans = plan_states()
    output.mkdir(parents=True, exist_ok=True)
    status_tmp = output / "document_status.parquet.partial"
    missing_tmp = output / "missing_documents.parquet.partial"
    status_writer = pq.ParquetWriter(status_tmp, STATUS_SCHEMA, compression="zstd")
    missing_writer = pq.ParquetWriter(missing_tmp, MISSING_SCHEMA, compression="zstd")
    final_counts = Counter()
    source_counts = Counter()
    pending_count = 0
    rows_written = 0
    try:
        for batch in pq.ParquetFile(ledger_path).iter_batches(batch_size=25_000):
            status_rows, missing_rows = [], []
            for row in batch.to_pylist():
                ident = int(row["id"])
                ledger_state = row["content_state"]
                selected_status = int(content_status[ident])
                group = source_names[int(content_source[ident])] or None
                if selected_status:
                    final_status = status_names[selected_status]
                elif ledger_state == "pubmed_metadata_only":
                    final_status = "metadata_only"
                elif ledger_state == "pubmed_unavailable":
                    final_status = "unavailable"
                elif ledger_state in {"pubmed_usable", "legacy_chunk_candidate", "needs_extract"}:
                    final_status = "pending_materialization"
                else:
                    attempt = status_names[int(attempt_status[ident])]
                    if attempt == "success":
                        final_status = "pending_materialization"
                    elif attempt:
                        final_status = "fetch_" + attempt
                    elif row["hostname"] in active_domains:
                        final_status = "pending_crawl"
                    else:
                        final_status = choose_domain_reason(domain_attempts[row["hostname"]])
                usable = final_status == "usable"
                reason = None if usable else final_status
                if final_status.startswith(("pending_", "unresolved_", "not_attempted")):
                    pending_count += 1
                final_counts[final_status] += 1
                source_counts[group or "none"] += 1
                record = {"id": ident, "official_url": row["official_url"],
                          "hostname": row["hostname"], "source_kind": row["source_kind"],
                          "ledger_state": ledger_state, "final_status": final_status,
                          "usable": usable, "content_group": group, "reason": reason}
                status_rows.append(record)
                if not usable:
                    missing_rows.append({key: record[key] for key in MISSING_SCHEMA.names})
            status_writer.write_table(pa.Table.from_pylist(status_rows, schema=STATUS_SCHEMA))
            if missing_rows:
                missing_writer.write_table(pa.Table.from_pylist(missing_rows, schema=MISSING_SCHEMA))
            rows_written += len(status_rows)
    finally:
        status_writer.close()
        missing_writer.close()
    if rows_written != ledger_manifest["rows"] or sum(final_counts.values()) != rows_written:
        raise ValueError("Final status coverage differs from official corpus")
    status_path = output / "document_status.parquet"
    missing_path = output / "missing_documents.parquet"
    os.replace(status_tmp, status_path)
    os.replace(missing_tmp, missing_path)
    complete = pending_count == 0
    manifest = {
        "status": "complete" if complete else "in_progress", "complete": complete,
        "corpus_revision": REVISION, "official_ids": rows_written,
        "usable_documents": final_counts["usable"],
        "missing_or_deferred_documents": rows_written - final_counts["usable"],
        "pending_documents": pending_count,
        "final_status_counts": dict(sorted(final_counts.items())),
        "content_group_counts": dict(sorted(source_counts.items())),
        "duplicate_content_rows": duplicate_content_rows,
        "overridden_content_rows": overridden_content_rows,
        "conflicting_content_rows": conflicting_content_rows,
        "document_status": {"path": str(status_path.relative_to(ROOT)),
                            "sha256": sha(status_path), "rows": rows_written},
        "missing_documents": {"path": str(missing_path.relative_to(ROOT)),
                              "sha256": sha(missing_path),
                              "rows": rows_written - final_counts["usable"]},
        "content_files": content_files, "crawl_result_files": attempt_files,
        "active_plan_domains": sorted(active_domains), "plans_by_domain": plans,
        "selection_policy": "Highest-priority materialized source per official ID; use document_status.content_group and ignore lower-priority duplicate rows.",
        "quality_limit": "Usable means automated extraction and exact-offset validation passed; it is not a manual relevance or clinical-quality judgment.",
    }
    manifest_tmp = output / "release_manifest.json.partial"
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(manifest_tmp, output / "release_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({key: result[key] for key in (
        "status", "official_ids", "usable_documents",
        "missing_or_deferred_documents", "pending_documents",
        "duplicate_content_rows", "conflicting_content_rows")}, indent=2))


if __name__ == "__main__":
    main()
