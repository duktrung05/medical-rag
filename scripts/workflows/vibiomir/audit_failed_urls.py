"""Build an immutable allowlist for isolated ViBioMIR failed-URL recovery.

This command is intentionally offline.  It reads the pinned official corpus and
the canonical step-2 status ledger, excludes every usable document, and writes
new recovery artifacts outside all protected data roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
CORPUS_SHA256 = "87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7"
STEP1 = ROOT / "outputs/vibiomir_step1_20261013/manifest.json"
STEP2 = ROOT / "outputs/vibiomir_step2_20261013"
DEFAULT_OUTPUT = ROOT / "outputs/crawl_failure_recovery"

ROW_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("source_kind", pa.string()), ("ledger_state", pa.string()),
    ("final_status", pa.string()), ("reason", pa.string()),
    ("recovery_group", pa.string()), ("recovery_action", pa.string()),
    ("network_allowed", pa.bool_()), ("last_attempt_at", pa.string()),
    ("next_retry_at", pa.string()), ("audit_revision", pa.string()),
])
EXCLUDED_SCHEMA = pa.schema(list(ROW_SCHEMA) + [("exclusion_reason", pa.string())])


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    os.replace(temporary, path)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def latest_attempts() -> tuple[dict[int, str], dict[str, str]]:
    """Return latest observed fetch time per ID and domain from existing logs."""
    by_id: dict[int, datetime] = {}
    by_domain: dict[str, datetime] = {}
    for path in sorted(STEP2.glob("**/crawl_results.parquet")):
        names = pq.read_schema(path).names
        if not {"id", "domain", "fetched_at"}.issubset(names):
            continue
        for row in pq.read_table(path, columns=["id", "domain", "fetched_at"]).to_pylist():
            when = parse_time(row["fetched_at"])
            if when is None:
                continue
            ident, domain = int(row["id"]), row["domain"]
            if ident not in by_id or when > by_id[ident]:
                by_id[ident] = when
            if domain not in by_domain or when > by_domain[domain]:
                by_domain[domain] = when
    return ({key: value.isoformat() for key, value in by_id.items()},
            {key: value.isoformat() for key, value in by_domain.items()})


def recovery_policy(status: str) -> tuple[str, str, bool, timedelta] | None:
    if "rate_limited" in status:
        return "rate_limited", "bounded_network_retry", True, timedelta(hours=24)
    if "robots_unavailable" in status:
        return "robots_unavailable", "robots_recheck", True, timedelta(hours=24)
    if "http_error" in status:
        return "http_error", "bounded_network_retry", True, timedelta(hours=1)
    if status.endswith(("timeout", "connection_error")):
        return "network_transient", "bounded_network_retry", True, timedelta(minutes=15)
    if status in {"extract_failed", "quality_review_empty", "quality_review_short"}:
        return "content_unusable", "extract_only", False, timedelta(0)
    return None


def exclusion_reason(status: str, usable: bool) -> str:
    if usable:
        return "verified_success"
    if "access_challenge" in status or "access_blocked" in status:
        return "access_restricted_no_automatic_retry"
    if "robots_denied" in status:
        return "robots_disallowed"
    if "not_found" in status:
        return "confirmed_not_found"
    if status in {"metadata_only", "unavailable"}:
        return "source_has_no_recoverable_content"
    return "status_not_eligible_for_automatic_recovery"


def protected_inventory() -> dict:
    roots = [
        ROOT / "data/vibiomir",
        STEP2 / "content",
        STEP2 / "release",
    ]
    entries = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                entries.append({"path": str(path.relative_to(ROOT)), "bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns})
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    canonical = [
        ROOT / "outputs/vibiomir_official_verify_20261012/links_corpus_vi_en_zh.parquet",
        STEP2 / "document_state.parquet",
        STEP2 / "ledger_manifest.json",
        STEP2 / "release/document_status.parquet",
        STEP2 / "release/missing_documents.parquet",
        STEP2 / "release/release_manifest.json",
        ROOT / "configs/v4_corpus_registry.json",
    ]
    return {
        "protected_roots": [str(path.relative_to(ROOT)) for path in roots],
        "file_count": len(entries),
        "total_bytes": sum(row["bytes"] for row in entries),
        "metadata_inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "files": entries,
        "canonical_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in canonical},
    }


def run(output: Path) -> dict:
    step1 = json.loads(STEP1.read_text())
    release = json.loads((STEP2 / "release/release_manifest.json").read_text())
    if step1["revision"] != REVISION or release["corpus_revision"] != REVISION:
        raise ValueError("Corpus revision differs from the pinned recovery revision")
    corpus_path = Path(step1["files"]["links_corpus_vi_en_zh.parquet"]["path"])
    if sha256(corpus_path) != CORPUS_SHA256:
        raise ValueError("Official corpus checksum mismatch")
    official_rows = pq.read_table(corpus_path, columns=["id", "url"]).to_pylist()
    official = {int(row["id"]): row["url"] for row in official_rows}
    if len(official) != len(official_rows) or len(set(official.values())) != len(official):
        raise ValueError("Official corpus contains duplicate ID or URL")
    status_rows = pq.read_table(STEP2 / "release/document_status.parquet").to_pylist()
    if len(status_rows) != len(official):
        raise ValueError("Canonical status ledger does not cover the official corpus")
    id_attempt, domain_attempt = latest_attempts()
    eligible, excluded = [], []
    eligible_groups, excluded_reasons = Counter(), Counter()
    status_counts = Counter()
    seen = set()
    for row in status_rows:
        ident = int(row["id"])
        if ident in seen or official.get(ident) != row["official_url"]:
            raise ValueError(f"Status-to-corpus mapping mismatch for ID {ident}")
        seen.add(ident)
        status = row["final_status"]
        status_counts[status] += 1
        policy = recovery_policy(status) if not row["usable"] else None
        common = {key: row.get(key) for key in (
            "id", "official_url", "hostname", "source_kind", "ledger_state",
            "final_status", "reason")}
        common["audit_revision"] = REVISION
        if policy is None:
            reason = exclusion_reason(status, bool(row["usable"]))
            excluded_reasons[reason] += 1
            excluded.append({**common, "recovery_group": None, "recovery_action": None,
                             "network_allowed": False, "last_attempt_at": None,
                             "next_retry_at": None, "exclusion_reason": reason})
            continue
        group, action, network_allowed, cooldown = policy
        last_value = id_attempt.get(ident) or domain_attempt.get(row["hostname"])
        last = parse_time(last_value)
        next_retry = (last + cooldown).isoformat() if last and network_allowed else None
        eligible_groups[group] += 1
        eligible.append({**common, "recovery_group": group,
                         "recovery_action": action, "network_allowed": network_allowed,
                         "last_attempt_at": last.isoformat() if last else None,
                         "next_retry_at": next_retry})
    if seen != set(official) or len(eligible) + len(excluded) != len(official):
        raise ValueError("Recovery partition does not exactly cover the official corpus")
    audit_dir = output / "audit"
    eligible_path = output / "eligible/FAILED_URLS_ELIGIBLE.parquet"
    excluded_path = output / "eligible/FAILED_URLS_EXCLUDED.parquet"
    atomic_parquet(eligible_path, eligible, ROW_SCHEMA)
    atomic_parquet(excluded_path, excluded, EXCLUDED_SCHEMA)
    baseline = {
        "created_at": datetime.now(UTC).isoformat(), "corpus_revision": REVISION,
        "corpus_sha256": CORPUS_SHA256, **protected_inventory(),
    }
    atomic_json(audit_dir / "PROTECTED_DATA_BASELINE.json", baseline)
    report = {
        "status": "complete", "network_requests": 0,
        "generated_at": datetime.now(UTC).isoformat(), "corpus_revision": REVISION,
        "official_ids": len(official), "eligible_ids": len(eligible),
        "excluded_ids": len(excluded), "eligible_groups": dict(sorted(eligible_groups.items())),
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "final_status_counts": dict(sorted(status_counts.items())),
        "verified_success_ids_excluded": excluded_reasons["verified_success"],
        "partition_complete": len(eligible) + len(excluded) == len(official),
        "eligible_path": str(eligible_path.relative_to(ROOT)),
        "excluded_path": str(excluded_path.relative_to(ROOT)),
        "baseline_path": str((audit_dir / "PROTECTED_DATA_BASELINE.json").relative_to(ROOT)),
        "policy": {
            "network_retry": ["rate_limited", "robots_unavailable", "http_error",
                              "network_transient"],
            "extract_only": ["extract_failed", "quality_review_empty",
                             "quality_review_short"],
            "automatic_retry_prohibited": ["access_restricted", "robots_disallowed",
                                           "confirmed_not_found", "metadata_only", "unavailable"],
        },
    }
    atomic_json(audit_dir / "FAILED_URL_AUDIT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
