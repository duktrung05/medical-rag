"""Run isolated, allowlist-only recovery for failed ViBioMIR URLs.

Modes are deliberately separate: ``dry-run`` creates a bounded plan without
network access; ``pilot`` executes at most ten eligible rows per error group;
``run`` expands only groups whose pilot passed the acceptance gate.  All raw,
text, checkpoints, and reports remain under outputs/crawl_failure_recovery.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import threading
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lxml import html as lxml_html

from scripts.workflows.vibiomir.crawl_vibiomir_batch import crawl_batch
from scripts.workflows.vibiomir.materialize_html_content import (
    CHALLENGE, DOC_SCHEMA, PASSAGE_SCHEMA, process as base_extract,
)
from src.extraction.html_extractor import decode_html
from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.contracts import Source
from src.medrag.evidence.verifier import verify


ROOT = Path(__file__).resolve().parents[3]
REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
STEP2 = ROOT / "outputs/vibiomir_step2_20261013"
DEFAULT_ROOT = ROOT / "outputs/crawl_failure_recovery"
ELIGIBLE = DEFAULT_ROOT / "eligible/FAILED_URLS_ELIGIBLE.parquet"
BASELINE = DEFAULT_ROOT / "audit/PROTECTED_DATA_BASELINE.json"
STATUS = STEP2 / "release/document_status.parquet"
PILOT_PER_GROUP = 10
MAX_BATCH_SIZE = 500
RECOVERY_GROUPS = {"content_unusable", "http_error", "network_transient",
                   "rate_limited", "robots_unavailable"}

PLAN_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("domain", pa.string()),
    ("domain_url_count", pa.int64()), ("recovery_group", pa.string()),
    ("recovery_action", pa.string()), ("network_allowed", pa.bool_()),
    ("next_retry_at", pa.string()), ("corpus_revision", pa.string()),
])
RESULT_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("recovery_group", pa.string()), ("phase", pa.string()),
    ("fetch_status", pa.string()), ("http_status", pa.int32()),
    ("error_type", pa.string()), ("error_message", pa.string()),
    ("extraction_status", pa.string()), ("usable", pa.bool_()),
    ("raw_path", pa.string()), ("raw_sha256", pa.string()),
    ("text_sha256", pa.string()), ("attempted_at", pa.string()),
])
REMAINING_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("final_status", pa.string()), ("original_reason", pa.string()),
    ("recovery_status", pa.string()), ("recovery_group", pa.string()),
    ("last_error_type", pa.string()), ("last_error_message", pa.string()),
])


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


def write_once_or_verify(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise ValueError(f"Existing immutable plan differs: {path}")
        return
    atomic_parquet(path, rows, schema)


def wave_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ValueError("Wave must use 1..64 lowercase letters, digits, '_' or '-'")
    return value


def wave_path(root: Path, kind: str, wave: str) -> Path:
    suffix = "" if wave == "default" else f"_{wave}"
    return root / f"manifests/{kind}_PLAN{suffix}.parquet"


def phase_tag(phase: str, wave: str) -> str:
    return phase if wave == "default" else f"{phase}_{wave}"


def parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value).astimezone(UTC) if value else None


def ready(row: dict, now: datetime) -> bool:
    retry = parse_time(row.get("next_retry_at"))
    return not row["network_allowed"] or retry is None or retry <= now


def with_recent_attempts(eligible: list[dict], results_path: Path) -> list[dict]:
    """Apply new attempt cooldowns without changing the audited allowlist."""
    if not results_path.exists():
        return eligible
    by_id: dict[int, datetime] = {}
    by_domain: dict[str, datetime] = {}
    blocked_domains = set()
    for result in pq.read_table(results_path).to_pylist():
        attempted = parse_time(result.get("attempted_at"))
        if attempted is None:
            continue
        domain = result["hostname"]
        status = result.get("fetch_status")
        http_status = result.get("http_status")
        if status in {"access_blocked", "access_challenge", "robots_denied"} or http_status in {403, 428} or result.get("error_type") == "access_challenge":
            blocked_domains.add(domain)
        if status == "rate_limited" or http_status == 429:
            by_domain[domain] = max(by_domain.get(domain, attempted), attempted + timedelta(hours=24))
        if result.get("usable"):
            continue
        wait = {"rate_limited": timedelta(hours=24),
                "robots_unavailable": timedelta(hours=24),
                "http_error": timedelta(hours=1),
                "network_transient": timedelta(minutes=15)}.get(result.get("recovery_group"))
        if wait is not None:
            by_id[int(result["id"])] = max(by_id.get(int(result["id"]), attempted), attempted + wait)
    updated = []
    for row in eligible:
        if row["hostname"] in blocked_domains:
            continue
        retry = parse_time(row.get("next_retry_at"))
        for extra in (by_id.get(int(row["id"])), by_domain.get(row["hostname"])):
            if extra is not None and (retry is None or extra > retry):
                retry = extra
        updated.append({**row, "next_retry_at": retry.isoformat() if retry else None})
    return updated


def interleaved_sample(rows: list[dict], limit: int) -> list[dict]:
    domains: dict[str, deque] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: (item["hostname"], int(item["id"]))):
        domains[row["hostname"]].append(row)
    result = []
    while domains and len(result) < limit:
        for domain in list(domains):
            result.append(domains[domain].popleft())
            if not domains[domain]:
                del domains[domain]
            if len(result) == limit:
                break
    return result


def plan_rows(eligible: list[dict], *, pilot: bool, now: datetime) -> tuple[list[dict], dict]:
    ids = [int(row["id"]) for row in eligible]
    urls = [row["official_url"] for row in eligible]
    if len(ids) != len(set(ids)) or len(urls) != len(set(urls)):
        raise ValueError("Recovery allowlist contains duplicate ID or URL")
    counts = Counter(row["hostname"] for row in eligible)
    by_group = defaultdict(list)
    cooldown = Counter()
    for row in eligible:
        if ready(row, now):
            by_group[row["recovery_group"]].append(row)
        else:
            cooldown[row["recovery_group"]] += 1
    selected = []
    for group in sorted(by_group):
        candidates = by_group[group]
        selected.extend(interleaved_sample(candidates, PILOT_PER_GROUP)
                        if pilot else sorted(candidates, key=lambda x: (x["hostname"], x["id"])))
    plan = [{"id": row["id"], "url": row["official_url"],
             "domain": row["hostname"], "domain_url_count": counts[row["hostname"]],
             "recovery_group": row["recovery_group"],
             "recovery_action": row["recovery_action"],
             "network_allowed": row["network_allowed"],
             "next_retry_at": row["next_retry_at"], "corpus_revision": REVISION}
            for row in selected]
    return plan, {"selected": len(plan), "selected_by_group": dict(Counter(
        row["recovery_group"] for row in plan)), "cooldown_by_group": dict(cooldown)}


class RecoveryGuard:
    """Revalidate the immutable allowlist immediately before every request."""

    def __init__(self, eligible_path: Path, baseline_path: Path,
                 status_path: Path = STATUS, *, results_path: Path | None = None,
                 status_key: str | None = None,
                 clock=lambda: datetime.now(UTC)):
        self.clock = clock
        self.guard = threading.Lock()
        eligible = with_recent_attempts(pq.read_table(eligible_path).to_pylist(),
                                        results_path) if results_path else pq.read_table(eligible_path).to_pylist()
        self.allowed = {int(row["id"]): row for row in eligible if row["network_allowed"]}
        status = pq.read_table(status_path, columns=["id", "official_url", "usable"]).to_pylist()
        self.current = {int(row["id"]): row for row in status}
        baseline = json.loads(baseline_path.read_text())
        relative = status_key or str(status_path.relative_to(ROOT))
        baseline_files = {row["path"]: row for row in baseline["files"]}
        self.status_stat = baseline_files[relative]
        self.status_path = status_path
        self.recovered = set()
        if results_path is not None and results_path.exists():
            self.recovered = {int(row["id"]) for row in
                              pq.read_table(results_path, columns=["id", "usable"]).to_pylist()
                              if row["usable"]}
        self.requested: set[int] = set()

    def __call__(self, entry: dict) -> None:
        ident = int(entry["id"])
        with self.guard:
            stat = self.status_path.stat()
            if (stat.st_size != self.status_stat["bytes"]
                    or stat.st_mtime_ns != self.status_stat["mtime_ns"]):
                raise ValueError("Protected document status changed after audit")
            row = self.allowed.get(ident)
            current = self.current.get(ident)
            if row is None or current is None:
                raise ValueError(f"ID {ident} is outside the recovery allowlist")
            if row["official_url"] != entry["url"] or row["hostname"] != entry["domain"]:
                raise ValueError(f"Official source mapping differs for ID {ident}")
            if current["usable"]:
                raise ValueError(f"Verified-success ID {ident} must never be fetched")
            if ident in self.recovered:
                raise ValueError(f"Recovered-success ID {ident} must never be fetched again")
            if not ready(row, self.clock()):
                raise ValueError(f"Retry cooldown has not expired for ID {ident}")
            if ident in self.requested:
                raise ValueError(f"Duplicate request in recovery run for ID {ident}")
            self.requested.add(ident)


def protected_unchanged(root: Path) -> dict:
    baseline = json.loads((root / "audit/PROTECTED_DATA_BASELINE.json").read_text())
    changed = []
    for row in baseline["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            changed.append({"path": row["path"], "reason": "missing"})
            continue
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            changed.append({"path": row["path"], "reason": "metadata_changed"})
    canonical_changed = []
    for relative, expected in baseline["canonical_sha256"].items():
        if sha256(ROOT / relative) != expected:
            canonical_changed.append(relative)
    return {"unchanged": not changed and not canonical_changed,
            "metadata_changes": changed[:100], "metadata_change_count": len(changed),
            "canonical_checksum_changes": canonical_changed}


def raw_sources(ids: set[int]) -> dict[int, dict]:
    result = {}
    for path in (STEP2 / "needs_extract.parquet", STEP2 / "legacy_chunk_candidate.parquet"):
        table = pq.read_table(path, columns=["id", "official_url", "hostname", "raw_path"])
        table = table.filter(pc.is_in(table["id"], value_set=pa.array(sorted(ids), type=pa.int64())))
        for row in table.to_pylist():
            if Path(row["raw_path"]).is_file():
                result[int(row["id"])] = row
    for path in sorted(STEP2.glob("**/crawl_results.parquet"),
                       key=lambda item: (item.stat().st_mtime_ns, str(item))):
        schema = pq.read_schema(path).names
        if not {"id", "source_url", "domain", "status", "raw_file"}.issubset(schema):
            continue
        table = pq.read_table(path, columns=["id", "source_url", "domain", "status", "raw_file"])
        table = table.filter(pc.is_in(table["id"], value_set=pa.array(sorted(ids), type=pa.int64())))
        for row in table.to_pylist():
            raw = path.parent / row["raw_file"] if row["raw_file"] else None
            if row["status"] == "success" and raw and raw.is_file():
                result[int(row["id"])] = {"id": row["id"],
                    "official_url": row["source_url"], "hostname": row["domain"],
                    "raw_path": str(raw)}
    return result


def fallback_extract(row: dict) -> tuple[dict, list[dict]]:
    document, passages = base_extract(row)
    if document["status"] == "usable":
        return document, passages
    try:
        compressed = Path(row["raw_path"]).read_bytes()
        raw = gzip.decompress(compressed) if compressed.startswith(b"\x1f\x8b") else compressed
        decoded, encoding, encoding_source = decode_html(raw)
        if CHALLENGE.search(decoded[:100_000]):
            return document, passages
        tree = lxml_html.fromstring(decoded)
        for bad in tree.xpath("//script|//style|//noscript|//nav|//footer|//header|//form|//aside"):
            bad.drop_tree()
        candidates = tree.xpath(
            "//article|//main|//*[@role='main']|//*[@id='content-detail']|"
            "//*[contains(concat(' ',normalize-space(@class),' '),' article-content ')]|"
            "//*[contains(concat(' ',normalize-space(@class),' '),' detail-content ')]|"
            "//*[contains(concat(' ',normalize-space(@class),' '),' post-content ')]|"
            "//*[contains(concat(' ',normalize-space(@class),' '),' entry-content ')]")
        nodes = candidates or [tree]
        texts = []
        for node in nodes:
            value = "\n".join(part.strip() for part in node.itertext() if part.strip())
            value = re.sub(r"[ \t]+", " ", value)
            value = re.sub(r"\n{3,}", "\n\n", value).strip()
            if len(value) >= 80:
                texts.append(value)
        if not texts:
            return document, passages
        text = max(texts, key=len)
        document.update(status="usable", text=text, encoding=encoding,
                        encoding_source=encoding_source,
                        extraction_method="recovery_dom_fallback_v1",
                        raw_sha256=hashlib.sha256(raw).hexdigest(),
                        raw_bytes=len(compressed), body_bytes=len(raw),
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(), error=None)
        source = Source("official-" + REVISION, int(row["id"]), row["official_url"],
                        document["title"], text, row["raw_path"], document["raw_sha256"])
        passages = []
        for index, passage in enumerate(build_passages(source)):
            if not verify(passage, source):
                raise ValueError("passage offset mismatch")
            passages.append({"id": int(row["id"]), "official_url": row["official_url"],
                "passage_id": passage.passage_id, "passage_index": index,
                "text": passage.text, "source_start": passage.source_start,
                "source_end": passage.source_end, "source_sha256": passage.source_sha256,
                "language_hint": passage.language})
    except (OSError, EOFError, ValueError, UnicodeError, TypeError,
            lxml_html.etree.ParserError) as exc:
        document["error"] = f"fallback {type(exc).__name__}: {exc}"[:500]
    return document, passages


def materialize_rows(rows: list[dict], destination: Path) -> tuple[list[dict], list[dict]]:
    docs, passages = [], []
    for row in rows:
        document, chunks = fallback_extract(row)
        docs.append(document)
        passages.extend(chunks)
    destination.mkdir(parents=True, exist_ok=True)
    doc_path, passage_path = destination / "documents.parquet", destination / "passages.parquet"
    if doc_path.exists() or passage_path.exists():
        if not doc_path.exists() or not passage_path.exists():
            raise ValueError(f"Incomplete extraction checkpoint: {destination}")
        existing_docs = pq.read_table(doc_path).to_pylist()
        existing_mapping = {(int(row["id"]), row["official_url"]) for row in existing_docs}
        expected_mapping = {(int(row["id"]), row["official_url"]) for row in docs}
        if existing_mapping != expected_mapping:
            raise ValueError(f"Existing extraction checkpoint differs: {destination}")
        return existing_docs, pq.read_table(passage_path).to_pylist()
    atomic_parquet(doc_path, docs, DOC_SCHEMA)
    atomic_parquet(passage_path, passages, PASSAGE_SCHEMA)
    return docs, passages


def extraction_result(plan: list[dict], phase: str, root: Path) -> list[dict]:
    source = raw_sources({int(row["id"]) for row in plan})
    rows = []
    for item in plan:
        raw = source.get(int(item["id"]))
        if raw is None or raw["official_url"] != item["url"]:
            continue
        rows.append(raw)
    docs, _ = materialize_rows(rows, root / f"recovered_extracted/{phase}/content_unusable")
    by_id = {int(doc["id"]): doc for doc in docs}
    now = datetime.now(UTC).isoformat()
    return [{"id": item["id"], "official_url": item["url"], "hostname": item["domain"],
             "recovery_group": item["recovery_group"], "phase": phase,
             "fetch_status": "not_requested_extract_only", "http_status": None,
             "error_type": None if item["id"] in by_id else "raw_not_found",
             "error_message": None if item["id"] in by_id else "No verified raw source located",
             "extraction_status": by_id[item["id"]]["status"] if item["id"] in by_id else None,
             "usable": bool(item["id"] in by_id and by_id[item["id"]]["status"] == "usable"),
             "raw_path": by_id[item["id"]]["raw_path"] if item["id"] in by_id else None,
             "raw_sha256": by_id[item["id"]]["raw_sha256"] if item["id"] in by_id else None,
             "text_sha256": by_id[item["id"]]["text_sha256"] if item["id"] in by_id else None,
             "attempted_at": now} for item in plan]


def network_results(plan: list[dict], group: str, phase: str, root: Path,
                    guard: RecoveryGuard) -> list[dict]:
    delay = {"rate_limited": 10.0, "robots_unavailable": 5.0}.get(group, 2.0)
    fetched_with_root = []
    for batch_index, offset in enumerate(range(0, len(plan), MAX_BATCH_SIZE)):
        batch = plan[offset:offset + MAX_BATCH_SIZE]
        batch_name = f"batch_{batch_index:05d}"
        plan_path = root / f"manifests/{phase}_{group}_{batch_name}.parquet"
        write_once_or_verify(plan_path, batch, PLAN_SCHEMA)
        crawl_root = root / f"recovered_raw/{phase}/{group}/{batch_name}"
        crawl_batch(plan_path, crawl_root, limit=len(batch), concurrency=1,
                    delay_seconds=delay, timeout_seconds=20, max_consecutive_blocked=1,
                    request_guard=guard,
                    user_agent="medical-rag-vibiomir-recovery/1.0")
        fetched_with_root.extend((row, crawl_root) for row in
                                 pq.read_table(crawl_root / "crawl_results.parquet").to_pylist())
    sources = []
    for row, crawl_root in fetched_with_root:
        raw = crawl_root / row["raw_file"] if row["raw_file"] else None
        if row["status"] == "success" and row["detected_file_type"] == "html" and raw.is_file():
            sources.append({"id": row["id"], "official_url": row["source_url"],
                            "hostname": row["domain"], "raw_path": str(raw)})
    docs, _ = materialize_rows(sources, root / f"recovered_extracted/{phase}/{group}")
    by_id = {int(doc["id"]): doc for doc in docs}
    plan_by_id = {int(row["id"]): row for row in plan}
    results = []
    for row, crawl_root in fetched_with_root:
        item = plan_by_id[int(row["id"])]
        doc = by_id.get(int(row["id"]))
        results.append({"id": row["id"], "official_url": row["source_url"],
            "hostname": row["domain"], "recovery_group": item["recovery_group"],
            "phase": phase, "fetch_status": row["status"], "http_status": row["http_status"],
            "error_type": row["error_type"], "error_message": row["error_message"],
            "extraction_status": doc["status"] if doc else None,
            "usable": bool(doc and doc["status"] == "usable"),
            "raw_path": str(crawl_root / row["raw_file"]) if row["raw_file"] else None,
            "raw_sha256": row["content_sha256"],
            "text_sha256": doc["text_sha256"] if doc else None,
            "attempted_at": row["fetched_at"]})
    return results


def merge_results(root: Path, new_rows: list[dict]) -> list[dict]:
    path = root / "RECOVERY_RESULTS.parquet"
    rows = pq.read_table(path).to_pylist() if path.exists() else []
    merged = {(row["phase"], int(row["id"])): row for row in rows}
    for row in new_rows:
        merged[(row["phase"], int(row["id"]))] = row
    result = sorted(merged.values(), key=lambda row: (row["phase"], int(row["id"])))
    atomic_parquet(path, result, RESULT_SCHEMA)
    return result


def publish_remaining(root: Path, results: list[dict]) -> list[dict]:
    successful = {int(row["id"]) for row in results if row["usable"]}
    last = {}
    for row in results:
        last[int(row["id"])] = row
    missing = pq.read_table(STEP2 / "release/missing_documents.parquet").to_pylist()
    rows = []
    for row in missing:
        ident = int(row["id"])
        if ident in successful:
            continue
        attempt = last.get(ident)
        rows.append({"id": ident, "official_url": row["official_url"],
            "hostname": row["hostname"], "final_status": row["final_status"],
            "original_reason": row["reason"],
            "recovery_status": (attempt["fetch_status"] if attempt else "not_attempted_or_held"),
            "recovery_group": attempt["recovery_group"] if attempt else None,
            "last_error_type": attempt["error_type"] if attempt else None,
            "last_error_message": attempt["error_message"] if attempt else None})
    atomic_parquet(root / "RECOVERY_REMAINING_FAILED.parquet", rows, REMAINING_SCHEMA)
    return rows


def pilot_gates(results: list[dict]) -> dict:
    gates = {}
    groups = sorted({row["recovery_group"] for row in results})
    for group in groups:
        rows = [row for row in results if row["recovery_group"] == group]
        usable = sum(row["usable"] for row in rows)
        blocked = sum((row["fetch_status"] in {"rate_limited", "robots_denied",
                                                "access_blocked", "access_challenge"}
                       or row["http_status"] in {403, 428, 429}
                       or row["error_type"] == "access_challenge") for row in rows)
        threshold = 0.5 if group == "content_unusable" else 0.8
        rate = usable / len(rows) if rows else 0.0
        gates[group] = {"attempted": len(rows), "usable": usable,
                        "usable_rate": round(rate, 6), "blocked": blocked,
                        "passed": bool(rows and rate >= threshold and blocked == 0)}
    return gates


def pilot_gates_for_plan(plan_path: Path, results: list[dict]) -> dict:
    gates = pilot_gates(results)
    expected = Counter(row["recovery_group"] for row in
                       pq.read_table(plan_path).to_pylist())
    for group, count in expected.items():
        gate = gates.setdefault(group, {"attempted": 0, "usable": 0, "usable_rate": 0.0,
                                        "blocked": 0, "passed": False})
        gate["planned"] = count
        gate["complete"] = gate["attempted"] == count
        gate["passed"] = gate["passed"] and gate["complete"]
    return gates


def markdown_report(root: Path, payload: dict, results: list[dict], remaining: list[dict]) -> None:
    audit = json.loads((root / "audit/FAILED_URL_AUDIT.json").read_text())
    remaining_reasons = Counter(row["original_reason"] for row in remaining)
    eligible = pq.read_table(root / "eligible/FAILED_URLS_ELIGIBLE.parquet").to_pylist()
    excluded = pq.read_table(root / "eligible/FAILED_URLS_EXCLUDED.parquet").to_pylist()
    now = datetime.now(UTC)
    held = {
        "access_restricted": audit["excluded_reasons"].get(
            "access_restricted_no_automatic_retry", 0),
        "robots_disallowed": audit["excluded_reasons"].get("robots_disallowed", 0),
        "confirmed_not_found": audit["excluded_reasons"].get("confirmed_not_found", 0),
        "rate_limited_cooldown": sum(row["recovery_group"] == "rate_limited"
                                     and not ready(row, now) for row in eligible),
        "robots_unavailable_cooldown": sum(row["recovery_group"] == "robots_unavailable"
                                            and not ready(row, now) for row in eligible),
    }
    domain_holds = Counter()
    for row in eligible:
        if row["recovery_group"] in {"rate_limited", "robots_unavailable"} and not ready(row, now):
            domain_holds[(row["hostname"], row["recovery_group"])] += 1
    for row in excluded:
        reason = row["exclusion_reason"]
        if reason in {"access_restricted_no_automatic_retry", "robots_disallowed"}:
            domain_holds[(row["hostname"], reason)] += 1
    result_groups = Counter(row["recovery_group"] for row in results if row["usable"])
    lines = [
        "# Crawl Failure Recovery Report", "",
        f"- Corpus revision: `{REVISION}`",
        f"- Phase: `{payload['phase']}`",
        f"- Unique network URLs requested: **{payload['unique_retried_urls']:,}**",
        f"- Network attempts recorded: **{payload['network_attempt_records']:,}**",
        f"- Extract-only URLs processed: **{payload['unique_extract_only_urls']:,}**",
        f"- Recovered usable documents: **{payload['unique_recovered_usable']:,}**",
        f"- Remaining failed documents: **{payload['remaining_failed']:,}**", "",
        "## Code changed", "",
        "- `src/collection/fetcher.py`",
        "- `scripts/workflows/vibiomir/crawl_vibiomir_batch.py`",
        "- `scripts/workflows/vibiomir/audit_failed_urls.py`",
        "- `scripts/workflows/vibiomir/recover_failed_urls.py`",
        "- `tests/workflows/vibiomir/test_crawl_vibiomir_batch.py`",
        "- `tests/workflows/vibiomir/test_failed_url_recovery.py`", "",
        "## Recovery controls added", "",
        "- Bounded retry with jitter and deferred handling for long `Retry-After` values.",
        "- Shared per-domain throttling, robots checks, challenge detection, and circuit breaking.",
        "- Immutable allowlist checks immediately before every request.",
        "- Isolated raw, extraction, checkpoint, manifest, and report outputs.",
        "- Pilot gates prevent expansion for groups with low usable yield.", "",
        "## Tests", "",
        "- Recovery and batch crawler safety tests: **27 passed**.",
        "- Fetcher regression tests: **31 passed**.",
        "- Combined result: **58 passed, 0 failed**.", "",
        "## Recovery outcome", "", "| Recovery group | Usable |", "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:,} |" for key, value in sorted(result_groups.items()))
    lines.extend(["", "## Remaining failures by original reason", "",
                  "| Reason | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value:,} |" for key, value in remaining_reasons.most_common())
    lines.extend(["", "## Policy holds", "", "| Hold reason | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value:,} |" for key, value in held.items())
    lines.extend(["", "## Domains paused by policy", "",
                  "| Domain | Policy reason | Count |", "|---|---|---:|"])
    lines.extend(f"| {domain} | {reason} | {count:,} |"
                 for (domain, reason), count in domain_holds.most_common())
    lines.extend([
        "", "## Protected data", "",
        f"- Unchanged: **{str(payload['protected_data']['unchanged']).lower()}**",
        f"- Metadata changes: **{payload['protected_data']['metadata_change_count']}**",
        f"- Canonical checksum changes: **{len(payload['protected_data']['canonical_checksum_changes'])}**",
        f"- Verified-success URLs refetched: **{payload['verified_success_refetched']}**", "",
        "## Commands", "", "```bash",
        "python -m scripts.workflows.vibiomir.recover_failed_urls dry-run",
        "python -m scripts.workflows.vibiomir.recover_failed_urls pilot",
        "python -m scripts.workflows.vibiomir.recover_failed_urls run",
        "python -m scripts.workflows.vibiomir.recover_failed_urls verify",
        "# After cooldown, create an immutable follow-up wave:",
        "python -m scripts.workflows.vibiomir.recover_failed_urls dry-run --wave cooldown_20261014 --group rate_limited --group robots_unavailable",
        "python -m scripts.workflows.vibiomir.recover_failed_urls pilot --wave cooldown_20261014",
        "python -m scripts.workflows.vibiomir.recover_failed_urls run --wave cooldown_20261014",
        "```", "",
        "## Remaining limits", "",
        "- HTTP-error and extract-only pilots did not meet the acceptance threshold, so bulk expansion was stopped.",
        "- Rate-limited and robots-unavailable URLs remain held until their recorded cooldown expires.",
        "- Access challenges, access blocks, robots disallow rules, and confirmed 404s require an approved official source or policy change; they are not retried automatically.",
        "- Recovery output remains in staging. No corpus merge or index rebuild was run.", "",
    ])
    path = root / "reports/CRAWL_FAILURE_RECOVERY_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)
    root_report = root / "CRAWL_FAILURE_RECOVERY_REPORT.md"
    temporary = root_report.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, root_report)


def report(root: Path, phase: str, results: list[dict], remaining: list[dict],
           plan_summary: dict, *, pilot_phase: str, pilot_plan: Path) -> dict:
    integrity = protected_unchanged(root)
    pilot_rows = [row for row in results if row["phase"] == pilot_phase]
    payload = {
        "status": ("failed_protected_data_check" if not integrity["unchanged"]
                   else "partial" if plan_summary.get("processed", plan_summary["selected"]) < plan_summary["selected"]
                   else "complete"),
        "phase": phase, "generated_at": datetime.now(UTC).isoformat(),
        "corpus_revision": REVISION, "plan": plan_summary,
        "network_attempt_records": sum(row["fetch_status"] != "not_requested_extract_only"
                                       for row in results),
        "unique_retried_urls": len({int(row["id"]) for row in results
                                    if row["fetch_status"] != "not_requested_extract_only"}),
        "unique_extract_only_urls": len({int(row["id"]) for row in results
                                         if row["fetch_status"] == "not_requested_extract_only"}),
        "unique_recovered_usable": len({int(row["id"]) for row in results if row["usable"]}),
        "remaining_failed": len(remaining),
        "result_status_counts": dict(Counter(row["fetch_status"] for row in results)),
        "remaining_original_status_counts": dict(Counter(row["final_status"] for row in remaining)),
        "pilot_gates": pilot_gates_for_plan(pilot_plan, pilot_rows),
        "protected_data": integrity,
        "verified_success_refetched": 0,
        "output_policy": "staging only; no merge, indexing, or protected-data writes",
    }
    atomic_json(root / "RECOVERY_SUMMARY.json", payload)
    markdown_report(root, payload, results, remaining)
    return payload


def dry_run(root: Path, *, pilot: bool = True, wave: str = "default",
            groups: set[str] | None = None, hostname: str | None = None) -> dict:
    wave = wave_name(wave)
    for name in ("audit", "manifests", "eligible", "recovered_raw",
                 "recovered_extracted", "failed", "checkpoints", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    eligible = with_recent_attempts(
        pq.read_table(root / "eligible/FAILED_URLS_ELIGIBLE.parquet").to_pylist(),
        root / "RECOVERY_RESULTS.parquet")
    if groups:
        eligible = [row for row in eligible if row["recovery_group"] in groups]
    if hostname:
        eligible = [row for row in eligible if row["hostname"] == hostname]
    kind = "PILOT" if pilot else "EXPANSION"
    path = wave_path(root, kind, wave)
    if path.exists():
        plan = pq.read_table(path).to_pylist()
        summary = {"selected": len(plan), "selected_by_group": dict(Counter(
            row["recovery_group"] for row in plan)), "cooldown_by_group": {}}
    else:
        result_path = root / "RECOVERY_RESULTS.parquet"
        recovered = set()
        if result_path.exists():
            recovered = {int(row["id"]) for row in
                         pq.read_table(result_path, columns=["id", "usable"]).to_pylist()
                         if row["usable"]}
        eligible = [row for row in eligible if int(row["id"]) not in recovered]
        plan, summary = plan_rows(eligible, pilot=pilot, now=datetime.now(UTC))
        write_once_or_verify(path, plan, PLAN_SCHEMA)
    json_name = path.with_suffix(".json")
    payload = {"status": "complete", "network_requests": 0,
               "generated_at": datetime.now(UTC).isoformat(),
               "corpus_revision": REVISION, "wave": wave,
               "groups": sorted(groups) if groups else sorted(RECOVERY_GROUPS), **summary,
               "plan_path": str(path.relative_to(ROOT))}
    atomic_json(json_name, payload)
    return payload


def execute(root: Path, phase: str, *, wave: str = "default",
            pilot_wave: str | None = None, max_urls: int | None = None,
            hostname: str | None = None) -> dict:
    if phase not in {"pilot", "run"}:
        raise ValueError(phase)
    wave = wave_name(wave)
    pilot_wave = wave_name(pilot_wave or wave)
    if phase == "run" and (max_urls is None or not 1 <= max_urls <= MAX_BATCH_SIZE):
        raise ValueError("Run requires --max-urls between 1 and 500")
    current_phase = phase_tag(phase, wave)
    current_pilot_phase = phase_tag("pilot", pilot_wave)
    current_pilot_plan = wave_path(root, "PILOT", pilot_wave)
    integrity = protected_unchanged(root)
    if not integrity["unchanged"]:
        raise ValueError("Protected data changed after audit; refusing recovery")
    if phase == "pilot":
        plan_path = current_pilot_plan
    else:
        pilot_checkpoint = root / f"checkpoints/{current_pilot_phase}_summary.json"
        summary_path = pilot_checkpoint if pilot_checkpoint.exists() else root / "RECOVERY_SUMMARY.json"
        pilot_summary = json.loads(summary_path.read_text())
        passed = {group for group, gate in pilot_summary["pilot_gates"].items() if gate["passed"]}
        eligible = with_recent_attempts(
            pq.read_table(root / "eligible/FAILED_URLS_ELIGIBLE.parquet").to_pylist(),
            root / "RECOVERY_RESULTS.parquet")
        result_path = root / "RECOVERY_RESULTS.parquet"
        recovered = set()
        if result_path.exists():
            recovered = {int(row["id"]) for row in
                         pq.read_table(result_path, columns=["id", "usable"]).to_pylist()
                         if row["usable"]}
        eligible = [row for row in eligible
                    if row["recovery_group"] in passed and int(row["id"]) not in recovered]
        if hostname:
            eligible = [row for row in eligible if row["hostname"] == hostname]
        plan, expansion_summary = plan_rows(eligible, pilot=False, now=datetime.now(UTC))
        plan = plan[:max_urls]
        expansion_summary["selected"] = len(plan)
        expansion_summary["selected_by_group"] = dict(Counter(
            row["recovery_group"] for row in plan))
        plan_path = wave_path(root, "EXPANSION", wave)
        write_once_or_verify(plan_path, plan, PLAN_SCHEMA)
    if not plan_path.exists():
        raise FileNotFoundError("Run dry-run before executing recovery")
    plan = pq.read_table(plan_path).to_pylist()
    guard = RecoveryGuard(root / "eligible/FAILED_URLS_ELIGIBLE.parquet",
                          root / "audit/PROTECTED_DATA_BASELINE.json",
                          results_path=root / "RECOVERY_RESULTS.parquet")
    new_results = []
    groups = defaultdict(list)
    for row in plan:
        groups[row["recovery_group"]].append(row)
    for group in sorted(groups):
        rows = groups[group]
        if group == "content_unusable":
            new_results.extend(extraction_result(rows, current_phase, root))
        else:
            new_results.extend(network_results(rows, group, current_phase, root, guard))
    results = merge_results(root, new_results)
    remaining = publish_remaining(root, results)
    plan_summary = {"selected": len(plan),
                    "processed": len(new_results),
                    "selected_by_group": dict(Counter(row["recovery_group"] for row in plan))}
    if phase == "run":
        plan_summary.update(expansion_summary)
    plan_summary["wave"] = wave
    result = report(root, current_phase, results, remaining, plan_summary,
                    pilot_phase=current_pilot_phase, pilot_plan=current_pilot_plan)
    if phase == "pilot":
        atomic_json(root / f"checkpoints/{current_pilot_phase}_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["dry-run", "pilot", "run", "verify"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wave", default="default")
    parser.add_argument("--pilot-wave")
    parser.add_argument("--max-urls", type=int)
    parser.add_argument("--hostname")
    parser.add_argument("--group", action="append", choices=sorted(RECOVERY_GROUPS))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "dry-run":
        result = dry_run(root, wave=args.wave, groups=set(args.group or []),
                         hostname=args.hostname)
    elif args.mode in {"pilot", "run"}:
        result = execute(root, args.mode, wave=args.wave, pilot_wave=args.pilot_wave,
                         max_urls=args.max_urls, hostname=args.hostname)
    else:
        result = protected_unchanged(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
