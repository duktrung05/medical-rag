"""Crawl at most 500 selected ViBioMIR URLs into raw, checkpointed files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_vibiomir import sha256_file
from src.collection.fetcher import BatchFetcher, BatchFetchState

DEFAULT_INPUT = Path("data/vibiomir/sample_urls.parquet")
DEFAULT_OUTPUT = Path("data/vibiomir/crawl/pilot500")
SCHEMA = pa.schema([
    ("id", pa.int64()), ("source_url", pa.string()),
    ("final_url", pa.string()), ("domain", pa.string()),
    ("status", pa.string()), ("attempts", pa.int32()),
    ("robots_status", pa.int32()), ("robots_allowed", pa.bool_()),
    ("robots_policy_status", pa.string()), ("robots_http_status", pa.int32()),
    ("robots_final_url", pa.string()), ("robots_redirect_count", pa.int32()),
    ("robots_error_type", pa.string()), ("robots_error_message", pa.string()),
    ("http_status", pa.int32()), ("content_type", pa.string()),
    ("detected_file_type", pa.string()), ("raw_file", pa.string()),
    ("bytes_downloaded", pa.int64()), ("content_sha256", pa.string()),
    ("redirect_count", pa.int32()), ("elapsed_ms", pa.float64()),
    ("fetched_at", pa.string()), ("error_type", pa.string()),
    ("error_message", pa.string()),
    ("previous_status", pa.string()), ("previous_attempts", pa.int32()),
])
FAILURE_STATUSES = {
    "robots_denied", "robots_unavailable", "not_found", "rate_limited",
    "http_error", "timeout", "connection_error", "too_large",
    "unsupported_content", "invalid_url",
}
TYPE_EXTENSIONS = {"html": "html", "pdf": "pdf", "text": "txt", "unknown": "bin"}


def detect_file_type(prefix: bytes, content_type: str | None) -> str | None:
    """Return raw file family; None means an explicitly unsupported format."""
    stripped = prefix.lstrip(b"\xef\xbb\xbf\x00\t\n\r ").lower()
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if stripped.startswith(b"%pdf-"):
        return "pdf"
    if (stripped.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))
            or b"<html" in stripped[:2048]):
        return "html"
    if prefix.startswith((b"\x89PNG\r\n", b"\xff\xd8\xff", b"GIF8", b"PK\x03\x04")):
        return None
    if mime in {"text/html", "application/xhtml+xml"}:
        return "html"
    if mime == "application/pdf":
        return "pdf"
    if mime in {"text/plain", "text/markdown", "application/json", "application/xml", "text/xml"}:
        return "text"
    if mime.startswith(("image/", "audio/", "video/")) or mime in {
        "text/css", "application/javascript", "text/javascript", "application/zip",
    }:
        return None
    return "unknown"


def _atomic_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                           dir=destination.parent)
    os.close(handle)
    return Path(name)


def _write_atomic(destination: Path, writer) -> None:
    temp = _atomic_path(destination)
    try:
        writer(temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _raw_path(output_dir: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    base = (output_dir / "raw").resolve()
    path = (output_dir / relative).resolve()
    return path if path.is_relative_to(base) else None


def _valid_success(output_dir: Path, row: dict) -> bool:
    path = _raw_path(output_dir, row.get("raw_file"))
    return bool(row.get("status") == "success" and path and path.is_file()
                and path.stat().st_size == row.get("bytes_downloaded")
                and sha256_file(path) == row.get("content_sha256"))


def _upgrade_row(row: dict) -> dict:
    """Add RFC 9309 fields to a legacy pilot manifest row without changing its result."""
    if "robots_policy_status" in row:
        return row
    status = row["status"]
    policy = ("unreachable" if status == "robots_unavailable" else
              "denied" if status == "robots_denied" else
              "unavailable_allow" if row["robots_status"] == 404 else "fetched")
    row.update(robots_policy_status=policy, robots_http_status=row["robots_status"],
               robots_final_url=None, robots_redirect_count=None,
               robots_error_type=row["error_type"] if status == "robots_unavailable" else None,
               robots_error_message=row["error_message"] if status == "robots_unavailable" else None,
               previous_status=None, previous_attempts=None)
    return row


def _clean_orphan_raw(output_dir: Path, selected: list[dict], results: dict[int, dict]) -> int:
    """Remove this pilot's uncheckpointed raw files left by an interrupted run."""
    selected_ids = {int(entry["id"]) for entry in selected}
    expected = {(_raw_path(output_dir, row.get("raw_file"))) for row in results.values()
                if row["status"] == "success"}
    removed = 0
    raw_root = output_dir / "raw"
    if not raw_root.exists():
        return removed
    for directory in TYPE_EXTENSIONS:
        for path in (raw_root / directory).glob("*"):
            if (path.is_file() and path.stem.isdigit()
                    and int(path.stem) in selected_ids and path.resolve() not in expected):
                path.unlink()
                removed += 1
    for path in raw_root.glob(".*.tmp"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def interleave_by_domain(entries: list[dict]) -> list[dict]:
    """Spread domains across workers while preserving order within each domain."""
    queues = defaultdict(deque)
    for entry in entries:
        queues[entry["domain"]].append(entry)
    ordered = []
    while queues:
        for domain in list(queues):
            ordered.append(queues[domain].popleft())
            if not queues[domain]:
                del queues[domain]
    return ordered


def _convert(probe: dict, output_dir: Path) -> dict:
    status = probe["status"]
    http_status = probe["http_status"]
    if status == "unsupported_scheme":
        status = "invalid_url"
    elif status == "network_error":
        status = "connection_error"
    elif status == "http_error" and http_status == 404:
        status = "not_found"
    elif status == "http_error" and http_status == 429:
        status = "rate_limited"
    detected = None
    raw_relative = None
    if status == "success":
        stage = _raw_path(output_dir, probe["raw_file"])
        if stage is None or not stage.is_file():
            raise ValueError("Successful fetch has no raw stage file")
        with stage.open("rb") as stream:
            prefix = stream.read(4096)
        detected = detect_file_type(prefix, probe["content_type"])
        if detected is None:
            stage.unlink()
            status = "unsupported_content"
        else:
            raw_relative = (f"raw/{detected}/{probe['id']}."
                            f"{TYPE_EXTENSIONS[detected]}")
            destination = output_dir / raw_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
    if status != "success":
        raw_relative = None
    return {
        "id": int(probe["id"]), "source_url": probe["url"],
        "final_url": probe["final_url"], "domain": probe["domain"],
        "status": status, "attempts": probe["attempt_count"],
        "robots_status": probe["robots_status"],
        "robots_allowed": probe["robots_allowed"],
        "robots_policy_status": probe["robots_policy_status"],
        "robots_http_status": probe["robots_http_status"],
        "robots_final_url": probe["robots_final_url"],
        "robots_redirect_count": probe["robots_redirect_count"],
        "robots_error_type": probe["robots_error_type"],
        "robots_error_message": probe["robots_error_message"],
        "http_status": http_status, "content_type": probe["content_type"],
        "detected_file_type": detected, "raw_file": raw_relative,
        "bytes_downloaded": probe["bytes_downloaded"],
        "content_sha256": probe["content_sha256"] if status == "success" else None,
        "redirect_count": probe["redirect_count"],
        "elapsed_ms": probe["elapsed_ms"], "fetched_at": probe["fetched_at"],
        "error_type": ("unsupported_content" if status == "unsupported_content"
                       else probe["error_type"]),
        "error_message": ("Unsupported media type or magic bytes"
                          if status == "unsupported_content" else probe["error_message"]),
        "previous_status": None, "previous_attempts": None,
    }


def _summary(selected: list[dict], results: dict[int, dict], *, duration: float,
             skipped: int, new_count: int, started_at: str) -> dict:
    rows = [results[x["id"]] for x in selected if x["id"] in results]
    status_counts = Counter(r["status"] for r in rows)
    http_counts = Counter(str(r["http_status"]) for r in rows if r["http_status"] is not None)
    type_counts = Counter(r["detected_file_type"] for r in rows if r["status"] == "success")
    errors = Counter(r["error_type"] or r["status"] for r in rows if r["status"] != "success")
    robots_policies = Counter(r["robots_policy_status"] for r in rows
                              if r.get("robots_policy_status"))
    robots_errors = Counter(r["robots_error_type"] for r in rows
                            if r.get("robots_error_type"))
    transitions = Counter(f"{r['previous_status']}->{r['status']}" for r in rows
                          if r.get("previous_status"))
    domains = defaultdict(lambda: {"processed": 0, "success": 0})
    for row in rows:
        domains[row["domain"]]["processed"] += 1
        domains[row["domain"]]["success"] += row["status"] == "success"
    low_success = sorted(({"domain": domain, **counts,
                           "success_rate": counts["success"] / counts["processed"]}
                          for domain, counts in domains.items()),
                         key=lambda x: (x["success_rate"], -x["processed"], x["domain"]))[:20]
    sample_domains = defaultdict(lambda: {"sample_count": 0, "corpus_url_count": 0})
    for entry in selected:
        sample_domains[entry["domain"]]["sample_count"] += 1
        sample_domains[entry["domain"]]["corpus_url_count"] = entry["domain_url_count"]
    largest = sorted(({"domain": domain, **counts} for domain, counts in sample_domains.items()),
                     key=lambda x: (-x["corpus_url_count"], x["domain"]))[:20]
    total_bytes = sum(r["bytes_downloaded"] for r in rows)
    raw_bytes = sum(r["bytes_downloaded"] for r in rows if r["status"] == "success")
    fetched_times = [datetime.fromisoformat(r["fetched_at"]) for r in rows if r["fetched_at"]]
    wall_span = ((max(fetched_times) - min(fetched_times)).total_seconds()
                 if fetched_times else 0.0)
    retry_times = [datetime.fromisoformat(r["fetched_at"]) for r in rows
                   if r.get("previous_status") and r["fetched_at"]]
    retry_span = ((max(retry_times) - min(retry_times)).total_seconds()
                  if retry_times else 0.0)
    return {
        "requested_urls": len(selected), "processed_urls": len(rows),
        "success_count": status_counts["success"],
        "status_counts": dict(sorted(status_counts.items())),
        "http_status_counts": dict(sorted(http_counts.items())),
        "file_type_counts": dict(sorted(type_counts.items())),
        "robots_policy_status_counts": dict(sorted(robots_policies.items())),
        "robots_error_type_counts": dict(sorted(robots_errors.items())),
        "retry_transition_counts": dict(sorted(transitions.items())),
        "retry_processed_count": len(retry_times),
        "retry_duration_seconds": round(retry_span, 3),
        "domain_counts": dict(sorted(domains.items())),
        "total_bytes": total_bytes, "total_raw_bytes": raw_bytes,
        "duration_seconds": round(max(duration, wall_span), 3),
        "run_duration_seconds": round(duration, 3),
        "average_pages_per_second": round(len(rows) / max(duration, wall_span), 5)
        if max(duration, wall_span) > 0 else 0,
        "skipped_due_to_resume": skipped,
        "common_errors": [{"error": k, "count": v} for k, v in errors.most_common(20)],
        "lowest_success_domains": low_success, "largest_domains": largest,
        "started_at": started_at, "completed_at": datetime.now(UTC).isoformat(),
        "first_fetched_at": min(fetched_times).isoformat() if fetched_times else None,
        "last_fetched_at": max(fetched_times).isoformat() if fetched_times else None,
    }


def _checkpoint(output_dir: Path, selected: list[dict], results: dict[int, dict],
                *, started: float, skipped: int, new_count: int,
                started_at: str) -> dict:
    ordered = [results[x["id"]] for x in selected if x["id"] in results]
    _write_atomic(output_dir / "crawl_results.parquet",
                  lambda path: pq.write_table(pa.Table.from_pylist(ordered, schema=SCHEMA),
                                              path, compression="zstd"))
    summary = _summary(selected, results, duration=time.monotonic() - started,
                       skipped=skipped, new_count=new_count, started_at=started_at)
    _write_atomic(output_dir / "crawl_summary.json",
                  lambda path: path.write_text(json.dumps(summary, ensure_ascii=False,
                                                          indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8"))
    return summary


def crawl_batch(input_file: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
                *, limit: int = 500, concurrency: int = 4,
                delay_seconds: float = 1, timeout_seconds: float = 30,
                max_bytes: int = 15_728_640,
                user_agent: str = "medical-rag-vibiomir-pilot/0.1",
                retry_failed: bool = False, force: bool = False,
                retry_status: set[str] | None = None,
                expected_retry_count: int | None = None,
                client_factory=None, sleep=time.sleep, progress=print) -> dict:
    if not 1 <= limit <= 500 or not 1 <= concurrency <= 4:
        raise ValueError("Pilot limit must be 1..500 and concurrency 1..4")
    if retry_status and (force or retry_failed):
        raise ValueError("--retry-status cannot be combined with --force or --retry-failed")
    input_file, output_dir = Path(input_file), Path(output_dir)
    sample = pq.read_table(input_file).to_pylist()
    if limit > len(sample):
        raise ValueError("limit exceeds sample row count")
    sample_summary = input_file.with_name("sample_summary.json")
    if sample_summary.exists():
        expected = json.loads(sample_summary.read_text(encoding="utf-8"))["output_sha256"]
        if sha256_file(input_file) != expected:
            raise ValueError("Sample SHA-256 differs from sample_summary.json")
    selected = sample[:limit]
    ids = [int(x["id"]) for x in selected]
    if len(ids) != len(set(ids)) or len({x["url"] for x in selected}) != len(selected):
        raise ValueError("Selected sample has duplicate official ID or URL")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "crawl_results.parquet"
    existing = {}
    if manifest.exists():
        table = pq.read_table(manifest)
        if not set(table.column_names).issubset(set(SCHEMA.names)):
            raise ValueError("Existing crawl manifest schema differs")
        existing = {int(row["id"]): _upgrade_row(row) for row in table.to_pylist()}
        if len(existing) != table.num_rows:
            raise ValueError("Duplicate IDs in existing crawl manifest")
    by_id = {int(x["id"]): x for x in selected}
    if any(id_ not in by_id or row["source_url"] != by_id[id_]["url"]
           for id_, row in existing.items()):
        raise ValueError("Existing manifest uses another sample selection")
    todo = []
    for entry in selected:
        old = existing.get(entry["id"])
        if retry_status:
            if old and old["status"] in retry_status:
                todo.append(entry)
        elif (force or old is None
              or (old["status"] == "success" and not _valid_success(output_dir, old))
              or (retry_failed and old["status"] != "success")):
            todo.append(entry)
    if expected_retry_count is not None and len(todo) != expected_retry_count:
        raise ValueError(f"Expected {expected_retry_count} retry URLs, found {len(todo)}")
    if retry_status:
        protected = {id_: row for id_, row in existing.items() if row["status"] == "success"}
        invalid = [id_ for id_, row in protected.items() if not _valid_success(output_dir, row)]
        if invalid:
            raise ValueError(f"Existing success raw checksum mismatch: {invalid[:10]}")
        archive = output_dir / "crawl_results_before_rfc9309.parquet"
        if not archive.exists():
            _write_atomic(archive, lambda path: path.write_bytes(manifest.read_bytes()))
        baseline = output_dir / "pre_retry_success_sha256.json"
        if not baseline.exists():
            payload = {str(id_): {"raw_file": row["raw_file"],
                                   "sha256": row["content_sha256"],
                                   "bytes": row["bytes_downloaded"]}
                       for id_, row in sorted(protected.items())}
            _write_atomic(baseline, lambda path: path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    skipped = len(selected) - len(todo)
    todo = interleave_by_domain(todo)
    domain_count = len({x["domain"] for x in selected})
    progress(f"Pilot plan: {len(selected)} URL, {domain_count} domains, concurrency={concurrency}, "
             f"per-domain delay={delay_seconds}s, timeout={timeout_seconds}s, "
             f"max_bytes={max_bytes}; pending={len(todo)}, resume_skip={skipped}.")
    progress("Estimated page-request floor ~2 minutes at 4 workers; network timeouts "
             "may extend the run beyond an hour. Redirects and robots checks add requests.")
    started = time.monotonic()
    fetched_before = [row["fetched_at"] for row in existing.values() if row["fetched_at"]]
    started_at = min(fetched_before) if fetched_before else datetime.now(UTC).isoformat()
    if not todo:
        removed = _clean_orphan_raw(output_dir, selected, existing)
        if removed:
            progress(f"Removed {removed} orphan raw file(s) from interrupted work.")
        return _checkpoint(output_dir, selected, existing, started=started,
                           skipped=skipped, new_count=0, started_at=started_at)
    state = BatchFetchState()
    local = threading.local()
    workers = []
    worker_guard = threading.Lock()

    def fetch_one(entry: dict) -> dict:
        if not hasattr(local, "fetcher"):
            client = client_factory() if client_factory is not None else None
            local.fetcher = BatchFetcher(output_dir, state=state,
                                         timeout_seconds=timeout_seconds,
                                         max_bytes=max_bytes, delay_seconds=delay_seconds,
                                         user_agent=user_agent, client=client, sleep=sleep)
            with worker_guard:
                workers.append((local.fetcher, client))
        probe = local.fetcher.fetch({"id": int(entry["id"]), "url": entry["url"],
                                    "domain": entry["domain"], "selected_group": "batch"})
        return _convert(probe, output_dir)

    new_count = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            iterator = iter(todo)
            pending = {}
            for _ in range(min(concurrency, len(todo))):
                item = next(iterator)
                pending[executor.submit(fetch_one, item)] = item
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = pending.pop(future)
                    row = future.result()
                    old = existing.get(entry["id"])
                    if old is not None:
                        row["previous_status"] = old["status"]
                        row["previous_attempts"] = old["attempts"]
                    if old and old.get("raw_file") and old["raw_file"] != row.get("raw_file"):
                        stale = _raw_path(output_dir, old["raw_file"])
                        if stale:
                            stale.unlink(missing_ok=True)
                    existing[entry["id"]] = row
                    new_count += 1
                    summary = _checkpoint(output_dir, selected, existing, started=started,
                                          skipped=skipped, new_count=new_count,
                                          started_at=started_at)
                    progress(f"{summary['processed_urls']}/{len(selected)} "
                             f"id={row['id']} {row['domain']} -> {row['status']} "
                             f"HTTP={row['http_status']} bytes={row['bytes_downloaded']}")
                    try:
                        item = next(iterator)
                    except StopIteration:
                        continue
                    pending[executor.submit(fetch_one, item)] = item
    finally:
        for fetcher, client in workers:
            fetcher.close()
            if client is not None:
                client.close()
    _clean_orphan_raw(output_dir, selected, existing)
    if retry_status:
        baseline = json.loads((output_dir / "pre_retry_success_sha256.json").read_text())
        changed = [id_ for id_, prior in baseline.items()
                   if not _valid_success(output_dir, {
                       "status": "success", "raw_file": prior["raw_file"],
                       "bytes_downloaded": prior["bytes"],
                       "content_sha256": prior["sha256"],
                   })]
        if changed:
            raise ValueError(f"Protected success raw changed: {changed[:10]}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--max-bytes", type=int, default=15_728_640)
    parser.add_argument("--user-agent", default="medical-rag-vibiomir-pilot/0.1")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-status", action="append", choices=sorted(FAILURE_STATUSES))
    parser.add_argument("--expected-retry-count", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = crawl_batch(args.input, args.output_dir, limit=args.limit,
                              concurrency=args.concurrency,
                              delay_seconds=args.delay_seconds,
                              timeout_seconds=args.timeout_seconds,
                              max_bytes=args.max_bytes, user_agent=args.user_agent,
                              retry_failed=args.retry_failed, force=args.force,
                              retry_status=set(args.retry_status or []),
                              expected_retry_count=args.expected_retry_count,
                              progress=lambda line: print(line, flush=True))
    except (OSError, ValueError, KeyError, httpx.HTTPError, pa.ArrowException) as exc:
        print(f"Batch crawl failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
