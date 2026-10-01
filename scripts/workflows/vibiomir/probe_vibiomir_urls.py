"""Probe a small, stratified ViBioMIR URL pilot with robots checks and resume."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workflows.vibiomir.prepare_vibiomir import OUTPUT_DIR, sha256_file
from src.collection.fetcher import PoliteFetcher

SAMPLE_PATH = OUTPUT_DIR / "sample_urls.parquet"
PROBE_DIR = OUTPUT_DIR / "probe"
FAILURE_STATUSES = {
    "robots_unavailable", "http_error", "timeout", "network_error",
    "too_large", "unsupported_scheme",
}
RESULT_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("domain", pa.string()),
    ("selected_group", pa.string()), ("status", pa.string()),
    ("attempt_count", pa.int32()), ("robots_url", pa.string()),
    ("robots_status", pa.int32()), ("robots_allowed", pa.bool_()),
    ("http_status", pa.int32()), ("final_url", pa.string()),
    ("redirect_count", pa.int32()), ("content_type", pa.string()),
    ("declared_content_length", pa.int64()), ("bytes_downloaded", pa.int64()),
    ("elapsed_ms", pa.float64()), ("fetched_at", pa.string()),
    ("content_sha256", pa.string()), ("raw_file", pa.string()),
    ("error_type", pa.string()), ("error_message", pa.string()),
])


def select_probe_urls(rows: list[dict], limit: int = 20) -> list[dict]:
    """Select by domain size strata, then lowest stable hash within each domain."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    by_domain: dict[str, list[dict]] = {}
    ids = set()
    urls = set()
    for row in rows:
        if row["id"] in ids or row["url"] in urls:
            raise ValueError("Sample contains duplicate ID or URL")
        ids.add(row["id"])
        urls.add(row["url"])
        by_domain.setdefault(row["domain"], []).append(row)
    if limit > len(by_domain):
        raise ValueError(f"Requested {limit} URLs but sample has only {len(by_domain)} domains")
    domain_counts = {}
    for domain, candidates in by_domain.items():
        counts = {row["domain_url_count"] for row in candidates}
        if len(counts) != 1 or next(iter(counts)) <= 0:
            raise ValueError(f"Inconsistent domain_url_count for {domain}")
        domain_counts[domain] = next(iter(counts))
    ordered = sorted(by_domain, key=lambda domain: (-domain_counts[domain], domain))
    if limit in (1, 2):
        large_n, medium_n = 1, 0
    else:
        large_n = max(1, round(limit * 0.4))
        medium_n = max(1, round(limit * 0.3))
    small_n = limit - large_n - medium_n
    if limit >= 3 and small_n < 1:
        medium_n -= 1
        small_n = 1
    large = ordered[:large_n]
    small = ordered[-small_n:] if small_n else []
    middle = ordered[large_n:len(ordered) - small_n if small_n else len(ordered)]
    middle_start = (len(middle) - medium_n) // 2
    medium = middle[middle_start:middle_start + medium_n]
    chosen = [(domain, "large") for domain in large]
    chosen += [(domain, "medium") for domain in medium]
    chosen += [(domain, "small") for domain in small]
    if len(chosen) != limit or len({domain for domain, _ in chosen}) != limit:
        raise ValueError("Domain strata overlap")
    result = []
    for domain, group in chosen:
        best = min(by_domain[domain], key=lambda row: (
            row["selection_hash"], row["id"], row["url"]))
        result.append({"id": best["id"], "url": best["url"], "domain": domain,
                       "domain_url_count": domain_counts[domain],
                       "selection_hash": best["selection_hash"], "selected_group": group})
    return result


def temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    return Path(name)


def write_json_atomic(destination: Path, payload: dict) -> None:
    temporary = temporary_path(destination)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_results_atomic(destination: Path, rows: list[dict]) -> None:
    temporary = temporary_path(destination)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=RESULT_SCHEMA), temporary,
                       compression="zstd", version="2.6")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def make_summary(selected: list[dict], results: list[dict], started_at: str) -> dict:
    status_counts = Counter(row["status"] for row in results)
    http_counts = Counter(str(row["http_status"]) for row in results
                          if row["http_status"] is not None)
    type_counts = Counter((row["content_type"] or "").split(";")[0].strip().lower()
                          for row in results if row["content_type"])
    total = sum(row["bytes_downloaded"] for row in results)
    return {
        "requested_urls": len(selected), "processed_urls": len(results),
        "success_count": status_counts["success"],
        "status_counts": dict(sorted(status_counts.items())),
        "http_status_counts": dict(sorted(http_counts.items(), key=lambda item: int(item[0]))),
        "content_type_counts": dict(sorted(type_counts.items())),
        "total_bytes": total,
        "average_bytes": round(total / len(results), 2) if results else 0,
        "average_elapsed_ms": round(sum(row["elapsed_ms"] for row in results) / len(results), 2) if results else 0,
        "robots_allowed_count": sum(row["robots_allowed"] is True for row in results),
        "robots_denied_count": status_counts["robots_denied"],
        "robots_unavailable_count": status_counts["robots_unavailable"],
        "started_at": started_at, "completed_at": datetime.now(UTC).isoformat(),
    }


def probe(sample_file: Path = SAMPLE_PATH, output_dir: Path = PROBE_DIR,
          *, limit: int = 20, timeout_seconds: float = 30,
          max_bytes: int = 5_242_880, delay_seconds: float = 0.5,
          user_agent: str = "medical-rag-research/0.1", force: bool = False,
          retry_failed: bool = False, client: httpx.Client | None = None,
          sleep=time.sleep, progress=print) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    sample_file, output_dir = Path(sample_file), Path(output_dir)
    sample_hash = sha256_file(sample_file)
    sample_summary = json.loads(sample_file.with_name("sample_summary.json").read_text())
    if sample_summary["output_sha256"] != sample_hash:
        raise ValueError("Sample SHA-256 does not match sample_summary.json")
    sample_rows = pq.read_table(sample_file).to_pylist()  # The sample has only 500 rows.
    if len(sample_rows) != sample_summary["sample_size_actual"]:
        raise ValueError("Sample row count does not match its summary")
    selected = select_probe_urls(sample_rows, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "probe_results.parquet"
    summary_path = output_dir / "probe_summary.json"
    selection_path = output_dir / "selected_urls.json"
    existing = {}
    if results_path.exists():
        table = pq.read_table(results_path)
        if table.schema != RESULT_SCHEMA:
            raise ValueError("Existing probe result schema does not match")
        for row in table.to_pylist():
            if row["id"] in existing:
                raise ValueError("Existing probe results contain duplicate IDs")
            existing[row["id"]] = row
    selected_by_id = {row["id"]: row for row in selected}
    if any(id_ not in selected_by_id or existing[id_]["url"] != selected_by_id[id_]["url"]
           for id_ in existing):
        raise ValueError("Existing probe results use another selection; choose a new output directory")
    previous_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    started_at = previous_summary.get("started_at") or datetime.now(UTC).isoformat()
    write_json_atomic(selection_path, {"sample_sha256": sample_hash, "requested_urls": limit,
                                       "selected_urls": selected})
    progress(f"Selected {limit} URLs across {len({row['domain'] for row in selected})} domains:")
    for item in selected:
        progress(f"  {item['selected_group']}: {item['domain']} | id={item['id']} | {item['url']}")
    results = {} if force else existing.copy()
    todo = [entry for entry in selected if entry["id"] not in results or
            (retry_failed and results[entry["id"]]["status"] in FAILURE_STATUSES)]
    if not todo and summary_path.exists():
        progress("All selected IDs already have terminal results; no HTTP request sent.")
        return previous_summary
    with PoliteFetcher(output_dir, timeout_seconds=timeout_seconds, max_bytes=max_bytes,
                       delay_seconds=delay_seconds, user_agent=user_agent,
                       client=client, sleep=sleep) as fetcher:
        for entry in todo:
            result = fetcher.fetch(entry)
            results[entry["id"]] = result
            ordered_results = [results[item["id"]] for item in selected if item["id"] in results]
            summary = make_summary(selected, ordered_results, started_at)
            write_results_atomic(results_path, ordered_results)
            write_json_atomic(summary_path, summary)
            progress(f"{len(ordered_results)}/{limit} id={entry['id']} "
                     f"{entry['domain']} -> {result['status']} "
                     f"HTTP={result['http_status']} bytes={result['bytes_downloaded']}")
    if not todo:
        ordered_results = [results[item["id"]] for item in selected if item["id"] in results]
        summary = make_summary(selected, ordered_results, started_at)
        write_results_atomic(results_path, ordered_results)
        write_json_atomic(summary_path, summary)
    return summary


def peak_memory_mib() -> float | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROBE_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--max-bytes", type=int, default=5_242_880)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--user-agent", default="medical-rag-research/0.1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        summary = probe(args.sample_file, args.output_dir, limit=args.limit,
                        timeout_seconds=args.timeout_seconds, max_bytes=args.max_bytes,
                        delay_seconds=args.delay_seconds, user_agent=args.user_agent,
                        force=args.force, retry_failed=args.retry_failed,
                        progress=lambda line: print(line, flush=True))
    except (OSError, ValueError, KeyError, httpx.HTTPError, pa.ArrowException) as error:
        print(f"Probe failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    peak = peak_memory_mib()
    print(f"Elapsed: {time.perf_counter() - started:.2f}s; "
          f"peak process memory: {f'{peak:.2f} MiB' if peak is not None else 'unavailable'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
