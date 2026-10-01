"""Inventory local ViBioMIR files; stream corpus records into disk-backed SQLite.

Duplicate counts are excess non-null occurrences. URL equality uses the original
string. Domain counts include duplicate valid URL rows. No URLs are fetched.
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data/raw/vibiomir"
QUERY_PATH = RAW_DATA_DIR / "query.parquet"
CORPUS_PATH = RAW_DATA_DIR / "links_corpus.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/vibiomir"
REPORT_FILES = ("domain_stats.csv", "invalid_urls.csv", "duplicate_urls.csv", "inventory.json")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file with bounded memory."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_schema(schema: pa.Schema, *, corpus: bool) -> None:
    names = ["id", "url" if corpus else "query"]
    if schema.names != names:
        raise ValueError(f"Invalid schema columns: expected {names}; got {schema.names}")
    valid_id = (schema.field("id").type == pa.int64() if corpus
                else pa.types.is_integer(schema.field("id").type))
    if not valid_id or schema.field(names[1]).type != pa.string():
        kind = "int64" if corpus else "integer"
        raise ValueError(f"Invalid {'corpus' if corpus else 'query'} schema: "
                         f"expected id: {kind}, {names[1]}: string; got {schema}")


def schema_inventory(schema: pa.Schema) -> list[dict]:
    return [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema]


def inspect_queries(parquet: pq.ParquetFile) -> dict:
    """Read the small query file in full, preserving integer IDs and nulls."""
    table = parquet.read(use_threads=False)
    ids = table.column("id").to_pylist()
    queries = table.column("query").to_pylist()
    valid_ids = [value for value in ids if value is not None]
    return {
        "rows": table.num_rows, "columns": table.column_names,
        "schema": schema_inventory(table.schema),
        "id_min": min(valid_ids, default=None), "id_max": max(valid_ids, default=None),
        "null_ids": len(ids) - len(valid_ids),
        "duplicate_ids": len(valid_ids) - len(set(valid_ids)),
        "null_queries": sum(value is None for value in queries),
        "empty_queries": sum(value is not None and not value.strip() for value in queries),
    }


def classify_url(value: str | None, *, strip_www: bool) -> tuple[str | None, str | None]:
    """Return domain/reason; stripping is for parsing, never for exact equality."""
    if value is None:
        return None, "null_url"
    if not value.strip():
        return None, "empty_url"
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return None, "unsupported_scheme"
        hostname = parsed.hostname
    except ValueError:
        return None, "malformed_url"
    domain = hostname.lower().rstrip(".") if hostname else ""
    if not domain:
        return None, "missing_hostname"
    if strip_www and domain.startswith("www."):
        domain = domain[4:]
    return domain, None


def duplicate_count(connection: sqlite3.Connection, column: str) -> int:
    # Only internal fixed column names are passed here.
    return connection.execute(
        f"SELECT COALESCE(SUM(n - 1), 0) FROM "
        f"(SELECT COUNT(*) AS n FROM records WHERE {column} IS NOT NULL "
        f"GROUP BY {column} HAVING COUNT(*) > 1)"
    ).fetchone()[0]


def inspect_corpus(parquet: pq.ParquetFile, staging: Path, *, batch_size: int,
                   strip_www: bool, progress) -> dict:
    stats = {
        "rows": parquet.metadata.num_rows, "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.schema_arrow.names, "schema": schema_inventory(parquet.schema_arrow),
        "id_min": None, "id_max": None, "null_ids": 0,
    }
    reasons = Counter()
    processed = 0
    with closing(sqlite3.connect(staging / "inventory.sqlite3")) as connection:
        # Disposable scratch database: bounded cache, disk-backed sorting.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("PRAGMA mmap_size=0")
        connection.execute("CREATE TABLE records (id INTEGER, url TEXT COLLATE BINARY)")
        connection.execute("CREATE TABLE domains (domain TEXT PRIMARY KEY, url_count INTEGER NOT NULL)")
        with (staging / "invalid_urls.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "url", "reason"])
            for index, batch in enumerate(parquet.iter_batches(batch_size=batch_size, use_threads=False), 1):
                ids, urls = batch.column(0).to_pylist(), batch.column(1).to_pylist()
                domains = Counter()
                for official_id, url in zip(ids, urls):
                    if official_id is None:
                        stats["null_ids"] += 1
                    else:
                        stats["id_min"] = official_id if stats["id_min"] is None else min(stats["id_min"], official_id)
                        stats["id_max"] = official_id if stats["id_max"] is None else max(stats["id_max"], official_id)
                    domain, reason = classify_url(url, strip_www=strip_www)
                    if reason:
                        reasons[reason] += 1
                        writer.writerow([official_id, url, reason])
                    else:
                        domains[domain] += 1
                with connection:
                    connection.executemany("INSERT INTO records VALUES (?, ?)", zip(ids, urls))
                    connection.executemany(
                        "INSERT INTO domains VALUES (?, ?) ON CONFLICT(domain) "
                        "DO UPDATE SET url_count=url_count+excluded.url_count", domains.items())
                processed += batch.num_rows
                progress(f"Corpus batch {index}: {processed:,}/{stats['rows']:,} rows")
        if processed != stats["rows"]:
            raise ValueError("Corpus metadata row count differs from scanned row count")
        progress("Building disk-backed ID/URL indexes and checking duplicates...")
        connection.execute("CREATE INDEX record_ids ON records(id)")
        connection.execute("CREATE INDEX record_urls ON records(url, id)")
        stats.update(
            scanned_rows=processed, duplicate_ids=duplicate_count(connection, "id"),
            duplicate_urls=duplicate_count(connection, "url"),
            null_urls=reasons["null_url"], empty_urls=reasons["empty_url"],
            invalid_urls=sum(reasons.values()), invalid_url_reasons=dict(sorted(reasons.items())),
            unique_domains=connection.execute("SELECT COUNT(*) FROM domains").fetchone()[0],
        )
        with (staging / "domain_stats.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["domain", "url_count"])
            writer.writerows(connection.execute(
                "SELECT domain, url_count FROM domains ORDER BY url_count DESC, domain ASC"))
        with (staging / "duplicate_urls.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["url", "id", "occurrence_count"])
            writer.writerows(connection.execute(
                "SELECT records.url, records.id, repeated.n FROM "
                "(SELECT url, COUNT(*) AS n FROM records WHERE url IS NOT NULL "
                "GROUP BY url HAVING COUNT(*) > 1) AS repeated "
                "JOIN records ON records.url=repeated.url "
                "ORDER BY records.url COLLATE BINARY ASC, records.id ASC"))
    return stats


def validation_errors(query: dict, corpus: dict) -> list[str]:
    errors = []
    for name, stats, fields in [
        ("query", query, ("null_ids", "duplicate_ids", "null_queries", "empty_queries")),
        ("corpus", corpus, ("null_ids", "duplicate_ids")),
    ]:
        errors.extend(f"{name}.{field}={stats[field]}" for field in fields if stats[field])
    return errors


def file_signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def prepare(query_path: Path = QUERY_PATH, corpus_path: Path = CORPUS_PATH,
            output_dir: Path = OUTPUT_DIR, *, batch_size: int = 50_000,
            strip_www: bool = True, progress=print) -> dict:
    """Stage every report, atomically replace files, then publish inventory last.

    Invalid IDs/queries still produce reports; main exits 1. Schema or processing
    failures publish nothing. SQLite and temporary reports are cleaned on exit.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    query_path, corpus_path, output_dir = map(Path, (query_path, corpus_path, output_dir))
    query_path, corpus_path = query_path.resolve(), corpus_path.resolve()
    signatures = {path: file_signature(path) for path in (query_path, corpus_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    with pq.ParquetFile(query_path) as query_parquet, pq.ParquetFile(corpus_path) as corpus_parquet:
        validate_schema(query_parquet.schema_arrow, corpus=False)
        validate_schema(corpus_parquet.schema_arrow, corpus=True)
        progress("Hashing both Parquet files in blocks...")
        hashes = {path: sha256_file(path) for path in signatures}
        query = {"path": str(query_path), "sha256": hashes[query_path],
                 **inspect_queries(query_parquet)}
        progress(f"Queries: {query['rows']:,}; corpus metadata: "
                 f"{corpus_parquet.metadata.num_rows:,} rows, {corpus_parquet.metadata.num_row_groups} row groups")
        with TemporaryDirectory(prefix=".inventory-", dir=output_dir) as scratch:
            staging = Path(scratch)
            corpus = {"path": str(corpus_path), "sha256": hashes[corpus_path],
                      **inspect_corpus(corpus_parquet, staging, batch_size=batch_size,
                                       strip_www=strip_www, progress=progress)}
            if any(file_signature(path) != original for path, original in signatures.items()):
                raise ValueError("Input Parquet file changed during inventory; no reports published")
            report = {
                "dataset": "AIGuruTinix/ViBioMIR", "inventory_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "query_file": query, "corpus_file": corpus,
                "domain_options": {"strip_www": strip_www},
                "validation_errors": validation_errors(query, corpus),
                "counting_policy": {
                    "duplicates": "excess non-null occurrences; original values compared exactly",
                    "invalid_urls": "all invalid rows including null/empty URLs",
                    "domains": "valid HTTP(S) URL occurrences, including duplicates",
                    "duplicate_url_report": "one row per occurrence, preserving every official ID",
                },
            }
            # Preserve creation time when all calculated content is unchanged.
            # Repeat runs in this output directory are then byte deterministic.
            previous_path = output_dir / "inventory.json"
            if previous_path.exists():
                try:
                    previous = json.loads(previous_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    previous = None
                if (isinstance(previous, dict) and isinstance(previous.get("created_at"), str)
                    and {k: v for k, v in previous.items() if k != "created_at"} == {
                        k: v for k, v in report.items() if k != "created_at"
                    }):
                    report["created_at"] = previous["created_at"]
            (staging / "inventory.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for name in REPORT_FILES:
                (staging / name).replace(output_dir / name)
    return report


def peak_memory_mib() -> float | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, default=QUERY_PATH)
    parser.add_argument("--corpus-file", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--keep-www", action="store_true", help="Keep www. in domain statistics")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    started = time.perf_counter()
    try:
        report = prepare(args.query_file, args.corpus_file, args.output_dir,
                         batch_size=args.batch_size, strip_www=not args.keep_www,
                         progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, sqlite3.Error, pa.ArrowException) as error:
        print(f"Inventory failed: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("Top 20 domains:", flush=True)
    with (args.output_dir / "domain_stats.csv").open(encoding="utf-8", newline="") as handle:
        for _, row in zip(range(20), csv.DictReader(handle)):
            print(f"  {row['domain']}: {int(row['url_count']):,}", flush=True)
    peak = peak_memory_mib()
    print(f"Elapsed: {time.perf_counter() - started:.2f}s; "
          f"peak process memory: {f'{peak:.2f} MiB' if peak is not None else 'unavailable'}", flush=True)
    if report["validation_errors"]:
        print("Validation failed: " + "; ".join(report["validation_errors"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
