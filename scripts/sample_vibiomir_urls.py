"""Deterministically sample ViBioMIR URLs by domain without network access."""

import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_vibiomir import (
    CORPUS_PATH,
    OUTPUT_DIR,
    classify_url,
    file_signature,
    sha256_file,
    validate_schema,
)

INVENTORY_PATH = OUTPUT_DIR / "inventory.json"
OUTPUT_PATH = OUTPUT_DIR / "sample_urls.parquet"
SUMMARY_PATH = OUTPUT_DIR / "sample_summary.json"
ALGORITHM_VERSION = "domain_quota_sha256_v1"
SAMPLE_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("domain", pa.string()),
    ("domain_url_count", pa.int64()), ("domain_quota", pa.int32()),
    ("sample_rank_in_domain", pa.int32()), ("selection_hash", pa.string()),
    ("sampling_reason", pa.string()),
])


def selection_hash(official_id: int, url: str) -> str:
    return hashlib.sha256(f"{official_id}\0{url}".encode()).hexdigest()


def read_domain_stats(path: Path) -> dict[str, int]:
    counts = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["domain", "url_count"]:
            raise ValueError(f"Invalid domain stats columns: {reader.fieldnames}")
        for row in reader:
            domain, count = row["domain"], int(row["url_count"])
            if not domain or domain in counts or count <= 0:
                raise ValueError(f"Invalid or duplicate domain in domain stats: {domain!r}")
            counts[domain] = count
    return counts


def allocate_quotas(counts: dict[str, int], sample_size: int,
                    per_domain_base: int) -> dict[str, int]:
    """Cover domains, fill base in rounds, then give extras to largest domains."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if per_domain_base < 0:
        raise ValueError("per_domain_base cannot be negative")
    total = sum(counts.values())
    if sample_size > total:
        raise ValueError(f"sample_size {sample_size} exceeds available corpus URL rows {total}")
    ordered = sorted(counts, key=lambda domain: (-counts[domain], domain))
    quotas = dict.fromkeys(ordered, 0)
    for domain in ordered[:sample_size]:
        quotas[domain] = 1
    remaining = sample_size - min(sample_size, len(ordered))
    target_base = max(per_domain_base, 1)
    for _ in range(1, target_base):
        if remaining == 0:
            break
        for domain in ordered:
            if remaining and quotas[domain] and quotas[domain] < counts[domain]:
                quotas[domain] += 1
                remaining -= 1
    while remaining:
        given = 0
        for domain in ordered:
            if remaining and quotas[domain] < counts[domain]:
                quotas[domain] += 1
                remaining -= 1
                given += 1
        if not given:
            raise ValueError("Cannot allocate the requested sample size")
    return quotas


def validate_sample(table: pa.Table, quotas: dict[str, int],
                    counts: dict[str, int], source_pairs: set[tuple[int, str]],
                    per_domain_base: int) -> None:
    """Check the staged Parquet as well as membership in streamed source rows."""
    if table.schema != SAMPLE_SCHEMA or table.num_rows != sum(quotas.values()):
        raise ValueError("Sample schema or row count mismatch")
    rows = table.to_pylist()  # At most sample_size rows, never the full corpus.
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate official ID in sample")
    if len({row["url"] for row in rows}) != len(rows):
        raise ValueError("Duplicate URL in sample")
    if {(row["id"], row["url"]) for row in rows} != source_pairs:
        raise ValueError("Sample contains an ID/URL pair absent from the source scan")
    expected_order = sorted(rows, key=lambda row: (row["domain"], row["sample_rank_in_domain"], row["id"]))
    if rows != expected_order:
        raise ValueError("Sample row order is not deterministic")
    selected = Counter(row["domain"] for row in rows)
    if selected != Counter({domain: quota for domain, quota in quotas.items() if quota}):
        raise ValueError("Domain quota mismatch")
    if len(rows) >= len(counts) and len(selected) != len(counts):
        raise ValueError("Sample does not cover every domain")
    ranks = {domain: [] for domain in selected}
    for row in rows:
        domain = row["domain"]
        ranks[domain].append(row["sample_rank_in_domain"])
        base = min(quotas[domain], max(1, per_domain_base))
        expected_reason = "domain_base" if row["sample_rank_in_domain"] < base else "large_domain_extra"
        if (row["domain_url_count"] != counts[domain]
                or row["domain_quota"] != quotas[domain]
                or row["selection_hash"] != selection_hash(row["id"], row["url"])
                or row["sampling_reason"] != expected_reason):
            raise ValueError("Sample metadata or stable hash mismatch")
    for domain, domain_ranks in ranks.items():
        if domain_ranks != list(range(quotas[domain])):
            raise ValueError(f"Non-contiguous sample ranks for {domain}")


def temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    return Path(name)


def sample_urls(corpus_file: Path = CORPUS_PATH, inventory_file: Path = INVENTORY_PATH,
                output_file: Path = OUTPUT_PATH, summary_file: Path = SUMMARY_PATH,
                *, sample_size: int = 500, per_domain_base: int = 5,
                batch_size: int = 50_000, progress=print) -> dict:
    if sample_size <= 0 or batch_size <= 0 or per_domain_base < 0:
        raise ValueError("sample_size and batch_size must be positive; per_domain_base cannot be negative")
    corpus_file, inventory_file, output_file, summary_file = map(
        Path, (corpus_file, inventory_file, output_file, summary_file))
    stats_file = inventory_file.parent / "domain_stats.csv"
    paths = [path.resolve() for path in (corpus_file, inventory_file, stats_file,
                                          output_file, summary_file)]
    if len(set(paths)) != len(paths):
        raise ValueError("Input and output file paths must be distinct")
    corpus_signature = file_signature(corpus_file)
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    corpus_info = inventory["corpus_file"]
    progress("Checking corpus SHA-256 against inventory...")
    actual_hash = sha256_file(corpus_file)
    if actual_hash != corpus_info["sha256"]:
        raise ValueError("Corpus SHA-256 mismatch with inventory; sampling stopped")
    if inventory.get("validation_errors") or any(corpus_info.get(name) for name in (
        "null_ids", "duplicate_ids", "duplicate_urls", "invalid_urls"
    )):
        raise ValueError("Inventory reports invalid IDs or URLs; cannot sample safely")
    if inventory.get("domain_options", {}).get("strip_www") is not True:
        raise ValueError("Inventory domain normalization must use strip_www=true")
    counts = read_domain_stats(stats_file)
    total_rows = corpus_info["rows"]
    if sum(counts.values()) != total_rows or len(counts) != corpus_info["unique_domains"]:
        raise ValueError("Domain stats totals disagree with inventory")
    quotas = allocate_quotas(counts, sample_size, per_domain_base)
    if sum(quotas.values()) != sample_size:
        raise ValueError("Internal quota calculation error")
    with pq.ParquetFile(corpus_file) as corpus:
        validate_schema(corpus.schema_arrow, corpus=True)
        if corpus.metadata.num_rows != total_rows:
            raise ValueError("Corpus row count disagrees with inventory")
        heaps = {domain: [] for domain, quota in quotas.items() if quota}
        observed = Counter()
        invalid_rows = processed = 0
        for index, batch in enumerate(corpus.iter_batches(batch_size=batch_size, use_threads=False), 1):
            ids, urls = batch.column(0).to_pylist(), batch.column(1).to_pylist()
            for official_id, url in zip(ids, urls):
                if official_id is None:
                    raise ValueError("Corpus contains null official ID")
                domain, reason = classify_url(url, strip_www=True)
                if reason:
                    invalid_rows += 1
                    continue
                if domain not in counts:
                    raise ValueError(f"Domain {domain!r} absent from domain stats")
                observed[domain] += 1
                quota = quotas[domain]
                if not quota:
                    continue
                digest = selection_hash(official_id, url)
                candidate = (-int(digest, 16), -official_id, official_id, url, digest)
                heap = heaps[domain]
                if len(heap) < quota:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
            processed += batch.num_rows
            progress(f"Corpus batch {index}: {processed:,}/{total_rows:,} rows")
    if processed != total_rows or invalid_rows != corpus_info["invalid_urls"] or observed != Counter(counts):
        raise ValueError("Corpus scan disagrees with inventory/domain stats")
    if file_signature(corpus_file) != corpus_signature:
        raise ValueError("Corpus changed during sampling; no output published")
    rows = []
    for domain in sorted(heaps):
        selected = sorted(heaps[domain], key=lambda item: (item[4], item[2], item[3]))
        if len(selected) != quotas[domain]:
            raise ValueError(f"Domain {domain!r} yielded too few URLs")
        base = min(quotas[domain], max(1, per_domain_base))
        for rank, (_, _, official_id, url, digest) in enumerate(selected):
            rows.append({
                "id": official_id, "url": url, "domain": domain,
                "domain_url_count": counts[domain], "domain_quota": quotas[domain],
                "sample_rank_in_domain": rank, "selection_hash": digest,
                "sampling_reason": "domain_base" if rank < base else "large_domain_extra",
            })
    table = pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA)
    source_pairs = {(row["id"], row["url"]) for row in rows}
    if len(source_pairs) != sample_size:
        raise ValueError("Selected ID/URL pairs are not unique")
    validate_sample(table, quotas, counts, source_pairs, per_domain_base)
    staged_sample = staged_summary = None
    try:
        staged_sample = temporary_path(output_file)
        pq.write_table(table, staged_sample, compression="zstd", version="2.6")
        validate_sample(pq.read_table(staged_sample), quotas, counts, source_pairs, per_domain_base)
        output_hash = sha256_file(staged_sample)
        ordered = sorted(counts, key=lambda domain: (-counts[domain], domain))
        allocations = []
        for domain in ordered:
            quota = quotas[domain]
            base = min(quota, max(1, per_domain_base))
            allocations.append({"domain": domain, "url_count": counts[domain],
                                "quota": quota, "domain_base": base,
                                "large_domain_extra": quota - base})
        active = [quota for quota in quotas.values() if quota]
        summary = {
            "corpus_sha256": actual_hash, "sample_size_requested": sample_size,
            "sample_size_actual": table.num_rows, "total_corpus_rows": total_rows,
            "total_domains": len(counts), "sampled_domains": len(active),
            "per_domain_base": per_domain_base, "min_samples_per_domain": min(active),
            "max_samples_per_domain": max(active), "domain_allocation": allocations,
            "algorithm_version": ALGORITHM_VERSION, "output_sha256": output_hash,
            "created_at": datetime.now(UTC).isoformat(),
        }
        if summary_file.exists():
            try:
                previous = json.loads(summary_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = None
            if (isinstance(previous, dict) and isinstance(previous.get("created_at"), str)
                and {key: value for key, value in previous.items() if key != "created_at"} == {
                    key: value for key, value in summary.items() if key != "created_at"
                }):
                summary["created_at"] = previous["created_at"]
        staged_summary = temporary_path(summary_file)
        staged_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
        if file_signature(corpus_file) != corpus_signature:
            raise ValueError("Corpus changed before publication; no output published")
        staged_sample.replace(output_file)
        staged_summary.replace(summary_file)
        return summary
    finally:
        for path in (staged_sample, staged_summary):
            if path is not None:
                path.unlink(missing_ok=True)


def peak_memory_mib() -> float | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, default=CORPUS_PATH)
    parser.add_argument("--inventory-file", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary-file", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--per-domain-base", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        summary = sample_urls(args.corpus_file, args.inventory_file,
                              args.output_file, args.summary_file,
                              sample_size=args.sample_size,
                              per_domain_base=args.per_domain_base,
                              batch_size=args.batch_size,
                              progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, KeyError, pa.ArrowException) as error:
        print(f"Sampling failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    peak = peak_memory_mib()
    print(f"Elapsed: {time.perf_counter() - started:.2f}s; "
          f"peak process memory: {f'{peak:.2f} MiB' if peak is not None else 'unavailable'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
