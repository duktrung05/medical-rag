"""Split selected missing official URLs into immutable, resumable crawl batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/vibiomir_step2_20261013"
REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
SCHEMA = pa.schema([("id", pa.int64()), ("url", pa.string()),
                    ("domain", pa.string()), ("domain_url_count", pa.int64())])


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(domains: set[str], output: Path, exclude: list[Path],
        *, interleave_domains: bool = False, batch_size: int = 500) -> dict:
    if not domains:
        raise ValueError("Select at least one domain")
    if not 1 <= batch_size <= 500:
        raise ValueError("Batch size must be 1..500")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Plan directory already contains files: {output}")
    ledger = json.loads((BASE / "ledger_manifest.json").read_text())
    if ledger["corpus_revision"] != REVISION:
        raise ValueError("Official corpus revision differs")
    omitted = set()
    for path in exclude:
        for row in pq.read_table(path, columns=["id", "status"]).to_pylist():
            if row["status"] == "success":
                omitted.add(row["id"])
    rows = [row for row in pq.read_table(BASE / "needs_fetch.parquet",
                                         columns=["id", "official_url", "hostname"]).to_pylist()
            if row["hostname"] in domains and row["id"] not in omitted]
    counts = Counter(row["hostname"] for row in rows)
    rows.sort(key=lambda row: row["id"])
    if interleave_domains:
        queues = {domain: queue for domain in sorted(domains)
                  if (queue := deque(row for row in rows
                                     if row["hostname"] == domain))}
        rows = []
        while queues:
            for domain in list(queues):
                rows.append(queues[domain].popleft())
                if not queues[domain]:
                    del queues[domain]
    output.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, start in enumerate(range(0, len(rows), batch_size)):
        selected = rows[start:start + batch_size]
        folder = output / f"batch_{index:05d}"
        folder.mkdir()
        input_path = folder / "input.parquet"
        records = [{"id": row["id"], "url": row["official_url"],
                    "domain": row["hostname"],
                    "domain_url_count": ledger["domains"][row["hostname"]]["needs_fetch"]}
                   for row in selected]
        pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), input_path,
                       compression="zstd")
        digest = sha(input_path)
        (folder / "sample_summary.json").write_text(json.dumps({
            "output_sha256": digest, "corpus_revision": REVISION,
            "rows": len(selected), "first_id": selected[0]["id"],
            "last_id": selected[-1]["id"]}, indent=2) + "\n")
        parts.append({"batch": index, "rows": len(selected), "sha256": digest,
                      "first_id": selected[0]["id"], "last_id": selected[-1]["id"]})
    plan = {"status": "prepared", "corpus_revision": REVISION,
            "corpus_sha256": ledger["corpus_sha256"], "domains": sorted(domains),
            "domain_counts": dict(counts), "rows": len(rows), "batches": len(parts),
            "excluded_success_ids": len(omitted),
            "interleave_domains": interleave_domains, "batch_size": batch_size,
            "batch_parts": parts}
    (output / "plan_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    return {key: plan[key] for key in ("domains", "domain_counts", "rows", "batches")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-results", type=Path, action="append", default=[])
    parser.add_argument("--interleave-domains", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(set(args.domain), args.output.resolve(),
                         [path.resolve() for path in args.exclude_results],
                         interleave_domains=args.interleave_domains,
                         batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
