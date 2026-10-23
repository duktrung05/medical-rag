"""Materialize one completed, bounded recovery pilot into verified passages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workflows.vibiomir.materialize_html_content import (
    DOC_SCHEMA, PASSAGE_SCHEMA, process,
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(crawl: Path, output: Path, *, allow_incomplete: bool = False) -> dict:
    summary = json.loads((crawl / "crawl_summary.json").read_text())
    complete = summary["requested_urls"] == summary["processed_urls"]
    if not complete and not allow_incomplete:
        raise ValueError("Crawl pilot is incomplete")
    source = crawl / "crawl_results.parquet"
    rows = pq.read_table(source).to_pylist()
    if len(rows) != summary["processed_urls"]:
        raise ValueError("Crawl results differ from summary")
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / "documents.parquet", output / "passages.parquet"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Pilot output exists; keep this version immutable")
    documents, passages = [], []
    for row in rows:
        if row["status"] != "success":
            continue
        raw = crawl / row["raw_file"]
        document, chunks = process({"id": row["id"], "official_url": row["source_url"],
                                    "hostname": row["domain"], "raw_path": str(raw)})
        if document["raw_sha256"] != row["content_sha256"]:
            raise ValueError(f"Raw checksum differs for ID {row['id']}")
        documents.append(document)
        passages.extend(chunks)
    for path, records, schema in zip(destinations, (documents, passages),
                                     (DOC_SCHEMA, PASSAGE_SCHEMA), strict=True):
        temporary = path.with_suffix(".partial")
        pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary,
                       compression="zstd")
        os.replace(temporary, path)
    manifest = {"status": "complete" if complete else "partial", "revision":
                "71be0e1239a19dfd7a717e2bc13e3bec21bd0334",
                "crawl_results_sha256": sha(source),
                "requested_urls": summary["requested_urls"],
                "processed_urls": summary["processed_urls"],
                "fetched_success": summary["success_count"],
                "documents": len(documents), "passages": len(passages),
                "status_counts": dict(Counter(row["status"] for row in documents)),
                "documents_sha256": sha(destinations[0]),
                "passages_sha256": sha(destinations[1])}
    if manifest["documents"] != manifest["fetched_success"]:
        raise ValueError("Some successful crawl records were not materialized")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.crawl.resolve(), args.output.resolve(),
                         allow_incomplete=args.allow_incomplete), indent=2))


if __name__ == "__main__":
    main()
