"""Convert successful recovery batches into versioned documents and passages."""

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


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/vibiomir_step2_20261013"
DEFAULT_BATCHES = BASE / "recovery_longchau_batches"
DEFAULT_OUTPUT = BASE / "content/recovery_longchau"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(batches: Path, output: Path, *, max_batches: int | None = None) -> dict:
    plan = json.loads((batches / "plan_manifest.json").read_text())
    if plan["corpus_revision"] != "71be0e1239a19dfd7a717e2bc13e3bec21bd0334":
        raise ValueError("Recovery plan revision differs from materializer")
    docs_dir, passages_dir, sidecars = [output / name for name in ("documents", "passages", "parts")]
    for directory in (docs_dir, passages_dir, sidecars):
        directory.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    completed = 0
    for entry in plan["batch_parts"]:
        if max_batches is not None and completed >= max_batches:
            break
        ident = entry["batch"]
        folder = batches / f"batch_{ident:05d}"
        crawl = folder / "crawl"
        summary_path = crawl / "crawl_summary.json"
        if not summary_path.exists():
            break
        summary = json.loads(summary_path.read_text())
        if summary["processed_urls"] != entry["rows"]:
            break
        name = f"part_{ident:05d}"
        doc_dest = docs_dir / f"{name}.parquet"
        passage_dest = passages_dir / f"{name}.parquet"
        sidecar = sidecars / f"{name}.json"
        crawl_result = crawl / "crawl_results.parquet"
        crawl_hash = sha(crawl_result)
        if sidecar.exists():
            prior = json.loads(sidecar.read_text())
            if (prior["crawl_results_sha256"] != crawl_hash or not doc_dest.exists()
                    or not passage_dest.exists() or prior["documents_sha256"] != sha(doc_dest)
                    or prior["passages_sha256"] != sha(passage_dest)):
                raise ValueError(f"Existing recovery content mismatch: {name}")
            counts.update(prior["status_counts"])
            counts["passages"] += prior["passages"]
            completed += 1
            continue
        documents, passages = [], []
        rows = pq.read_table(crawl_result).to_pylist()
        for row in rows:
            if row["status"] != "success":
                continue
            raw_file = crawl / row["raw_file"]
            source = {"id": row["id"], "official_url": row["source_url"],
                      "hostname": row["domain"], "raw_path": str(raw_file)}
            document, chunks = process(source)
            if document["raw_sha256"] != row["content_sha256"]:
                raise ValueError(f"Recovery raw checksum mismatch for {row['id']}")
            documents.append(document)
            passages.extend(chunks)
        for destination, records, schema in ((doc_dest, documents, DOC_SCHEMA),
                                             (passage_dest, passages, PASSAGE_SCHEMA)):
            temporary = destination.with_suffix(".partial")
            if temporary.exists() or destination.exists():
                raise FileExistsError(destination)
            pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression="zstd")
            os.replace(temporary, destination)
        status_counts = dict(Counter(d["status"] for d in documents))
        part_report = {"crawl_results_sha256": crawl_hash, "documents": len(documents),
                       "passages": len(passages), "status_counts": status_counts,
                       "documents_sha256": sha(doc_dest), "passages_sha256": sha(passage_dest)}
        sidecar.write_text(json.dumps(part_report, indent=2) + "\n")
        counts.update(status_counts)
        counts["passages"] += len(passages)
        completed += 1
        print(f"{name}: {len(documents)} documents, {len(passages)} passages, {status_counts}", flush=True)
    result = {"status": "complete" if completed == plan["batches"] else "in_progress",
              "corpus_revision": plan["corpus_revision"], "planned_batches": plan["batches"],
              "materialized_batches": completed,
              "documents": sum(v for k, v in counts.items() if k != "passages"),
              "passages": counts["passages"],
              "status_counts": {k: v for k, v in counts.items() if k != "passages"}}
    (output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.batches.resolve(), args.output.resolve(),
                         max_batches=args.max_batches), indent=2))


if __name__ == "__main__":
    main()
