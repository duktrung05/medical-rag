"""Materialize verified PubMed text and exact-offset passages for step 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.contracts import Source
from src.medrag.evidence.verifier import verify


ROOT = Path(__file__).resolve().parents[3]
REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
DEFAULT_INPUT = ROOT / "data/vibiomir/pubmed" / REVISION / "run_20261013_2gpu"
DEFAULT_OUTPUT = ROOT / "outputs/vibiomir_step2_20261013/content/pubmed"
DOC_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("source_kind", pa.string()),
    ("title", pa.string()), ("text", pa.string()), ("language_metadata", pa.string()),
    ("pmid", pa.string()), ("raw_file", pa.string()), ("raw_sha256", pa.string()),
    ("text_sha256", pa.string()), ("extractor_version", pa.string()),
])
PASSAGE_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("passage_id", pa.string()),
    ("passage_index", pa.int32()), ("text", pa.string()),
    ("source_start", pa.int64()), ("source_end", pa.int64()),
    ("source_sha256", pa.string()), ("language_hint", pa.string()),
])


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(source: Path, output: Path) -> dict:
    release = json.loads((source / "release_manifest.json").read_text())
    if not release["complete"] or release["corpus_revision"] != REVISION:
        raise ValueError("Unverified PubMed release")
    docs_dir, passages_dir, parts_dir = [output / name for name in ("documents", "passages", "parts")]
    for directory in (docs_dir, passages_dir, parts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for part in sorted((source / "documents").glob("part_*.parquet")):
        dest_doc = docs_dir / part.name
        dest_passages = passages_dir / part.name
        sidecar = parts_dir / (part.stem + ".json")
        source_hash = sha(part)
        if sidecar.exists():
            prior = json.loads(sidecar.read_text())
            if (prior["input_sha256"] != source_hash or not dest_doc.exists()
                    or not dest_passages.exists() or sha(dest_doc) != prior["documents_sha256"]
                    or sha(dest_passages) != prior["passages_sha256"]):
                raise ValueError(f"Existing materialized part differs: {part.name}")
            counts["documents"] += prior["documents"]
            counts["passages"] += prior["passages"]
            counts["parts"] += 1
            continue
        docs, passages = [], []
        for batch in pq.ParquetFile(part).iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                if row["status"] != "usable_abstract":
                    continue
                text = row["text"]
                if not text or hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
                    raise ValueError(f"PubMed text checksum mismatch for {row['id']}")
                source_record = Source("official-" + REVISION, row["id"], row["official_url"],
                                       row["title"], text, row["raw_file"], row["raw_sha256"])
                docs.append({"id": row["id"], "official_url": row["official_url"],
                             "source_kind": "pubmed", "title": row["title"], "text": text,
                             "language_metadata": row["language"], "pmid": row["pmid"],
                             "raw_file": row["raw_file"], "raw_sha256": row["raw_sha256"],
                             "text_sha256": row["text_sha256"],
                             "extractor_version": "pubmed_xml_20261013"})
                for index, passage in enumerate(build_passages(source_record)):
                    if not verify(passage, source_record):
                        raise ValueError(f"Passage offsets mismatch for {row['id']}")
                    passages.append({"id": row["id"], "official_url": row["official_url"],
                                     "passage_id": passage.passage_id, "passage_index": index,
                                     "text": passage.text, "source_start": passage.source_start,
                                     "source_end": passage.source_end,
                                     "source_sha256": passage.source_sha256,
                                     "language_hint": passage.language})
        for destination, rows, schema in ((dest_doc, docs, DOC_SCHEMA),
                                          (dest_passages, passages, PASSAGE_SCHEMA)):
            temporary = destination.with_suffix(".partial")
            if temporary.exists() or destination.exists():
                raise FileExistsError(destination)
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
            os.replace(temporary, destination)
        part_report = {"input": str(part), "input_sha256": source_hash,
                       "documents": len(docs), "passages": len(passages),
                       "documents_sha256": sha(dest_doc), "passages_sha256": sha(dest_passages)}
        sidecar.write_text(json.dumps(part_report, indent=2) + "\n")
        counts["documents"] += len(docs)
        counts["passages"] += len(passages)
        counts["parts"] += 1
        print(f"{part.name}: {len(docs):,} documents, {len(passages):,} passages", flush=True)
    if counts["documents"] != release["states"]["usable_abstract"]:
        raise ValueError("Materialized PubMed coverage differs from release")
    result = {"status": "complete", "revision": REVISION, "source_release": str(source / "release_manifest.json"),
              **dict(counts), "passage_policy": {"max_tokens": 256, "overlap_tokens": 32,
                                                 "exact_offsets": True}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
