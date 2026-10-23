"""Build a verified disk BM25/store canary from the locked V4 Phase 2 snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.contracts import Passage, Source
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.data.phase2_prepared import FORMAT
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.data.disk_store import DiskCorpusStore


def build(snapshot: Path, output: Path, registry: CorpusRegistry) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    version = registry.get(str(source_manifest["corpus_version"]))
    if (source_manifest.get("format") != FORMAT or source_manifest.get("status") != "complete"
            or source_manifest.get("corpus_revision") != version.revision
            or source_manifest.get("corpus_sha256") != version.sha256):
        raise ValueError("Snapshot and pinned registry disagree")
    for name, details in source_manifest["files"].items():
        if sha256_file(snapshot / name) != details["sha256"]:
            raise ValueError(f"Snapshot file checksum mismatch: {name}")
    fingerprint = sha256_file(snapshot / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    staging.mkdir()
    store = DiskCorpusStore(staging / "corpus.sqlite", version.name, fingerprint, create=True)
    docs = DiskBM25(staging / "documents.sqlite", "document_bm25", fingerprint, create=True)
    passage_index = DiskBM25(staging / "passages.sqlite", "passage_bm25", fingerprint, create=True)
    source_count = 0
    with (snapshot / "sources.jsonl").open(encoding="utf-8") as stream:
        batch = []
        for line in stream:
            if not line.strip():
                continue
            source = Source(**json.loads(line))
            if (source.corpus_version != version.name
                    or source.text_sha256 != source.metadata.get("normalized_text_sha256")):
                raise ValueError(f"Source hash/version mismatch: {source.official_id}")
            batch.append(source)
            if len(batch) >= 128:
                store.append_sources(batch)
                docs.append_batch((s.official_id, None, f"{s.title} {s.text}") for s in batch)
                source_count += len(batch)
                batch = []
        if batch:
            store.append_sources(batch)
            docs.append_batch((s.official_id, None, f"{s.title} {s.text}") for s in batch)
            source_count += len(batch)
    passage_count = 0
    for batch in pq.ParquetFile(snapshot / "passages.parquet").iter_batches(batch_size=256):
        passages = [Passage(**row) for row in batch.to_pylist()]
        store.append_passages(passages)
        passage_index.append_batch((p.official_id, p.passage_id, p.text) for p in passages)
        passage_count += len(passages)
    if (source_count != int(source_manifest["source_count"])
            or passage_count != int(source_manifest["passage_count"])):
        raise ValueError("Snapshot row count mismatch")
    store.complete()
    docs.complete()
    passage_index.complete()
    files = {name: sha256_file(staging / name) for name in
             ("corpus.sqlite", "documents.sqlite", "passages.sqlite")}
    manifest = {"format": "medrag-v4-disk-index-v1", "status": "complete",
                "corpus_version": version.name, "fingerprint": fingerprint,
                "source_count": source_count, "passage_count": passage_count,
                "source_snapshot_manifest_sha256": fingerprint,
                "files": files, "dense": "not_built", "scope": "locked_subset_canary"}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.snapshot.resolve(), args.output.resolve(),
                           CorpusRegistry.load(args.registry.resolve())), indent=2))
