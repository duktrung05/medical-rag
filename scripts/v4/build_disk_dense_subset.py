"""Add a real BGE-M3 disk vector index to a verified V4 lexical canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.bge_encoder import BGEM3Encoder
from src.medrag.retrieval.disk_dense import DiskDense, input_sha256, payload


def build(snapshot: Path, lexical: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    registry = CorpusRegistry.load(config_path.resolve().parents[1] / config["registry"])
    version = registry.get(config["corpus_version"])
    dense_config, reranker_config = config["dense"], config["reranker"]
    if not dense_config.get("enabled") or not reranker_config.get("enabled"):
        raise ValueError("Canary requires explicit dense and reranker configuration")
    source_manifest = json.loads((snapshot / "manifest.json").read_text())
    lexical_manifest = json.loads((lexical / "manifest.json").read_text())
    snapshot_sha = sha256_file(snapshot / "manifest.json")
    if (source_manifest["corpus_version"] != version.name
            or lexical_manifest.get("source_snapshot_manifest_sha256") != snapshot_sha
            or lexical_manifest.get("status") != "complete"):
        raise ValueError("Lexical canary and source snapshot differ")
    for name, details in source_manifest["files"].items():
        if sha256_file(snapshot / name) != details["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
    for name, expected in lexical_manifest["files"].items():
        if sha256_file(lexical / name) != expected:
            raise ValueError(f"Lexical canary checksum mismatch: {name}")
    contract = {"source_snapshot_manifest_sha256": snapshot_sha,
                "model": dense_config["model"], "revision": dense_config["revision"],
                "document_max_length": int(dense_config["document_max_length"]),
                "query_max_length": int(dense_config["query_max_length"]),
                "input_format": "title_newline_passage-v1", "normalize_embeddings": True,
                "dtype": "float32"}
    dense_fp = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    staging.mkdir()
    for name in ("corpus.sqlite", "documents.sqlite", "passages.sqlite"):
        shutil.copyfile(lexical / name, staging / name)
    store = DiskCorpusStore(staging / "corpus.sqlite", version.name, snapshot_sha)
    encoder = BGEM3Encoder(str(dense_config["model"]), revision=str(dense_config["revision"]),
                           document_max_length=int(dense_config["document_max_length"]),
                           query_max_length=int(dense_config["query_max_length"]),
                           device=dense_config.get("device"))
    dense = None
    count = 0
    for batch in pq.ParquetFile(snapshot / "passages.parquet").iter_batches(batch_size=128):
        rows = batch.to_pylist()
        inputs = []
        identities = []
        for row in rows:
            source = store.source(int(row["official_id"]))
            text = row["text"]
            title = source.title
            inputs.append(payload(text, title))
            identities.append((int(row["official_id"]), row["passage_id"],
                               input_sha256(text, title)))
        vectors = encoder.encode(inputs)
        if dense is None:
            dense = DiskDense(staging / "dense", encoder, dense_fp, create=True,
                              expected_rows=int(source_manifest["passage_count"]),
                              dimension=len(vectors[0]))
        dense.append_batch(identities, vectors)
        count += len(rows)
    if dense is None or count != int(source_manifest["passage_count"]):
        raise ValueError("Dense canary did not encode every passage")
    dense.complete()
    files = {name: sha256_file(staging / name) for name in
             ("corpus.sqlite", "documents.sqlite", "passages.sqlite",
              "dense/mapping.sqlite", "dense/vectors.npy")}
    manifest = {**lexical_manifest, "files": files,
                "dense": {**contract, "fingerprint": dense_fp, "rows": count,
                          "dimension": int(dense._check(require_complete=True)["dimension"])},
                "reranker": {key: reranker_config[key] for key in
                             ("model", "revision", "max_length")},
                "scope": "locked_subset_dense_canary"}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--lexical", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.snapshot.resolve(), args.lexical.resolve(),
                           args.config.resolve(), args.output.resolve()), indent=2))
