"""Encode every passage in a completed V4 sparse release index with resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.core.paths import project_root
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.bge_encoder import BGEM3Encoder
from src.medrag.retrieval.disk_dense import DiskDense, input_sha256, payload


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build(root: Path, sparse_index: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    sparse_manifest_path = sparse_index / "manifest.json"
    sparse_manifest = _json(sparse_manifest_path)
    if sparse_manifest.get("status") != "complete" or sparse_manifest.get("format") != "medrag-v4-disk-index-v1":
        raise ValueError("Sparse release index is incomplete")
    for name, expected in sparse_manifest["files"].items():
        if sha256_file(sparse_index / name) != expected:
            raise ValueError(f"Sparse index artifact checksum mismatch: {name}")
    config = load_config(config_path)
    if (config["corpus_version"] != sparse_manifest["corpus_version"]
            or not config["dense"].get("enabled")):
        raise ValueError("Dense model/config does not match sparse corpus")
    dense_config = config["dense"]
    p2_root = root / "outputs/vibiomir_phase2_full_v2"
    p2_manifest_path = p2_root / "manifests/phase2_release_manifest.json"
    p2 = _json(p2_manifest_path)
    sparse_lock = _json(sparse_index / "lock.json")
    if sha256_file(p2_manifest_path) != sparse_lock["phase2_release_sha256"]:
        raise ValueError("Sparse and dense builders reference different Phase 2 releases")
    exclusions = {int(row["doc_id"]) for row in _json(
        root / "configs/vibiomir/phase2_semantic_staging_20261015.json")["documents"]}
    contract = {"format": "medrag-v4-release-dense-contract-v1",
                "sparse_manifest_sha256": sha256_file(sparse_manifest_path),
                "sparse_fingerprint": sparse_manifest["fingerprint"],
                "model": dense_config["model"], "revision": dense_config["revision"],
                "document_max_length": int(dense_config["document_max_length"]),
                "query_max_length": int(dense_config["query_max_length"]),
                "input_format": "title_newline_passage-v1", "normalize_embeddings": True,
                "dtype": "float32",
                "implementation_sha256": {name: sha256_file(root / name) for name in (
                    "scripts/v4/build_release_dense.py", "src/medrag/retrieval/disk_dense.py",
                    "src/medrag/retrieval/bge_encoder.py")}}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    store = DiskCorpusStore(sparse_index / "corpus.sqlite", config["corpus_version"],
                            sparse_manifest["fingerprint"])
    if store.counts() != (sparse_manifest["source_count"], sparse_manifest["passage_count"]):
        raise ValueError("Sparse store rows differ from manifest")
    staging = output.with_name(output.name + ".building")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        if _json(staging / "contract.json") != contract:
            raise ValueError("Existing dense build fingerprint/contract mismatch")
    else:
        staging.mkdir()
        _write(staging / "contract.json", contract)
    encoder = BGEM3Encoder(str(dense_config["model"]), revision=str(dense_config["revision"]),
                           document_max_length=int(dense_config["document_max_length"]),
                           query_max_length=int(dense_config["query_max_length"]),
                           device=dense_config.get("device"))
    dense = (DiskDense(staging / "dense", encoder, fingerprint)
             if (staging / "dense").exists() else None)
    committed = dense.count() if dense else 0
    cursor = 0
    for part_index in range(int(sparse_manifest["parts"])):
        part = p2["passage_parts"][part_index]
        name = str(part["part"])
        path = p2_root / "passages/strategy_C" / f"{name}.parquet"
        if sha256_file(path) != part["sha256"]:
            raise ValueError(f"Phase 2 passage shard checksum mismatch: {name}")
        rows = [row for row in pq.read_table(path, columns=[
            "passage_id", "doc_id", "passage_index", "passage_text",
            "text_representation_sha256"]).to_pylist()
                if int(row["doc_id"]) not in exclusions]
        rows.sort(key=lambda row: (int(row["doc_id"]), int(row["passage_index"])))
        sources = store.sources_batch({int(row["doc_id"]) for row in rows})
        passages = store.passages_batch([str(row["passage_id"]) for row in rows])
        for offset in range(0, len(rows), 128):
            batch = rows[offset:offset + 128]
            records = []
            inputs = []
            for row in batch:
                doc_id, passage_id = int(row["doc_id"]), str(row["passage_id"])
                source, passage = sources[doc_id], passages[passage_id]
                if (passage.official_id != doc_id or passage.text != row["passage_text"]
                        or row["text_representation_sha256"] != source.text_sha256):
                    raise ValueError(f"Dense input/identity mismatch: {passage_id}")
                records.append((doc_id, passage_id, input_sha256(passage.text, source.title)))
                inputs.append(payload(passage.text, source.title))
            to_skip = max(0, min(committed - cursor, len(batch)))
            if to_skip:
                if dense is None or dense.identity(cursor + to_skip - 1) != records[to_skip - 1]:
                    raise ValueError("Dense resume row/input mapping mismatch")
            if to_skip < len(batch):
                vectors = encoder.encode(inputs[to_skip:])
                if dense is None:
                    dense = DiskDense(staging / "dense", encoder, fingerprint, create=True,
                                      expected_rows=int(sparse_manifest["passage_count"]),
                                      dimension=len(vectors[0]))
                dense.append_batch(records[to_skip:], vectors)
            cursor += len(batch)
        print(f"dense part {part_index + 1}/{sparse_manifest['parts']}: "
              f"{dense.count() if dense else 0}/{sparse_manifest['passage_count']}", flush=True)
    if dense is None or cursor != sparse_manifest["passage_count"] or dense.count() != cursor:
        raise ValueError("Dense release coverage differs from sparse eligible passage count")
    if dense._check(require_complete=False)["status"] == "building":
        dense_result = dense.complete()
    else:
        dense_result = {"status": "complete", "rows": dense.count(),
                        "dimension": int(dense._check(require_complete=True)["dimension"])}
    files = {name: sha256_file(staging / name) for name in
             ("dense/vectors.npy", "dense/mapping.sqlite", "contract.json")}
    manifest = {"format": "medrag-v4-release-dense-v1", "status": "complete",
                "corpus_version": config["corpus_version"], "fingerprint": fingerprint,
                "sparse_fingerprint": sparse_manifest["fingerprint"],
                "rows": cursor, "dimension": dense_result["dimension"],
                "parts": sparse_manifest["parts"], "contract": contract, "files": files}
    _write(staging / "manifest.json", manifest)
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--sparse-index", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.sparse_index.resolve(),
                           args.config.resolve(), args.output.resolve()), indent=2))
