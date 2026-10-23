"""Publish one V4 runtime index from immutable sparse and dense release artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.medrag.core.config import load_config
from src.medrag.data.phase2_batch import sha256_file


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(sparse: Path, dense: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    s, d, config = _json(sparse / "manifest.json"), _json(dense / "manifest.json"), load_config(config_path)
    if s.get("status") != d.get("status") or s.get("status") != "complete":
        raise ValueError("Sparse/dense artifacts must both be complete")
    if (s.get("fingerprint") != d.get("sparse_fingerprint")
            or s.get("corpus_version") != d.get("corpus_version")
            or s.get("passage_count") != d.get("rows")
            or s.get("parts") != d.get("parts")
            or config["corpus_version"] != s["corpus_version"]):
        raise ValueError("Sparse/dense/config identity mismatch")
    if d["contract"]["sparse_manifest_sha256"] != sha256_file(sparse / "manifest.json"):
        raise ValueError("Dense contract is linked to another sparse manifest")
    for path, manifest in ((sparse, s), (dense, d)):
        for name, expected in manifest["files"].items():
            if sha256_file(path / name) != expected:
                raise ValueError(f"Index artifact checksum mismatch: {path / name}")
    dense_config, reranker_config = config["dense"], config["reranker"]
    if not dense_config.get("enabled") or not reranker_config.get("enabled"):
        raise ValueError("Production V4 requires dense and reranker explicitly enabled")
    for key in ("model", "revision", "document_max_length", "query_max_length"):
        if dense_config[key] != d["contract"][key]:
            raise ValueError(f"Dense model/input config mismatch: {key}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    staging.mkdir()
    for name in ("corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json"):
        (staging / name).symlink_to((sparse / name).resolve())
    (staging / "dense").symlink_to((dense / "dense").resolve(), target_is_directory=True)
    files = {name: sha256_file(staging / name) for name in (
        "corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json",
        "dense/mapping.sqlite", "dense/vectors.npy")}
    manifest = {**s, "files": files,
                "dense": {"fingerprint": d["fingerprint"], "rows": d["rows"],
                          "dimension": d["dimension"], **{key: dense_config[key] for key in
                          ("model", "revision", "document_max_length", "query_max_length")}},
                "reranker": {key: reranker_config[key] for key in
                             ("model", "revision", "max_length")},
                "source_sparse_manifest_sha256": sha256_file(sparse / "manifest.json"),
                "source_dense_manifest_sha256": sha256_file(dense / "manifest.json")}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", required=True, type=Path)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.sparse.resolve(), args.dense.resolve(),
                              args.config.resolve(), args.output.resolve()), indent=2))
