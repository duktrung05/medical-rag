"""Check combined V4 index provenance and published artifact integrity on CPU."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from src.medrag.core.config import load_config
from src.medrag.data.phase2_batch import sha256_file
from scripts.v4.verify_release_dense import write_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_links(sparse: Path, dense: Path) -> dict[str, Path]:
    return {name: (sparse / name).resolve() for name in
            ("corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json")} | {
                "dense": (dense / "dense").resolve()}


def check_links(index: Path, sparse: Path, dense: Path) -> dict[str, str]:
    links = {}
    for name, expected in expected_links(sparse, dense).items():
        path = index / name
        if not path.is_symlink() or not path.exists() or path.resolve() != expected:
            raise ValueError(f"Combined link is missing or points elsewhere: {name}")
        links[name] = str(expected)
    return links


def verify_index(index: Path, sparse: Path, dense: Path, config_path: Path,
                 reports: Path) -> dict:
    index_manifest_path = index / "manifest.json"
    sparse_manifest_path = sparse / "manifest.json"
    dense_manifest_path = dense / "manifest.json"
    manifest, s, d, config = (read_json(index_manifest_path),
                             read_json(sparse_manifest_path),
                             read_json(dense_manifest_path), load_config(config_path))
    dense_report_path = reports / "dense_full_integrity.json"
    sparse_report_path = index.parents[0] / "completion_20261019/reports/sparse_integrity.json"
    sparse_coverage_path = index.parents[0] / "completion_20261019/reports/sparse_coverage.json"
    dense_report, sparse_report, sparse_coverage = (
        read_json(dense_report_path), read_json(sparse_report_path),
        read_json(sparse_coverage_path))
    sparse_hash, dense_hash = sha256_file(sparse_manifest_path), sha256_file(dense_manifest_path)
    if (dense_report.get("status") != "DENSE_FULL_PASS"
            or dense_report.get("dense_manifest_sha256") != dense_hash
            or dense_report.get("sparse_manifest_sha256") != sparse_hash
            or sparse_report.get("status") != "SPARSE_FULL_PASS"
            or sparse_report.get("manifest_sha256") != sparse_hash
            or sparse_coverage.get("status") != "SPARSE_FULL_PASS"
            or sparse_coverage.get("index_manifest_sha256") != sparse_hash):
        raise ValueError("Dense/sparse acceptance reports are absent or stale")
    if (manifest.get("format") != "medrag-v4-disk-index-v1"
            or manifest.get("status") != "complete"
            or manifest.get("scope") != "full_release"
            or manifest.get("parts") != 922
            or manifest.get("source_count") != 1_073_525
            or manifest.get("passage_count") != 2_951_224
            or manifest.get("corpus_version") != config["corpus_version"]
            or manifest.get("fingerprint") != s["fingerprint"]
            or manifest.get("source_sparse_manifest_sha256") != sparse_hash
            or manifest.get("source_dense_manifest_sha256") != dense_hash
            or manifest.get("dense", {}).get("fingerprint") != d["fingerprint"]
            or manifest.get("dense", {}).get("rows") != 2_951_224
            or manifest.get("dense", {}).get("dimension") != 1024):
        raise ValueError("Combined manifest differs from accepted dense/sparse sources")
    for stage, keys in (("dense", ("model", "revision", "document_max_length", "query_max_length")),
                        ("reranker", ("model", "revision", "max_length"))):
        for key in keys:
            if manifest[stage].get(key) != config[stage].get(key):
                raise ValueError(f"Combined {stage} config mismatch: {key}")
    links = check_links(index, sparse, dense)
    expected_files = {"corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json",
                      "dense/mapping.sqlite", "dense/vectors.npy"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("Combined manifest file list differs")
    hashes = {}
    for name in sorted(expected_files):
        expected = (s["files"].get(name) if name in s["files"] else d["files"].get(name))
        if manifest["files"].get(name) != expected:
            raise ValueError(f"Combined manifest checksum differs from source: {name}")
        actual = sha256_file(index / name)
        if actual != expected:
            raise ValueError(f"Combined file checksum mismatch: {name}")
        hashes[name] = actual
        print(f"combined hash PASS {name}", flush=True)
    with sqlite3.connect(f"file:{index / 'dense/mapping.sqlite'}?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
    if (meta.get("status") != "complete" or meta.get("rows") != "2951224"
            or meta.get("fingerprint") != d["fingerprint"]):
        raise ValueError("Combined dense mapping metadata mismatch")
    report = {"format": "medrag-v4-full-index-cpu-acceptance-v1",
              "status": "FULL_INDEX_ARTIFACTS_PASS",
              "index_manifest_sha256": sha256_file(index_manifest_path),
              "sparse_manifest_sha256": sparse_hash,
              "dense_manifest_sha256": dense_hash,
              "config_sha256": sha256_file(config_path),
              "sparse_integrity_report_sha256": sha256_file(sparse_report_path),
              "sparse_coverage_report_sha256": sha256_file(sparse_coverage_path),
              "dense_integrity_report_sha256": sha256_file(dense_report_path),
              "links": links, "files_sha256": hashes,
              "counts": {"parts": 922, "sources": 1_073_525, "passages": 2_951_224},
              "coverage": "sparse_and_dense_accepted_by_exact_manifest_hash"}
    write_json(reports / "full_integrity.json", report)
    write_json(reports / "full_coverage.json", {
        "status": "FULL_INDEX_ARTIFACTS_PASS",
        "index_manifest_sha256": report["index_manifest_sha256"],
        "counts": report["counts"], "coverage": report["coverage"]})
    write_json(reports / "cpu_acceptance.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    reports = args.reports.resolve()
    try:
        report = verify_index(args.index.resolve(), args.sparse.resolve(),
                              args.dense.resolve(), args.config.resolve(), reports)
    except Exception as exc:
        write_json(reports / "cpu_acceptance.json",
                   {"status": "FULL_INDEX_ARTIFACTS_FAIL",
                    "error": f"{type(exc).__name__}: {exc}"})
        print(f"FULL_INDEX_ARTIFACTS_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"],
                      "index_manifest_sha256": report["index_manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
