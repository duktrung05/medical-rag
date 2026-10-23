"""Independently verify a published V4 dense release using CPU and bounded memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.verifier import verify as verify_passage
from src.medrag.retrieval.disk_dense import input_sha256

EXPECTED_ROWS = 2_951_224
EXPECTED_SOURCES = 1_073_525
EXPECTED_PARTS = 922
EXPECTED_DIMENSION = 1024
EXPECTED_EXCLUDED_PASSAGES = 49


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_mapping_structure(db: sqlite3.Connection, *, fingerprint: str,
                            expected_rows: int, dimension: int,
                            expected_sources: int = EXPECTED_SOURCES) -> dict:
    meta = dict(db.execute("SELECT key,value FROM meta"))
    expected = {"format": "medrag-v4-dense-fp32-v1", "status": "complete",
                "fingerprint": fingerprint, "rows": str(expected_rows),
                "expected_rows": str(expected_rows), "dimension": str(dimension)}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Dense mapping meta {key} differs: {meta.get(key)!r}")
    result = db.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise ValueError(f"Dense mapping SQLite integrity_check: {result}")
    count, low, high, unique_parents = db.execute(
        "SELECT count(*),min(row_id),max(row_id),count(DISTINCT official_id) FROM rows").fetchone()
    if (count, low, high, unique_parents) != (
            expected_rows, 0, expected_rows - 1, expected_sources):
        raise ValueError("Dense mapping count/sequence/parent coverage mismatch")
    if db.execute("SELECT count(*) FROM rows WHERE input_sha256 IS NULL OR "
                  "length(input_sha256)!=64").fetchone()[0]:
        raise ValueError("Dense mapping has invalid input hashes")
    return {"meta": meta, "rows": count, "min_row_id": low, "max_row_id": high,
            "unique_parents": unique_parents, "sqlite_integrity_check": result}


def check_vectors(path: Path, *, expected_rows: int, dimension: int,
                  block_rows: int = 8192) -> dict:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (expected_rows, dimension) or matrix.dtype != np.dtype("float32"):
        raise ValueError(f"Dense matrix shape/dtype differs: {matrix.shape}, {matrix.dtype}")
    minimum, maximum = float("inf"), float("-inf")
    norm_min, norm_max = float("inf"), float("-inf")
    for start in range(0, expected_rows, block_rows):
        block = np.asarray(matrix[start:start + block_rows])
        if not np.isfinite(block).all():
            raise ValueError(f"Dense vectors contain non-finite value near row {start}")
        norms = np.linalg.norm(block, axis=1)
        if (norms < 0.99).any() or (norms > 1.01).any():
            raise ValueError(f"Dense vector norm differs near row {start}")
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    return {"rows": expected_rows, "dimension": dimension, "dtype": "float32",
            "min_value": minimum, "max_value": maximum,
            "min_norm": norm_min, "max_norm": norm_max,
            "max_norm_deviation": max(abs(norm_min - 1), abs(norm_max - 1)),
            "block_rows": block_rows}


def check_ordered_mapping(root: Path, sparse: Path, db: sqlite3.Connection,
                          sparse_manifest: dict, p2: dict, exclusions: set[int],
                          progress_path: Path) -> dict:
    store = DiskCorpusStore(sparse / "corpus.sqlite", sparse_manifest["corpus_version"],
                            sparse_manifest["fingerprint"])
    mapping = db.execute("SELECT row_id,official_id,passage_id,input_sha256 "
                         "FROM rows ORDER BY row_id")
    cursor = 0
    excluded = 0
    last_progress = time.monotonic()
    p2_root = root / "outputs/vibiomir_phase2_full_v2"
    for part_index, part in enumerate(p2["passage_parts"]):
        name = str(part["part"])
        path = p2_root / "passages/strategy_C" / f"{name}.parquet"
        if sha256_file(path) != part["sha256"]:
            raise ValueError(f"Phase 2 passage part checksum mismatch: {name}")
        all_rows = pq.read_table(path, columns=[
            "passage_id", "doc_id", "passage_index", "passage_text",
            "text_representation_sha256"]).to_pylist()
        if len(all_rows) != int(part["rows"]):
            raise ValueError(f"Phase 2 passage part count mismatch: {name}")
        rows = [row for row in all_rows if int(row["doc_id"]) not in exclusions]
        excluded += len(all_rows) - len(rows)
        rows.sort(key=lambda row: (int(row["doc_id"]), int(row["passage_index"])))
        sources = store.sources_batch({int(row["doc_id"]) for row in rows})
        passages = store.passages_batch([str(row["passage_id"]) for row in rows])
        for row in rows:
            doc_id, passage_id = int(row["doc_id"]), str(row["passage_id"])
            source, passage = sources[doc_id], passages[passage_id]
            if (passage.official_id != doc_id or passage.text != row["passage_text"]
                    or row["text_representation_sha256"] != source.text_sha256
                    or not verify_passage(passage, source)):
                raise ValueError(f"Passage source/offset/provenance mismatch: {passage_id}")
            actual = mapping.fetchone()
            expected = (cursor, doc_id, passage_id, input_sha256(passage.text, source.title))
            if actual != expected:
                raise ValueError(f"Dense mapping row {cursor} differs: {actual!r} != {expected!r}")
            cursor += 1
        if time.monotonic() - last_progress >= 60 or part_index + 1 == len(p2["passage_parts"]):
            write_json(progress_path, {"stage": "ordered_mapping", "pid": os.getpid(),
                                       "parts_checked": part_index + 1, "rows_checked": cursor,
                                       "excluded_passages": excluded, "timestamp": time.time()})
            print(f"verified dense part {part_index + 1}/{len(p2['passage_parts'])}: "
                  f"{cursor} rows", flush=True)
            last_progress = time.monotonic()
    if mapping.fetchone() is not None or cursor != EXPECTED_ROWS or excluded != EXPECTED_EXCLUDED_PASSAGES:
        raise ValueError("Dense mapping tail/count/exclusions differ from release")
    return {"rows_checked": cursor, "parts_checked": len(p2["passage_parts"]),
            "excluded_passages": excluded, "passage_identity_coverage": "exact_ordered_match"}


def verify_dense(root: Path, sparse: Path, dense: Path, config_path: Path,
                 report_dir: Path) -> dict:
    sparse_manifest_path, dense_manifest_path = sparse / "manifest.json", dense / "manifest.json"
    initial_dense_manifest_hash = sha256_file(dense_manifest_path)
    initial_sparse_manifest_hash = sha256_file(sparse_manifest_path)
    sparse_manifest, dense_manifest = load_json(sparse_manifest_path), load_json(dense_manifest_path)
    contract, sparse_lock = load_json(dense / "contract.json"), load_json(sparse / "lock.json")
    config = load_config(config_path)
    if (sparse_manifest.get("status") != "complete"
            or sparse_manifest.get("scope") != "full_release"
            or sparse_manifest.get("parts") != EXPECTED_PARTS
            or sparse_manifest.get("source_count") != EXPECTED_SOURCES
            or sparse_manifest.get("passage_count") != EXPECTED_ROWS):
        raise ValueError("Sparse release manifest differs from full release")
    if (dense_manifest.get("format") != "medrag-v4-release-dense-v1"
            or dense_manifest.get("status") != "complete"
            or dense_manifest.get("rows") != EXPECTED_ROWS
            or dense_manifest.get("parts") != EXPECTED_PARTS
            or dense_manifest.get("dimension") != EXPECTED_DIMENSION
            or dense_manifest.get("contract") != contract
            or dense_manifest.get("sparse_fingerprint") != sparse_manifest["fingerprint"]
            or dense_manifest.get("corpus_version") != sparse_manifest["corpus_version"]
            or contract.get("sparse_manifest_sha256") != initial_sparse_manifest_hash
            or contract.get("sparse_fingerprint") != sparse_manifest["fingerprint"]
            or config["corpus_version"] != sparse_manifest["corpus_version"]):
        raise ValueError("Dense release manifest/contract differs from sparse release")
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if dense_manifest.get("fingerprint") != fingerprint:
        raise ValueError("Dense fingerprint differs from builder contract")
    dense_config = config["dense"]
    for key in ("model", "revision", "document_max_length", "query_max_length"):
        if contract.get(key) != dense_config.get(key):
            raise ValueError(f"Dense model contract mismatch: {key}")
    if (contract.get("format") != "medrag-v4-release-dense-contract-v1"
            or contract.get("input_format") != "title_newline_passage-v1"
            or contract.get("dtype") != "float32"
            or contract.get("normalize_embeddings") is not True):
        raise ValueError("Dense representation contract differs")
    for name, expected in contract["implementation_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Dense implementation changed: {name}")
    p2_manifest_path = root / "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json"
    if (sha256_file(p2_manifest_path) != sparse_lock["phase2_release_sha256"]
            or sha256_file(root / "configs/vibiomir/phase2_semantic_staging_20261015.json")
            != sparse_lock["semantic_exclusions_sha256"]):
        raise ValueError("Dense Phase 2 release/exclusions differ from sparse lock")
    p2 = load_json(p2_manifest_path)
    if len(p2["passage_parts"]) != EXPECTED_PARTS:
        raise ValueError("Dense Phase 2 part count differs")
    exclusions = {int(row["doc_id"]) for row in load_json(
        root / "configs/vibiomir/phase2_semantic_staging_20261015.json")["documents"]}
    if len(exclusions) != 6:
        raise ValueError("Dense semantic exclusions differ")
    required_files = {"contract.json", "dense/vectors.npy", "dense/mapping.sqlite"}
    if set(dense_manifest.get("files", {})) != required_files:
        raise ValueError("Dense manifest file list differs")
    hashes = {}
    for name in sorted(required_files):
        actual = sha256_file(dense / name)
        if actual != dense_manifest["files"][name]:
            raise ValueError(f"Dense file checksum mismatch: {name}")
        hashes[name] = actual
        print(f"dense hash PASS {name}", flush=True)
    mapping_path = dense / "dense/mapping.sqlite"
    with sqlite3.connect(f"file:{mapping_path}?mode=ro", uri=True) as db:
        mapping_stats = check_mapping_structure(db, fingerprint=fingerprint,
                                                expected_rows=EXPECTED_ROWS,
                                                dimension=EXPECTED_DIMENSION)
        ordered = check_ordered_mapping(root, sparse, db, sparse_manifest, p2, exclusions,
                                        report_dir / "dense_progress.json")
    vector_stats = check_vectors(dense / "dense/vectors.npy", expected_rows=EXPECTED_ROWS,
                                 dimension=EXPECTED_DIMENSION)
    if (sha256_file(dense_manifest_path) != initial_dense_manifest_hash
            or sha256_file(sparse_manifest_path) != initial_sparse_manifest_hash):
        raise ValueError("Dense or sparse manifest changed during verification")
    report = {"format": "medrag-v4-dense-integrity-v1", "status": "DENSE_FULL_PASS",
              "dense_manifest_sha256": initial_dense_manifest_hash,
              "sparse_manifest_sha256": initial_sparse_manifest_hash,
              "config_sha256": sha256_file(config_path), "fingerprint": fingerprint,
              "files_sha256": hashes, "mapping": mapping_stats,
              "ordered_identity_coverage": ordered, "vectors": vector_stats}
    write_json(report_dir / "dense_full_integrity.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    reports = args.reports.resolve()
    try:
        report = verify_dense(args.root.resolve(), args.sparse.resolve(), args.dense.resolve(),
                              args.config.resolve(), reports)
    except Exception as exc:
        write_json(reports / "dense_full_integrity.json",
                   {"status": "DENSE_FULL_FAIL", "error": f"{type(exc).__name__}: {exc}"})
        print(f"DENSE_FULL_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"status": report["status"], "rows": report["vectors"]["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
