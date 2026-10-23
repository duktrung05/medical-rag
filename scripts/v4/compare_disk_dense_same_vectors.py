"""Isolate disk dense search parity using identical stored passage/query vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.retrieval.bge_encoder import BGEM3Encoder
from src.medrag.retrieval.core import Dense
from src.medrag.retrieval.disk_dense import DiskDense


class FixedQueryEncoder:
    def __init__(self, vector: list[float]):
        self.vector = vector

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        if len(texts) != 1:
            raise ValueError("Expected one query")
        return [self.vector]

    encode = encode_query


def run(snapshot: Path, index: Path, config_path: Path, output: Path, device: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    dense_config = config["dense"]
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["dense"]
    encoder = BGEM3Encoder(dense_config["model"], revision=dense_config["revision"],
                           document_max_length=dense_config["document_max_length"],
                           query_max_length=dense_config["query_max_length"], device=device)
    matrix = np.load(index / "dense/vectors.npy", mmap_mode="r")
    with sqlite3.connect(index / "dense/mapping.sqlite") as db:
        rows = db.execute("SELECT row_id,official_id,passage_id FROM rows ORDER BY row_id").fetchall()
    if len(rows) != len(matrix) or any(row_id != i for i, (row_id, _, _) in enumerate(rows)):
        raise ValueError("Dense row mapping is incomplete")
    passages = [SimpleNamespace(official_id=parent, passage_id=passage_id)
                for _, parent, passage_id in rows]
    reference = object.__new__(Dense)
    reference.passages = passages
    reference.vectors = matrix.tolist()
    queries = pq.read_table(snapshot / "queries.parquet", columns=["id", "query"]).to_pylist()
    mismatches = []
    max_abs_score_delta = 0.0
    for row in queries:
        qid, query = int(row["id"]), str(row["query"])
        query_vector = encoder.encode_query([query])[0]
        fixed = FixedQueryEncoder(query_vector)
        reference.encoder = fixed
        disk = DiskDense(index / "dense", fixed, contract["fingerprint"])
        left, right = reference.search(query, 200), disk.search(query, 200)
        left_ids = [(hit.official_id, hit.passage_id) for hit in left]
        right_ids = [(hit.official_id, hit.passage_id) for hit in right]
        if left_ids != right_ids:
            first = next(i for i, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1])
            mismatches.append({"query_id": qid, "first_rank": first + 1,
                               "same_top200_set": set(left_ids) == set(right_ids),
                               "reference": left_ids[first], "disk": right_ids[first],
                               "reference_score": left[first].score,
                               "disk_score": right[first].score})
        max_abs_score_delta = max(max_abs_score_delta,
                                  *(abs(a.score - b.score) for a, b in zip(left, right, strict=True)))
    report = {"status": "pass" if not mismatches else "fail",
              "scope": "locked_subset_same_fp32_passage_and_query_vectors",
              "query_count": len(queries), "top200_order_mismatches": mismatches,
              "max_abs_score_delta": max_abs_score_delta,
              "index_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "device": device}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.rename(temporary, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot.resolve(), args.index.resolve(), args.config.resolve(),
                         args.output.resolve(), args.device), indent=2))
