"""Compare real-model in-memory V4 to a complete independent disk backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.prepared import load_prepared


def run(snapshot: Path, index: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    registry = CorpusRegistry.load(config_path.resolve().parents[1] / config["registry"])
    baseline = load_prepared(snapshot, registry, config=config)
    candidate = DiskPipeline(index, config)
    if baseline.corpus_version != candidate.corpus_version:
        raise ValueError("Canary corpus version mismatch")
    queries = pq.read_table(snapshot / "queries.parquet", columns=["id", "query"]).to_pylist()
    errors = []
    max_dense_delta = 0.0
    dense_order_mismatches = []
    predictions = [[], []]
    for position, row in enumerate(queries, 1):
        qid, query = int(row["id"]), str(row["query"])
        left_dense = baseline.dense.search(query, 200)
        right_dense = candidate.dense.search(query, 200)
        if [(hit.official_id, hit.passage_id) for hit in left_dense] != [
                (hit.official_id, hit.passage_id) for hit in right_dense]:
            dense_order_mismatches.append(qid)
        max_dense_delta = max(max_dense_delta, *(abs(a.score - b.score)
                                                 for a, b in zip(left_dense, right_dense, strict=True)))
        left = baseline.retrieve(query, 10)
        right = candidate.retrieve(query, 10)
        if left != right:
            errors.append(qid)
        predictions[0].extend(records([qid], {qid: left}))
        predictions[1].extend(records([qid], {qid: right}))
        print(f"{position}/{len(queries)}", flush=True)
    canonical = [hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest() for rows in predictions]
    report = {"status": "pass" if not errors and not dense_order_mismatches and
              canonical[0] == canonical[1] else "fail",
              "scope": "locked_1000_doc_20_query_real_model_disk_dense_canary",
              "query_count": len(queries), "dense_top200_order_mismatch_query_ids": dense_order_mismatches,
              "max_abs_dense_score_delta": max_dense_delta,
              "selected_result_mismatch_query_ids": errors,
              "baseline_canonical_sha256": canonical[0], "disk_canonical_sha256": canonical[1],
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "index_manifest_sha256": hashlib.sha256((index / "manifest.json").read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.rename(temporary, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot.resolve(), args.index.resolve(), args.config.resolve(),
                         args.output.resolve()), indent=2))
