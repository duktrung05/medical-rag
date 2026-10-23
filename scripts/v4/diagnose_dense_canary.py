"""Report numerical/order differences between the locked memory and disk dense canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.prepared import load_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Model device (default: cpu)")
    args = parser.parse_args()
    config = load_config(args.config)
    config["dense"]["device"] = args.device
    config["reranker"]["enabled"] = False
    registry = CorpusRegistry.load(args.config.resolve().parents[1] / config["registry"])
    baseline = load_prepared(args.snapshot, registry, config=config)
    candidate = DiskPipeline(args.index, config)
    left_vectors = np.asarray(baseline.dense.vectors, dtype=np.float32)
    right_vectors = np.load(args.index / "dense/vectors.npy", mmap_mode="r")
    disk_rows = {}
    for row_id in range(candidate.dense.count()):
        parent, passage_id, _ = candidate.dense.identity(row_id)
        disk_rows[(parent, passage_id)] = row_id
    deltas = [float(np.max(np.abs(left_vectors[i] - right_vectors[disk_rows[(p.official_id, p.passage_id)]])))
              for i, p in enumerate(baseline.dense.passages)]
    details = []
    for row in pq.read_table(args.snapshot / "queries.parquet", columns=["id", "query"]).to_pylist():
        qid, query = int(row["id"]), str(row["query"])
        if qid not in (92, 927):
            continue
        left = baseline.dense.search(query, 200)
        right = candidate.dense.search(query, 200)
        differences = []
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if (a.official_id, a.passage_id) != (b.official_id, b.passage_id):
                other_right = next((h.score for h in right if h.passage_id == a.passage_id), None)
                other_left = next((h.score for h in left if h.passage_id == b.passage_id), None)
                differences.append({"rank": i + 1, "memory_id": a.passage_id,
                                    "disk_id": b.passage_id, "memory_score": a.score,
                                    "disk_score": b.score, "memory_score_of_disk_hit": other_left,
                                    "disk_score_of_memory_hit": other_right})
        details.append({"query_id": qid, "same_top200_set":
                        {(h.official_id, h.passage_id) for h in left} ==
                        {(h.official_id, h.passage_id) for h in right},
                        "differences": differences[:10]})
    print(json.dumps({"max_vector_component_delta": max(deltas),
                      "nonidentical_vectors": sum(x > 0 for x in deltas),
                      "queries": details}, indent=2))


if __name__ == "__main__":
    main()
