"""Run the standalone B0 challenger on frozen queries against the registered full index."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.medrag.adapters.validation import validate_selected
from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.core.passage_first import run_b0
from src.medrag.evidence.selector import SelectionPolicy


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def main(config_path: Path, split_path: Path, partition: str, count: int,
         output: Path, device: str | None) -> None:
    if output.exists() or count <= 0:
        raise ValueError("Need fresh output and positive query count")
    config = load_config(config_path)
    if device:
        config["dense"]["device"] = device
        config["reranker"]["device"] = device
    index = ROOT / config["index_namespace"]
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["scope"] != "full_release":
        raise ValueError("B0 benchmark requires the registered full-release index")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = split[f"{partition}_ids"][:count]
    if len(ids) != count or split["label_status"] != "unlabeled":
        raise ValueError("Frozen unlabeled query split does not cover requested count")
    migration = json.loads((ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
                           .read_text(encoding="utf-8"))
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query fingerprint changed")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    load_start = time.monotonic()
    pipeline = DiskPipeline(index, config)
    load_seconds = time.monotonic() - load_start
    policy_config = config["selection"]
    policy = SelectionPolicy(**policy_config)
    output.mkdir(parents=True)
    metrics = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for qid in ids:
            started = time.monotonic()
            selected, diagnostic = run_b0(pipeline, queries[qid], limit=200,
                                          pair_budget=200, chunk_budget=60)
            elapsed = time.monotonic() - started
            # Registered source IDs and exact passage/context offsets are checked here.
            records = validate_selected([qid], {qid: selected}, pipeline.sources.store,
                                        policy, result_limit=200, reranker_enabled=True,
                                        official_ids={row["official_id"] for row in selected})
            if diagnostic["rerank_pairs"] != 200 or diagnostic["selected_chunks"] > 60:
                raise ValueError(f"B0 budget validation failed for query {qid}")
            stream.write(json.dumps(records[0], ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
            metrics.append({"query_id": qid, "elapsed_seconds": elapsed,
                            "rerank_pairs": diagnostic["rerank_pairs"],
                            "scored_parents": diagnostic["scored_parents"],
                            "selected_documents": diagnostic["selected_documents"],
                            "selected_chunks": diagnostic["selected_chunks"]})
    write_json(output / "per_query.json", metrics)
    latencies = sorted(row["elapsed_seconds"] for row in metrics)
    write_json(output / "manifest.json", {
        "status": "standalone_b0_full_index_run", "partition": partition,
        "query_count": len(ids), "model_pair_budget_per_query": 200,
        "dual_path_budget_claim": False, "index_manifest_sha256": sha(index / "manifest.json"),
        "config_sha256": sha(config_path), "query_split_sha256": sha(split_path),
        "load_seconds": load_seconds,
        "latency_seconds": {"mean": statistics.mean(latencies),
                            "p50": statistics.median(latencies),
                            "p95_nearest_rank": latencies[max(0, int(.95 * len(latencies) + .999999) - 1)]},
        "predictions_sha256": sha(output / "predictions.jsonl"),
        "per_query_sha256": sha(output / "per_query.json"),
        "relevance_status": "unlabeled; no evidence-quality conclusion",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT /
                        "configs/v4_phase2_full_independent_d200_context_gpu3.yaml")
    parser.add_argument("--split", type=Path, default=ROOT /
                        "outputs/v4/p3_query_split_20261021/query_split.json")
    parser.add_argument("--partition", choices=("development", "holdout"), default="development")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    args = parser.parse_args()
    main(args.config, args.split, args.partition, args.count, args.output, args.device)
