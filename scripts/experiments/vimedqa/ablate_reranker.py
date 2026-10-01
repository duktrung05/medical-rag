"""Compare no-rerank and BGE reranking depths with quality and resource profiles."""

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil
import torch

from src.adapters import adapt_query
from src.config import load_pipeline_config
from src.data.loader import DataLoader
from src.evaluation.evaluator import Evaluator
from src.reranking.factory import create_reranker
from src.service import build_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", default="outputs/reranker_ablation.json")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if not config.reranker.enabled:
        raise ValueError("Ablation config must enable reranking to provide the model settings")
    queries = [adapt_query(q) for q in DataLoader.load_queries(args.queries)]
    ground_truth = DataLoader.load_ground_truth(args.ground_truth)
    retrieval_only_config = config.model_copy(update={
        "reranker": config.reranker.model_copy(update={"enabled": False})
    })
    pipeline = build_pipeline(retrieval_only_config)
    process = psutil.Process()
    rows = []

    def run_arm(name, depth):
        pipeline.reranker_top_k = depth or 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        durations = []
        predictions = []
        for query in queries:
            started = time.perf_counter()
            predictions.append(pipeline.run_query(query.query_id, query.text))
            durations.append(time.perf_counter() - started)
        metrics = Evaluator().evaluate(ground_truth, predictions, doc_map=pipeline.doc_map).to_dict()
        rows.append({
            "arm": name,
            "candidate_depth": depth,
            "latency_ms_mean": statistics.mean(durations) * 1000 if durations else 0,
            "latency_ms_p95": sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))] * 1000 if durations else 0,
            "rss_mb_after": process.memory_info().rss / (1024 * 1024),
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None,
            "metrics": metrics,
        })

    pipeline.reranker = None
    run_arm("no-rerank", None)
    reranker = create_reranker(config.reranker)  # one model instance reused across rerank arms
    reranker._load_model()  # exclude checkpoint loading from per-query latency measurements
    for depth in (50, 100, 150):
        pipeline.reranker = reranker
        run_arm(f"rerank-{depth}", depth)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved ablation profile to {output}")


if __name__ == "__main__":
    main()
