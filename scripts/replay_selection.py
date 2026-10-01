"""Replay a selection config against cached rankings and save per-query diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.benchmark_vimed_validation import read_rows
from scripts.run_metadata import write_run_manifest
from src.config import load_pipeline_config
from src.data.loader import DataLoader, DocumentChunkMap
from src.selection.prediction import select_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vimed_dynamic_selection.yaml"))
    parser.add_argument("--rankings", type=Path, default=Path("outputs/vimed_validation/rerank/rankings.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/vimed/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/selection_calibration/dynamic_runtime_replay"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new --output: {args.output}")
    config = load_pipeline_config(args.config)
    rows = read_rows(args.rankings)
    chunks = DataLoader.load_chunks(args.chunks)
    doc_map = DocumentChunkMap.from_chunks(chunks)
    selection = config.selection
    predictions, chunk_counts, doc_counts = [], Counter(), Counter()
    for row in rows:
        prediction, diagnostics = select_prediction(
            row["id"], row["results"], doc_map,
            doc_aggregation=config.scoring.document.aggregation,
            doc_top_n_mean=config.scoring.document.top_n_mean,
            chunk_threshold=selection.chunk.threshold,
            chunk_delta=selection.chunk.relative_delta,
            chunk_min_k=selection.chunk.min_k,
            chunk_max_k=selection.chunk.max_k,
            doc_threshold=selection.document.threshold,
            doc_delta=selection.document.relative_delta,
            doc_min_k=selection.document.min_k,
            doc_max_k=selection.document.max_k,
        )
        chunk_counts[len(prediction.relevant_chunks)] += 1
        doc_counts[len(prediction.relevant_docs)] += 1
        predictions.append({"prediction": prediction.model_dump(), "selection_diagnostics": diagnostics})
    args.output.mkdir(parents=True)
    write_run_manifest(args.output, task="offline_selection_replay", root=Path.cwd(), inputs={
        "config": args.config, "rankings": args.rankings, "chunks": args.chunks,
    }, parameters={"num_queries": len(rows), "model_inference": False})
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    summary = {"num_queries": len(rows), "chunk_count_distribution": dict(sorted(chunk_counts.items())),
               "document_count_distribution": dict(sorted(doc_counts.items())),
               "mean_chunks": sum(k * v for k, v in chunk_counts.items()) / len(rows),
               "mean_documents": sum(k * v for k, v in doc_counts.items()) / len(rows),
               "model_inference": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
