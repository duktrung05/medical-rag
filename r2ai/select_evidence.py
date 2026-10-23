"""Replay final selection from a ranking cache without models or a GPU."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.evidence_io import read_ranking_cache, selection_summary, write_predictions
from r2ai.evidence_selector import SelectionConfig, select_evidence


def add_selection_arguments(parser):
    parser.add_argument("--selector", choices=("topk", "threshold"), default="topk")
    parser.add_argument("--chunk-threshold", type=float)
    parser.add_argument("--chunk-delta", type=float, help="Maximum score gap from the best reranked chunk")
    parser.add_argument("--document-mode", choices=("parents", "baseline", "threshold"), default="parents")
    parser.add_argument("--doc-threshold", type=float)
    parser.add_argument("--doc-delta", type=float)
    parser.add_argument("--chunk-order", choices=("document", "rerank_score"), default="document",
                        help="Candidate priority for final chunk selection")
    parser.add_argument("--selection-config", type=Path, help="Load all selector settings, including limits, from JSON")


def config_from_args(args, *, max_docs: int, max_chunks: int, chunks_per_doc: int) -> SelectionConfig:
    if args.selection_config:
        return SelectionConfig(**json.loads(args.selection_config.read_text(encoding="utf-8")))
    return SelectionConfig(mode=args.selector, max_docs=max_docs, max_chunks=max_chunks,
                           chunks_per_doc=chunks_per_doc, chunk_threshold=args.chunk_threshold,
                           chunk_delta=args.chunk_delta, document_mode=args.document_mode,
                           doc_threshold=args.doc_threshold, doc_delta=args.doc_delta,
                           chunk_order=args.chunk_order)


def replay(rankings, metadata, config, output: Path) -> dict:
    if config.mode == "threshold" and metadata.get("reranker", {}).get("enabled") is False:
        raise ValueError("Threshold replay requires a cache with reranking enabled")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    predictions, diagnostics = [], []
    for ranking in rankings:
        prediction, diagnostic = select_evidence(ranking, config)
        predictions.append(prediction)
        diagnostics.append(diagnostic)
    output.mkdir(parents=True)
    write_predictions(output / "submission.zip", predictions)
    (output / "diagnostics.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n" for d in diagnostics), encoding="utf-8")
    (output / "selection_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    summary = {**selection_summary(predictions), "config": asdict(config), "ranking_metadata": metadata,
               "model_inference": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--k-docs", type=int, default=10)
    parser.add_argument("--k-chunks", type=int, default=10, help="Hard cap; zero emits no chunks")
    parser.add_argument("--chunks-per-doc", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    add_selection_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args, max_docs=args.k_docs, max_chunks=args.k_chunks,
                              chunks_per_doc=args.chunks_per_doc)
    rankings, metadata = read_ranking_cache(args.rankings)
    print(json.dumps(replay(rankings, metadata, config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
