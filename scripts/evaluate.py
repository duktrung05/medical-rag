"""CLI script to evaluate predictions against ground truth annotations."""

import argparse
import csv
import json
from pathlib import Path

from src.data.loader import DataLoader, DocumentChunkMap
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions using Macro P, R, F2.")
    parser.add_argument("--prediction", type=str, required=True, help="Path to predictions JSONL file")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSONL file")
    parser.add_argument("--corpus-chunks", type=str, help="Optional chunks JSONL for ID and parent checks")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path for evaluation report")
    parser.add_argument("--experiment-name", type=str, default="evaluation", help="Name of experiment for log tracking")
    parser.add_argument("--log-csv", type=str, default=None, help="Optional CSV file to append results (e.g. experiments/experiment_log.csv)")

    args = parser.parse_args()

    pred_path = Path(args.prediction)
    gt_path = Path(args.ground_truth)

    predictions = DataLoader.load_predictions(pred_path)
    ground_truth = DataLoader.load_ground_truth(gt_path)
    doc_map = (
        DocumentChunkMap.from_chunks(DataLoader.load_chunks(args.corpus_chunks))
        if args.corpus_chunks else None
    )

    evaluator = Evaluator(beta=2.0)
    summary = evaluator.evaluate(ground_truth=ground_truth, predictions=predictions, doc_map=doc_map)

    print(summary.format_table(title=f"EVALUATION: {args.experiment_name}"))

    if args.output_json:
        out_json_p = Path(args.output_json)
        out_json_p.parent.mkdir(parents=True, exist_ok=True)
        with out_json_p.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Summary saved to: {out_json_p}")

    if args.log_csv:
        csv_p = Path(args.log_csv)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_p.is_file()

        with csv_p.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "experiment_name",
                    "num_queries",
                    "doc_p",
                    "doc_r",
                    "doc_f2",
                    "chunk_p",
                    "chunk_r",
                    "chunk_f2",
                    "macro_f2",
                ])
            writer.writerow([
                args.experiment_name,
                summary.num_queries,
                f"{summary.doc_precision:.4f}",
                f"{summary.doc_recall:.4f}",
                f"{summary.doc_f2:.4f}",
                f"{summary.chunk_precision:.4f}",
                f"{summary.chunk_recall:.4f}",
                f"{summary.chunk_f2:.4f}",
                f"{summary.macro_f2:.4f}",
            ])
        print(f"Logged results to: {csv_p}")


if __name__ == "__main__":
    main()
