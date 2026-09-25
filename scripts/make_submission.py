"""Constructs competition submission file and validates parent consistency."""

import argparse
from pathlib import Path

from src.data.loader import DataLoader, DocumentChunkMap
from src.submission.build import build_submission_file
from src.submission.validate import validate_submission_file


def main():
    parser = argparse.ArgumentParser(description="Build and validate submission file.")
    parser.add_argument("--predictions", type=str, required=True, help="Input predictions JSONL")
    parser.add_argument("--chunks", type=str, required=True, help="Corpus chunks.jsonl or doc_map.parquet")
    parser.add_argument("--queries", type=str, required=True, help="Test queries JSONL")
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.jsonl", help="Output submission JSONL")

    args = parser.parse_args()

    chunks_p = Path(args.chunks)
    if chunks_p.suffix == ".parquet":
        doc_map = DocumentChunkMap.load_parquet(chunks_p)
    else:
        chunks = DataLoader.load_chunks(chunks_p)
        doc_map = DocumentChunkMap.from_chunks(chunks)

    preds = DataLoader.load_predictions(args.predictions)

    out_file = build_submission_file(
        predictions=preds,
        output_path=args.output,
        doc_map=doc_map,
        auto_fix_parent=True,
    )
    print(f"Built submission with parent consistency enforcement: {out_file}")

    print("\nRunning submission verification:")
    report = validate_submission_file(out_file, doc_map=doc_map, expected_queries_path=args.queries)
    report.print_summary()
    if not report.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
