"""Builds sparse lexical index (Tantivy / BM25). Prepared for Day 2."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build BM25/Tantivy sparse index.")
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.parquet")
    parser.add_argument("--output-index", type=str, default="artifacts/sparse_index")
    args = parser.parse_args()

    out_p = Path(args.output_index)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"Sparse index target directory: {out_p} (Ready for Day 2 BM25 implementation)")


if __name__ == "__main__":
    main()
