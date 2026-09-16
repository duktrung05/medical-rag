"""Builds dense vector embeddings and FAISS index. Prepared for Day 3."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build BGE-M3 dense FAISS index.")
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.parquet")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--output-index", type=str, default="artifacts/dense_index")
    args = parser.parse_args()

    out_p = Path(args.output_index)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"Dense index target directory: {out_p} (Ready for Day 3 Dense Retrieval)")


if __name__ == "__main__":
    main()
