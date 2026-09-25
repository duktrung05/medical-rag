"""Build a persistent BM25 sparse index from a chunks JSONL corpus."""

import argparse
from pathlib import Path

from src.config import SparseRetrievalConfig
from src.data.loader import DataLoader
from src.retrieval.bm25 import BM25Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=str, default="data/medquad/dev/chunks.jsonl")
    parser.add_argument("--output-index", type=str, default="artifacts/medquad_bm25_index")
    parser.add_argument("--tokenizer", choices=("word", "char_ngram"), default="word")
    parser.add_argument("--char-ngram-size", type=int, default=3)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    chunks = DataLoader.load_chunks(args.chunks)
    config = SparseRetrievalConfig(
        enabled=True,
        engine="bm25",
        tokenizer=args.tokenizer,
        char_ngram_size=args.char_ngram_size,
        k1=args.k1,
        b=args.b,
        index_path=Path(args.output_index),
    )
    corpus_hash = BM25Retriever.build_index(chunks, args.output_index, config)
    print(
        f"Built {args.tokenizer} BM25 index with {len(chunks)} chunks at "
        f"{Path(args.output_index).resolve()} (corpus_sha256={corpus_hash})"
    )


if __name__ == "__main__":
    main()
