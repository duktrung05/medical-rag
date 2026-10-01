"""Prepares and validates raw data, converting to optimized Parquet and doc_map."""

import argparse
from pathlib import Path
import pandas as pd

from src.data.loader import DataLoader, DocumentChunkMap
from src.data.preprocess import normalize_text
from src.data.validator import validate_chunks_integrity, validate_queries_integrity


def prepare_datasets(
    input_chunks: Path,
    input_queries: Path = None,
    output_dir: Path = Path("data/processed"),
    normalize: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from: {input_chunks}")
    chunks = DataLoader.load_chunks(input_chunks)
    chunk_errors = validate_chunks_integrity(chunks)
    if chunk_errors:
        print(f"WARNING: Found {len(chunk_errors)} chunk integrity issues:")
        for err in chunk_errors[:5]:
            print(f"  - {err}")
    else:
        print(f"Chunks integrity OK. Loaded {len(chunks)} chunks.")

    doc_map = DocumentChunkMap.from_chunks(chunks)
    doc_map_path = output_dir / "doc_map.parquet"
    doc_map.save_parquet(doc_map_path)
    print(f"Saved doc_map to: {doc_map_path} (Unique docs: {len(doc_map.all_doc_ids)}, chunks: {len(doc_map.all_chunk_ids)})")

    chunks_data = []
    for c in chunks:
        c_text = normalize_text(c.text, lowercase=False) if normalize else c.text
        chunks_data.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "language": c.language,
            "text": c_text,
        })
    chunks_df = pd.DataFrame(chunks_data)
    chunks_parquet = output_dir / "chunks.parquet"
    chunks_df.to_parquet(chunks_parquet, index=False)
    print(f"Saved chunks to: {chunks_parquet}")

    if input_queries and input_queries.is_file():
        print(f"Loading queries from: {input_queries}")
        queries = DataLoader.load_queries(input_queries)
        q_errors = validate_queries_integrity(queries)
        if q_errors:
            print(f"WARNING: Found {len(q_errors)} query integrity issues:")
            for err in q_errors[:5]:
                print(f"  - {err}")
        else:
            print(f"Queries integrity OK. Loaded {len(queries)} queries.")

        q_data = []
        for q in queries:
            q_norm = normalize_text(q.query, lowercase=True) if normalize else q.query
            q_data.append({
                "id": q.id,
                "query": q_norm,
                "raw_query": q.query,
            })
        queries_df = pd.DataFrame(q_data)
        queries_parquet = output_dir / "queries.parquet"
        queries_df.to_parquet(queries_parquet, index=False)
        print(f"Saved queries to: {queries_parquet}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate raw chunks and queries into Parquet.")
    parser.add_argument("--input-chunks", type=str, default="data/dev/sample_chunks.jsonl", help="Path to input chunks.jsonl")
    parser.add_argument("--input-queries", type=str, default="data/dev/sample_queries.jsonl", help="Path to input queries.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for parquet files")
    parser.add_argument("--no-normalize", action="store_true", help="Disable text normalization")

    args = parser.parse_args()
    prepare_datasets(
        input_chunks=Path(args.input_chunks),
        input_queries=Path(args.input_queries) if args.input_queries else None,
        output_dir=Path(args.output_dir),
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
