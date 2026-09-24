"""Build a cached exact dense passage index from a configured corpus."""

import argparse
from pathlib import Path

from src.config import PipelineConfig, load_pipeline_config
from src.data.loader import DataLoader
from src.retrieval.dense import DenseRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_dense.yaml")
    parser.add_argument("--chunks", type=str, help="Optional chunks JSONL override")
    parser.add_argument("--output-index", type=str, help="Optional index directory override")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if not config.retrieval.dense.enabled:
        raise ValueError("Dense index build requires retrieval.dense.enabled=true")
    dense_config = config.retrieval.dense
    if args.output_index:
        dense_config = dense_config.model_copy(update={"index_path": Path(args.output_index).resolve()})
    chunks = DataLoader.load_chunks(args.chunks or config.corpus)
    manifest = DenseRetriever.build_index(chunks, dense_config, device=args.device)
    print(
        f"Built exact dense index with {manifest['num_passages']} chunks at "
        f"{dense_config.index_path} using model commit {manifest['model_revision']}"
    )


if __name__ == "__main__":
    main()
