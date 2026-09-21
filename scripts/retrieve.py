"""Runs retrieval pipeline given a YAML configuration file."""

import argparse

from src.config import load_config
from src.data.loader import DataLoader, save_jsonl_records
from src.data.validator import validate_queries_integrity
from src.service import build_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run retrieval pipeline with config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--queries", type=str, default="data/dev/sample_queries.jsonl")
    parser.add_argument("--output", type=str, default="outputs/predictions/predictions.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    queries = DataLoader.load_queries(args.queries)
    errors = validate_queries_integrity(queries)
    if errors:
        raise ValueError("Invalid queries: " + "; ".join(errors))
    pipeline = build_pipeline(config)
    save_jsonl_records(args.output, pipeline.run_batch(queries))
    print(f"{config.experiment_name}: wrote {len(queries)} predictions to {args.output}")


if __name__ == "__main__":
    main()
