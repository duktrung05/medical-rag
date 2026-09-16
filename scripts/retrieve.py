"""Runs retrieval pipeline given a YAML configuration file."""

import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description="Run retrieval pipeline with config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--queries", type=str, default="data/dev/sample_queries.jsonl")
    parser.add_argument("--output", type=str, default="outputs/predictions/predictions.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {args.config} (Experiment: {config.get('experiment_name')})")
    print(f"Target output: {args.output}")


if __name__ == "__main__":
    main()
