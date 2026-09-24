"""Command line interface dispatcher for r2ai-medical-retrieval."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="r2ai",
        description="R2AI Multilingual Medical Information Retrieval System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate
    val_parser = subparsers.add_parser("validate", help="Validate submission file")
    val_parser.add_argument("submission", type=str, help="Path to submission.jsonl")
    val_parser.add_argument("--corpus-chunks", type=str, required=False, help="Path to chunks.jsonl")
    val_parser.add_argument("--test-queries", type=str, required=False, help="Path to queries.jsonl")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against ground truth")
    eval_parser.add_argument("--prediction", type=str, required=True, help="Path to predictions.jsonl")
    eval_parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground_truth.jsonl")

    # Config validation
    config_parser = subparsers.add_parser("validate-config", help="Validate a runtime or pipeline YAML config")
    config_parser.add_argument("config", type=str, help="Path to YAML config")

    # Retrieval run
    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval using a YAML config")
    retrieve_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    retrieve_parser.add_argument("--queries", type=str, default="data/dev/sample_queries.jsonl")
    retrieve_parser.add_argument("--output", type=str, default="outputs/predictions/predictions.jsonl")

    args = parser.parse_args()

    if args.command == "validate":
        from src.submission.validate import main as val_main
        sys.argv = [sys.argv[0], args.submission]
        if args.corpus_chunks:
            sys.argv.extend(["--corpus-chunks", args.corpus_chunks])
        if args.test_queries:
            sys.argv.extend(["--test-queries", args.test_queries])
        val_main()
    elif args.command == "evaluate":
        from scripts.evaluate import main as eval_main
        sys.argv = [sys.argv[0], "--prediction", args.prediction, "--ground-truth", args.ground_truth]
        eval_main()
    elif args.command == "validate-config":
        from src.config import load_config
        config = load_config(args.config)
        print(f"Valid {type(config).__name__}: {config.experiment_name}")
    elif args.command == "retrieve":
        from scripts.retrieve import main as retrieve_main
        sys.argv = [sys.argv[0], "--config", args.config, "--queries", args.queries, "--output", args.output]
        retrieve_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
