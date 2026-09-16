"""Grid search threshold calibration script across dev set."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Calibrate threshold and relative delta on dev set.")
    parser.add_argument("--ground-truth", type=str, default="data/dev/sample_ground_truth.jsonl")
    parser.add_argument("--output-csv", type=str, default="experiments/calibration_results.csv")
    args = parser.parse_args()

    print(f"Calibration script target output: {args.output_csv} (Ready for Day 6 Calibration)")


if __name__ == "__main__":
    main()
