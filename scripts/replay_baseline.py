"""Verify an archived ViMed ranking baseline without loading models."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.benchmark_vimed_validation import read_rows
from src.evaluation.ranking_metrics import summarize_rankings


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="baseline_20260930")
    args = parser.parse_args()
    root = Path.cwd()
    snapshot = root / "outputs/baseline_reference" / args.baseline
    expected = json.loads((snapshot / "snapshot_manifest.json").read_text())
    for relative, digest in expected["files_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Baseline snapshot checksum mismatch: {relative}")
    samples = read_rows(root / "data/vimed/validation/samples.jsonl")
    baseline = json.loads((snapshot / "baseline_metrics.json").read_text())
    if len(samples) != baseline["num_queries"]:
        raise ValueError("Baseline validation query count differs from current samples")
    results = {}
    for stage in ("bm25", "dense", "hybrid", "rerank"):
        rows = read_rows(snapshot / stage / "rankings.jsonl")
        recomputed = summarize_rankings(samples, rows)
        saved = baseline["stages"][stage]
        for key in recomputed:
            if key not in saved or recomputed[key] != saved[key]:
                raise ValueError(f"{stage} replay mismatch at {key}")
        results[stage] = {
            "num_queries": len(rows), "recall_at_k": recomputed["recall_at_k"],
            "document_recall_at_k": recomputed["document_recall_at_k"],
            "mrr_at_100": recomputed["mrr_at_100"],
            "zero_hit_queries_at_k": recomputed["zero_hit_queries_at_k"],
        }
    print(json.dumps({"baseline": args.baseline, "commit": baseline["source_commit_sha"],
                      "corpus_sha256": baseline["corpus_sha256"], "replay": "passed",
                      "stages": results}, indent=2))


if __name__ == "__main__":
    main()
