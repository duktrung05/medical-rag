"""Calibrate final selection offline from cached reranker rankings."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from scripts.workflows.vimedqa.benchmark_vimed_validation import read_rows
from scripts.tooling.run_metadata import write_run_manifest
from src.data.loader import DataLoader, DocumentChunkMap
from src.evaluation.evaluator import Evaluator
from src.selection.prediction import select_prediction


CHUNK_MINS = (0, 1)
CHUNK_MAXS = (1, 2, 3, 4, 5, 7, 10, 15, 20)
DOC_MINS = (0, 1)
DOC_MAXS = (1, 2, 3, 5, 7, 10)
THRESHOLD_PERCENTILES = (10, 30, 50, 70, 90)
DELTA_PERCENTILES = (10, 30, 50, 70, 90)


def quantiles(values: list[float], percentiles: tuple[int, ...]) -> list[float]:
    return sorted({float(np.percentile(values, p)) for p in percentiles})


def evaluate_params(rows, truths, doc_map, params):
    predictions, chunk_counts, doc_counts, zero_count = [], [], [], 0
    for row in rows:
        prediction, _ = select_prediction(row["id"], row["results"], doc_map, **params)
        predictions.append(prediction)
        chunk_counts.append(len(prediction.relevant_chunks))
        doc_counts.append(len(prediction.relevant_docs))
        zero_count += not prediction.relevant_chunks and not prediction.relevant_docs
    metrics = Evaluator().evaluate(truths, predictions, doc_map=doc_map).to_dict()
    return {
        **params, **metrics,
        "average_chunk_count": statistics.mean(chunk_counts),
        "average_doc_count": statistics.mean(doc_counts),
        "p95_chunk_count": sorted(chunk_counts)[math.ceil(.95 * len(chunk_counts)) - 1],
        "p95_doc_count": sorted(doc_counts)[math.ceil(.95 * len(doc_counts)) - 1],
        "empty_prediction_rate": zero_count / len(rows),
    }


def score_key(result):
    return (result["internal_macro_f2"], result["chunk_f2"], result["doc_f2"],
            -result["average_chunk_count"], -result["average_doc_count"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No calibration results")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, default=Path("outputs/vimed_validation/rerank/rankings.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/vimed/validation/ground_truth.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/vimed/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/selection_calibration/run_20260930"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose another --output: {args.output}")
    rows = read_rows(args.rankings)
    truths = DataLoader.load_ground_truth(args.ground_truth)
    chunks = DataLoader.load_chunks(args.chunks)
    if len(rows) != len(truths) or [r["id"] for r in rows] != [t.id for t in truths]:
        raise ValueError("Ranking and ground-truth query IDs/order differ")
    doc_map = DocumentChunkMap.from_chunks(chunks)
    truth_map = {t.id: t for t in truths}
    score_values = [float(score) for row in rows for _, score in row["results"]]
    gaps = [max(0.0, float(row["results"][0][1]) - float(score))
            for row in rows if row["results"] for _, score in row["results"]]
    positive_scores = [float(score) for row in rows for cid, score in row["results"]
                       if cid in set(truth_map[row["id"]].relevant_chunks)]
    threshold_values = quantiles(score_values, THRESHOLD_PERCENTILES)
    delta_values = quantiles(gaps, DELTA_PERCENTILES)
    distribution = {
        "num_queries": len(rows), "candidate_depth_min": min(map(lambda r: len(r["results"]), rows)),
        "candidate_depth_max": max(map(lambda r: len(r["results"]), rows)),
        "score": {"min": min(score_values), "max": max(score_values),
                  "percentiles": {str(p): float(np.percentile(score_values, p)) for p in (0, 10, 25, 50, 75, 90, 95, 99, 100)}},
        "top1_score": {"min": min(float(r["results"][0][1]) for r in rows),
                       "max": max(float(r["results"][0][1]) for r in rows),
                       "percentiles": {str(p): float(np.percentile([r["results"][0][1] for r in rows], p))
                                       for p in (0, 10, 25, 50, 75, 90, 95, 99, 100)}},
        "top1_gap_percentiles": {str(p): float(np.percentile(gaps, p)) for p in (0, 10, 25, 50, 75, 90, 95, 99, 100)},
        "positive_score_count": len(positive_scores),
        "positive_score_percentiles": ({str(p): float(np.percentile(positive_scores, p))
                                         for p in (0, 10, 25, 50, 75, 90, 100)} if positive_scores else {}),
        "threshold_grid_from_score_percentiles": threshold_values,
        "delta_grid_from_top1_gap_percentiles": delta_values,
    }

    base = dict(doc_aggregation="max", doc_top_n_mean=3)
    fixed_results = []
    for k in (1, 3, 5, 10):
        fixed = dict(base, chunk_threshold=min(score_values) - 1, chunk_delta=max(gaps) + 1,
                     doc_threshold=min(score_values) - 1, doc_delta=max(gaps) + 1,
                     chunk_min_k=k, chunk_max_k=k, doc_min_k=min(k, k), doc_max_k=k)
        result = evaluate_params(rows, truths, doc_map, fixed)
        result["selector"] = f"fixed_top{k}"
        fixed_results.append(result)

    # Coordinate search bounds the CPU-only grid while tuning chunk and document
    # choices independently with the same runtime selection implementation.
    chunk_candidates = [dict(base, chunk_threshold=t, chunk_delta=d, chunk_min_k=mn, chunk_max_k=mx,
                             doc_threshold=min(score_values) - 1, doc_delta=max(gaps) + 1,
                             doc_min_k=1, doc_max_k=10)
                        for t in threshold_values for d in delta_values
                        for mn in CHUNK_MINS for mx in CHUNK_MAXS if mn <= mx]
    chunk_results = [evaluate_params(rows, truths, doc_map, params) for params in chunk_candidates]
    best_chunk = max(chunk_results, key=score_key)
    doc_candidates = [dict(base, chunk_threshold=best_chunk["chunk_threshold"],
                           chunk_delta=best_chunk["chunk_delta"], chunk_min_k=best_chunk["chunk_min_k"],
                           chunk_max_k=best_chunk["chunk_max_k"], doc_threshold=t, doc_delta=d,
                           doc_min_k=mn, doc_max_k=mx)
                      for t in threshold_values for d in delta_values
                      for mn in DOC_MINS for mx in DOC_MAXS if mn <= mx]
    dynamic_results = [evaluate_params(rows, truths, doc_map, params) for params in doc_candidates]
    best_dynamic = max(dynamic_results, key=score_key)
    all_results = fixed_results + [dict(result, selector="dynamic") for result in dynamic_results]
    all_results += [dict(result, selector="chunk_search") for result in chunk_results]

    args.output.mkdir(parents=True)
    write_run_manifest(args.output, task="offline_selection_calibration", root=Path.cwd(), inputs={
        "rankings": args.rankings, "ground_truth": args.ground_truth, "chunks": args.chunks,
        "dynamic_config": Path("configs/vimed_dynamic_selection.yaml"),
    }, parameters={"threshold_percentiles": THRESHOLD_PERCENTILES,
                   "delta_percentiles": DELTA_PERCENTILES, "chunk_min": CHUNK_MINS,
                   "chunk_max": CHUNK_MAXS, "doc_min": DOC_MINS, "doc_max": DOC_MAXS,
                   "objective": "internal_macro_f2", "model_inference": False})
    write_csv(args.output / "all_results.csv", all_results)
    (args.output / "score_distribution.json").write_text(json.dumps(distribution, indent=2) + "\n")
    best_config = {key: best_dynamic[key] for key in (
        "doc_aggregation", "doc_top_n_mean", "chunk_threshold", "chunk_delta", "chunk_min_k", "chunk_max_k",
        "doc_threshold", "doc_delta", "doc_min_k", "doc_max_k")}
    best_config.update({"objective": "internal_macro_f2", "internal_macro_f2": best_dynamic["internal_macro_f2"],
                        "chunk_f2": best_dynamic["chunk_f2"], "doc_f2": best_dynamic["doc_f2"],
                        "split": "validation", "num_queries": len(rows),
                        "search_method": "coordinate search; best configuration found in searched grid, not guaranteed global optimum"})
    (args.output / "best_config.json").write_text(json.dumps(best_config, indent=2) + "\n")

    sorted_fixed = sorted(fixed_results, key=score_key, reverse=True)
    sorted_dynamic = sorted(dynamic_results, key=score_key, reverse=True)
    fold_members = {fold: [] for fold in range(5)}
    for i, row in enumerate(rows):
        fold = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % 5
        fold_members[fold].append(i)
    robustness = {}
    robustness_configs = [(r["selector"], r) for r in fixed_results]
    robustness_configs.append(("best_dynamic", best_dynamic))
    for label, result in robustness_configs:
        params = {key: result[key] for key in (
            "doc_aggregation", "doc_top_n_mean", "chunk_threshold", "chunk_delta", "chunk_min_k", "chunk_max_k",
            "doc_threshold", "doc_delta", "doc_min_k", "doc_max_k")}
        robustness[label] = {}
        for fold, indices in fold_members.items():
            robustness[label][str(fold)] = evaluate_params(
                [rows[i] for i in indices], [truths[i] for i in indices], doc_map, params)
    (args.output / "fold_metrics.json").write_text(json.dumps({
        "method": "fixed SHA256(query_id) modulo 5 descriptive validation slices",
        "warning": "Configurations were selected using all validation queries; fold values are stability slices, not unbiased cross-validation.",
        "fold_sizes": {str(fold): len(indices) for fold, indices in fold_members.items()},
        "metrics": robustness,
    }, indent=2) + "\n")
    lines = ["# Offline selection calibration", "",
             f"Validation queries: {len(rows)}. Reranker rankings were read from `{args.rankings}`; no model was run.",
             "Objective: `internal_macro_f2`, equal mean of per-query document and chunk F2.",
             "Thresholds come from score percentiles; deltas come from top1-to-candidate score-gap percentiles.",
             "The coordinate search tunes chunk parameters, then document parameters, against validation; the reported best is best in this search, not a proven global optimum. Scores are selection estimates and may be optimistic.", "",
             "## Fixed top K comparison", "",
             "| Selector | Internal Macro F2 | Doc P/R/F2 | Chunk P/R/F2 | Avg chunks | Avg docs |",
             "|---|---:|---:|---:|---:|---:|"]
    for result in sorted_fixed:
        lines.append(f"| {result['selector']} | {result['internal_macro_f2']:.4f} | "
                     f"{result['doc_precision']:.4f}/{result['doc_recall']:.4f}/{result['doc_f2']:.4f} | "
                     f"{result['chunk_precision']:.4f}/{result['chunk_recall']:.4f}/{result['chunk_f2']:.4f} | "
                     f"{result['average_chunk_count']:.2f} | {result['average_doc_count']:.2f} |")
    lines += ["", "## Top 10 dynamic configurations", "", "| Rank | Internal Macro F2 | Chunk threshold/delta/min/max | Doc threshold/delta/min/max | Doc F2 | Chunk F2 | Avg chunks/docs |", "|---:|---:|---|---|---:|---:|---:|"]
    for rank, result in enumerate(sorted_dynamic[:10], 1):
        lines.append(f"| {rank} | {result['internal_macro_f2']:.4f} | "
                     f"{result['chunk_threshold']:.4g}/{result['chunk_delta']:.4g}/{result['chunk_min_k']}/{result['chunk_max_k']} | "
                     f"{result['doc_threshold']:.4g}/{result['doc_delta']:.4g}/{result['doc_min_k']}/{result['doc_max_k']} | "
                     f"{result['doc_f2']:.4f} | {result['chunk_f2']:.4f} | "
                     f"{result['average_chunk_count']:.2f}/{result['average_doc_count']:.2f} |")
    lines += ["", "## Validation slice stability", "", "Cấu hình tốt nhất và fixed Top K được chấm riêng trên 5 fold SHA256 cố định.", "| Selector | Mean fold Internal Macro F2 | Min | Max |", "|---|---:|---:|---:|"]
    for label, folds in robustness.items():
        values = [metrics["internal_macro_f2"] for metrics in folds.values()]
        lines.append(f"| {label} | {statistics.mean(values):.4f} | {min(values):.4f} | {max(values):.4f} |")
    lines += ["", "## Caveat", "", "`internal_macro_f2` là contract nội bộ. Ground truth hiện có một source context suy ra mỗi query, chưa phải nhãn relevance đầy đủ. Các fold là lát cắt độ ổn định; config được chọn từ toàn validation nên metric fold không phải ước lượng độc lập. Giữ test split cho đánh giá cuối.", ""]
    (args.output / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate_thresholds": threshold_values,
                      "candidate_deltas": delta_values, "fixed_top5": next(r for r in fixed_results if r["selector"] == "fixed_top5")["internal_macro_f2"],
                      "best_dynamic": best_config}, indent=2))


if __name__ == "__main__":
    main()
