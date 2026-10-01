"""Replay cached ViMed validation rankings and final predictions into a report."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.benchmark_vimed_validation import CONFIGS, load_dependency, read_rows
from scripts.run_metadata import write_run_manifest
from src.data.loader import DataLoader, DocumentChunkMap
from src.data.schema import PredictionRecord
from src.evaluation.evaluator import Evaluator
from src.evaluation.ranking_metrics import summarize_rankings, validate_ranking
from src.indexing.sparse_index import corpus_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-objective", choices=("retrieval", "final_selection"), default="retrieval")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path("outputs/vimed_validation")
    samples = read_rows(Path("data/vimed/validation/samples.jsonl"))
    truths = DataLoader.load_ground_truth("data/vimed/validation/ground_truth.jsonl")
    chunks = DataLoader.load_chunks("data/vimed/chunks.jsonl")
    doc_map = DocumentChunkMap.from_chunks(chunks)
    corpus_hash = corpus_sha256(chunks)
    sample_hash = hashlib.sha256(
        json.dumps(samples, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    rankings, stage_metrics, final_metrics = {}, {}, {}
    for stage in CONFIGS:
        rows = load_dependency(root, stage, samples, corpus_hash, sample_hash)
        for row in rows:
            validate_ranking(row["results"], doc_map.all_chunk_ids)
        saved = json.loads((root / stage / "metrics.json").read_text(encoding="utf-8"))
        replayed = summarize_rankings(samples, rows)
        for key, value in replayed.items():
            if saved.get(key) != value:
                raise ValueError(f"{stage} ranking metric mismatch: {key}")
        predictions = [PredictionRecord.model_validate(row["prediction"]) for row in rows]
        selection = Evaluator().evaluate(truths, predictions, doc_map=doc_map).to_dict()
        rankings[stage] = rows
        stage_metrics[stage] = saved
        final_metrics[stage] = selection

    if args.winner_objective == "retrieval":
        winner = max(CONFIGS, key=lambda stage: (
            stage_metrics[stage]["recall_at_k"]["10"],
            stage_metrics[stage]["mrr_at_100"],
            -stage_metrics[stage]["latency_ms"]["mean"],
        ))
        objective_label = "Recall@10 → MRR@100 → latency"
    else:
        winner = max(CONFIGS, key=lambda stage: (
            final_metrics[stage]["internal_macro_f2"],
            final_metrics[stage]["chunk_f2"],
            final_metrics[stage]["doc_f2"],
            -sum(len(row["prediction"]["relevant_chunks"]) for row in rankings[stage]) / len(samples),
        ))
        objective_label = "internal_macro_f2 → chunk_f2 → doc_f2 → lower average chunk count"

    report = {
        "split": "validation",
        "num_queries": len(samples),
        "corpus_sha256": corpus_hash,
        "winner_objective": args.winner_objective,
        "winner_rule": objective_label,
        "winner": winner,
        "retrieval_quality": {
            stage: {key: metrics[key] for key in (
                "recall_at_k", "document_recall_at_k", "mrr_at_100", "zero_hit_queries_at_k",
                "latency_ms", "candidate_depth", "latency_method",
                "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "union_candidate_recall") if key in metrics}
            for stage, metrics in stage_metrics.items()
        },
        "final_selection_quality": final_metrics,
        "warning": "internal_macro_f2 là metric nội bộ; ground truth mỗi query hiện chỉ có source context suy ra từ QA.",
    }
    out = args.output or Path("outputs/vimed_validation_reports") / args.winner_objective
    if out.exists():
        raise FileExistsError(f"Report destination exists; choose a new --output: {out}")
    out.mkdir(parents=True)
    write_run_manifest(out, task="cached_vimed_validation_report", root=Path.cwd(), inputs={
        **{f"{stage}_rankings": root / stage / "rankings.jsonl" for stage in CONFIGS},
        "ground_truth": Path("data/vimed/validation/ground_truth.jsonl"),
        "chunks": Path("data/vimed/chunks.jsonl"),
        **{f"{stage}_config": Path(path) for stage, path in CONFIGS.items()},
    }, parameters={"winner_objective": args.winner_objective, "split": "validation"})
    report["source_commit_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# ViMed validation report: {args.winner_objective}", "",
             f"Validation queries: {len(samples)}. Winner: **{winner}**.",
             f"Rule: `{objective_label}`. Config files are not changed.", "",
             "## Retrieval", "",
             "| Stage | Recall@10 | Candidate Recall@100 | Union Candidate Recall | MRR@100 | Zero hit @10 | Mean latency ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for stage, metrics in stage_metrics.items():
        union = metrics.get("union_candidate_recall")
        union_cell = f"{union:.4f}" if union is not None else "—"
        lines.append(f"| {stage} | {metrics['recall_at_k']['10']:.4f} | {metrics['recall_at_k']['100']:.4f} | "
                     f"{union_cell} | {metrics['mrr_at_100']:.4f} | {metrics['zero_hit_queries_at_k']['10']} | "
                     f"{metrics['latency_ms']['mean']:.1f} |")
    lines += ["", "## Final Selection", "",
              "| Stage | Doc P | Doc R | Doc F2 | Chunk P | Chunk R | Chunk F2 | Internal Macro F2 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stage, metrics in final_metrics.items():
        lines.append(f"| {stage} | {metrics['doc_precision']:.4f} | {metrics['doc_recall']:.4f} | "
                     f"{metrics['doc_f2']:.4f} | {metrics['chunk_precision']:.4f} | "
                     f"{metrics['chunk_recall']:.4f} | {metrics['chunk_f2']:.4f} | "
                     f"{metrics['internal_macro_f2']:.4f} |")
    lines += ["", "`internal_macro_f2` là điểm composite nội bộ, không phải điểm chính thức của cuộc thi.",
              "Ground truth chưa có đủ relevance judgments; các metric phản ánh source-context retrieval.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"winner": winner, "objective": args.winner_objective,
                      "output": str(out), "final_selection": final_metrics[winner]}, indent=2))


if __name__ == "__main__":
    main()
