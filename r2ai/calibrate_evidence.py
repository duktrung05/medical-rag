"""Label a cached candidate pool, then calibrate selection on dev and check holdout.

Metrics are internal macro set precision/recall/F2, not an official scorer.
Judged candidate pools measure pool recall; complete corpus gold can be supplied
as relevant_docs and relevant_chunks (doc_id/chunk_index), with label_scope=corpus.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.evidence_io import fingerprint, read_ranking_cache, selection_summary
from r2ai.evidence_selector import SelectionConfig, select_evidence


def prepare_labels(rankings, path: Path, sample_size: int, seed: int) -> None:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    sample = random.Random(seed).sample(rankings, min(sample_size, len(rankings)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in sample:
            annotation = {"id": row.query_id, "query": row.query_text, "label_scope": "candidate_pool",
                          "judgments": [{"doc_id": c.doc_id, "chunk_index": c.chunk_index,
                                         "chunk_text": c.chunk_text, "rerank_score": c.rerank_score,
                                         "relevant": None} for c in row.candidates]}
            stream.write(json.dumps(annotation, ensure_ascii=False, allow_nan=False) + "\n")


def chunk_key(row) -> tuple[int, int]:
    doc, index = row["doc_id"], row["chunk_index"]
    if type(doc) is not int or type(index) is not int or index < 0:
        raise ValueError("Ground truth chunk requires integer doc_id and nonnegative chunk_index")
    return doc, index


def load_truth(path: Path, rankings) -> dict[int, dict]:
    by_id = {r.query_id: r for r in rankings}
    truths = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["id"]
        if type(qid) is not int or qid in truths or qid not in by_id:
            raise ValueError(f"Invalid, duplicate or unknown ground truth query ID: {qid}")
        scope = row.get("label_scope")
        if scope not in {"candidate_pool", "corpus"}:
            raise ValueError("Every label row must declare label_scope=candidate_pool or corpus")
        if "judgments" in row:
            if scope != "candidate_pool":
                raise ValueError("Pool judgments cannot claim corpus recall")
            judgments = row["judgments"]
            if any(type(j.get("relevant")) is not bool for j in judgments):
                raise ValueError(f"Query {qid} has unjudged candidates; fill every relevant field with true/false")
            keys = [chunk_key(j) for j in judgments]
            if len(set(keys)) != len(keys) or set(keys) != {c.key for c in by_id[qid].candidates}:
                raise ValueError(f"Query {qid} judgments must cover exactly its cached candidate pool")
            texts = {c.key: c.chunk_text for c in by_id[qid].candidates}
            if any(j.get("chunk_text") != texts[chunk_key(j)] for j in judgments):
                raise ValueError(f"Query {qid} judgment text differs from cache")
            chunks = {chunk_key(j) for j in judgments if j["relevant"]}
            docs = {doc for doc, _ in chunks}
        else:
            docs = set(row["relevant_docs"])
            chunks_list = [chunk_key(c) for c in row["relevant_chunks"]]
            chunks = set(chunks_list)
            if any(type(doc) is not int for doc in docs):
                raise ValueError("Ground truth document IDs must be integers")
            if len(docs) != len(row["relevant_docs"]) or len(chunks) != len(chunks_list):
                raise ValueError("Duplicate ground truth IDs")
            if not {doc for doc, _ in chunks} <= docs:
                raise ValueError("Ground truth chunks require their parent documents")
            if scope == "candidate_pool" and (not chunks <= {c.key for c in by_id[qid].candidates}
                                               or not docs <= set(by_id[qid].document_order)):
                raise ValueError("Pool ground truth contains items outside the cached pool")
        truths[qid] = {"docs": docs, "chunks": chunks, "scope": scope}
    if not truths:
        raise ValueError("Ground truth must contain at least one query")
    if len({t["scope"] for t in truths.values()}) != 1:
        raise ValueError("Do not mix corpus and candidate_pool labels in one evaluation")
    return truths


def set_metrics(actual: set, predicted: set) -> tuple[float, float, float]:
    tp = len(actual & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision and recall else 0.0
    return precision, recall, f2


def evaluate(rankings, truths, config: SelectionConfig) -> tuple[dict, list[dict]]:
    totals = dict.fromkeys(("doc_precision", "doc_recall", "doc_f2",
                          "chunk_precision", "chunk_recall", "chunk_f2"), 0.0)
    predictions, details = [], []
    for ranking in rankings:
        if ranking.query_id not in truths:
            continue
        truth = truths[ranking.query_id]
        prediction, diagnostic = select_evidence(ranking, config)
        chosen = {(d["doc_id"], d["chunk_index"]) for d in diagnostic["decisions"] if d["reason"] == "selected"}
        doc_metrics = set_metrics(truth["docs"], set(prediction["relevant_docs"]))
        chunk_metrics = set_metrics(truth["chunks"], chosen)
        per_query = dict(zip(totals, doc_metrics + chunk_metrics, strict=True))
        for key, value in per_query.items():
            totals[key] += value
        details.append({"id": ranking.query_id, **per_query,
                        "selected_chunks": [list(c) for c in sorted(chosen)],
                        "false_positive_chunks": [list(c) for c in sorted(chosen - truth["chunks"])],
                        "missed_positive_chunks": [list(c) for c in sorted(truth["chunks"] - chosen)]})
        predictions.append(prediction)
    if len(predictions) != len(truths):
        raise ValueError("Cache must contain every ground truth query")
    metrics = {key: value / len(truths) for key, value in totals.items()}
    metrics["internal_macro_f2"] = (metrics["doc_f2"] + metrics["chunk_f2"]) / 2
    return {**metrics, **selection_summary(predictions), "label_scope": next(iter(truths.values()))["scope"]}, details


def improves(metrics: dict, baseline: dict, recall_drop: float) -> bool:
    return (metrics["chunk_precision"] > baseline["chunk_precision"]
            and metrics["chunk_recall"] >= baseline["chunk_recall"] - recall_drop
            and metrics["doc_recall"] >= baseline["doc_recall"] - recall_drop
            and metrics["internal_macro_f2"] >= baseline["internal_macro_f2"])


def percentiles(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("Cache has no reranked scores to calibrate")
    values = sorted(values)
    output = {values[0] - 1.0}
    for p in (0, .1, .3, .5, .7, .9, 1):
        position = p * (len(values) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        output.add(values[lo] + (values[hi] - values[lo]) * (position - lo))
    return sorted(output)


def optional_floats(value: str) -> list[float | None]:
    return [None if item.strip().lower() == "none" else float(item) for item in value.split(",")]


def score_distribution(rankings, truths) -> dict:
    groups = {"positive": [], "negative": []}
    unscored = 0
    for ranking in rankings:
        if ranking.query_id not in truths:
            continue
        for candidate in ranking.candidates:
            if candidate.rerank_score is None:
                unscored += 1
                continue
            label = "positive" if candidate.key in truths[ranking.query_id]["chunks"] else "negative"
            groups[label].append(candidate.rerank_score)
    result = {"not_reranked": unscored}
    for label, values in groups.items():
        ordered = sorted(values)
        result[label] = {"count": len(values), "min": min(values) if values else None,
                         "max": max(values) if values else None,
                         "median": ordered[len(ordered) // 2] if ordered else None}
    return result


def calibrate(rankings, truths, baseline, configs, recall_drop: float, output: Path,
              metadata: dict, holdout=None) -> dict:
    if not 0 <= recall_drop <= 1:
        raise ValueError("max_recall_drop must be between 0 and 1")
    if holdout and truths.keys() & holdout.keys():
        raise ValueError("Dev and holdout query IDs must be disjoint")
    if holdout and {t["scope"] for t in truths.values()} != {t["scope"] for t in holdout.values()}:
        raise ValueError("Dev and holdout must use the same label scope")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    baseline_metrics, _ = evaluate(rankings, truths, baseline)
    results = []
    best_config, best_metrics = baseline, baseline_metrics
    for config in configs:
        metrics, _ = evaluate(rankings, truths, config)
        eligible = improves(metrics, baseline_metrics, recall_drop)
        results.append({**asdict(config), **metrics, "meets_objective": eligible})
        if eligible and (metrics["chunk_precision"], metrics["internal_macro_f2"]) > (
                best_metrics["chunk_precision"], best_metrics["internal_macro_f2"]):
            best_config, best_metrics = config, metrics
    report = {"status": "improving_configuration_found" if best_config != baseline else "no_improving_configuration",
              "baseline_config": asdict(baseline), "baseline_dev": baseline_metrics,
              "selected_config": asdict(best_config), "selected_dev": best_metrics,
              "max_recall_drop": recall_drop, "ranking_metadata": metadata,
              "scoring": "internal macro set metrics; empty predictions score zero; all labeled queries included",
              "model_inference": False}
    if holdout:
        baseline_holdout, _ = evaluate(rankings, holdout, baseline)
        selected_holdout, holdout_details = evaluate(rankings, holdout, best_config)
        report.update(baseline_holdout=baseline_holdout, selected_holdout=selected_holdout,
                      holdout_meets_objective=improves(selected_holdout, baseline_holdout, recall_drop))
    _, baseline_details = evaluate(rankings, truths, baseline)
    _, details = evaluate(rankings, truths, best_config)
    baseline_by_id = {row["id"]: row for row in baseline_details}
    for row in details:
        previous = baseline_by_id[row["id"]]
        selected = {tuple(c) for c in row["selected_chunks"]}
        removed = {tuple(c) for c in previous["selected_chunks"]} - selected
        row["removed_false_positive_chunks"] = [list(c) for c in sorted(removed - truths[row["id"]]["chunks"])]
        row["removed_positive_chunks"] = [list(c) for c in sorted(removed & truths[row["id"]]["chunks"])]
    report["score_distribution_dev"] = score_distribution(rankings, truths)
    output.mkdir(parents=True)
    with (output / "all_results.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = [{**asdict(baseline), **baseline_metrics, "meets_objective": False}] + results
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "best_config.json").write_text(json.dumps(asdict(best_config), indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["| Configuration | Chunk P | Chunk R | Doc P | Doc R | Internal F2 | Empty chunks |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    comparisons = [("Baseline / dev", baseline_metrics), ("Selected / dev", best_metrics)]
    if holdout:
        comparisons += [("Baseline / holdout", baseline_holdout), ("Selected / holdout", selected_holdout)]
    for name, metrics in comparisons:
        lines.append(f"| {name} | " + " | ".join(f"{metrics[k]:.4f}" for k in
                     ("chunk_precision", "chunk_recall", "doc_precision", "doc_recall",
                      "internal_macro_f2", "empty_chunk_rate")) + " |")
    scope = next(iter(truths.values()))["scope"]
    (output / "report.md").write_text(
        f"Selection status: {report['status']}\n\nLabel scope: {scope}. "
        "Internal macro set metrics; empty predictions score zero.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    (output / "dev_details.jsonl").write_text("".join(json.dumps(d) + "\n" for d in details), encoding="utf-8")
    if holdout:
        (output / "holdout_details.jsonl").write_text("".join(json.dumps(d) + "\n" for d in holdout_details), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--prepare-labels", type=Path, help="Export candidate judgments with relevant=null; no automatic labels")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--holdout-ground-truth", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-k-docs", type=int, default=10)
    parser.add_argument("--baseline-k-chunks", type=int, default=10)
    parser.add_argument("--max-docs", default="5,10,20")
    parser.add_argument("--max-chunks", default="5,10,20")
    parser.add_argument("--chunks-per-doc", type=int, default=1)
    parser.add_argument("--chunk-thresholds", help="Comma-separated raw scores; default derives percentiles from dev")
    parser.add_argument("--chunk-deltas", default="none")
    parser.add_argument("--document-modes", default="parents,baseline,threshold")
    parser.add_argument("--doc-thresholds", default="none", help="none uses the chunk threshold")
    parser.add_argument("--doc-deltas", default="none")
    parser.add_argument("--max-recall-drop", type=float, default=.02, help="Absolute recall allowance, .02 = 2 percentage points")
    args = parser.parse_args()
    rankings, metadata = read_ranking_cache(args.rankings)
    if args.prepare_labels:
        prepare_labels(rankings, args.prepare_labels, args.sample_size, args.seed)
        print(f"Wrote {args.prepare_labels}; label candidates and split dev/holdout before calibration")
        return
    if not args.ground_truth or not args.output:
        parser.error("Calibration requires --ground-truth and --output")
    truths = load_truth(args.ground_truth, rankings)
    holdout = load_truth(args.holdout_ground_truth, rankings) if args.holdout_ground_truth else None
    baseline = SelectionConfig(max_docs=args.baseline_k_docs, max_chunks=args.baseline_k_chunks,
                               chunks_per_doc=args.chunks_per_doc)
    thresholds = ([float(t) for t in args.chunk_thresholds.split(",")] if args.chunk_thresholds else
                  percentiles([c.rerank_score for r in rankings if r.query_id in truths
                               for c in r.candidates if c.rerank_score is not None]))
    doc_caps = [int(k) for k in args.max_docs.split(",")]
    chunk_caps = [int(k) for k in args.max_chunks.split(",")]
    configs = [SelectionConfig(max_docs=d, max_chunks=c, chunks_per_doc=args.chunks_per_doc)
               for d, c in itertools.product(doc_caps, chunk_caps)]
    for d, c, t, delta, mode in itertools.product(doc_caps, chunk_caps, thresholds,
                                                  optional_floats(args.chunk_deltas), args.document_modes.split(",")):
        doc_gates = (itertools.product(optional_floats(args.doc_thresholds), optional_floats(args.doc_deltas))
                     if mode == "threshold" else [(None, None)])
        for dt, dd in doc_gates:
            configs.append(SelectionConfig(mode="threshold", max_docs=d, max_chunks=c,
                                           chunks_per_doc=args.chunks_per_doc, chunk_threshold=t,
                                           chunk_delta=delta, document_mode=mode, doc_threshold=dt, doc_delta=dd))
    metadata = {**metadata, "rankings_file": fingerprint(args.rankings), "dev_labels": fingerprint(args.ground_truth)}
    if args.holdout_ground_truth:
        metadata["holdout_labels"] = fingerprint(args.holdout_ground_truth)
    report = calibrate(rankings, truths, baseline, configs, args.max_recall_drop, args.output, metadata, holdout)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
