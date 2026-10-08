"""Freeze a ViBioMIR baseline and prepare a disjoint, stratified label audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from r2ai.evidence_io import read_ranking_cache

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_diagnostics(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            query_id = row["id"]
            if query_id in rows:
                raise ValueError(f"Duplicate diagnostic query id: {query_id}")
            rows[query_id] = row
    return rows


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def query_features(rankings, diagnostics: dict[int, dict]) -> tuple[dict[int, dict], dict]:
    top_scores = [
        max(c.rerank_score for c in row.candidates if c.rerank_score is not None)
        for row in rankings if any(c.rerank_score is not None for c in row.candidates)
    ]
    if not top_scores:
        raise ValueError("Ranking cache has no reranker scores")
    score_low, score_high = _quantile(top_scores, 1 / 3), _quantile(top_scores, 2 / 3)
    features: dict[int, dict] = {}
    for row in rankings:
        scores = [c.rerank_score for c in row.candidates if c.rerank_score is not None]
        top_score = max(scores) if scores else float("-inf")
        cjk_candidates = sum(bool(_CJK.search(c.chunk_text)) for c in row.candidates)
        cjk_share = cjk_candidates / len(row.candidates) if row.candidates else 0.0
        language_bin = "non_cjk" if cjk_share == 0 else ("mixed" if cjk_share < 0.5 else "cjk_heavy")
        score_bin = "low" if top_score <= score_low else ("mid" if top_score <= score_high else "high")
        diagnostic = diagnostics[row.query_id]
        max_chunks = diagnostic["config"]["max_chunks"]
        selected = diagnostic["selected_chunk_count"]
        cap_bin = "at_cap" if selected >= max_chunks else "below_cap"
        features[row.query_id] = {
            "language_bin": language_bin,
            "score_bin": score_bin,
            "cap_bin": cap_bin,
            "cjk_candidate_share": cjk_share,
            "top_rerank_score": None if top_score == float("-inf") else top_score,
            "selected_chunk_count": selected,
            "selected_doc_count": diagnostic["selected_doc_count"],
            "stratum": f"{language_bin}/{score_bin}/{cap_bin}",
        }
    return features, {"top_score_tertiles": [score_low, score_high]}


def stratified_sample(rankings, features: dict[int, dict], sample_size: int, seed: int):
    if not 0 < sample_size <= len(rankings):
        raise ValueError("sample_size must be within the ranking count")
    groups = defaultdict(list)
    for row in rankings:
        groups[features[row.query_id]["stratum"]].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    names = sorted(groups)
    while len(selected) < sample_size:
        progressed = False
        for name in names:
            if groups[name] and len(selected) < sample_size:
                selected.append(groups[name].pop())
                progressed = True
        if not progressed:
            break
    return selected


def annotation_row(row, feature: dict) -> dict:
    return {
        "id": row.query_id,
        "query": row.query_text,
        "label_scope": "candidate_pool",
        "audit_stratum": feature["stratum"],
        "audit_features": {key: feature[key] for key in (
            "language_bin", "score_bin", "cap_bin", "cjk_candidate_share",
            "top_rerank_score", "selected_chunk_count", "selected_doc_count"
        )},
        "judgments": [
            {
                "doc_id": candidate.doc_id,
                "chunk_index": candidate.chunk_index,
                "chunk_text": candidate.chunk_text,
                "rerank_score": candidate.rerank_score,
                "relevant": None,
            }
            for candidate in row.candidates
        ],
    }


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(rankings, diagnostics: dict[int, dict], features: dict[int, dict]) -> dict:
    reasons = Counter()
    unique_docs = []
    docs_with_multiple_chunks = []
    max_chunks_per_doc = []
    for row in rankings:
        counts = Counter(candidate.doc_id for candidate in row.candidates)
        unique_docs.append(len(counts))
        docs_with_multiple_chunks.append(sum(count >= 2 for count in counts.values()))
        max_chunks_per_doc.append(max(counts.values(), default=0))
        reasons.update(diagnostics[row.query_id]["reasons"])
    selected_counts = [diagnostics[row.query_id]["selected_chunk_count"] for row in rankings]
    max_chunk_limits = [diagnostics[row.query_id]["config"]["max_chunks"] for row in rankings]
    return {
        "num_queries": len(rankings),
        "candidate_pairs": sum(len(row.candidates) for row in rankings),
        "unscored_pairs": sum(candidate.rerank_score is None for row in rankings for candidate in row.candidates),
        "unique_docs_per_query": {
            "mean": mean(unique_docs), "median": median(unique_docs),
            "min": min(unique_docs), "max": max(unique_docs),
        },
        "docs_with_multiple_chunks_per_query_mean": mean(docs_with_multiple_chunks),
        "max_chunks_from_one_doc_per_query_mean": mean(max_chunks_per_doc),
        "selected_chunks_per_query_mean": mean(selected_counts),
        "queries_at_chunk_cap": sum(value >= limit for value, limit in zip(selected_counts, max_chunk_limits)),
        "decision_reasons": dict(sorted(reasons.items())),
        "population_strata": dict(sorted(Counter(feature["stratum"] for feature in features.values()).items())),
    }


def write_metrics(path: Path, feedback: dict, baseline_sha: str, candidate_sha: str) -> None:
    fields = ["run", "sha256", "score_status", "FINAL_SCORE", "DOCS_F2MACRO", "CHUNKS_F2MACRO",
              "DOCS_PRECISION", "DOCS_RECALL", "CHUNKS_PRECISION", "CHUNKS_RECALL"]
    metrics = feedback["metrics"]
    rows = [
        {"run": "threshold_neg2", "sha256": baseline_sha, "score_status": "recorded_user_feedback", **metrics},
        {"run": "threshold_neg2p5", "sha256": candidate_sha, "score_status": "not_scored_or_feedback_missing"},
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True, help="Scored threshold -2 run directory")
    parser.add_argument("--candidate-run", type=Path, required=True, help="Current threshold -2.5 run directory")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if not 0 < args.dev_size < args.sample_size:
        raise ValueError("dev_size must be positive and smaller than sample_size")

    rankings, ranking_metadata = read_ranking_cache(args.rankings)
    candidate_diagnostics_path = args.candidate_run / "diagnostics.jsonl"
    diagnostics = read_diagnostics(candidate_diagnostics_path)
    ranking_ids = {row.query_id for row in rankings}
    if set(diagnostics) != ranking_ids:
        raise ValueError("Diagnostics and ranking cache query IDs differ")
    feedback = json.loads(args.feedback.read_text(encoding="utf-8"))
    required_metrics = {"FINAL_SCORE", "DOCS_F2MACRO", "CHUNKS_F2MACRO", "DOCS_PRECISION",
                        "DOCS_RECALL", "CHUNKS_PRECISION", "CHUNKS_RECALL"}
    if set(feedback.get("metrics", {})) != required_metrics:
        raise ValueError("Feedback must contain exactly the seven official metrics")

    args.output.mkdir(parents=True)
    features, stratification = query_features(rankings, diagnostics)
    sample = stratified_sample(rankings, features, args.sample_size, args.seed)
    holdout_size = args.sample_size - args.dev_size
    holdout_positions = {round(i * (args.sample_size - 1) / (holdout_size - 1)) for i in range(holdout_size)}
    holdout = [row for index, row in enumerate(sample) if index in holdout_positions]
    dev = [row for index, row in enumerate(sample) if index not in holdout_positions]
    if len(dev) != args.dev_size or len(holdout) != holdout_size:
        raise AssertionError("Unexpected deterministic split size")
    write_jsonl(args.output / "labels_dev_unjudged.jsonl", (annotation_row(row, features[row.query_id]) for row in dev))
    write_jsonl(args.output / "labels_holdout_unjudged.jsonl",
                (annotation_row(row, features[row.query_id]) for row in holdout))

    baseline_zip = args.baseline_run / "submission.zip"
    candidate_zip = args.candidate_run / "submission.zip"
    audit = summarize(rankings, diagnostics, features)
    audit["stratification"] = stratification
    audit["sample"] = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "dev_size": len(dev),
        "holdout_size": len(holdout),
        "dev_query_ids": [row.query_id for row in dev],
        "holdout_query_ids": [row.query_id for row in holdout],
        "dev_strata": dict(sorted(Counter(features[row.query_id]["stratum"] for row in dev).items())),
        "holdout_strata": dict(sorted(Counter(features[row.query_id]["stratum"] for row in holdout).items())),
    }
    (args.output / "error_breakdown.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_metrics(args.output / "leaderboard_metrics.csv", feedback,
                  fingerprint(baseline_zip)["sha256"], fingerprint(candidate_zip)["sha256"])
    manifest = {
        "task": "ViBioMIR score improvement P0 baseline freeze and error audit",
        "score_provenance": feedback.get("source"),
        "scorer_status": "official aggregate metric names verified from public competition page; exact chunk matching implementation unavailable",
        "artifacts": {
            "ranking_cache": fingerprint(args.rankings),
            "baseline_submission": fingerprint(baseline_zip),
            "baseline_selection_config": fingerprint(args.baseline_run / "selection_config.json"),
            "candidate_submission": fingerprint(candidate_zip),
            "candidate_selection_config": fingerprint(args.candidate_run / "selection_config.json"),
            "candidate_diagnostics": fingerprint(candidate_diagnostics_path),
            "leaderboard_feedback": fingerprint(args.feedback),
        },
        "ranking_metadata": ranking_metadata,
        "effective_configs": {
            "baseline_threshold_neg2": json.loads((args.baseline_run / "selection_config.json").read_text()),
            "candidate_threshold_neg2p5": json.loads((args.candidate_run / "selection_config.json").read_text()),
        },
        "notes": [
            "A null doc_threshold means the selector uses chunk_threshold as the effective document threshold.",
            "The -2.5 candidate has no linked leaderboard feedback in the workspace.",
            "Unjudged templates measure candidate-pool recall only after every relevant field is filled.",
        ],
    }
    (args.output / "baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# ViBioMIR Task 0 audit

- Scored baseline: threshold -2, FINAL_SCORE {feedback['metrics']['FINAL_SCORE']:.4f}.
- Current candidate: threshold -2.5; no linked leaderboard feedback was found.
- Ranking cache: {audit['candidate_pairs']:,} scored pairs across {audit['num_queries']:,} queries.
- Distinct reranked documents/query: mean {audit['unique_docs_per_query']['mean']:.2f}, median {audit['unique_docs_per_query']['median']:.0f}.
- Candidate selector output: mean {audit['selected_chunks_per_query_mean']:.2f} chunks/query; {audit['queries_at_chunk_cap']}/{audit['num_queries']} queries hit their chunk cap.
- Annotation split: {len(dev)} dev + {len(holdout)} holdout, disjoint and deterministically stratified; all `relevant` values remain null pending human judgment.

The public competition page confirms document/chunk retrieval and recall-weighted Macro F2. The exact official chunk normalization/matching implementation was not publicly retrievable, so LCS/tokenization assumptions remain unverified.
"""
    (args.output / "report.md").write_text(report, encoding="utf-8")
    guide = """# Annotation guide

For every candidate, replace `relevant: null` with `true` or `false`. Judge whether
the passage contains evidence that directly addresses the Vietnamese query; being
about the same disease or body part is not sufficient. Keep source text and IDs
unchanged. The dev and holdout query IDs must remain disjoint.

After judging a query, optionally add one `error_category` field to its top-level
object using one of these values:

- `no_relevant_document_in_candidate_pool`
- `relevant_document_but_no_relevant_passage`
- `relevant_passage_below_reranker_or_selector_gate`
- `extraction_or_boilerplate_failure`
- `passage_boundary_or_missing_context`
- `matching_rule_uncertain`
- `no_pipeline_error_observed`

These labels describe the cached candidate pool. They do not establish recall over
the full 4.39M-document corpus.
"""
    (args.output / "annotation_guide.md").write_text(guide, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "queries": len(rankings), "candidate_pairs": audit["candidate_pairs"],
        "dev": len(dev), "holdout": len(holdout), "queries_at_chunk_cap": audit["queries_at_chunk_cap"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
