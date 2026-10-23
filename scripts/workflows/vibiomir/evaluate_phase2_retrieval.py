"""Compare A/B/C passage retrieval on manually evidenced pilot queries."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflows.vibiomir.review_phase2_pilot import _load_documents
from src.medrag.retrieval.core import BM25


CUTOFFS = (1, 5, 10, 20)


def _resolve_qrels(config_path: Path, query_path: Path, candidate: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    query_rows = pq.read_table(query_path).to_pylist()
    query_map = {int(row["id"]): row["query"] for row in query_rows}
    labels = config.get("queries", [])
    documents = _load_documents(candidate, {int(row["doc_id"]) for row in labels})
    resolved = []
    seen = set()
    for row in labels:
        query_id, doc_id = int(row["query_id"]), int(row["doc_id"])
        query = row.get("query") or query_map.get(query_id)
        if query_id in seen or not query:
            raise ValueError(f"Duplicate query_id or missing query text: {query_id}")
        seen.add(query_id)
        text = documents[doc_id]["normalized_text"]
        evidence = row["evidence_text"]
        start = text.find(evidence)
        if start < 0 or text.find(evidence, start + 1) >= 0:
            raise ValueError(f"Evidence must occur exactly once: query_id={query_id}")
        resolved.append({
            **row, "query": query, "evidence_start": start,
            "evidence_end": start + len(evidence),
        })
    return resolved


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return float(ordered[index])


def _passage_contexts(passages: list[dict], adjacent: bool) -> dict[str, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for passage in passages:
        grouped[int(passage["doc_id"])].append(passage)
    contexts = {}
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["passage_index"]))
        for index, row in enumerate(rows):
            members = rows[max(0, index - 1):min(len(rows), index + 2)] if adjacent else [row]
            start = min(int(member["char_start"]) for member in members)
            end = max(int(member["char_end"]) for member in members)
            member_chars = sum(
                int(member["char_end"]) - int(member["char_start"]) for member in members
            )
            contexts[row["passage_id"]] = {
                "context_char_start": start,
                "context_char_end": end,
                "context_chars": end - start,
                "added_context_chars": (end - start) - (
                    int(row["char_end"]) - int(row["char_start"])
                ),
                "redundant_overlap_chars": max(0, member_chars - (end - start)),
                "context_token_sum": sum(int(member["token_count"]) for member in members),
                "adjacent_passage_ids": [member["passage_id"] for member in members],
            }
    return contexts


def _evaluate_strategy(passages: list[dict], qrels: list[dict], strategy: str,
                       *, adjacent: bool = False) -> tuple[dict, list[dict], list[dict]]:
    index = BM25([
        (int(row["doc_id"]), row["passage_id"], row["passage_text"])
        for row in passages
    ], f"phase2_{strategy}_bm25")
    by_id = {row["passage_id"]: row for row in passages}
    contexts = _passage_contexts(passages, adjacent)
    relevant = defaultdict(set)
    for label in qrels:
        for passage in passages:
            context = contexts[passage["passage_id"]]
            if (
                int(passage["doc_id"]) == int(label["doc_id"])
                and context["context_char_start"] <= int(label["evidence_start"])
                and context["context_char_end"] >= int(label["evidence_end"])
            ):
                relevant[int(label["query_id"])].add(passage["passage_id"])
    missing = [label["query_id"] for label in qrels if not relevant[int(label["query_id"])] ]
    rankings = []
    query_metrics = []
    reciprocal_ranks = []
    hits_at = {cutoff: 0 for cutoff in CUTOFFS}
    latencies = []
    top10_contexts = []
    for label in qrels:
        query_id = int(label["query_id"])
        started = time.perf_counter()
        hits = index.search(label["query"], 100)
        # Materializing the expansion is intentionally inside the timed region.
        expanded = [(hit, contexts[hit.passage_id]) for hit in hits]
        latencies.append((time.perf_counter() - started) * 1_000)
        first = next((hit.rank for hit in hits if hit.passage_id in relevant[query_id]), None)
        reciprocal_rank = 1 / first if first else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        for cutoff in CUTOFFS:
            hits_at[cutoff] += bool(first and first <= cutoff)
        query_metrics.append({
            "strategy_id": strategy,
            "query_id": query_id,
            "first_relevant_rank": first,
            "reciprocal_rank_at_100": reciprocal_rank,
            **{f"hit_at_{cutoff}": bool(first and first <= cutoff) for cutoff in CUTOFFS},
        })
        top10_contexts.extend(context for hit, context in expanded if hit.rank <= 10)
        for hit, context in expanded:
            passage = by_id[hit.passage_id]
            rankings.append({
                "strategy_id": strategy, "query_id": query_id, "query": label["query"],
                "rank": hit.rank, "score": hit.score, "passage_id": hit.passage_id,
                "doc_id": int(hit.official_id), "char_start": int(passage["char_start"]),
                "char_end": int(passage["char_end"]),
                "is_relevant": hit.passage_id in relevant[query_id],
                **context,
                "passage_text": passage["passage_text"],
            })
    count = len(qrels)
    base_chars = [
        int(by_id[hit["passage_id"]]["char_end"]) - int(by_id[hit["passage_id"]]["char_start"])
        for hit in rankings if int(hit["rank"]) <= 10
    ]
    context_chars = [row["context_chars"] for row in top10_contexts]
    metrics = {
        "strategy_id": strategy,
        "adjacent_expansion": adjacent,
        "passages_indexed": len(passages),
        "queries": count,
        "queries_with_containing_passage": count - len(missing),
        "missing_evidence_query_ids": missing,
        **{f"recall_at_{cutoff}": hits_at[cutoff] / count for cutoff in CUTOFFS},
        "mrr_at_100": sum(reciprocal_ranks) / count,
        "zero_hit_at_10": count - hits_at[10],
        "mean_query_latency_ms": statistics.fmean(latencies),
        "p95_query_latency_ms": _percentile(latencies, 0.95),
        "mean_context_chars_at_10": statistics.fmean(context_chars),
        "mean_added_context_chars_at_10": statistics.fmean(
            row["added_context_chars"] for row in top10_contexts
        ),
        "p95_added_context_chars_at_10": _percentile(
            [row["added_context_chars"] for row in top10_contexts], 0.95
        ),
        "mean_redundant_overlap_chars_at_10": statistics.fmean(
            row["redundant_overlap_chars"] for row in top10_contexts
        ),
        "context_char_multiplier_at_10": (
            statistics.fmean(context_chars) / statistics.fmean(base_chars)
        ),
    }
    return metrics, rankings, query_metrics


def _paired_bootstrap(query_metrics: list[dict], comparisons: list[tuple[str, str]],
                      *, samples: int = 10_000, seed: int = 20261015) -> list[dict]:
    by_strategy = {
        strategy: {int(row["query_id"]): row for row in query_metrics if row["strategy_id"] == strategy}
        for strategy in {row["strategy_id"] for row in query_metrics}
    }
    rng = random.Random(seed)
    output = []
    for left, right in comparisons:
        ids = sorted(set(by_strategy[left]) & set(by_strategy[right]))
        for field in ("hit_at_10", "reciprocal_rank_at_100"):
            paired = [
                float(by_strategy[left][query_id][field]) - float(by_strategy[right][query_id][field])
                for query_id in ids
            ]
            estimates = [
                statistics.fmean(paired[rng.randrange(len(paired))] for _ in paired)
                for _ in range(samples)
            ]
            output.append({
                "left": left,
                "right": right,
                "metric": "recall_at_10" if field == "hit_at_10" else "mrr_at_100",
                "queries": len(ids),
                "mean_difference": statistics.fmean(paired),
                "ci95_low": _percentile(estimates, 0.025),
                "ci95_high": _percentile(estimates, 0.975),
                "bootstrap_samples": samples,
                "seed": seed,
            })
    return output


def evaluate(pilot: Path, candidate: Path, query_path: Path, qrels_path: Path,
             staging_ledger: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    qrels = _resolve_qrels(qrels_path, query_path, candidate)
    excluded = set(map(int, pq.read_table(staging_ledger)["doc_id"].to_pylist()))
    qrel_docs = {int(row["doc_id"]) for row in qrels}
    if qrel_docs & excluded:
        raise ValueError(f"Qrel documents are excluded by staging: {sorted(qrel_docs & excluded)}")
    all_metrics, all_rankings, all_query_metrics = [], [], []
    for strategy in "ABC":
        passages = [
            row for row in pq.read_table(
                pilot / f"pilot/strategy_{strategy}/part-00000.parquet"
            ).to_pylist()
            if int(row["doc_id"]) not in excluded
        ]
        evaluations = [(strategy, False)]
        if strategy == "B":
            evaluations.append(("B_adjacent", True))
        for label, adjacent in evaluations:
            metrics, rankings, query_metrics = _evaluate_strategy(
                passages, qrels, label, adjacent=adjacent
            )
            all_metrics.append(metrics)
            all_rankings.extend(rankings)
            all_query_metrics.extend(query_metrics)
    bootstrap = _paired_bootstrap(
        all_query_metrics,
        [("B_adjacent", "B"), ("B_adjacent", "C"), ("C", "B")],
    )
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(qrels), output / "resolved_qrels.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(all_rankings), output / "rankings_top100.parquet", compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(all_query_metrics), output / "per_query_metrics.parquet", compression="zstd"
    )
    result = {
        "status": "complete",
        "index_scope": "independent_in_memory_bm25",
        "primary_index_modified": False,
        "query_labels": len(qrels),
        "staging_excluded_documents": len(excluded),
        "relevance_rule": "passage_fully_contains_manually_labeled_evidence_span",
        "metrics": all_metrics,
        "paired_bootstrap_95ci": bootstrap,
    }
    (output / "retrieval_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--staging-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(
            args.pilot, args.candidate, args.queries, args.qrels,
            args.staging_ledger, args.output,
        )
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Phase 2 retrieval evaluation failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
