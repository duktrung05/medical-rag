"""Benchmark candidate fusion variants from cached BM25 and dense rankings."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_vimed_validation import read_rows
from scripts.run_metadata import write_run_manifest
from src.retrieval.fusion import reciprocal_rank_fusion


DEPTHS = (50, 100, 150, 200)


def weighted_rrf(sparse, dense, *, k, sparse_weight, top_k):
    scores = defaultdict(float)
    for weight, ranking in ((sparse_weight, sparse), (1.0 - sparse_weight, dense)):
        for rank, (cid, _) in enumerate(ranking, 1):
            scores[cid] += weight / (k + rank)
    return sorted(((cid, score) for cid, score in scores.items() if score > 0),
                  key=lambda pair: (-pair[1], pair[0]))[:top_k]


def ordered_union(sparse, dense):
    ranks = {}
    for ranking in (sparse, dense):
        for rank, (cid, _) in enumerate(ranking, 1):
            ranks.setdefault(cid, []).append(rank)
    return [(cid, -float(min(source_ranks))) for cid, source_ranks in sorted(
        ranks.items(), key=lambda item: (min(item[1]), sum(item[1]), item[0]))]


def candidate_metrics(samples, rankings, depth):
    total_recall = 0.0
    hit_queries = 0
    lost_positives = 0
    for sample, ranking in zip(samples, rankings, strict=True):
        positives = set(sample["relevant_chunks"])
        found = {cid for cid, _ in ranking[:depth]} & positives
        total_recall += len(found) / len(positives)
        hit_queries += bool(found)
        lost_positives += len(positives - found)
    result = {"candidate_recall": total_recall / len(samples),
            "query_hit_rate": hit_queries / len(samples),
            "lost_positive_count": lost_positives,
            "mean_returned_candidates": sum(min(depth, len(row)) for row in rankings) / len(rankings)}
    for cutoff in DEPTHS:
        cutoff_recall = 0.0
        cutoff_hits = 0
        cutoff_lost = 0
        for sample, ranking in zip(samples, rankings, strict=True):
            positives = set(sample["relevant_chunks"])
            found = {cid for cid, _ in ranking[:cutoff]} & positives
            cutoff_recall += len(found) / len(positives)
            cutoff_hits += bool(found)
            cutoff_lost += len(positives - found)
        result[f"candidate_recall_at_{cutoff}"] = cutoff_recall / len(samples)
        result[f"query_hit_rate_at_{cutoff}"] = cutoff_hits / len(samples)
        result[f"lost_positive_count_at_{cutoff}"] = cutoff_lost
    return result


def signature(sparse, dense, strategy, *, k=60, sparse_weight=None, depth=100):
    if strategy == "rrf":
        return reciprocal_rank_fusion([sparse, dense], k=k, top_k=depth)
    if strategy == "weighted_rrf":
        return weighted_rrf(sparse, dense, k=k, sparse_weight=sparse_weight, top_k=depth)
    if strategy == "union":
        return ordered_union(sparse, dense)[:depth]
    raise ValueError(strategy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings-root", type=Path, default=Path("outputs/vimed_validation"))
    parser.add_argument("--data", type=Path, default=Path("data/vimed/validation/samples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_ablation/run_20260930"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new --output: {args.output}")
    samples = read_rows(args.data)
    stage_rows = {stage: read_rows(args.rankings_root / stage / "rankings.jsonl") for stage in ("bm25", "dense", "hybrid")}
    if any(len(rows) != len(samples) for rows in stage_rows.values()):
        raise ValueError("Rankings must contain every validation query")
    candidates_by_config = {}
    config_meta = {}
    for i, sample in enumerate(samples):
        if any(stage_rows[s][i]["id"] != sample["id"] for s in stage_rows):
            raise ValueError(f"Ranking order mismatch for {sample['id']}")
    for k in (10, 20, 40, 60, 80, 100):
        for depth in DEPTHS:
            name = f"rrf_k{k}_top{depth}"
            candidates_by_config[name] = [signature(stage_rows["bm25"][i]["results"], stage_rows["dense"][i]["results"],
                                                    "rrf", k=k, depth=depth) for i in range(len(samples))]
            config_meta[name] = {"strategy": "rrf", "rrf_k": k, "depth": depth}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        for depth in DEPTHS:
            name = f"weighted_rrf_bm{weight:.2f}_de{depth}"
            candidates_by_config[name] = [signature(stage_rows["bm25"][i]["results"], stage_rows["dense"][i]["results"],
                                                    "weighted_rrf", k=60, sparse_weight=weight, depth=depth)
                                          for i in range(len(samples))]
            config_meta[name] = {"strategy": "weighted_rrf", "rrf_k": 60,
                                 "bm25_weight": weight, "dense_weight": 1.0-weight, "depth": depth}
    for depth in DEPTHS:
        name = f"union_min_rank_top{depth}"
        candidates_by_config[name] = [signature(stage_rows["bm25"][i]["results"], stage_rows["dense"][i]["results"],
                                                "union", depth=depth) for i in range(len(samples))]
        config_meta[name] = {"strategy": "union_min_rank_then_rank_sum", "depth": depth}

    current = [row["results"] for row in stage_rows["hybrid"]]
    replay = candidates_by_config["rrf_k60_top100"]
    if any([cid for cid, _ in left] != [cid for cid, _ in right] or
           any(abs(a[1]-b[1]) > 1e-12 for a, b in zip(left, right, strict=True))
           for left, right in zip(current, replay, strict=True)):
        raise ValueError("Cached current RRF k=60 top100 did not replay exactly")

    result_rows = []
    for name, rankings in candidates_by_config.items():
        depth = config_meta[name]["depth"]
        result_rows.append({"config": name, **config_meta[name], **candidate_metrics(samples, rankings, depth)})
    grouped_best = {}
    for depth in DEPTHS:
        candidates = [row for row in result_rows if row["depth"] == depth]
        grouped_best[str(depth)] = max(candidates, key=lambda row: (
            row["candidate_recall"], row["query_hit_rate"], -row["mean_returned_candidates"], row["config"]))
    current_metrics = candidate_metrics(samples, current, 100)
    best_at_100 = candidates_by_config[grouped_best["100"]["config"]]
    current_hits, best_hits = [], []
    for sample, cur, best in zip(samples, current, best_at_100, strict=True):
        gold = set(sample["relevant_chunks"])
        cur_hit = bool(gold & {cid for cid, _ in cur[:100]})
        best_hit = bool(gold & {cid for cid, _ in best[:100]})
        if best_hit and not cur_hit:
            current_hits.append({"id": sample["id"], "query": sample["query"], "topic": sample["topic"], "change": "recovered"})
        elif cur_hit and not best_hit:
            best_hits.append({"id": sample["id"], "query": sample["query"], "topic": sample["topic"], "change": "regressed"})

    args.output.mkdir(parents=True)
    write_run_manifest(args.output, task="offline_fusion_ablation", root=Path.cwd(), inputs={
        "bm25_rankings": args.rankings_root / "bm25/rankings.jsonl",
        "dense_rankings": args.rankings_root / "dense/rankings.jsonl",
        "hybrid_rankings": args.rankings_root / "hybrid/rankings.jsonl",
        "samples": args.data,
    }, parameters={"rrf_k": [10, 20, 40, 60, 80, 100], "depths": DEPTHS,
                   "weighted_rrf_bm25_weights": [0, .25, .5, .75, 1],
                   "cache_depth_limit": 100, "reranker_run": False})
    with (args.output / "fusion_results.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = list(dict.fromkeys(key for row in result_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(result_rows)
    best_report = {"current_rrf_k60_top100": current_metrics, "best_by_depth": grouped_best,
                   "best_top100_config": grouped_best["100"]["config"],
                   "candidate_pool_limits": {"sparse_cache_depth": min(len(r["results"]) for r in stage_rows["bm25"]),
                                              "dense_cache_depth": min(len(r["results"]) for r in stage_rows["dense"])},
                   "reranker_not_run": True}
    (args.output / "best_fusion.json").write_text(json.dumps(best_report, indent=2) + "\n")
    (args.output / "recovered_queries.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in current_hits))
    (args.output / "regressed_queries.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in best_hits))
    lines = ["# Offline fusion ablation", "", "BM25 and dense rankings were read from top-depth caches; no model was run.",
             "Current RRF k=60 top100 is replay-checked against the saved hybrid ranking.",
             "Union ordering: best source rank, then sum of source ranks, then chunk ID. It is an ordering heuristic, not an oracle.", "",
             f"Current RRF candidate Recall@100: **{current_metrics['candidate_recall']:.4f}**; "
             f"query hit rate {current_metrics['query_hit_rate']:.4f}; lost positive labels {current_metrics['lost_positive_count']}.", "",
             "| Depth | Best configuration | Candidate Recall | Query hit rate | Lost positives | Mean candidates |",
             "|---:|---|---:|---:|---:|---:|"]
    for depth in DEPTHS:
        row = grouped_best[str(depth)]
        lines.append(f"| {depth} | {row['config']} | {row['candidate_recall']:.4f} | {row['query_hit_rate']:.4f} | "
                     f"{row['lost_positive_count']} | {row['mean_returned_candidates']:.1f} |")
    lines += ["", f"At depth 100, best candidate hit changes recover {len(current_hits)} queries and regress {len(best_hits)} versus current RRF.",
              "Cache depths cap each source at 100, so depth above 200 was not tested. Candidate recall does not predict reranked top-k or final F2.", ""]
    (args.output / "fusion_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(best_report, indent=2))


if __name__ == "__main__":
    main()
