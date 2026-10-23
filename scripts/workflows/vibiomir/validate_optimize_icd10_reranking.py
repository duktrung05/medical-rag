"""Validate and optimize only ICD-10 candidate reranking on the frozen split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflows.vibiomir.run_icd10_retrieval_experiment import (
    CUTOFFS, _eligible_passages, _hit_rows, _query_metrics, _relevance, _summary,
    sha256_file,
)
from src.medrag.retrieval.core import BM25
from src.medrag.retrieval.icd10 import ICD10QueryMapper, rerank_candidates_cached


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trusted(matches, max_query_codes: int) -> bool:
    return bool(matches) and len(matches) <= max_query_codes and all(
        match.method == "verified_dictionary" and match.confidence >= 1.0 for match in matches
    )


def _cached_ranking(candidates, matches, feature_cache, config):
    if not _trusted(matches, config["max_query_codes"]):
        return [{**row} for row in candidates]
    return rerank_candidates_cached(
        candidates, query_codes={match.code for match in matches},
        passage_codes=feature_cache, concept_weight=config["concept_weight"],
        rescore_depth=config["rescore_depth"],
    )


def _bootstrap(per_query: list[dict], *, subset: str, samples: int = 10_000) -> list[dict]:
    selected = [row for row in per_query if row["split"] == "evaluation" and row["chunk_evaluable"]]
    if subset == "mapped":
        selected = [row for row in selected if row["mapping_status"] == "MAPPED"]
    output = []
    for field in ("delta_recall_at_10", "delta_mrr_at_100"):
        values = [float(row[field]) for row in selected]
        rng = random.Random(f"20261016:{subset}:{field}")
        estimates = sorted(
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(samples)
        )
        output.append({
            "subset": subset, "metric": field.removeprefix("delta_"),
            "queries": len(values), "mean_delta": statistics.fmean(values),
            "ci95_low": estimates[int(.025 * samples)],
            "ci95_high": estimates[math.ceil(.975 * samples) - 1],
            "samples": samples,
        })
    return output


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    taxonomy = pq.read_table(args.taxonomy).to_pylist()
    alias_config = json.loads(args.aliases.read_text(encoding="utf-8"))
    mapper = ICD10QueryMapper(taxonomy, alias_config["aliases"])
    mappings = {int(row["query_id"]): row for row in pq.read_table(args.query_mapping).to_pylist()}
    qrels = pq.read_table(args.qrels).to_pylist()
    for query in qrels:
        query["split"] = mappings[int(query["query_id"])]["split"]
    if Counter(row["split"] for row in qrels) != {"evaluation": 72, "tuning": 35}:
        raise ValueError("Frozen split differs from the accepted 72/35 partition")
    passages = _eligible_passages(args.passages, args.staging)
    passages_by_id = {str(row["passage_id"]): row for row in passages}
    chunk_rel, doc_rel = _relevance(qrels, passages)
    index = BM25([
        (int(row["doc_id"]), row["passage_id"], row["passage_text"]) for row in passages
    ], "icd10_final_bm25")

    feature_started = time.perf_counter()
    feature_rows = []
    feature_cache = {}
    for row in passages:
        codes = frozenset(mapper.codes_in_text(row["passage_text"], row["language"]))
        feature_cache[str(row["passage_id"])] = codes
        feature_rows.append({
            "passage_id": str(row["passage_id"]), "language": row["language"],
            "icd_codes": sorted(codes),
        })
    feature_ms = (time.perf_counter() - feature_started) * 1000
    pq.write_table(
        pa.Table.from_pylist(feature_rows), args.output / "icd10_passage_feature_cache.parquet",
        compression="zstd",
    )

    # Baseline rankings are computed once for tuning. The final latency measurement below
    # executes both systems independently.
    baseline_cache = {}
    match_cache = {}
    for query in qrels:
        qid = int(query["query_id"])
        baseline_cache[qid] = _hit_rows(index.search(query["query"], 100), passages_by_id)
        match_cache[qid] = mapper.map_query(query["query"], query["query_language"])

    grid = []
    for weight in (0.02, 0.05, 0.10, 0.20):
        for depth in (10, 20, 50, 100):
            for max_codes in (1, 2, 3):
                config = {"concept_weight": weight, "rescore_depth": depth, "max_query_codes": max_codes}
                rows = []
                for query in qrels:
                    if query["split"] != "tuning":
                        continue
                    qid = int(query["query_id"])
                    ranking = _cached_ranking(baseline_cache[qid], match_cache[qid], feature_cache, config)
                    rows.append(_query_metrics(query, "C_OPT", ranking, 0.0, chunk_rel, doc_rel))
                summary = _summary(rows, "C_OPT", "tuning")
                grid.append({**config, **summary})
    best = max(grid, key=lambda row: (
        float(row["recall_at_10"]), float(row["mrr_at_100"]),
        -int(row["rescore_depth"]), -int(row["max_query_codes"]), -float(row["concept_weight"]),
    ))
    config = {key: best[key] for key in ("concept_weight", "rescore_depth", "max_query_codes")}

    metric_rows = []
    comparison_rows = []
    candidate_sets_preserved = True
    for query in qrels:
        qid = int(query["query_id"])
        started = time.perf_counter()
        baseline = _hit_rows(index.search(query["query"], 100), passages_by_id)
        baseline_latency = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        matches = mapper.map_query(query["query"], query["query_language"])
        optimized_base = _hit_rows(index.search(query["query"], 100), passages_by_id)
        optimized = _cached_ranking(optimized_base, matches, feature_cache, config)
        optimized_latency = (time.perf_counter() - started) * 1000
        candidate_sets_preserved &= {
            row["passage_id"] for row in baseline
        } == {row["passage_id"] for row in optimized}
        a = _query_metrics(query, "A", baseline, baseline_latency, chunk_rel, doc_rel)
        c = _query_metrics(query, "C_OPT", optimized, optimized_latency, chunk_rel, doc_rel)
        metric_rows.extend((a, c))
        mapping = mappings[qid]
        comparison_rows.append({
            "query_id": qid, "split": query["split"], "language": query["query_language"],
            "mapping_status": mapping["mapping_status"], "icd_codes": "|".join(mapping["icd_codes"]),
            "mapping_trusted": _trusted(matches, int(config["max_query_codes"])),
            "baseline_first_chunk_rank": a["first_relevant_chunk_rank"],
            "optimized_first_chunk_rank": c["first_relevant_chunk_rank"],
            "baseline_mrr_at_100": a["mrr_at_100"], "optimized_mrr_at_100": c["mrr_at_100"],
            "delta_mrr_at_100": c["mrr_at_100"] - a["mrr_at_100"],
            "baseline_recall_at_10": a["chunk_recall_at_10"],
            "optimized_recall_at_10": c["chunk_recall_at_10"],
            "delta_recall_at_10": c["chunk_recall_at_10"] - a["chunk_recall_at_10"],
            "baseline_chunk_f2_at_10": a["chunk_f2_at_10"],
            "optimized_chunk_f2_at_10": c["chunk_f2_at_10"],
            "baseline_doc_f2_at_10": a["doc_f2_at_10"],
            "optimized_doc_f2_at_10": c["doc_f2_at_10"],
            "baseline_latency_ms": baseline_latency, "optimized_latency_ms": optimized_latency,
            "chunk_evaluable": a["chunk_evaluable"],
        })
    if not candidate_sets_preserved:
        raise ValueError("Optimized reranker changed the top-100 candidate set")

    _write_csv(args.output / "icd10_reranking_per_query_comparison.csv", comparison_rows)
    summaries = [
        _summary(metric_rows, variant, split)
        for split in ("tuning", "evaluation", "all") for variant in ("A", "C_OPT")
    ]
    _write_csv(args.output / "icd10_reranking_metrics.csv", summaries)
    eval_summary = {row["variant"]: row for row in summaries if row["split"] == "evaluation"}
    bootstrap = _bootstrap(comparison_rows, subset="all") + _bootstrap(comparison_rows, subset="mapped")

    groups = []
    for status in ("MAPPED", "UNKNOWN"):
        for language in ("vi", "en", "zh"):
            rows = [row for row in comparison_rows if row["split"] == "evaluation"
                    and row["mapping_status"] == status and row["language"] == language]
            if not rows:
                continue
            groups.append({
                "mapping_status": status, "language": language, "queries": len(rows),
                "improved": sum(float(row["delta_mrr_at_100"]) > 0 for row in rows),
                "degraded": sum(float(row["delta_mrr_at_100"]) < 0 for row in rows),
                "unchanged": sum(float(row["delta_mrr_at_100"]) == 0 for row in rows),
                "mean_mrr_delta": statistics.fmean(float(row["delta_mrr_at_100"]) for row in rows),
                "mean_recall10_delta": statistics.fmean(float(row["delta_recall_at_10"]) for row in rows),
            })
    _write_csv(args.output / "icd10_reranking_group_analysis.csv", groups)

    eval_comparisons = [row for row in comparison_rows if row["split"] == "evaluation"]
    evaluable = [row for row in eval_comparisons if row["chunk_evaluable"]]
    baseline_100_coverage = statistics.fmean(
        float(row["baseline_first_chunk_rank"] not in (None, "") and int(row["baseline_first_chunk_rank"]) <= 100)
        for row in evaluable
    )
    rescore_opportunity = statistics.fmean(
        float(row["baseline_first_chunk_rank"] not in (None, "") and int(row["baseline_first_chunk_rank"]) <= int(config["rescore_depth"]))
        for row in evaluable
    )
    all_ci = {row["metric"]: row for row in bootstrap if row["subset"] == "all"}
    quality_reliable = (
        eval_summary["C_OPT"]["recall_at_10"] >= eval_summary["A"]["recall_at_10"]
        and eval_summary["C_OPT"]["mrr_at_100"] >= eval_summary["A"]["mrr_at_100"]
        and any(row["ci95_low"] > 0 for row in all_ci.values())
    )
    latency_ratio = eval_summary["C_OPT"]["mean_latency_ms"] / eval_summary["A"]["mean_latency_ms"]
    decision = "GO" if quality_reliable and latency_ratio <= 1.5 else "NO-GO"
    config_artifact = {
        "format": "vibiomir-icd10-reranking-config-v1", "enabled": decision == "GO",
        "decision": decision, "activation": "verified_dictionary_mapping_only",
        **config, "candidate_pool": 100, "candidate_set_preserved": candidate_sets_preserved,
        "selection_split": "tuning", "selection_queries": 35,
        "evaluation_split": "evaluation", "evaluation_queries": 72,
        "taxonomy_sha256": sha256_file(args.taxonomy), "aliases_sha256": sha256_file(args.aliases),
        "fallback": "unchanged_baseline",
    }
    (args.output / "icd10_reranking_best_config.json").write_text(
        json.dumps(config_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "decision": decision, "quality_reliable": quality_reliable,
        "latency_ratio": latency_ratio, "feature_precompute_ms": feature_ms,
        "feature_cache_rows": len(feature_rows), "candidate_set_preserved": candidate_sets_preserved,
        "baseline_top100_known_positive_coverage": baseline_100_coverage,
        "rescore_depth_known_positive_opportunity": rescore_opportunity,
        "config": config, "metrics": eval_summary, "bootstrap": bootstrap,
        "groups": groups,
    }
    (args.output / "validation_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    a, c = eval_summary["A"], eval_summary["C_OPT"]
    report = f"""# ICD-10 Reranking Final Evaluation

Date: 2026-10-16

## Decision: {decision}

This conclusion uses only the **72 frozen evaluation queries**. The 35 tuning queries
were used only to select `{json.dumps(config, sort_keys=True)}`.

## Protocol

- Same 9,277 eligible Strategy B passages and top-100 BM25 candidate budget.
- Baseline and optimized reranker were executed independently per query.
- Passage ICD features were precomputed once for this pilot ({feature_ms:.2f} ms) and cached by `passage_id`.
- Reranking activates only for verified dictionary mappings with at most {config['max_query_codes']} code(s); otherwise it returns the baseline unchanged.
- The tuning grid tested weights 0.02/0.05/0.10/0.20, rescore depths 10/20/50/100, and maximum mapped-code counts 1/2/3. Shallower rescoring is promoted only when tuning quality supports it.
- Only the first {config['rescore_depth']} candidates are rescored, but all top-100 candidates are retained. Candidate-set equality: `{candidate_sets_preserved}`.
- Baseline top-100 known-positive coverage: {baseline_100_coverage:.4f}; known-positive opportunity within rescore depth: {rescore_opportunity:.4f}.

## Evaluation metrics

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@100 | Chunk F2@10* | Doc F2@10* | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | {a['recall_at_1']:.4f} | {a['recall_at_5']:.4f} | {a['recall_at_10']:.4f} | {a['recall_at_20']:.4f} | {a['mrr_at_100']:.4f} | {a['chunk_f2_at_10']:.4f} | {a['doc_f2_at_10']:.4f} | {a['mean_latency_ms']:.2f} | {a['p95_latency_ms']:.2f} |
| Cached conditional C | {c['recall_at_1']:.4f} | {c['recall_at_5']:.4f} | {c['recall_at_10']:.4f} | {c['recall_at_20']:.4f} | {c['mrr_at_100']:.4f} | {c['chunk_f2_at_10']:.4f} | {c['doc_f2_at_10']:.4f} | {c['mean_latency_ms']:.2f} | {c['p95_latency_ms']:.2f} |

Latency ratio: **{latency_ratio:.2f}x**.

## Statistical validation

"""
    for row in bootstrap:
        report += f"- {row['subset']} / {row['metric']}: delta {row['mean_delta']:+.4f}, 95% CI [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}], n={row['queries']}.\n"
    report += "\n## MAPPED / UNKNOWN and language analysis\n\n"
    report += "| Status | Language | Queries | Improved | Degraded | Unchanged | Mean MRR delta |\n|---|---|---:|---:|---:|---:|---:|\n"
    for row in groups:
        report += f"| {row['mapping_status']} | {row['language']} | {row['queries']} | {row['improved']} | {row['degraded']} | {row['unchanged']} | {row['mean_mrr_delta']:+.4f} |\n"
    report += f"""

UNKNOWN queries deliberately use the unchanged baseline. English and Chinese subgroup
sizes are too small for independent claims.

## F2 protocol

For each query and cutoff 10, precision is `known relevant retrieved / retrieved`, recall
is `known relevant retrieved / known relevant`, and `F2 = 5PR / (4P + R)`. Chunk F2 is
computed only for the 70 evaluation queries whose exact evidence span is fully contained
by at least one Strategy B passage; Doc F2 uses all 72 evaluation queries.

The labels are valid manually evidenced positives, but they are not exhaustive judgments.
Unjudged results are treated as non-relevant, so both F2 values are **known-positive proxy
metrics**, not official ViBioMIR Chunk/Doc F2.

## Conclusion

The optimized implementation removes repeated passage dictionary matching and avoids ICD
work for UNKNOWN/untrusted queries. The release decision nevertheless requires both a
statistically reliable quality gain and latency no worse than 1.5x baseline. That rule was
{'satisfied' if decision == 'GO' else 'not satisfied'}, so the saved config is
`enabled: {str(decision == 'GO').lower()}` and the production fallback remains baseline.

No corpus audit, full Phase 3 execution, NER, Knowledge Graph, full-corpus annotation, or
primary-index modification was performed.
"""
    (args.output / "ICD10_RERANKING_FINAL_EVALUATION.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--query-mapping", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
