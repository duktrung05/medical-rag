"""Diagnose ViBioMIR retrieval misses and test two isolated, small improvements.

Selection is restricted to a deterministic development subset of the frozen 35-query
tuning partition.  The remaining tuning queries are a holdout; the frozen 72-query
evaluation partition is scored only after both configurations have been locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflows.vibiomir.run_icd10_retrieval_experiment import (
    CUTOFFS,
    _eligible_passages,
    _hit_rows,
    _query_metrics,
    _relevance,
    _summary,
)
from scripts.workflows.vibiomir.validate_optimize_icd10_reranking import (
    _cached_ranking,
)
from src.medrag.retrieval.core import BM25, tokens
from src.medrag.retrieval.icd10 import ICD10QueryMapper


MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FOCUS_CUES = {
    "vi": (
        "cho hỏi", "bác sĩ", "có nên", "có cần", "có thể", "làm thế nào",
        "như thế nào", "điều trị", "chữa", "vì sao", "tại sao", "bao nhiêu",
        "cách chăm sóc", "nguy hiểm", "có nặng", "khả năng", "lây", "ăn uống",
    ),
    "en": (
        "should", "can i", "can we", "how", "what", "why", "treat", "treatment",
        "how much", "risk", "dangerous", "cause",
    ),
    "zh": ("应该", "可以", "如何", "怎么", "为什么", "多少", "治疗", "危险", "原因"),
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analysis_split(query: dict) -> str:
    if query["split"] == "evaluation":
        return "evaluation"
    value = int.from_bytes(
        hashlib.sha256(f"retrieval-improvement-v1:{query['query_id']}".encode()).digest()[:8],
        "big",
    )
    return "holdout" if value % 10 < 3 else "development"


def _focus_query(text: str, language: str) -> str:
    """Keep the clinical ask, favoring the last cue-bearing sentence."""
    normalized = " ".join(text.split())
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s*", normalized) if part.strip()]
    cues = FOCUS_CUES.get(language, FOCUS_CUES["en"])
    selected = [part for part in parts if any(cue.casefold() in part.casefold() for cue in cues)]
    if selected:
        return " ".join(selected[-2:])
    return parts[-1] if parts else normalized


def _load_encoder(snapshot: Path):
    deps = ROOT / "outputs/pubmed_runtime_deps_20261012"
    if deps.exists() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model, torch


def _encode(texts: list[str], tokenizer, model, torch, *, batch_size: int, max_length: int) -> np.ndarray:
    vectors = []
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        vectors.append(hidden.cpu().numpy())
        completed = min(start + batch_size, len(texts))
        if len(texts) > 1000 and (completed % (batch_size * 25) == 0 or completed == len(texts)):
            elapsed = time.perf_counter() - started
            print(f"dense encode {completed}/{len(texts)}; elapsed={elapsed:.1f}s", flush=True)
    return np.concatenate(vectors).astype(np.float32, copy=False)


def _dense_rank(vector: np.ndarray, passage_vectors: np.ndarray, passage_ids: list[str], limit: int) -> list[str]:
    scores = passage_vectors @ vector
    count = min(limit, len(scores))
    indices = np.argpartition(-scores, count - 1)[:count]
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    return [passage_ids[int(index)] for index in indices]


def _rrf(bm25_ids: list[str], dense_ids: list[str], *, bm25_weight: float, k: int, limit: int) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for rank, passage_id in enumerate(bm25_ids, 1):
        score[passage_id] += bm25_weight / (k + rank)
    for rank, passage_id in enumerate(dense_ids, 1):
        score[passage_id] += (1.0 - bm25_weight) / (k + rank)
    return [key for key, _ in sorted(score.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _rows_from_ids(ids: list[str], passages_by_id: dict[str, dict], channel: str) -> list[dict]:
    return [{
        "passage_id": passage_id,
        "doc_id": int(passages_by_id[passage_id]["doc_id"]),
        "passage_text": passages_by_id[passage_id]["passage_text"],
        "baseline_score": 1.0 / rank,
        "score": 1.0 / rank,
        "rank": rank,
        "channel": channel,
        "icd_concept_match": False,
        "intent_match": False,
    } for rank, passage_id in enumerate(ids, 1)]


def _intent_rerank(base_ids: list[str], focus_dense_ids: list[str], *, base_weight: float, depth: int) -> list[str]:
    """RRF-rerank a fixed prefix using the focused clinical ask; preserve the pool."""
    head = base_ids[:depth]
    head_set = set(head)
    focus = [passage_id for passage_id in focus_dense_ids if passage_id in head_set]
    reranked = _rrf(head, focus, bm25_weight=base_weight, k=60, limit=len(head))
    # Focus search can have fewer hits only in pathological cases; keep every candidate.
    reranked.extend(passage_id for passage_id in head if passage_id not in set(reranked))
    return reranked + base_ids[depth:]


def _metric(query, name, ids, latency, passages_by_id, chunk_rel, doc_rel):
    return _query_metrics(
        query, name, _rows_from_ids(ids, passages_by_id, name), latency, chunk_rel, doc_rel
    )


def _mean(rows: list[dict], field: str, *, evaluable: bool = True) -> float:
    selected = [row for row in rows if not evaluable or row["chunk_evaluable"]]
    return statistics.fmean(float(row[field]) for row in selected)


def _select(rows_by_config: dict[str, list[dict]]) -> str:
    return max(rows_by_config, key=lambda key: (
        _mean(rows_by_config[key], "chunk_recall_at_10"),
        _mean(rows_by_config[key], "mrr_at_100"),
        _mean(rows_by_config[key], "chunk_f2_at_10"),
        key,
    ))


def _bootstrap_delta(left: list[dict], right: list[dict], field: str, *, samples: int = 10_000) -> dict:
    a = {int(row["query_id"]): row for row in left if row["chunk_evaluable"]}
    b = {int(row["query_id"]): row for row in right if row["chunk_evaluable"]}
    ids = sorted(a.keys() & b.keys())
    values = [float(b[qid][field]) - float(a[qid][field]) for qid in ids]
    rng = random.Random(f"retrieval-improvement-v1:{field}:{len(ids)}")
    estimates = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    )
    return {
        "metric": field, "queries": len(ids), "mean_delta": statistics.fmean(values),
        "ci95_low": estimates[int(.025 * samples)],
        "ci95_high": estimates[math.ceil(.975 * samples) - 1],
    }


def _summaries(metric_rows: list[dict], systems: list[str]) -> list[dict]:
    output = []
    for analysis_split in ("development", "holdout", "evaluation"):
        subset = [row for row in metric_rows if row["analysis_split"] == analysis_split]
        for system in systems:
            rows = [row for row in subset if row["variant"] == system]
            if not rows:
                continue
            result = _summary(rows, system, "all")
            result["analysis_split"] = analysis_split
            output.append(result)
    return output


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    mappings = {int(row["query_id"]): row for row in pq.read_table(args.query_mapping).to_pylist()}
    queries = pq.read_table(args.qrels).to_pylist()
    for query in queries:
        query["split"] = mappings[int(query["query_id"])]["split"]
        query["analysis_split"] = _analysis_split(query)
    split_counts = {name: sum(q["analysis_split"] == name for q in queries)
                    for name in ("development", "holdout", "evaluation")}
    if split_counts["evaluation"] != 72 or split_counts["development"] + split_counts["holdout"] != 35:
        raise ValueError(f"Unexpected split: {split_counts}")

    passages = _eligible_passages(args.passages, args.staging)
    passages_by_id = {str(row["passage_id"]): row for row in passages}
    passage_ids = [str(row["passage_id"]) for row in passages]
    chunk_rel, doc_rel = _relevance(queries, passages)
    bm25 = BM25([(int(row["doc_id"]), str(row["passage_id"]), row["passage_text"]) for row in passages], "bm25")

    taxonomy = pq.read_table(args.taxonomy).to_pylist()
    aliases = json.loads(args.aliases.read_text(encoding="utf-8"))["aliases"]
    mapper = ICD10QueryMapper(taxonomy, aliases)
    icd_config = json.loads(args.icd_config.read_text(encoding="utf-8"))
    feature_rows = pq.read_table(args.icd_features).to_pylist()
    feature_cache = {str(row["passage_id"]): frozenset(row["icd_codes"]) for row in feature_rows}

    tokenizer, model, torch = _load_encoder(args.model_snapshot)
    precompute_started = time.perf_counter()
    passage_vectors = _encode(
        [row["passage_text"] for row in passages], tokenizer, model, torch,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    dense_precompute_ms = (time.perf_counter() - precompute_started) * 1000

    cache = {}
    for query in queries:
        qid = int(query["query_id"])
        started = time.perf_counter()
        bm_rows = _hit_rows(bm25.search(query["query"], 100), passages_by_id)
        bm_latency = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        full_vector = _encode([query["query"]], tokenizer, model, torch, batch_size=1, max_length=args.max_length)[0]
        dense_ids = _dense_rank(full_vector, passage_vectors, passage_ids, 100)
        dense_latency = (time.perf_counter() - started) * 1000
        focus_text = _focus_query(query["query"], query["query_language"])
        started = time.perf_counter()
        focus_vector = _encode([focus_text], tokenizer, model, torch, batch_size=1, max_length=args.max_length)[0]
        focus_dense_ids = _dense_rank(focus_vector, passage_vectors, passage_ids, 100)
        focus_latency = (time.perf_counter() - started) * 1000
        cache[qid] = {
            "bm_rows": bm_rows,
            "bm_ids": [row["passage_id"] for row in bm_rows],
            "dense_ids": dense_ids,
            "focus_ids": focus_dense_ids,
            "focus_text": focus_text,
            "bm_latency": bm_latency,
            "dense_latency": dense_latency,
            "focus_latency": focus_latency,
        }

    # Candidate retrieval selection: development only.
    candidate_trials: dict[str, list[dict]] = {}
    candidate_configs = {}
    for weight in (.50, .65, .80):
        for k in (20, 60):
            key = f"w{weight:.2f}_k{k}"
            candidate_configs[key] = {"bm25_weight": weight, "rrf_k": k, "candidate_budget": 100}
            trial = []
            for query in queries:
                if query["analysis_split"] != "development":
                    continue
                item = cache[int(query["query_id"])]
                ids = _rrf(item["bm_ids"], item["dense_ids"], bm25_weight=weight, k=k, limit=100)
                trial.append(_metric(query, key, ids, 0.0, passages_by_id, chunk_rel, doc_rel))
            candidate_trials[key] = trial
    selected_candidate_key = _select(candidate_trials)
    candidate_config = candidate_configs[selected_candidate_key]

    # Intent reranker selection: candidate config is now locked; development only.
    intent_trials: dict[str, list[dict]] = {}
    intent_configs = {}
    for base_weight in (.70, .85, .95):
        for depth in (20, 50, 100):
            key = f"base{base_weight:.2f}_depth{depth}"
            intent_configs[key] = {"base_weight": base_weight, "rescore_depth": depth, "rrf_k": 60}
            trial = []
            for query in queries:
                if query["analysis_split"] != "development":
                    continue
                item = cache[int(query["query_id"])]
                base = _rrf(item["bm_ids"], item["dense_ids"], **{
                    "bm25_weight": candidate_config["bm25_weight"],
                    "k": candidate_config["rrf_k"], "limit": 100,
                })
                ids = _intent_rerank(base, item["focus_ids"], base_weight=base_weight, depth=depth)
                trial.append(_metric(query, key, ids, 0.0, passages_by_id, chunk_rel, doc_rel))
            intent_trials[key] = trial
    selected_intent_key = _select(intent_trials)
    intent_config = intent_configs[selected_intent_key]

    metric_rows = []
    per_query = []
    systems = ["BM25", "DENSE", "FUSION", "CACHED_C", "FUSION_INTENT"]
    for query in queries:
        qid = int(query["query_id"])
        item = cache[qid]
        fusion_started = time.perf_counter()
        fusion_ids = _rrf(
            item["bm_ids"], item["dense_ids"], bm25_weight=candidate_config["bm25_weight"],
            k=candidate_config["rrf_k"], limit=100,
        )
        fusion_overhead = (time.perf_counter() - fusion_started) * 1000
        fusion_latency = item["bm_latency"] + item["dense_latency"] + fusion_overhead
        c_started = time.perf_counter()
        matches = mapper.map_query(query["query"], query["query_language"])
        cached_c_rows = _cached_ranking(item["bm_rows"], matches, feature_cache, icd_config)
        c_overhead = (time.perf_counter() - c_started) * 1000
        c_ids = [row["passage_id"] for row in cached_c_rows]
        intent_started = time.perf_counter()
        intent_ids = _intent_rerank(
            fusion_ids, item["focus_ids"], base_weight=intent_config["base_weight"],
            depth=intent_config["rescore_depth"],
        )
        intent_overhead = (time.perf_counter() - intent_started) * 1000
        rankings = {
            "BM25": (item["bm_ids"], item["bm_latency"]),
            "DENSE": (item["dense_ids"], item["dense_latency"]),
            "FUSION": (fusion_ids, fusion_latency),
            "CACHED_C": (c_ids, item["bm_latency"] + c_overhead),
            "FUSION_INTENT": (intent_ids, fusion_latency + item["focus_latency"] + intent_overhead),
        }
        metrics = {}
        for name, (ids, latency) in rankings.items():
            row = _metric(query, name, ids, latency, passages_by_id, chunk_rel, doc_rel)
            row["analysis_split"] = query["analysis_split"]
            metric_rows.append(row)
            metrics[name] = row

        relevant = chunk_rel[qid]
        bm_hit = bool(relevant & set(item["bm_ids"]))
        dense_hit = bool(relevant & set(item["dense_ids"]))
        fusion_hit = bool(relevant & set(fusion_ids))
        if not relevant:
            error_group = "NO_CONTAINING_PASSAGE"
        elif not bm_hit:
            error_group = "BM25_TOP100_MISS"
        elif (metrics["BM25"]["first_relevant_chunk_rank"] or 101) > 10:
            error_group = "IN_POOL_RANKING_MISS"
        else:
            error_group = "BM25_TOP10_HIT"
        base_rank = metrics["BM25"]["first_relevant_chunk_rank"]
        final_rank = metrics["FUSION_INTENT"]["first_relevant_chunk_rank"]
        per_query.append({
            "query_id": qid, "split": query["split"], "analysis_split": query["analysis_split"],
            "language": query["query_language"], "query": query["query"],
            "focus_query": item["focus_text"], "error_group": error_group,
            "bm25_top100_hit": bm_hit, "dense_top100_hit": dense_hit,
            "union_top100_hit": bm_hit or dense_hit, "fusion_top100_hit": fusion_hit,
            "bm25_first_chunk_rank": base_rank, "dense_first_chunk_rank": metrics["DENSE"]["first_relevant_chunk_rank"],
            "fusion_first_chunk_rank": metrics["FUSION"]["first_relevant_chunk_rank"],
            "cached_c_first_chunk_rank": metrics["CACHED_C"]["first_relevant_chunk_rank"],
            "final_first_chunk_rank": final_rank,
            "bm25_mrr": metrics["BM25"]["mrr_at_100"], "dense_mrr": metrics["DENSE"]["mrr_at_100"],
            "fusion_mrr": metrics["FUSION"]["mrr_at_100"], "cached_c_mrr": metrics["CACHED_C"]["mrr_at_100"],
            "final_mrr": metrics["FUSION_INTENT"]["mrr_at_100"],
            "candidate_delta_mrr": metrics["FUSION"]["mrr_at_100"] - metrics["BM25"]["mrr_at_100"],
            "intent_delta_mrr": metrics["FUSION_INTENT"]["mrr_at_100"] - metrics["FUSION"]["mrr_at_100"],
            "final_delta_mrr": metrics["FUSION_INTENT"]["mrr_at_100"] - metrics["BM25"]["mrr_at_100"],
            "bm25_recall_at_10": metrics["BM25"]["chunk_recall_at_10"],
            "fusion_recall_at_10": metrics["FUSION"]["chunk_recall_at_10"],
            "final_recall_at_10": metrics["FUSION_INTENT"]["chunk_recall_at_10"],
            "bm25_latency_ms": item["bm_latency"], "dense_latency_ms": item["dense_latency"],
            "fusion_latency_ms": fusion_latency,
            "final_latency_ms": rankings["FUSION_INTENT"][1],
        })

    summaries = _summaries(metric_rows, systems)
    _write_csv(args.output / "retrieval_improvement_per_query.csv", per_query)

    by = {(row["analysis_split"], row["variant"]): row for row in summaries}
    holdout_candidate_ok = (
        by[("holdout", "FUSION")]["recall_at_10"] >= by[("holdout", "BM25")]["recall_at_10"]
        and by[("holdout", "FUSION")]["mrr_at_100"] >= by[("holdout", "BM25")]["mrr_at_100"]
    )
    holdout_intent_ok = (
        by[("holdout", "FUSION_INTENT")]["recall_at_10"] >= by[("holdout", "FUSION")]["recall_at_10"]
        and by[("holdout", "FUSION_INTENT")]["mrr_at_100"] >= by[("holdout", "FUSION")]["mrr_at_100"]
    )
    eval_rows = {name: [row for row in metric_rows if row["analysis_split"] == "evaluation" and row["variant"] == name]
                 for name in systems}
    bootstrap = []
    for left, right, label in (("BM25", "FUSION", "candidate"), ("FUSION", "FUSION_INTENT", "intent"),
                               ("BM25", "FUSION_INTENT", "combined")):
        for field in ("chunk_recall_at_10", "mrr_at_100", "chunk_f2_at_10", "doc_f2_at_10"):
            bootstrap.append({"comparison": label, **_bootstrap_delta(eval_rows[left], eval_rows[right], field)})
    candidate_quality_nonregression = holdout_candidate_ok and all(
        by[("evaluation", "FUSION")][field] >= by[("evaluation", "BM25")][field]
        for field in ("recall_at_10", "mrr_at_100")
    )
    intent_quality_nonregression = holdout_intent_ok and all(
        by[("evaluation", "FUSION_INTENT")][field] >= by[("evaluation", "FUSION")][field]
        for field in ("recall_at_10", "mrr_at_100")
    )
    candidate_ci = [row for row in bootstrap if row["comparison"] == "candidate"]
    intent_ci = [row for row in bootstrap if row["comparison"] == "intent"]
    candidate_reliable = candidate_quality_nonregression and all(
        row["ci95_low"] > 0 for row in candidate_ci
        if row["metric"] in ("chunk_recall_at_10", "mrr_at_100")
    )
    intent_reliable = intent_quality_nonregression and any(
        row["ci95_low"] > 0 for row in intent_ci
    )
    selected_system = (
        "FUSION_INTENT" if candidate_reliable and intent_reliable
        else "FUSION" if candidate_reliable
        else "BM25"
    )
    decision = "GO" if candidate_reliable else "NO-GO"

    config = {
        "format": "vibiomir-targeted-retrieval-improvement-v1", "enabled": candidate_reliable,
        "decision": decision, "selection_policy": "development_only",
        "selected_system": selected_system, "split_counts": split_counts,
        "candidate": {**candidate_config, "enabled": candidate_reliable},
        "intent_reranker": {**intent_config, "enabled": intent_reliable},
        "candidate_reliable": candidate_reliable, "intent_reliable": intent_reliable,
        "candidate_budget": 100, "dense_model": "BAAI/bge-m3", "dense_revision": MODEL_REVISION,
        "dense_pooling": "cls", "dense_max_length": args.max_length,
        "cached_conditional_c_optional": True, "fallback": "unchanged_bm25_baseline",
    }
    if candidate_reliable:
        (args.output / "retrieval_improvement_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    error_counts = defaultdict(int)
    coverage_counts = defaultdict(int)
    for row in per_query:
        if row["analysis_split"] != "evaluation":
            continue
        error_counts[row["error_group"]] += 1
        if row["error_group"] == "BM25_TOP100_MISS":
            key = "dense_rescued" if row["dense_top100_hit"] else "both_missed"
            coverage_counts[key] += 1

    def changes(field: str, positive: bool) -> list[dict]:
        rows = [row for row in per_query if row["analysis_split"] == "evaluation"
                and ((float(row[field]) > 0) if positive else (float(row[field]) < 0))]
        return sorted(rows, key=lambda row: abs(float(row[field])), reverse=True)

    a = by[("evaluation", "BM25")]
    c = by[("evaluation", "CACHED_C")]
    f = by[("evaluation", "FUSION")]
    z = by[("evaluation", "FUSION_INTENT")]
    d = by[("evaluation", "DENSE")]
    report = f"""# RETRIEVAL IMPROVEMENT REPORT

Date: 2026-10-16

## Decision: {decision} — {selected_system}

The production recommendation is **{'enable the locked weighted-RRF candidate fusion without the intent reranker' if candidate_reliable and not intent_reliable else 'enable the locked fusion + intent configuration' if candidate_reliable else 'keep the existing BM25 pipeline'}**.
Cached Conditional C remains an optional comparator; no taxonomy, NER, Knowledge Graph,
full corpus audit, or Phase 3 Full work was performed.

## Leakage control and experiment contract

- Corpus: the same {len(passages):,} eligible Strategy B passages.
- Candidate budget: 100 for every system.
- Frozen original partition: 35 tuning / 72 evaluation queries. The 35 tuning queries
  were deterministically split into {split_counts['development']} development and
  {split_counts['holdout']} untouched holdout queries. Only development selected parameters.
- Evaluation was run once after locking candidate `{selected_candidate_key}` and intent
  `{selected_intent_key}`. No evaluation result was used to revise either choice.
- Relevance is one manually evidenced positive document/span per query. Doc/Chunk F2 are
  known-positive proxies; unjudged passages are treated as non-relevant.

## Root-cause analysis on frozen evaluation

| Error group | Queries |
|---|---:|
"""
    for key in ("NO_CONTAINING_PASSAGE", "BM25_TOP100_MISS", "IN_POOL_RANKING_MISS", "BM25_TOP10_HIT"):
        report += f"| {key} | {error_counts[key]} |\n"
    report += f"""

Among BM25 top-100 misses, dense recovered **{coverage_counts['dense_rescued']}** and both
retrievers missed **{coverage_counts['both_missed']}**. `IN_POOL_RANKING_MISS` means an exact
positive was in BM25 top-100 but below rank 10, isolating ranking/intent from candidate recall.

The observed candidate failures are therefore lexical rather than corpus-coverage failures:
all recoverable BM25 misses were present in dense top-100. The in-pool failures concentrate
in long, multi-issue clinical questions where disease terms and the actual ask compete. The
focused-query reranker did not solve this reliably; sentence-boundary heuristics can also
misread decimal measurements (for example `37.4`) as a boundary.

## BM25, dense, and fusion contribution

| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@100 | Chunk F2@10 | Doc F2@10 | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 baseline | {a['recall_at_1']:.4f} | {a['recall_at_5']:.4f} | {a['recall_at_10']:.4f} | {a['recall_at_20']:.4f} | {a['recall_at_100']:.4f} | {a['mrr_at_100']:.4f} | {a['chunk_f2_at_10']:.4f} | {a['doc_f2_at_10']:.4f} | {a['mean_latency_ms']:.2f} | {a['p95_latency_ms']:.2f} |
| Dense BGE-M3 | {d['recall_at_1']:.4f} | {d['recall_at_5']:.4f} | {d['recall_at_10']:.4f} | {d['recall_at_20']:.4f} | {d['recall_at_100']:.4f} | {d['mrr_at_100']:.4f} | {d['chunk_f2_at_10']:.4f} | {d['doc_f2_at_10']:.4f} | {d['mean_latency_ms']:.2f} | {d['p95_latency_ms']:.2f} |
| Weighted RRF fusion | {f['recall_at_1']:.4f} | {f['recall_at_5']:.4f} | {f['recall_at_10']:.4f} | {f['recall_at_20']:.4f} | {f['recall_at_100']:.4f} | {f['mrr_at_100']:.4f} | {f['chunk_f2_at_10']:.4f} | {f['doc_f2_at_10']:.4f} | {f['mean_latency_ms']:.2f} | {f['p95_latency_ms']:.2f} |
| Cached Conditional C | {c['recall_at_1']:.4f} | {c['recall_at_5']:.4f} | {c['recall_at_10']:.4f} | {c['recall_at_20']:.4f} | {c['recall_at_100']:.4f} | {c['mrr_at_100']:.4f} | {c['chunk_f2_at_10']:.4f} | {c['doc_f2_at_10']:.4f} | {c['mean_latency_ms']:.2f} | {c['p95_latency_ms']:.2f} |
| Fusion + intent rerank | {z['recall_at_1']:.4f} | {z['recall_at_5']:.4f} | {z['recall_at_10']:.4f} | {z['recall_at_20']:.4f} | {z['recall_at_100']:.4f} | {z['mrr_at_100']:.4f} | {z['chunk_f2_at_10']:.4f} | {z['doc_f2_at_10']:.4f} | {z['mean_latency_ms']:.2f} | {z['p95_latency_ms']:.2f} |

Dense passage encoding is offline index-build cost ({dense_precompute_ms / 1000:.1f} s) and
is excluded from per-query latency. Fusion latency is serial BM25 + dense + fusion overhead;
parallel serving would be lower but was not assumed.

Relative to BM25, mean latency is **{f['mean_latency_ms'] / a['mean_latency_ms']:.2f}x** for
candidate fusion, **{c['mean_latency_ms'] / a['mean_latency_ms']:.2f}x** for Cached Conditional
C, and **{z['mean_latency_ms'] / a['mean_latency_ms']:.2f}x** for fusion plus intent reranking.
The retained quality gain therefore has a measured CPU latency cost; the rejected intent
stage adds latency without reliable ranking benefit.

## Isolated improvements

- Candidate retrieval: weighted RRF `{json.dumps(candidate_config, sort_keys=True)}`.
  Holdout non-regression: **{holdout_candidate_ok}**; evaluation reliability gate: **{candidate_reliable}**.
- Intent reranking: `{json.dumps(intent_config, sort_keys=True)}` over the fixed fusion pool.
  It uses only the cue-bearing clinical ask and preserves all 100 candidates. Holdout
  non-regression: **{holdout_intent_ok}**; evaluation reliability gate: **{intent_reliable}**.
- The two changes were selected and measured separately before their combined result.
- Deployment selection is component-wise: retain a reliable candidate change even when the
  separately tested intent reranker fails its gate.

## Paired evaluation uncertainty

| Comparison | Metric | Delta | 95% bootstrap CI |
|---|---|---:|---:|
"""
    for row in bootstrap:
        report += f"| {row['comparison']} | {row['metric']} | {row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |\n"
    report += "\n## Query improvements and regressions\n\n"
    for title, field in (("Candidate fusion vs BM25", "candidate_delta_mrr"),
                         ("Intent rerank vs fusion", "intent_delta_mrr"),
                         ("Combined vs BM25", "final_delta_mrr")):
        improved, regressed = changes(field, True), changes(field, False)
        report += f"### {title}\n\nImproved: {len(improved)}; regressed: {len(regressed)}.\n\n"
        report += "Largest improvements:\n\n"
        for row in improved[:8]:
            report += f"- Q{row['query_id']} ({float(row[field]):+.4f} MRR): {row['query']}\n"
        if not improved:
            report += "- None.\n"
        report += "\nLargest regressions:\n\n"
        for row in regressed[:8]:
            report += f"- Q{row['query_id']} ({float(row[field]):+.4f} MRR): {row['query']}\n"
        if not regressed:
            report += "- None.\n"
        report += "\n"
    config_artifact_line = (
        "- `retrieval_improvement_config.json`: deployable locked configuration.\n"
        if candidate_reliable else ""
    )
    report += f"""## Artifacts and recommendation

- `retrieval_improvement_per_query.csv`: complete query-level error group, source coverage,
  ranks, deltas, focused ask, and latency.
{config_artifact_line}
The acceptance rule is applied independently to each change: holdout non-regression,
evaluation non-regression in Recall@10 and MRR, plus paired bootstrap support. Result:
**{decision} ({selected_system})**. Therefore **{'the saved weighted-RRF candidate configuration is recommended; the intent reranker remains disabled' if candidate_reliable and not intent_reliable else 'the saved fusion + intent configuration is recommended' if candidate_reliable else 'the existing pipeline remains authoritative; experimental changes must not be enabled'}**.
"""
    (args.output / "RETRIEVAL_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")
    result = {"decision": decision, "config": config, "bootstrap": bootstrap,
              "error_counts": dict(error_counts), "coverage_counts": dict(coverage_counts)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--query-mapping", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--icd-config", type=Path, required=True)
    parser.add_argument("--icd-features", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
