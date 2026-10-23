"""Replay chunk allocation arms from locked P0 cached scores without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4.audit_evidence_survival import anchor_coverage, normalize, sha
from src.medrag.adapters.validation import validate_selected
from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy


QUERY_IDS = [53, 130, 380, 754, 771, 1144]
RERANK_POOL_CAP = 200
CHUNK_CAP = 60
HYBRID_WEIGHTS = {
    "A2_hybrid_5_3_2": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
    "A2_hybrid_4_3_3": {"alpha": 0.4, "beta": 0.3, "gamma": 0.3},
    "A2_hybrid_6_2_2": {"alpha": 0.6, "beta": 0.2, "gamma": 0.2},
}
QUOTA_ARM = "A3_hybrid40_dense10_coverage10"
EXPECTED_TOP60_POSITIVES = {
    (53, 756661), (130, 546076), (130, 553119), (130, 2968260),
    (754, 2713777), (771, 484565), (771, 494605),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).lower()
    return list(dict.fromkeys(re.findall(r"[^\W_]+", value, flags=re.UNICODE)))


def query_coverage(query: str, passage: str) -> float:
    terms = query_tokens(query)
    if not terms:
        return 0.0
    passage_terms = set(query_tokens(passage))
    return sum(term in passage_terms for term in terms) / len(terms)


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def reciprocal_rank_strength(rank: int | None, cap: int) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return max(0.0, (cap - rank + 1) / cap)


def build_parent_pool(qid: int, query: str, checkpoint: dict, cache_path: Path,
                      query_hash: str, max_document_rank: int) -> list[dict]:
    candidates = checkpoint["diagnostics"]["ranked_candidates"]
    by_parent = {int(row["official_id"]): row for row in candidates}
    if len(by_parent) != len(candidates):
        raise ValueError(f"Duplicate ranked parent: query={qid}")
    selected = checkpoint["selection"]
    selected_by_parent = {int(row["official_id"]): row for row in selected}
    if len(selected_by_parent) != len(selected) or len(selected) != RERANK_POOL_CAP:
        raise ValueError(f"Expected 200 distinct selected parents: query={qid}")
    pre_rerank = sorted(candidates, key=lambda row: (-float(row["score"]),
                                                     int(row["official_id"])))
    fused_parent_rank = {int(row["official_id"]): position
                         for position, row in enumerate(pre_rerank, 1)}
    final_doc_rank = {int(row["official_id"]): position
                      for position, row in enumerate(selected, 1)}
    pair_rows = checkpoint["pair_trace"]
    pair_by_parent: dict[int, list[dict]] = {}
    for pair_position, pair in enumerate(pair_rows, 1):
        parent = int(pair["official_id"])
        pair_by_parent.setdefault(parent, []).append({"position": pair_position, **pair})
    if len(pair_rows) != RERANK_POOL_CAP or len(pair_by_parent) != RERANK_POOL_CAP:
        raise ValueError(f"P0 pair trace is not 200 distinct parents: query={qid}")
    if set(pair_by_parent) != set(selected_by_parent):
        raise ValueError(f"Scored parent set differs from relevant_docs: query={qid}")

    branch = checkpoint["branch_trace"]["branch_hits"]
    dense_rank = {}
    for hit in branch["dense"]:
        parent = int(hit["official_id"])
        rank = int(hit["rank"])
        dense_rank[parent] = min(rank, dense_rank.get(parent, rank))
    bm25_rank = {}
    for name in ("document_bm25", "passage_bm25"):
        for hit in branch[name]:
            parent = int(hit["official_id"])
            rank = int(hit["rank"])
            bm25_rank[parent] = min(rank, bm25_rank.get(parent, rank))

    pair_cache = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    records = []
    try:
        for parent, pair_items in pair_by_parent.items():
            if len(pair_items) != 1:
                raise ValueError(f"Expected one parent-first score per parent: {qid}/{parent}")
            pair = pair_items[0]
            score = float(pair["score"])
            if not math.isfinite(score):
                raise ValueError(f"Invalid cached score: query={qid}, parent={parent}")
            cache_rows = pair_cache.execute(
                "SELECT score,input_sha256 FROM scores WHERE query_sha256=? AND official_id=? "
                "AND passage_id=?", (query_hash, parent, pair["passage_id"])).fetchall()
            if cache_rows != [(score, pair["input_sha256"])]:
                raise ValueError(f"P0 score cache mismatch: query={qid}, parent={parent}")
            doc = by_parent.get(parent)
            selected_doc = selected_by_parent.get(parent)
            if doc is None or selected_doc is None:
                raise ValueError(f"Scored parent missing ranked/selected trace: {qid}/{parent}")
            evidence_matches = [item for item in doc["evidence"]
                                if item["passage_id"] == pair["passage_id"]
                                and item.get("score_status") == "reranked"
                                and float(item.get("rerank_score", float("nan"))) == score]
            if len(evidence_matches) != 1:
                raise ValueError(f"P0 scored passage missing from ranked evidence: {qid}/{parent}")
            evidence = evidence_matches[0]
            if (selected_doc["document_evidence"]["passage_id"] != pair["passage_id"]
                    or float(selected_doc["document_evidence"]["rerank_score"]) != score):
                raise ValueError(f"P0 selected anchor differs from cached scored passage: {qid}/{parent}")
            records.append({
                "query_id": qid,
                "doc_id": parent,
                "final_doc_rank": final_doc_rank[parent],
                "fused_rank": fused_parent_rank[parent],
                "dense_rank": dense_rank.get(parent),
                "bm25_rank": bm25_rank.get(parent),
                "passage_id": pair["passage_id"],
                "rerank_score": score,
                "query_coverage": query_coverage(query, evidence["text"]),
                "chunk_selection_score": 0.0,
                "document_fused_score": float(doc["score"]),
                "pair_position": pair["position"],
                "pair_cache_hit": bool(pair["cache_hit"]),
                "pair_input_sha256": pair["input_sha256"],
                "passage_text": evidence["text"],
                "passage_start": evidence["start"],
                "passage_end": evidence["end"],
                "document_evidence": selected_doc["document_evidence"],
                "retrieval_ranks": dict(doc["ranks"]),
            })
    finally:
        pair_cache.close()
    records.sort(key=lambda row: row["final_doc_rank"])
    if len(records) != RERANK_POOL_CAP:
        raise ValueError(f"Pool cap mismatch: query={qid}")

    rerank_norm = minmax([row["rerank_score"] for row in records])
    fused_cap = max(fused_parent_rank.values(), default=1)
    retrieval_scores = []
    for row in records:
        fused = reciprocal_rank_strength(row["fused_rank"], fused_cap)
        dense = reciprocal_rank_strength(row["dense_rank"], 200)
        bm25 = reciprocal_rank_strength(row["bm25_rank"], 400)
        retrieval_scores.append((fused + dense + bm25) / 3.0)
    retrieval_norm = minmax(retrieval_scores)
    for row, normalized_rerank, normalized_retrieval in zip(
            records, rerank_norm, retrieval_norm, strict=True):
        row["rerank_normalized"] = normalized_rerank
        row["retrieval_signal"] = normalized_retrieval
        row["hybrid_scores"] = {
            name: weights["alpha"] * normalized_rerank
            + weights["beta"] * normalized_retrieval
            + weights["gamma"] * row["query_coverage"]
            for name, weights in HYBRID_WEIGHTS.items()
        }
    return records


def score_records(records: list[dict], arm: str) -> list[dict]:
    result = [dict(row) for row in records]
    if arm == "A0":
        score = minmax([-float(row["final_doc_rank"]) for row in result])
    elif arm == "A1":
        score = [row["rerank_normalized"] for row in result]
    elif arm in HYBRID_WEIGHTS:
        score = [row["hybrid_scores"][arm] for row in result]
    elif arm == QUOTA_ARM:
        score = [row["hybrid_scores"]["A2_hybrid_5_3_2"] for row in result]
    else:
        raise ValueError(f"Unknown allocation arm: {arm}")
    for row, value in zip(result, score, strict=True):
        row["chunk_selection_score"] = float(value)
    return result


def order_by(rows: list[dict], key: str, *, reverse: bool = True) -> list[dict]:
    if reverse:
        return sorted(rows, key=lambda row: (-float(row[key]), int(row["final_doc_rank"]),
                                             int(row["doc_id"])))
    return sorted(rows, key=lambda row: (float(row[key]) if row[key] is not None else math.inf,
                                         int(row["final_doc_rank"]), int(row["doc_id"])))


def choose_arm(records: list[dict], arm: str) -> tuple[list[dict], dict]:
    scored = score_records(records, arm)
    if arm == "A3_hybrid40_dense10_coverage10":
        chosen: list[dict] = []
        chosen_ids: set[int] = set()
        quota_counts = {"hybrid": 0, "dense": 0, "coverage": 0, "hybrid_backfill": 0}

        def take(candidates: list[dict], count: int, category: str) -> None:
            for row in candidates:
                if quota_counts[category] >= count or len(chosen) >= CHUNK_CAP:
                    break
                doc_id = int(row["doc_id"])
                if doc_id in chosen_ids:
                    continue
                chosen_ids.add(doc_id)
                chosen.append(row)
                quota_counts[category] += 1

        hybrid = order_by(scored, "chunk_selection_score")
        dense_candidates = order_by(
            [row for row in scored if row["dense_rank"] is not None], "dense_rank", reverse=False)
        coverage_candidates = order_by(scored, "query_coverage")
        take(hybrid, 40, "hybrid")
        take(dense_candidates, 10, "dense")
        take(coverage_candidates, 10, "coverage")
        take(hybrid, CHUNK_CAP, "hybrid_backfill")
        if len(chosen) != CHUNK_CAP:
            raise ValueError("A3 failed to backfill to 60 unique parents")
        return chosen, {"quota_targets": {"hybrid": 40, "dense": 10, "coverage": 10},
                       "quota_filled": quota_counts, "deduplication": "doc_id",
                       "backfill": "hybrid ranking until 60 unique parents"}
    ordered = order_by(scored, "chunk_selection_score")
    return ordered[:CHUNK_CAP], {"quota_targets": None, "quota_filled": None,
                                 "deduplication": "doc_id", "backfill": "not_needed"}


def build_export_docs(relevant_docs: list[dict], selected_chunks: list[dict], sources,
                      passages, policy: SelectionPolicy) -> list[dict]:
    by_parent = {int(row["doc_id"]): row for row in selected_chunks}
    docs = []
    for document in relevant_docs:
        doc_id = int(document["official_id"])
        selected = by_parent.get(doc_id)
        row = {**document, "evidence": []}
        if selected:
            scored = {
                "passage_id": selected["passage_id"],
                "text": selected["passage_text"],
                "start": selected["passage_start"],
                "end": selected["passage_end"],
                "score_status": "reranked",
                "rerank_score": selected["rerank_score"],
            }
            row["evidence"] = [scored]
        docs.append(row)
    contextual, _ = apply_chunk_context(docs, policy, sources, passages)
    return contextual


def positive_metrics(query_rows: list[dict], selected: list[dict], baseline: dict,
                     policy: SelectionPolicy, sources, passages) -> dict:
    qid = int(query_rows[0]["query_id"])
    by_parent = {int(row["official_id"]): row for row in selected}
    values = []
    for anchor in query_rows:
        doc_id = int(anchor["doc_id"])
        chunk_order = int(anchor["baseline_chunk_order"])
        text = normalize(baseline[qid]["relevant_chunks"][chunk_order]["chunk_text"])
        exported = by_parent.get(doc_id, {}).get("evidence", [])
        contexts = [item["text"] for item in exported]
        values.append({"query_id": qid, "doc_id": doc_id,
                       "exact_anchor": any(text in normalize(value) for value in contexts),
                       "lcs_proxy": anchor_coverage(text, contexts),
                       "parent_selected": doc_id in by_parent,
                       "chunk_selected": bool(exported)})
    count = len(values)
    return {
        "query_id": qid,
        "historical_positive_anchors": count,
        "exact_anchor_count": sum(row["exact_anchor"] for row in values),
        "exact_anchor_rate": sum(row["exact_anchor"] for row in values) / count,
        "lcs_proxy_mean": sum(row["lcs_proxy"] for row in values) / count,
        "lcs_proxy_at_least_0_4": sum(row["lcs_proxy"] >= 0.4 for row in values),
        "selected_positive_parents": sum(row["parent_selected"] for row in values),
        "positive_chunk_slots": sum(row["chunk_selected"] for row in values),
        "anchor_rows": values,
    }


def main(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    p0 = root / "outputs/v4/p0_positive_path_replay_20261021_v3"
    p0_summary = read_json(p0 / "summary.json")
    p0_manifest = read_json(p0 / "manifest.json")
    pilot = root / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
    smoke = root / "outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550"
    context_replay = root / "outputs/v4/adjacent_context_replay_d200_20261021"
    audit = root / "outputs/v4/evidence_survival_audit_20261021_v2"
    baseline_zip = root / "outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip"
    context_config = root / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml"
    config = load_config(context_config)
    pilot_summary = read_json(pilot / "summary.json")
    pilot_lock = read_json(pilot / "run_lock.json")
    smoke_lock = read_json(smoke / "smoke_lock.json")
    context_manifest = read_json(context_replay / "manifest.json")
    if p0_summary.get("status") != "P0_POSITIVE_PATH_REPLAY_PASS":
        raise ValueError("Chunk allocation requires completed P0 positive-path replay")
    for name, digest in p0_manifest["artifacts_sha256"].items():
        if sha(p0 / name) != digest:
            raise ValueError(f"P0 artifact checksum mismatch: {name}")
    if sha(smoke / "smoke_lock.json") != pilot_lock["a0_smoke_lock_sha256"]:
        raise ValueError("Baseline smoke lock differs from P0 pilot")
    if context_manifest.get("status") != "ADJACENT_CONTEXT_REPLAY_PASS":
        raise ValueError("Adjacent context replay is incomplete")
    if context_manifest.get("config_sha256") != sha(context_config):
        raise ValueError("Adjacent context artifact uses a different selection config")
    if config["retrieval"]["rerank_candidate_limit"] != RERANK_POOL_CAP:
        raise ValueError("Unexpected rerank candidate budget")
    if config["selection"]["max_chunks"] != CHUNK_CAP:
        raise ValueError("Unexpected chunk cap")
    if config["selection"]["chunks_per_doc"] != 1:
        raise ValueError("Chunk allocation requires one chunk per document")
    query_ids = [int(value) for value in pilot_summary["query_ids"]]
    if query_ids != QUERY_IDS or query_ids != pilot_lock["query_ids"]:
        raise ValueError("Pilot query scope differs from locked six-query set")
    index = root / config["index_namespace"]
    index_manifest = read_json(index / "manifest.json")
    if sha(index / "manifest.json") != pilot_lock["full_index_manifest_sha256"]:
        raise ValueError("Index manifest differs from locked P0 pilot")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)
    policy = SelectionPolicy(**config["selection"])

    migration = read_json(root / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = root / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query file fingerprint mismatch")
    import pyarrow.parquet as pq

    query_rows = {int(row["id"]): str(row["query"])
                  for row in pq.read_table(query_path, columns=["id", "query"]).to_pylist()}
    if not set(query_ids).issubset(query_rows):
        raise ValueError("Official query text missing pilot IDs")
    if sha(baseline_zip) != pilot_lock["baseline_submission_sha256"]:
        raise ValueError("Historical baseline ZIP differs from P0 lock")
    with zipfile.ZipFile(baseline_zip) as archive:
        baseline = {int(row["id"]): row
                    for row in json.loads(archive.read("predictions.json"))}
    with (audit / "historical_positive_stages.csv").open(encoding="utf-8") as stream:
        historical = [row for row in csv.DictReader(stream)
                      if int(row["query_id"]) in set(query_ids)]
    with (p0 / "positive_paths.csv").open(encoding="utf-8") as stream:
        p0_rows = list(csv.DictReader(stream))
    p0_by_key = {(int(row["query_id"]), int(row["doc_id"])): row for row in p0_rows}
    if len(historical) != 37 or len(p0_by_key) != len(historical):
        raise ValueError("P0 positive anchors differ from replay labels")

    reference_context = {
        int(row["id"]): row["selected"]
        for row in map(json.loads, (context_replay / "selected_results.jsonl").open(
            encoding="utf-8"))
    }
    arms = ["A0", "A1", *HYBRID_WEIGHTS, QUOTA_ARM]
    pool_by_query: dict[int, list[dict]] = {}
    relevant_docs_by_query: dict[int, list[dict]] = {}
    all_pool_rows: list[dict] = []
    diagnostics_by_query: dict[int, dict] = {}
    input_paths = [
        p0 / "manifest.json", p0 / "summary.json", p0 / "positive_paths.csv",
        pilot / "summary.json", pilot / "run_lock.json", pilot / "scores.sqlite",
        smoke / "smoke_lock.json",
        context_replay / "manifest.json", context_replay / "selected_results.jsonl",
        audit / "historical_positive_stages.csv", baseline_zip, context_config,
        query_path, index / "manifest.json",
    ]
    input_paths += [pilot / "queries" / f"{qid}.json" for qid in query_ids]
    input_paths += [smoke / "smoke_queries" / f"{qid}.json" for qid in query_ids]
    input_hashes_before = {str(path.relative_to(root)): sha(path) for path in input_paths}

    for qid in query_ids:
        current = read_json(pilot / "queries" / f"{qid}.json")
        a0 = read_json(smoke / "smoke_queries" / f"{qid}.json")
        query_hash = current["query_sha256"]
        if query_hash != smoke_lock["query_sha256"][str(qid)]:
            raise ValueError(f"Query hash differs from locked P0 checkpoint: {qid}")
        rows = build_parent_pool(qid, query_rows[qid], current,
                                 pilot / "scores.sqlite", query_hash, 400)
        pool_by_query[qid] = rows
        selected_parent_ids = [int(doc["official_id"]) for doc in current["selection"]]
        if len(selected_parent_ids) != 200 or len(set(selected_parent_ids)) != 200:
            raise ValueError(f"relevant_docs must remain 200 unique parents: {qid}")
        relevant_docs_by_query[qid] = current["selection"]
        p0_selected = reference_context[qid]
        a0_chunk_ids = [int(doc["official_id"]) for doc in p0_selected if doc["evidence"]]
        expected_a0 = selected_parent_ids[:CHUNK_CAP]
        if a0_chunk_ids != expected_a0:
            raise ValueError(f"A0 selected parents do not match P0 output: {qid}")
        for arm in arms:
            arm_records, allocator = choose_arm(rows, arm)
            if len(arm_records) != CHUNK_CAP or len({r["doc_id"] for r in arm_records}) != CHUNK_CAP:
                raise ValueError(f"Arm violates 60 unique chunk parents: {qid}/{arm}")
            for record in arm_records:
                if int(record["doc_id"]) not in selected_parent_ids:
                    raise ValueError(f"Arm changed relevant_docs: {qid}/{arm}")
            if arm == "A0":
                selected_a0 = {row["doc_id"] for row in arm_records}
                if selected_a0 != set(a0_chunk_ids):
                    raise ValueError(f"A0 did not replay P0 top-60: {qid}")
            diagnostics_by_query[(qid, arm)] = allocator
        all_pool_rows.extend({"query_id": qid, **row} for row in rows)

    with zipfile.ZipFile(baseline_zip) as archive:
        baseline = {int(row["id"]): row
                    for row in json.loads(archive.read("predictions.json"))}
    positives_by_query: dict[int, list[dict]] = {qid: [] for qid in query_ids}
    for row in historical:
        qid, parent = int(row["query_id"]), int(row["doc_id"])
        positives_by_query[qid].append(row)
    row_by_key = {(int(row["query_id"]), int(row["doc_id"])): row
                  for row in historical}
    if set(row_by_key) != set(p0_by_key):
        raise ValueError("P0 positive anchors do not exactly match historical labels")

    selected_by_arm_query: dict[tuple[str, int], list[dict]] = {}
    metrics_by_arm: dict[str, list[dict]] = {arm: [] for arm in arms}
    selected_chunk_rows = []
    allocation_changes = []
    displaced_rows = []
    rescued = {arm: {"rescued_exact_anchors": [], "displaced_exact_anchors": [],
                     "seven_top60_cases": []} for arm in arms}
    a0_metrics_by_q: dict[int, dict] = {}
    arm_metrics_total: dict[str, dict] = {}

    for arm in arms:
        total_exact, total_lcs, total_lcs40, total_positives = 0, 0.0, 0, 0
        total_chunks = total_query_coverage = 0
        for qid in query_ids:
            pool = pool_by_query[qid]
            chosen, allocator = choose_arm(pool, arm)
            relevant_docs = relevant_docs_by_query[qid]
            contextual_docs = build_export_docs(relevant_docs, chosen,
                                                sources, passages, policy)
            official_ids = {int(doc["official_id"]) for doc in relevant_docs}
            payload_rows = validate_selected(
                [qid], {qid: contextual_docs}, store, policy,
                result_limit=200, reranker_enabled=True, official_ids=official_ids)
            if len(payload_rows[0]["relevant_docs"]) != 200:
                raise ValueError(f"relevant_docs changed under {arm}: query={qid}")
            if len(payload_rows[0]["relevant_chunks"]) != 60:
                raise ValueError(f"chunk count mismatch under {arm}: query={qid}")
            selected_by_arm_query[(arm, qid)] = contextual_docs
            if arm == "A0":
                ref = reference_context[qid]
                if contextual_docs != ref:
                    raise ValueError(f"A0 full selected/context output differs from P0: {qid}")

            q_positives = positive_metrics(positives_by_query[qid], contextual_docs,
                                           baseline, policy, sources, passages)
            q_positives["arm"] = arm
            q_positives["selected_chunks"] = len(chosen)
            q_positives["mean_query_coverage"] = sum(
                float(row["query_coverage"]) for row in chosen) / len(chosen)
            q_positives["mean_rerank_score"] = sum(
                float(row["rerank_score"]) for row in chosen) / len(chosen)
            q_positives["allocator"] = allocator
            metrics_by_arm[arm].append(q_positives)
            total_exact += q_positives["exact_anchor_count"]
            total_lcs += sum(row["lcs_proxy"] for row in q_positives["anchor_rows"])
            total_lcs40 += q_positives["lcs_proxy_at_least_0_4"]
            total_positives += q_positives["historical_positive_anchors"]
            total_chunks += len(chosen)
            total_query_coverage += q_positives["mean_query_coverage"] * len(chosen)

            if arm == "A0":
                a0_metrics_by_q[qid] = q_positives

            chosen_ids = {int(row["doc_id"]) for row in chosen}
            a0_ids = {int(doc["official_id"]) for doc in reference_context[qid]
                      if doc["evidence"]}
            score_by_id = {int(row["doc_id"]): row for row in pool}
            for row in chosen:
                selected_chunk_rows.append({
                    "query_id": qid,
                    "arm": arm,
                    "doc_id": int(row["doc_id"]),
                    "passage_id": row["passage_id"],
                    "rerank_score": row["rerank_score"],
                    "query_coverage": row["query_coverage"],
                    "chunk_selection_score": row["chunk_selection_score"],
                    "final_doc_rank": row["final_doc_rank"],
                    "fused_rank": row["fused_rank"],
                    "dense_rank": row["dense_rank"],
                    "bm25_rank": row["bm25_rank"],
                    "score_status": "reranked",
                    "context_text": next(
                        evidence["text"] for document in contextual_docs
                        if int(document["official_id"]) == int(row["doc_id"])
                        for evidence in document["evidence"]),
                    "context_start": next(
                        evidence["start"] for document in contextual_docs
                        if int(document["official_id"]) == int(row["doc_id"])
                        for evidence in document["evidence"]),
                    "context_end": next(
                        evidence["end"] for document in contextual_docs
                        if int(document["official_id"]) == int(row["doc_id"])
                        for evidence in document["evidence"]),
                    "pair_input_sha256": row["pair_input_sha256"],
                    "pair_cache_hit": row["pair_cache_hit"],
                })
            added_ids, removed_ids = chosen_ids - a0_ids, a0_ids - chosen_ids
            for action, doc_ids in (("added_vs_A0", added_ids), ("removed_vs_A0", removed_ids)):
                for doc_id in sorted(doc_ids):
                    history = row_by_key.get((qid, doc_id))
                    a0_anchor = next((anchor for anchor in a0_metrics_by_q.get(
                        qid, {}).get("anchor_rows", []) if anchor["doc_id"] == doc_id), None)
                    allocation_changes.append({
                        "query_id": qid, "arm": arm, "action": action,
                        "doc_id": doc_id,
                        "historical_positive": history is not None,
                        "a0_context_exact": bool(a0_anchor and a0_anchor["exact_anchor"]),
                        "final_doc_rank": score_by_id[doc_id]["final_doc_rank"],
                        "fused_rank": score_by_id[doc_id]["fused_rank"],
                        "rerank_score": score_by_id[doc_id]["rerank_score"],
                        "chunk_selection_score": score_by_id[doc_id]["chunk_selection_score"],
                    })
                    if action == "removed_vs_A0":
                        displaced_rows.append({
                            "query_id": qid, "arm": arm, "doc_id": doc_id,
                            "historical_positive": history is not None,
                            "a0_context_exact": bool(history and next(
                                anchor["exact_anchor"] for anchor in a0_metrics_by_q[qid]["anchor_rows"]
                                if anchor["doc_id"] == doc_id)),
                            "final_doc_rank": score_by_id[doc_id]["final_doc_rank"],
                            "fused_rank": score_by_id[doc_id]["fused_rank"],
                            "rerank_score": score_by_id[doc_id]["rerank_score"],
                            "chunk_selection_score": score_by_id[doc_id]["chunk_selection_score"],
                        })
            if arm != "A0":
                this_by_id = {row["doc_id"]: row for row in q_positives["anchor_rows"]}
                base_by_id = {row["doc_id"]: row for row in a0_metrics_by_q[qid]["anchor_rows"]}
                for parent, value in this_by_id.items():
                    if value["exact_anchor"] and not base_by_id[parent]["exact_anchor"]:
                        rescued[arm]["rescued_exact_anchors"].append(
                            {"query_id": qid, "doc_id": parent})
                    if base_by_id[parent]["exact_anchor"] and not value["exact_anchor"]:
                        rescued[arm]["displaced_exact_anchors"].append(
                            {"query_id": qid, "doc_id": parent})
            top60_rows = [row for row in historical
                          if int(row["query_id"]) == qid
                          and (qid, int(row["doc_id"])) in EXPECTED_TOP60_POSITIVES]
            top60_eval = []
            q_anchor_by_doc = {row["doc_id"]: row for row in q_positives["anchor_rows"]}
            for top in top60_rows:
                parent = int(top["doc_id"])
                anchor_eval = q_anchor_by_doc[parent]
                top60_eval.append({
                    "query_id": qid,
                    "doc_id": parent,
                    "selected_for_chunk": parent in chosen_ids,
                    "exact_context_recovered": anchor_eval["exact_anchor"],
                    "lcs_proxy": anchor_eval["lcs_proxy"],
                    "selection_score": score_by_id[parent]["chunk_selection_score"],
                    "final_doc_rank": score_by_id[parent]["final_doc_rank"],
                    "rerank_score": score_by_id[parent]["rerank_score"],
                    "displaced_a0_docs": sorted(removed_ids) if parent in added_ids else [],
                })
            rescued[arm]["seven_top60_cases"].extend(top60_eval)

        arm_metrics_total[arm] = {
            "queries": len(query_ids),
            "historical_positive_anchors": total_positives,
            "exact_anchor_count": total_exact,
            "exact_anchor_rate": total_exact / total_positives,
            "lcs_proxy_mean": total_lcs / total_positives,
            "lcs_proxy_at_least_0_4": total_lcs40,
            "selected_chunks": total_chunks,
            "mean_query_coverage": total_query_coverage / total_chunks,
            "documents_per_query": 200,
            "chunks_per_query": CHUNK_CAP,
        }

    a0_total = arm_metrics_total["A0"]
    for arm in arms:
        total = arm_metrics_total[arm]
        total["quality_gate_vs_A0"] = {
            "exact_anchor_non_decrease": total["exact_anchor_count"] >= a0_total["exact_anchor_count"],
            "lcs_proxy_non_decrease": total["lcs_proxy_mean"] + 1e-12 >= a0_total["lcs_proxy_mean"],
            "per_query_exact_non_decrease": all(
                next(row for row in metrics_by_arm[arm] if row["query_id"] == qid)[
                    "exact_anchor_count"] >= a0_metrics_by_q[qid]["exact_anchor_count"]
                for qid in query_ids),
            "per_query_lcs_non_decrease": all(
                math.isclose(
                    next(row for row in metrics_by_arm[arm] if row["query_id"] == qid)[
                        "lcs_proxy_mean"],
                    a0_metrics_by_q[qid]["lcs_proxy_mean"], rel_tol=0.0, abs_tol=1e-12)
                or next(row for row in metrics_by_arm[arm] if row["query_id"] == qid)[
                    "lcs_proxy_mean"] > a0_metrics_by_q[qid]["lcs_proxy_mean"]
                for qid in query_ids),
        }
        total["quality_gate_vs_A0"]["pass"] = all(total["quality_gate_vs_A0"].values())

    # The seven P0 top-60 cases contain five exact recoveries from their current
    # selected seed/context and two requiring a different reranked passage.
    exact_rescuable = [
        (53, 756661), (130, 553119), (130, 2968260), (754, 2713777), (771, 484565),
    ]
    five_case_results = {}
    for arm in arms:
        evaluated = [row for row in rescued[arm]["seven_top60_cases"]
                     if (row["query_id"], row["doc_id"]) in exact_rescuable]
        five_case_results[arm] = {
            "exact_recoverable_cases": len(exact_rescuable),
            "selected_for_chunk": sum(row["selected_for_chunk"] for row in evaluated),
            "exact_context_recovered": sum(row["exact_context_recovered"] for row in evaluated),
            "cases": evaluated,
        }

    input_hashes_after = {str(path.relative_to(root)): sha(path) for path in input_paths}
    if input_hashes_before != input_hashes_after:
        raise ValueError("Allocation replay input changed during execution")

    output.mkdir(parents=True)
    comparison = {
        "status": "CHUNK_ALLOCATION_REPLAY_COMPLETE",
        "scope": "six locked full-index queries; cached P0 scored-parent pool only",
        "query_ids": query_ids,
        "pool_contract": {"max_scored_parents_per_query": RERANK_POOL_CAP,
                          "actual_scored_parents_per_query": RERANK_POOL_CAP,
                          "relevant_docs_per_query": 200,
                          "chunks_per_query": CHUNK_CAP,
                          "chunks_per_document": 1,
                          "model_calls": 0,
                          "retrieval_calls": 0},
        "score_definitions": {
            "rerank_normalized": "per-query min-max over 200 P0 cached parent scores; constant vector maps to 0.5",
            "retrieval_signal": "min-max of mean rank strength across fused parent rank, best dense rank (cap 200), and best document/passage BM25 rank (cap 400); missing branch is zero",
            "query_coverage": "unique NFKC/lowercase query word tokens found in scored passage divided by unique query word tokens; no labels",
            "A2_formula": "alpha*rerank_normalized + beta*retrieval_signal + gamma*query_coverage",
            "A2_weight_sets": HYBRID_WEIGHTS,
            "A3_quota": "40 hybrid(0.5/0.3/0.2) + 10 best dense rank + 10 query coverage, deduplicate doc_id, hybrid backfill",
        },
        "arms": arm_metrics_total,
        "a0_p0_parity": True,
        "seven_top60_cases": {
            "exact_recoverable_cases": exact_rescuable,
            "per_arm": five_case_results,
        },
        "allocation_tradeoff": {
            "note": "Historical anchors are incomplete and are diagnostic only, not official precision/F2.",
            "per_arm_rescued_exact_anchors_vs_A0": {
                arm: rescued[arm]["rescued_exact_anchors"] for arm in arms},
            "per_arm_displaced_exact_anchors_vs_A0": {
                arm: rescued[arm]["displaced_exact_anchors"] for arm in arms},
            "per_arm_changed_chunk_slots_vs_A0": {
                arm: sum(1 for row in allocation_changes
                         if row["arm"] == arm and row["action"] == "added_vs_A0")
                for arm in arms},
            "quality_gate_pass_arms": [arm for arm in arms
                                        if arm_metrics_total[arm]["quality_gate_vs_A0"]["pass"]],
        },
        "official_f2": "UNMEASURED",
    }

    # Store a record for each relevant_docs parent, with the requested fields
    # and score values for every arm. `chunk_selection_score` defaults to the
    # main A2 0.5/0.3/0.2 score; per-arm values are retained alongside it.
    pool_with_arm_scores = []
    for qid in query_ids:
        for row in pool_by_query[qid]:
            pool_with_arm_scores.append({
                key: row[key] for key in (
                    "query_id", "doc_id", "final_doc_rank", "fused_rank", "dense_rank",
                    "bm25_rank", "passage_id", "rerank_score", "query_coverage",
                )} | {
                    "chunk_selection_score": row["hybrid_scores"]["A2_hybrid_5_3_2"],
                    "chunk_selection_scores": {"A0": score_records([row], "A0")[0][
                        "chunk_selection_score"],
                        "A1": score_records([row], "A1")[0]["chunk_selection_score"],
                        **row["hybrid_scores"], QUOTA_ARM: row["hybrid_scores"][
                            "A2_hybrid_5_3_2"]},
                    "retrieval_signal": row["retrieval_signal"],
                    "rerank_normalized": row["rerank_normalized"],
                    "pair_position": row["pair_position"],
                    "pair_input_sha256": row["pair_input_sha256"],
                    "pair_cache_hit": row["pair_cache_hit"],
                })

    write_json(output / "arm_comparison.json", comparison)
    write_json(output / "per_query_metrics.json", metrics_by_arm)
    write_jsonl(output / "scored_parent_pool.jsonl", pool_with_arm_scores)
    write_jsonl(output / "selected_chunks.jsonl", selected_chunk_rows)
    write_csv(output / "allocation_changes.csv", allocation_changes)
    write_json(output / "rescued_positives.json", rescued)
    write_json(output / "displaced_documents.json", displaced_rows)
    validation = {
        "status": "PASS",
        "checks": {
            "a0_matches_p0_selected_chunks": True,
            "all_arms_keep_same_200_relevant_docs": True,
            "all_arms_select_exactly_60_unique_parents_per_query": True,
            "all_arms_use_one_cached_scored_passage_per_chunk": True,
            "all_cache_scores_verified_by_query_parent_passage_input_hash": True,
            "all_selected_contexts_source_validated": True,
            "deduplication_key": "doc_id",
            "no_retrieval_calls": True,
            "no_model_calls": True,
            "historical_labels_used_only_for_metrics": True,
            "allocation_weights": "three interpretable A2 sets plus one A3 quota; no search/tuning",
        },
        "historical_top60_positive_validation": {
            "positive_cases": len(EXPECTED_TOP60_POSITIVES),
            "a0_replay_exactly_matches_p0": True,
            "five_current_seed_context_recoverable_checked_for_each_arm": True,
        },
        "a3_quota_by_query": {
            str(row["query_id"]): row["allocator"]["quota_filled"]
            for row in metrics_by_arm[QUOTA_ARM]
        },
        "quality_gates": {arm: value["quality_gate_vs_A0"]
                           for arm, value in arm_metrics_total.items()},
        "pilot_acceptance": "diagnostic_only; quality gate is not generalization evidence",
    }
    write_json(output / "validation.json", validation)
    write_json(output / "input_hashes.json", input_hashes_before)
    artifact_names = ["arm_comparison.json", "per_query_metrics.json",
                      "scored_parent_pool.jsonl", "selected_chunks.jsonl",
                      "allocation_changes.csv", "rescued_positives.json",
                      "displaced_documents.json", "validation.json", "input_hashes.json"]
    manifest = {
        "format": "v4-p1-chunk-allocation-replay-v1",
        "status": "complete",
        "query_ids": query_ids,
        "historical_positive_anchors": len(historical),
        "arms": arms,
        "pool_parents_per_query": RERANK_POOL_CAP,
        "chunk_slots_per_query": CHUNK_CAP,
        "model_calls": 0,
        "retrieval_calls": 0,
        "input_hashes_sha256": sha(output / "input_hashes.json"),
        "artifacts_sha256": {name: sha(output / name) for name in artifact_names},
        "implementation_sha256": sha(Path(__file__).resolve()),
        "summary_sha256": canonical_sha(comparison),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "arms": arm_metrics_total,
                      "allocation_ceiling_reference": p0_summary["allocation_ceiling"],
                      "quality_gate_pass_arms": comparison["allocation_tradeoff"][
                          "quality_gate_pass_arms"]}, ensure_ascii=False, indent=2))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
