"""Audit B0 candidate deltas and cached-score readiness without retrieval or inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.contracts import Hit
from src.medrag.core.passage_first import passage_candidates
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.evidence.verifier import verify
from src.medrag.ranking.score_cache import ScoreCache


ROOT = Path(__file__).resolve().parents[2]
A0_CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
A0_CONFIG_SHA256 = "bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03"
INDEX = ROOT / "outputs/v4/task4/full_index_gpu3_20261019"
TRACES = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2/queries"
P4 = ROOT / "outputs/v4/p4_b0_candidate_replay_20261021_v2"
P2_PAIRS = ROOT / "outputs/v4/p2_passage_candidate_replay_20261021/pair_allocation.csv"
P0_PATHS = ROOT / "outputs/v4/p0_positive_path_replay_20261021_v3/positive_paths.jsonl"
P1_POOL = ROOT / "outputs/v4/p1_chunk_allocation_20261021_v2/scored_parent_pool.jsonl"
MIGRATION = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"
PRIMARY_LABELS = ROOT / "outputs/v4/p2_relevance_review_packet_20261021_v2/annotation_template.csv"
SECONDARY_LABELS = ROOT / "outputs/v4/p3_secondary_review_assignment_20261022/secondary_annotation_template.csv"
QUERY_SPLIT = ROOT / "outputs/v4/p3_query_split_20261021/query_split.json"
SEED_CACHE = ROOT / "outputs/v4/p2_missing_pair_scores_20261021/scores.sqlite"
TARGET_CONTRACT_FINGERPRINT = "42350936c327dc01630472257a4444a8df27713140b6fc7e518ba01ad4239c41"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def nonempty_label_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        return sum(bool(row["answers_question_0_1_2"].strip()) for row in csv.DictReader(stream))


def cache_metadata(path: Path) -> dict:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key,value FROM meta"))
            count = int(db.execute("SELECT count(*) FROM scores").fetchone()[0])
        contract = json.loads(meta.get("contract", "null"))
        expected = hashlib.sha256(canonical(contract).encode()).hexdigest()
        return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "entries": count,
                "format": meta.get("format"), "fingerprint": meta.get("fingerprint"),
                "calculated_fingerprint": expected, "contract": contract, "read_error": None}
    except Exception as exc:  # report incompatibility; never silently merge
        return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "entries": None,
                "format": None, "fingerprint": None, "calculated_fingerprint": None,
                "contract": None, "read_error": f"{type(exc).__name__}: {exc}"}


def branch_sets(trace: dict) -> dict[str, set]:
    branches = trace["branch_trace"]["branch_hits"]
    result = {
        "document_bm25_parents": {int(row["official_id"]) for row in branches["document_bm25"]},
        "passage_bm25": {(int(row["official_id"]), row["passage_id"])
                         for row in branches["passage_bm25"]},
        "dense": {(int(row["official_id"]), row["passage_id"])
                  for row in branches["dense"]},
        "fallback": set(),
    }
    for parent in trace["branch_trace"]["fallback_hits"]:
        result["fallback"].update((int(row["official_id"]), row["passage_id"])
                                  for row in parent["hits"])
    return result


def branch_summary(pairs: set[tuple[int, str]], branches: dict[str, set]) -> dict:
    counts = Counter()
    for pair in pairs:
        in_bm25, in_dense = pair in branches["passage_bm25"], pair in branches["dense"]
        if in_bm25 and in_dense:
            counts["both_passage_branches"] += 1
        elif in_bm25:
            counts["passage_bm25_only"] += 1
        elif in_dense:
            counts["dense_only"] += 1
        else:
            counts["neither_passage_branch"] += 1
        if pair in branches["fallback"]:
            counts["fallback_passage"] += 1
        if pair[0] in branches["document_bm25_parents"]:
            counts["document_bm25_parent"] += 1
    return {key: counts.get(key, 0) for key in (
        "both_passage_branches", "passage_bm25_only", "dense_only",
        "neither_passage_branch", "fallback_passage", "document_bm25_parent")}


def rank_summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def main(output: Path, comparison_path: Path, readiness_path: Path) -> None:
    output = output.resolve()
    if output.exists() or comparison_path.exists() or readiness_path.exists():
        raise FileExistsError("B0 audit outputs must be fresh")
    if sha(A0_CONFIG) != A0_CONFIG_SHA256:
        raise ValueError("Frozen A0 config hash mismatch")
    config = read(A0_CONFIG)
    index_manifest_path = INDEX / "manifest.json"
    index_manifest = read(index_manifest_path)
    if (index_manifest.get("status") != "complete"
            or index_manifest.get("scope") != "full_release"
            or index_manifest.get("corpus_version") != config["corpus_version"]):
        raise ValueError("Full index/corpus identity mismatch")
    p4_manifest = read(P4 / "manifest.json")
    p4_rows = read(P4 / "candidate_comparison.json")
    if (p4_manifest.get("status") != "candidate_replay_only"
            or p4_manifest.get("query_count") != 6
            or p4_manifest.get("candidate_comparison_sha256") != sha(P4 / "candidate_comparison.json")
            or p4_manifest.get("full_index_manifest_sha256") != sha(index_manifest_path)
            or p4_manifest.get("baseline_pair_allocation_sha256") != sha(P2_PAIRS)):
        raise ValueError("P4 replay provenance mismatch")
    query_ids = [int(row["query_id"]) for row in p4_rows]
    if query_ids != sorted(query_ids) or len(query_ids) != 6:
        raise ValueError("P4 pilot query coverage/order mismatch")
    for qid in query_ids:
        if p4_manifest["trace_sha256"].get(str(qid)) != sha(TRACES / f"{qid}.json"):
            raise ValueError(f"P4 trace hash mismatch: {qid}")

    migration = read(MIGRATION)
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query hash mismatch")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist() if int(row["id"]) in query_ids}
    if set(queries) != set(query_ids):
        raise ValueError("Pilot queries missing from official query artifact")

    a0_by_query: dict[int, list[dict]] = defaultdict(list)
    with P2_PAIRS.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] == "P2-A0":
                a0_by_query[int(row["query_id"])].append(row)
    if any(len(a0_by_query[qid]) != 200 for qid in query_ids):
        raise ValueError("A0 candidate pool does not contain 200 pairs/query")

    store = DiskCorpusStore(INDEX / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    all_b0_pairs = {(int(doc), passage_id) for row in p4_rows
                    for doc, passage_id in row["b0_pair_ids"]}
    passages = store.passages_batch(sorted({passage_id for _, passage_id in all_b0_pairs}))
    sources = store.sources_batch({doc for doc, _ in all_b0_pairs})
    source_invalid: list[dict] = []
    wrong_parent: list[dict] = []
    for doc, passage_id in sorted(all_b0_pairs):
        passage = passages[passage_id]
        if passage.official_id != doc:
            wrong_parent.append({"doc_id": doc, "passage_id": passage_id,
                                 "registered_parent": passage.official_id})
        elif not verify(passage, sources[doc]):
            source_invalid.append({"doc_id": doc, "passage_id": passage_id})

    cache_paths = sorted(ROOT.glob("outputs/v4/**/scores.sqlite"))
    caches = [cache_metadata(path) for path in cache_paths]
    target_contract = read(ROOT / "outputs/v4/a0_t4_v1_full_20261022/lock.json")["score_contract"]
    target_fingerprint = hashlib.sha256(canonical(target_contract).encode()).hexdigest()
    if target_fingerprint != TARGET_CONTRACT_FINGERPRINT:
        raise ValueError("A0 score contract fingerprint changed")
    compatible_paths = [ROOT / row["path"] for row in caches
                        if row["read_error"] is None
                        and row["format"] == "medrag-v4-score-cache-v1"
                        and row["fingerprint"] == target_fingerprint
                        and row["calculated_fingerprint"] == target_fingerprint
                        and row["contract"] == target_contract]
    incompatible_caches = [row for row in caches if ROOT / row["path"] not in compatible_paths]
    if not compatible_paths:
        raise ValueError("No score cache matches the frozen A0 scoring contract")
    key_builder = ScoreCache(compatible_paths[0], target_contract)

    score_by_key: dict[str, float] = {}
    score_sources: dict[str, list[str]] = defaultdict(list)
    conflicting_scores: list[dict] = []
    incompatible_rows: list[dict] = []
    pair_rows_by_query: dict[int, list[dict]] = defaultdict(list)
    for p4_row in p4_rows:
        qid = int(p4_row["query_id"])
        query = queries[qid]
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        for b0_rank, (doc_raw, passage_id) in enumerate(p4_row["b0_pair_ids"], 1):
            doc = int(doc_raw)
            passage, source = passages[passage_id], sources[doc]
            payload = f"{source.title}\n{passage.text}" if source.title else passage.text
            input_hash = hashlib.sha256(payload.encode()).hexdigest()
            cache_key = key_builder._key(query, doc, passage_id, payload)
            exact_values: list[tuple[str, float]] = []
            mismatched = []
            for cache_path in compatible_paths:
                with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as db:
                    row = db.execute(
                        "SELECT score,query_sha256,official_id,passage_id,input_sha256 "
                        "FROM scores WHERE cache_key=?", (cache_key,)).fetchone()
                    if row is not None:
                        expected = (query_hash, doc, passage_id, input_hash)
                        if row[1:] == expected and math.isfinite(float(row[0])):
                            exact_values.append((str(cache_path.relative_to(ROOT)), float(row[0])))
                        else:
                            mismatched.append({"cache": str(cache_path.relative_to(ROOT)),
                                               "row": list(row), "expected": list(expected)})
                    for other in db.execute(
                            "SELECT cache_key,score,input_sha256 FROM scores "
                            "WHERE query_sha256=? AND official_id=? AND passage_id=? AND cache_key<>?",
                            (query_hash, doc, passage_id, cache_key)):
                        mismatched.append({"cache": str(cache_path.relative_to(ROOT)),
                                           "cache_key": other[0], "score": other[1],
                                           "input_sha256": other[2], "expected_input_sha256": input_hash})
            unique_values = sorted({value for _, value in exact_values})
            if len(unique_values) > 1:
                conflicting_scores.append({"query_id": qid, "doc_id": doc,
                                           "passage_id": passage_id, "values": exact_values})
            if mismatched:
                incompatible_rows.append({"query_id": qid, "doc_id": doc,
                                          "passage_id": passage_id, "rows": mismatched})
            valid_score = unique_values[0] if len(unique_values) == 1 else None
            if valid_score is not None:
                score_by_key[cache_key] = valid_score
                score_sources[cache_key] = sorted(source for source, _ in exact_values)
            pair_rows_by_query[qid].append({
                "query_id": qid, "b0_fused_rank": b0_rank, "official_id": doc,
                "passage_id": passage_id, "query": query, "query_sha256": query_hash,
                "payload": payload, "input_sha256": input_hash, "cache_key": cache_key,
                "cached_score": valid_score, "cache_sources": score_sources.get(cache_key, []),
            })

    top60 = set()
    with P1_POOL.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["final_doc_rank"]) <= 60:
                top60.add((int(row["query_id"]), int(row["doc_id"])))
    historical_by_query: dict[int, set[int]] = defaultdict(set)
    with P0_PATHS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            historical_by_query[int(row["identity"]["query_id"])].add(
                int(row["identity"]["doc_id"]))

    comparison_rows = []
    potential_better = []
    total_duplicate_pairs = 0
    for p4_row in p4_rows:
        qid = int(p4_row["query_id"])
        trace = read(TRACES / f"{qid}.json")
        branches = branch_sets(trace)
        rankings = [[Hit(int(item["official_id"]), item["passage_id"], float(item["score"]),
                         channel, int(item["rank"]))
                     for item in trace["branch_trace"]["branch_hits"][channel]]
                    for channel in ("passage_bm25", "dense")]
        b0_fused = passage_candidates(rankings, limit=400)
        b0_pairs = [(hit.official_id, hit.passage_id) for hit in b0_fused[:200]]
        expected_b0 = [(int(doc), passage_id) for doc, passage_id in p4_row["b0_pair_ids"]]
        if b0_pairs != expected_b0:
            raise ValueError(f"B0 direct passage fusion differs from frozen P4 replay: {qid}")
        a0_rows = sorted(a0_by_query[qid], key=lambda row: int(row["pair_order"]))
        a0_pairs = [(int(row["doc_id"]), row["passage_id"]) for row in a0_rows]
        a_set, b_set = set(a0_pairs), set(b0_pairs)
        total_duplicate_pairs += (len(a0_pairs) - len(a_set)) + (len(b0_pairs) - len(b_set))
        a_parents, b_parents = {doc for doc, _ in a_set}, {doc for doc, _ in b_set}
        a_passage = {doc: passage_id for doc, passage_id in a0_pairs}
        b_passages: dict[int, set[str]] = defaultdict(set)
        for doc, passage_id in b0_pairs:
            b_passages[doc].add(passage_id)
        shared_parents = a_parents & b_parents
        same_a0_passage = {doc for doc in shared_parents if a_passage[doc] in b_passages[doc]}
        different_only = shared_parents - same_a0_passage
        additional_b0 = {doc for doc in shared_parents
                         if any(pid != a_passage[doc] for pid in b_passages[doc])}
        a_rank = {pair: int(row["pair_order"]) for pair, row in zip(a0_pairs, a0_rows, strict=True)}
        b_rank = {pair: rank for rank, pair in enumerate(b0_pairs, 1)}
        shared_deltas = [b_rank[pair] - a_rank[pair] for pair in sorted(a_set & b_set)]
        b_pair_data = {(row["official_id"], row["passage_id"]): row
                       for row in pair_rows_by_query[qid]}
        a_score_by_pair = {}
        for row in a0_rows:
            pair = (int(row["doc_id"]), row["passage_id"])
            value = row["cached_rerank_score"]
            a_score_by_pair[pair] = None if value == "" else float(value)
        for doc in sorted(additional_b0):
            a_pair = (doc, a_passage[doc])
            a_score = a_score_by_pair.get(a_pair)
            candidates = [b_pair_data[(doc, pid)] for pid in sorted(b_passages[doc])
                          if pid != a_passage[doc] and b_pair_data[(doc, pid)]["cached_score"] is not None]
            if a_score is None or not candidates:
                continue
            best = max(candidates, key=lambda row: (row["cached_score"], row["passage_id"]))
            if best["cached_score"] > a_score:
                potential_better.append({
                    "query_id": qid, "doc_id": doc, "a0_passage_id": a_passage[doc],
                    "a0_cached_score": a_score, "b0_alternative_passage_id": best["passage_id"],
                    "b0_cached_score": best["cached_score"],
                    "score_delta": best["cached_score"] - a_score,
                    "b0_fused_rank": best["b0_fused_rank"],
                    "a0_top60_parent": (qid, doc) in top60,
                    "independent_relevance_judgment": "UNKNOWN",
                    "interpretation": "reranker-score hypothesis only; not a relevance label",
                })
        historical = historical_by_query[qid]
        comparison_rows.append({
            "query_id": qid,
            "a0_candidate_pairs": len(a0_pairs), "b0_candidate_pairs": len(b0_pairs),
            "a0_only_passages": [{"official_id": doc, "passage_id": pid}
                                  for doc, pid in sorted(a_set - b_set)],
            "b0_only_passages": [{"official_id": doc, "passage_id": pid}
                                  for doc, pid in sorted(b_set - a_set)],
            "shared_passages": [{"official_id": doc, "passage_id": pid}
                                for doc, pid in sorted(a_set & b_set)],
            "a0_unique_parents": len(a_parents), "b0_unique_parents": len(b_parents),
            "shared_parents": len(shared_parents),
            "shared_parents_same_a0_passage": len(same_a0_passage),
            "shared_parents_a0_passage_absent_from_b0": len(different_only),
            "shared_parents_with_additional_b0_passages": len(additional_b0),
            "parent_ids": {"a0_only": sorted(a_parents - b_parents),
                           "b0_only": sorted(b_parents - a_parents),
                           "shared_different_only": sorted(different_only),
                           "shared_with_additional_b0_passages": sorted(additional_b0)},
            "passage_fusion_rank_delta_b0_minus_a0_for_shared_pairs": {
                **rank_summary(shared_deltas),
                "b0_ranked_better": sum(delta < 0 for delta in shared_deltas),
                "same_rank": sum(delta == 0 for delta in shared_deltas),
                "b0_ranked_worse": sum(delta > 0 for delta in shared_deltas)},
            "candidate_survival_by_branch": {
                "a0_pool": branch_summary(a_set, branches),
                "b0_pool": branch_summary(b_set, branches),
                "a0_only": branch_summary(a_set - b_set, branches),
                "b0_only": branch_summary(b_set - a_set, branches),
                "shared": branch_summary(a_set & b_set, branches)},
            "historical_parent_survival_diagnostic_only": {
                "historical_parents": len(historical),
                "in_a0_pool": len(historical & a_parents),
                "in_b0_pool": len(historical & b_parents),
                "a0_only": sorted(historical & (a_parents - b_parents)),
                "b0_only": sorted(historical & (b_parents - a_parents))},
            "source_validated_b0_pairs": len(b0_pairs),
        })

    all_pair_rows = [row for qid in query_ids for row in pair_rows_by_query[qid]]
    missing = [row for row in all_pair_rows if row["cached_score"] is None]
    valid = [row for row in all_pair_rows if row["cached_score"] is not None]
    if len(all_pair_rows) != 1200:
        raise ValueError("B0 total pair count differs from 1,200")
    output.mkdir(parents=True)
    missing_dir = output / "missing_pairs"
    missing_dir.mkdir()
    shard_hashes = {}
    for qid in query_ids:
        rows = [{key: value for key, value in row.items() if key != "cache_sources"}
                for row in missing if row["query_id"] == qid]
        shard = {"format": "medrag-v4-b0-missing-pairs-v1", "query_id": qid,
                 "score_contract_fingerprint": target_fingerprint,
                 "query_sha256": hashlib.sha256(queries[qid].encode()).hexdigest(),
                 "missing_pair_count": len(rows), "pairs": rows}
        path = missing_dir / f"{qid}.json"
        write_once(path, shard)
        shard_hashes[str(qid)] = sha(path)

    comparison = {
        "format": "medrag-v4-b0-a0-candidate-comparison-v1",
        "status": "PASS_CANDIDATE_DELTA",
        "scope": "six frozen diagnostic queries; no retrieval/model calls",
        "query_ids": query_ids,
        "totals": {
            "a0_pairs": sum(row["a0_candidate_pairs"] for row in comparison_rows),
            "b0_pairs": sum(row["b0_candidate_pairs"] for row in comparison_rows),
            "shared_passages": sum(len(row["shared_passages"]) for row in comparison_rows),
            "a0_only_passages": sum(len(row["a0_only_passages"]) for row in comparison_rows),
            "b0_only_passages": sum(len(row["b0_only_passages"]) for row in comparison_rows),
            "a0_unique_parents": sum(row["a0_unique_parents"] for row in comparison_rows),
            "b0_unique_parents": sum(row["b0_unique_parents"] for row in comparison_rows),
            "duplicated_pairs_within_query_pools": total_duplicate_pairs,
        },
        "per_query": comparison_rows,
        "offline_quality": {
            "independent_labels_available": False,
            "primary_completed_labels": nonempty_label_count(PRIMARY_LABELS),
            "secondary_completed_labels": nonempty_label_count(SECONDARY_LABELS),
            "development_holdout_label_status": read(QUERY_SPLIT)["label_status"],
            "potentially_better_within_a0_parent_by_cached_reranker_only": potential_better,
            "candidate_judgment_default": "UNKNOWN",
            "warning": "retrieval overlap and reranker score are not independent relevance labels",
        },
        "provenance": {
            "a0_config_sha256": sha(A0_CONFIG),
            "p4_manifest_sha256": sha(P4 / "manifest.json"),
            "p4_candidate_comparison_sha256": sha(P4 / "candidate_comparison.json"),
            "p2_pair_allocation_sha256": sha(P2_PAIRS),
            "index_manifest_sha256": sha(index_manifest_path),
            "official_queries_sha256": sha(query_path),
            "trace_sha256": p4_manifest["trace_sha256"],
        },
    }
    write_once(comparison_path, comparison)

    per_query_readiness = []
    for qid in query_ids:
        rows = pair_rows_by_query[qid]
        per_query_readiness.append({
            "query_id": qid, "total_pairs": len(rows),
            "cached_valid_scores": sum(row["cached_score"] is not None for row in rows),
            "missing_scores": sum(row["cached_score"] is None for row in rows),
            "source_invalid_pairs": sum(any(item["doc_id"] == row["official_id"]
                                             and item["passage_id"] == row["passage_id"]
                                             for item in source_invalid) for row in rows),
            "end_to_end_replay_ready": all(row["cached_score"] is not None for row in rows),
        })
    readiness = {
        "format": "medrag-v4-b0-rerank-readiness-v1",
        "status": "BLOCKED_MISSING_SCORES" if missing else "READY_FOR_CPU_REPLAY",
        "query_ids": query_ids,
        "score_contract": target_contract,
        "score_contract_fingerprint": target_fingerprint,
        "counts": {
            "total_b0_pairs": len(all_pair_rows),
            "cached_valid_scores": len(valid),
            "missing_scores": len(missing),
            "incompatible_cached_scores": len(incompatible_rows) + len(conflicting_scores),
            "duplicated_pairs": total_duplicate_pairs,
            "source_invalid_pairs": len(source_invalid) + len(wrong_parent),
        },
        "per_query": per_query_readiness,
        "cache_inventory": {"compatible": [row for row in caches if ROOT / row["path"] in compatible_paths],
                            "incompatible": incompatible_caches,
                            "incompatible_pair_rows": incompatible_rows,
                            "conflicting_exact_scores": conflicting_scores,
                            "seed_cache_for_gpu_job": str(SEED_CACHE.relative_to(ROOT)),
                            "seed_cache_sha256": sha(SEED_CACHE)},
        "missing_score_manifest": {
            "directory": str(missing_dir.relative_to(ROOT)),
            "shards_sha256": shard_hashes,
            "total": len(missing),
        },
        "source_validation": {
            "validated_pairs": len(all_pair_rows) - len(source_invalid) - len(wrong_parent),
            "invalid": source_invalid, "parent_mapping_mismatch": wrong_parent,
            "prior_p4_validated_pairs": p4_manifest["registered_source_validated_pairs"],
        },
        "end_to_end": {
            "reranked_passage_candidates": "BLOCKED_MISSING_SCORES" if missing else "READY",
            "document_ranking": "BLOCKED_MISSING_SCORES" if missing else "READY",
            "top60_unique_parent_chunks": "BLOCKED_MISSING_SCORES" if missing else "READY",
            "source_grounded_contexts": "STRUCTURALLY_VALIDATED; FINAL_REPLAY_PENDING_SCORES",
            "submission_compatible_objects": "IMPLEMENTED; FINAL_REPLAY_PENDING_SCORES",
            "cpu_replay_script": "scripts/v4/replay_b0_cached.py",
        },
        "execution": {"retrieval_calls": 0, "model_calls": 0, "gpu_used": False,
                      "full_index_loaded": False,
                      "targeted_read_only_corpus_rows": len(all_pair_rows)},
    }
    write_once(readiness_path, readiness)
    gpu_job = {
        "format": "medrag-v4-b0-gpu-missing-score-job-v1", "status": "PREPARED_NOT_EXECUTED",
        "missing_pairs": len(missing), "query_shards": query_ids,
        "score_contract_fingerprint": target_fingerprint,
        "input_shards_sha256": shard_hashes,
        "output_namespace": "outputs/v4/b0_missing_scores_20261022",
        "command": [".venv-phase2-batch/bin/python", "scripts/v4/run_single_gpu.py",
                    "--config", "configs/v4_submission_p2_a0.yaml", "--report",
                    "outputs/v4/b0_missing_scores_20261022/gpu_preflight.json", "--",
                    "scripts/v4/score_b0_missing_pairs.py", "--readiness",
                    str(readiness_path.relative_to(ROOT)), "--output",
                    "outputs/v4/b0_missing_scores_20261022", "--device", "cuda:0"],
        "guard": "DO NOT EXECUTE while A0 has priority; use separate output and GPU preflight",
    }
    write_once(output / "gpu_scoring_job.json", gpu_job)
    artifact_manifest = {
        "format": "medrag-v4-b0-cached-first-audit-artifacts-v1",
        "status": "PASS",
        "artifacts_sha256": {
            str(comparison_path.relative_to(ROOT)): sha(comparison_path),
            str(readiness_path.relative_to(ROOT)): sha(readiness_path),
            "gpu_scoring_job.json": sha(output / "gpu_scoring_job.json"),
            **{f"missing_pairs/{qid}.json": value for qid, value in shard_hashes.items()},
        },
    }
    write_once(output / "manifest.json", artifact_manifest)
    print(json.dumps({"status": readiness["status"], "counts": readiness["counts"],
                      "potential_better_unknown": len(potential_better),
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=ROOT / "B0_A0_COMPARISON.json")
    parser.add_argument("--readiness", type=Path, default=ROOT / "B0_RERANK_READINESS.json")
    args = parser.parse_args()
    main(args.output, args.comparison.resolve(), args.readiness.resolve())
