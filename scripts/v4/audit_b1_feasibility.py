"""CPU-only B0 passage-gain and B1 breadth-preserving feasibility audit.

This script consumes frozen candidate traces and exact cached reranker scores only.
It performs no retrieval, model loading, inference, or GPU work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.evidence.verifier import verify
from src.medrag.ranking.score_cache import ScoreCache


ROOT = Path(__file__).resolve().parents[2]
A0_CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
A0_CONFIG_SHA256 = "bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03"
INDEX = ROOT / "outputs/v4/task4/full_index_gpu3_20261019"
P4 = ROOT / "outputs/v4/p4_b0_candidate_replay_20261021_v2"
TRACES = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2/queries"
P2_PAIRS = ROOT / "outputs/v4/p2_passage_candidate_replay_20261021/pair_allocation.csv"
P1_POOL = ROOT / "outputs/v4/p1_chunk_allocation_20261021_v2/scored_parent_pool.jsonl"
MIGRATION = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"
SEED_CACHE = ROOT / "outputs/v4/p2_missing_pair_scores_20261021/scores.sqlite"
B0_COMPARISON = ROOT / "B0_A0_COMPARISON.json"
B0_READINESS = ROOT / "B0_RERANK_READINESS.json"
B0_NAMESPACE = ROOT / "outputs/v4/b0_cached_first_readiness_20261022"
CONTRACT_FINGERPRINT = "42350936c327dc01630472257a4444a8df27713140b6fc7e518ba01ad4239c41"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def main(output: Path, review_csv: Path, top60_json: Path) -> None:
    output = output.resolve()
    review_csv = review_csv.resolve()
    top60_json = top60_json.resolve()
    if output.exists() or review_csv.exists() or top60_json.exists():
        raise FileExistsError("B1 audit outputs must be fresh")
    if sha(A0_CONFIG) != A0_CONFIG_SHA256:
        raise ValueError("Frozen A0 config hash mismatch")

    config = __import__("yaml").safe_load(A0_CONFIG.read_text(encoding="utf-8"))
    index_manifest_path = INDEX / "manifest.json"
    index_manifest = read(index_manifest_path)
    p4_manifest = read(P4 / "manifest.json")
    p4_rows = read(P4 / "candidate_comparison.json")
    prior_comparison = read(B0_COMPARISON)
    prior_readiness = read(B0_READINESS)
    query_ids = [int(row["query_id"]) for row in p4_rows]
    if query_ids != [53, 130, 380, 754, 771, 1144]:
        raise ValueError(f"Unexpected frozen pilot queries: {query_ids}")
    if (p4_manifest["candidate_comparison_sha256"] != sha(P4 / "candidate_comparison.json")
            or p4_manifest["baseline_pair_allocation_sha256"] != sha(P2_PAIRS)
            or p4_manifest["full_index_manifest_sha256"] != sha(index_manifest_path)):
        raise ValueError("P4 artifact provenance mismatch")
    if prior_comparison["totals"] != {
            "a0_pairs": 1200, "b0_pairs": 1200, "shared_passages": 851,
            "a0_only_passages": 349, "b0_only_passages": 349,
            "a0_unique_parents": 1200, "b0_unique_parents": 878,
            "duplicated_pairs_within_query_pools": 0}:
        raise ValueError("Prior B0/A0 counts changed")
    if prior_readiness["counts"] != {
            "total_b0_pairs": 1200, "cached_valid_scores": 1099,
            "missing_scores": 101, "incompatible_cached_scores": 0,
            "duplicated_pairs": 0, "source_invalid_pairs": 0}:
        raise ValueError("Prior B0 score readiness changed")
    contract = prior_readiness["score_contract"]
    if hashlib.sha256(canonical(contract).encode()).hexdigest() != CONTRACT_FINGERPRINT:
        raise ValueError("Score contract fingerprint mismatch")
    if prior_readiness["score_contract_fingerprint"] != CONTRACT_FINGERPRINT:
        raise ValueError("Recorded score contract fingerprint mismatch")

    migration = read(MIGRATION)
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query artifact mismatch")
    queries = {int(row["id"]): str(row["query"]) for row in
               pq.read_table(query_path, columns=["id", "query"]).to_pylist()
               if int(row["id"]) in query_ids}
    if set(queries) != set(query_ids):
        raise ValueError("Pilot query missing from official query artifact")

    a0_by_query: dict[int, list[dict]] = defaultdict(list)
    a1_by_query: dict[int, list[dict]] = defaultdict(list)
    with P2_PAIRS.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] == "P2-A0" and int(row["query_id"]) in query_ids:
                a0_by_query[int(row["query_id"])].append(row)
            elif row["arm"] == "P2-A1" and int(row["query_id"]) in query_ids:
                a1_by_query[int(row["query_id"])].append(row)
    for qid in query_ids:
        a0_by_query[qid].sort(key=lambda row: int(row["pair_order"]))
        a1_by_query[qid].sort(key=lambda row: int(row["pair_order"]))
        if len(a0_by_query[qid]) != 200 or len(a1_by_query[qid]) != 200:
            raise ValueError(f"A0/A1 does not have 200 pairs for query {qid}")

    p1_by_query: dict[int, dict[int, dict]] = defaultdict(dict)
    with P1_POOL.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid, doc = int(row["query_id"]), int(row["doc_id"])
            if qid in query_ids:
                p1_by_query[qid][doc] = row
    if any(len(p1_by_query[qid]) != 200 for qid in query_ids):
        raise ValueError("Frozen A0 scored parent pool is not 200 parents/query")

    b0_ordered = {int(row["query_id"]): [(int(doc), pid) for doc, pid in row["b0_pair_ids"]]
                  for row in p4_rows}
    all_pairs = {(int(row["doc_id"]), row["passage_id"])
                 for rows in a0_by_query.values() for row in rows}
    all_pairs.update(pair for rows in b0_ordered.values() for pair in rows)
    passage_ids = sorted({pid for _, pid in all_pairs})
    doc_ids = {doc for doc, _ in all_pairs}
    store = DiskCorpusStore(INDEX / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    passages = store.passages_batch(passage_ids)
    sources = store.sources_batch(doc_ids)
    invalid = []
    for doc, pid in sorted(all_pairs):
        passage = passages[pid]
        if passage.official_id != doc or not verify(passage, sources[doc]):
            invalid.append({"query_independent_doc_id": doc, "passage_id": pid,
                            "registered_parent": passage.official_id})
    if invalid:
        raise ValueError(f"Source validation failed: {invalid[:3]}")

    key_builder = ScoreCache(SEED_CACHE, contract)
    b0_score: dict[tuple[int, int, str], float | None] = {}
    b0_cache_key: dict[tuple[int, int, str], str] = {}
    with sqlite3.connect(f"file:{SEED_CACHE}?mode=ro", uri=True) as db:
        for qid in query_ids:
            query = queries[qid]
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            for doc, pid in b0_ordered[qid]:
                title = sources[doc].title
                payload = f"{title}\n{passages[pid].text}" if title else passages[pid].text
                input_hash = hashlib.sha256(payload.encode()).hexdigest()
                cache_key = key_builder._key(query, doc, pid, payload)
                b0_cache_key[(qid, doc, pid)] = cache_key
                row = db.execute(
                    "SELECT score,query_sha256,official_id,passage_id,input_sha256 "
                    "FROM scores WHERE cache_key=?", (cache_key,)).fetchone()
                score = None
                if row is not None:
                    expected = (query_hash, doc, pid, input_hash)
                    if row[1:] != expected or not math.isfinite(float(row[0])):
                        raise ValueError(f"Incompatible cached score row: {(qid, doc, pid)}")
                    score = float(row[0])
                b0_score[(qid, doc, pid)] = score
    valid_count = sum(value is not None for value in b0_score.values())
    if valid_count != 1099 or len(b0_score) - valid_count != 101:
        raise ValueError(f"B0 cache counts changed: {valid_count}/{len(b0_score)}")

    alternative_rows: list[dict] = []
    higher_keys = {
        (int(row["query_id"]), int(row["doc_id"]), row["b0_alternative_passage_id"])
        for row in prior_comparison["offline_quality"]
        ["potentially_better_within_a0_parent_by_cached_reranker_only"]
    }
    per_query: list[dict] = []
    review_candidates_by_q: dict[int, dict[str, list[dict]]] = {}
    b1_plan: list[dict] = []
    cached_counterfactual: list[dict] = []

    for qid in query_ids:
        arows = a0_by_query[qid]
        amap = {int(row["doc_id"]): row for row in arows}
        a1map = {int(row["doc_id"]): row for row in a1_by_query[qid]}
        if set(a1map) != set(amap):
            raise ValueError(f"A1 parent breadth differs from A0 for query {qid}")
        arank = {int(row["doc_id"]): int(row["pair_order"]) for row in arows}
        bgroup: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for rank, (doc, pid) in enumerate(b0_ordered[qid], 1):
            bgroup[doc].append((rank, pid))
        aparent, bparent = set(amap), set(bgroup)
        shared = aparent & bparent
        new_parents = bparent - aparent
        lost_parents = aparent - bparent
        q_alts: list[dict] = []
        for doc in sorted(shared):
            arow = amap[doc]
            apid = arow["passage_id"]
            ascore = float(arow["cached_rerank_score"])
            for brank, pid in bgroup[doc]:
                if pid == apid:
                    continue
                score = b0_score[(qid, doc, pid)]
                if score is None:
                    outcome, delta = "MISSING", None
                else:
                    delta = score - ascore
                    outcome = "HIGHER" if delta > 0 else ("LOWER" if delta < 0 else "EQUAL")
                item = {
                    "query_id": qid, "doc_id": doc,
                    "a0_passage_id": apid, "a0_pair_rank": arank[doc],
                    "a0_final_parent_rank": int(p1_by_query[qid][doc]["final_doc_rank"]),
                    "a0_top60": int(p1_by_query[qid][doc]["final_doc_rank"]) <= 60,
                    "a0_cached_score": ascore,
                    "alternative_passage_id": pid, "b0_passage_fusion_rank": brank,
                    "alternative_cached_score": score, "score_delta": delta,
                    "cached_score_comparison": outcome,
                    "in_locked_57_score_higher_set": (qid, doc, pid) in higher_keys,
                    "independent_relevance": "UNKNOWN",
                }
                alternative_rows.append(item)
                q_alts.append(item)

        # B1 primary: preserve A0's 200 parents. Substitute the label-free dense
        # representative only when that exact pair also survived frozen B0 top-200.
        # This is the non-trivial breadth-preserving intersection of prior A1 and B0.
        q_b1 = []
        for doc, arow in amap.items():
            apid = arow["passage_id"]
            a1row = a1map[doc]
            a1pid = a1row["passage_id"]
            if a1pid != apid and any(pid == a1pid for _, pid in bgroup.get(doc, [])):
                chosen_pid = a1pid
                chosen_rank = next(rank for rank, pid in bgroup[doc] if pid == chosen_pid)
                mechanism = "DENSE_REPRESENTATIVE_IF_IN_FROZEN_B0_POOL"
                score = float(a1row["cached_rerank_score"])
                cache_state = "EXACT_CACHED_VALID"
            else:
                chosen_pid = apid
                chosen_rank = next((rank for rank, pid in bgroup.get(doc, []) if pid == apid), None)
                mechanism = "KEEP_A0_PASSAGE"
                score = float(arow["cached_rerank_score"])
                cache_state = "EXACT_CACHED_VALID"
            if chosen_pid != apid and b0_score[(qid, doc, chosen_pid)] != score:
                raise ValueError(f"B1/P2-A1 exact score mismatch: {(qid, doc, chosen_pid)}")
            q_b1.append({
                "query_id": qid, "doc_id": doc, "selected_passage_id": chosen_pid,
                "a0_passage_id": apid, "substituted": chosen_pid != apid,
                "selection_mechanism": mechanism, "b0_passage_fusion_rank": chosen_rank,
                "cached_score": score, "score_status": cache_state,
                "a0_cached_score": float(arow["cached_rerank_score"]),
                "a0_final_parent_rank": int(p1_by_query[qid][doc]["final_doc_rank"]),
            })
        if len(q_b1) != 200 or len({row["doc_id"] for row in q_b1}) != 200:
            raise ValueError(f"B1 breadth/budget invariant failed for query {qid}")
        b1_plan.extend(sorted(q_b1, key=lambda row: row["a0_final_parent_rank"]))

        # Exact cached B1 rerank. The tie-break reproduces the locked A0 runtime
        # ordering exactly when B1 scores are replaced by A0 scores.
        trace = read(TRACES / f"{qid}.json")
        parent_retrieval_score = {int(row["official_id"]): float(row["score"])
                                  for row in trace["diagnostics"]["ranked_candidates"]}
        baseline_reproduction = sorted(
            amap, key=lambda doc: (-float(amap[doc]["cached_rerank_score"]),
                                   -parent_retrieval_score[doc], doc))
        if any(int(p1_by_query[qid][doc]["final_doc_rank"]) != rank
               for rank, doc in enumerate(baseline_reproduction, 1)):
            raise ValueError(f"A0 runtime rank reproduction failed for query {qid}")
        b1_order = sorted(q_b1, key=lambda row: (
            -float(row["cached_score"]), -parent_retrieval_score[row["doc_id"]], row["doc_id"]))
        b1_rank = {row["doc_id"]: rank for rank, row in enumerate(b1_order, 1)}
        a0_top_exact = {doc for doc in amap if int(p1_by_query[qid][doc]["final_doc_rank"]) <= 60}
        b1_top_exact = {doc for doc, rank in b1_rank.items() if rank <= 60}
        b1_changed_passages_in_top60 = sorted(
            row["doc_id"] for row in q_b1 if row["doc_id"] in b1_top_exact and row["substituted"])

        # Cached-only counterfactual: keep all A0 parents and replace an A0 passage
        # only where the best valid cached alternative has a strictly higher score.
        base_order = sorted(aparent, key=lambda doc: int(p1_by_query[qid][doc]["final_doc_rank"]))
        cf_rows = []
        for doc in base_order:
            arow = amap[doc]
            apid, ascore = arow["passage_id"], float(arow["cached_rerank_score"])
            candidates = [(b0_score[(qid, doc, pid)], rank, pid)
                          for rank, pid in bgroup.get(doc, []) if pid != apid
                          and b0_score[(qid, doc, pid)] is not None]
            best = max(candidates, default=None, key=lambda value: (value[0], -value[1], value[2]))
            if best is not None and best[0] > ascore:
                score, _, pid = best
            else:
                score, pid = ascore, apid
            cf_rows.append({"doc_id": doc, "passage_id": pid, "score": score,
                            "baseline_rank": int(p1_by_query[qid][doc]["final_doc_rank"]),
                            "passage_changed": pid != apid})
        cf_order = sorted(cf_rows, key=lambda row: (-row["score"], row["baseline_rank"], row["doc_id"]))
        cf_rank = {row["doc_id"]: rank for rank, row in enumerate(cf_order, 1)}
        base_top = {doc for doc in aparent if int(p1_by_query[qid][doc]["final_doc_rank"]) <= 60}
        cf_top = {doc for doc, rank in cf_rank.items() if rank <= 60}
        cached_counterfactual.append({
            "query_id": qid,
            "interpretation": "cached-score diagnostic only; not an independent relevance result or B1 runtime replay",
            "passage_replacements": sum(row["passage_changed"] for row in cf_rows),
            "baseline_top60_parents": sorted(base_top),
            "counterfactual_top60_parents": sorted(cf_top),
            "promoted_parents": sorted(cf_top - base_top),
            "demoted_parents": sorted(base_top - cf_top),
            "top60_parents_with_changed_passage": sorted(
                row["doc_id"] for row in cf_rows if row["doc_id"] in cf_top
                and row["passage_changed"]),
        })

        new_rows = []
        for doc in sorted(new_parents):
            rank, pid = min(bgroup[doc])
            new_rows.append({"query_id": qid, "doc_id": doc, "passage_id": pid,
                             "b0_passage_fusion_rank": rank,
                             "cached_score": b0_score[(qid, doc, pid)]})
        lost_rows = [{"query_id": qid, "doc_id": doc,
                      "passage_id": amap[doc]["passage_id"],
                      "a0_pair_rank": arank[doc],
                      "a0_final_parent_rank": int(p1_by_query[qid][doc]["final_doc_rank"]),
                      "a0_cached_score": float(amap[doc]["cached_rerank_score"])}
                     for doc in sorted(lost_parents)]
        review_candidates_by_q[qid] = {"alternatives": q_alts, "new": new_rows,
                                      "lost": lost_rows}
        b1_valid = sum(row["cached_score"] is not None for row in q_b1)
        top60_lost = [row for row in lost_rows if row["a0_final_parent_rank"] <= 60]
        cutoff = min(float(row["rerank_score"]) for row in p1_by_query[qid].values()
                     if int(row["final_doc_rank"]) <= 60)
        per_query.append({
            "query_id": qid, "shared_parent_alternatives": len(q_alts),
            "alternative_score_outcomes": dict(Counter(row["cached_score_comparison"] for row in q_alts)),
            "locked_score_higher_cases": sum(row["in_locked_57_score_higher_set"] for row in q_alts),
            "locked_score_higher_cases_on_a0_top60_parent": sum(
                row["in_locked_57_score_higher_set"] and row["a0_top60"] for row in q_alts),
            "new_b0_parents": len(new_rows), "a0_only_parents": len(lost_rows),
            "a0_only_parents_in_baseline_top60": len(top60_lost),
            "a0_only_parent_ids_in_baseline_top60": [row["doc_id"] for row in top60_lost],
            "a0_rank60_score_cutoff": cutoff,
            "new_b0_parent_best_passages": {
                "cached_valid": sum(row["cached_score"] is not None for row in new_rows),
                "missing": sum(row["cached_score"] is None for row in new_rows),
                "cached_score_above_a0_rank60_cutoff": sum(
                    row["cached_score"] is not None and row["cached_score"] > cutoff for row in new_rows),
            },
            "b1_budget_pairs": len(q_b1), "b1_unique_parents": len({row["doc_id"] for row in q_b1}),
            "b1_passage_substitutions": sum(row["substituted"] for row in q_b1),
            "b1_selected_scores_valid": b1_valid,
            "b1_selected_scores_missing": len(q_b1) - b1_valid,
            "b1_exact_top60_status": "READY" if b1_valid == 200 else "BLOCKED_MISSING_SCORES",
            "b1_top60_parent_promotions": sorted(b1_top_exact - a0_top_exact),
            "b1_top60_parent_demotions": sorted(a0_top_exact - b1_top_exact),
            "b1_top60_changed_passage_parent_ids": b1_changed_passages_in_top60,
            "b1_top60_changed_passages": len(b1_changed_passages_in_top60),
        })

    if len(alternative_rows) != 322:
        raise ValueError(f"Expected 322 shared-parent alternatives, got {len(alternative_rows)}")
    locked = [row for row in alternative_rows if row["in_locked_57_score_higher_set"]]
    if len(locked) != 57 or set((row["query_id"], row["doc_id"], row["alternative_passage_id"])
                                for row in locked) != higher_keys:
        raise ValueError("Locked 57 score-higher cases do not reproduce prior audit")

    # Construct a deterministic, blinded packet: all 57 hypotheses, 57 query-matched
    # controls from other alternatives, and all 27 new-parent vs lost-parent pairs.
    packet_internal: list[dict] = []
    for qid in query_ids:
        qalts = review_candidates_by_q[qid]["alternatives"]
        qhigher = [row for row in qalts if row["in_locked_57_score_higher_set"]]
        packet_internal.extend({"stratum": "LOCKED_SCORE_HIGHER", "query_id": qid,
                                "left_arm": "A0", "left_doc": row["doc_id"],
                                "left_pid": row["a0_passage_id"], "right_arm": "B0_ALT",
                                "right_doc": row["doc_id"], "right_pid": row["alternative_passage_id"],
                                "metadata": row} for row in qhigher)
        higher_parents = {row["doc_id"] for row in qhigher}
        control_pool = [row for row in qalts if row["doc_id"] not in higher_parents]
        control_pool.sort(key=lambda row: stable_key("control", qid, row["doc_id"],
                                                    row["alternative_passage_id"]))
        controls = control_pool[:len(qhigher)]
        if len(controls) != len(qhigher):
            raise ValueError(f"Insufficient query-matched controls for {qid}")
        packet_internal.extend({"stratum": "QUERY_MATCHED_CONTROL", "query_id": qid,
                                "left_arm": "A0", "left_doc": row["doc_id"],
                                "left_pid": row["a0_passage_id"], "right_arm": "B0_ALT",
                                "right_doc": row["doc_id"], "right_pid": row["alternative_passage_id"],
                                "metadata": row} for row in controls)

        lost_available = review_candidates_by_q[qid]["lost"][:]
        for new in sorted(review_candidates_by_q[qid]["new"], key=lambda row: row["b0_passage_fusion_rank"]):
            lost = min(lost_available, key=lambda row: (
                abs(row["a0_pair_rank"] - new["b0_passage_fusion_rank"]),
                row["a0_pair_rank"], row["doc_id"]))
            lost_available.remove(lost)
            packet_internal.append({
                "stratum": "NEW_PARENT_VS_A0_ONLY_PARENT", "query_id": qid,
                "left_arm": "A0_ONLY_PARENT", "left_doc": lost["doc_id"],
                "left_pid": lost["passage_id"], "right_arm": "B0_NEW_PARENT",
                "right_doc": new["doc_id"], "right_pid": new["passage_id"],
                "metadata": {"new": new, "lost": lost},
            })
    if Counter(row["stratum"] for row in packet_internal) != {
            "LOCKED_SCORE_HIGHER": 57, "QUERY_MATCHED_CONTROL": 57,
            "NEW_PARENT_VS_A0_ONLY_PARENT": 27}:
        raise ValueError("Review packet stratum counts failed")

    review_rows: list[dict] = []
    hidden_rows: list[dict] = []
    for item in packet_internal:
        raw_id = stable_key("b1-review-v1", item["stratum"], item["query_id"],
                            item["left_doc"], item["left_pid"], item["right_doc"], item["right_pid"])
        pair_id = f"PAIR-{raw_id[:16].upper()}"
        swap = int(stable_key("orientation", raw_id), 16) % 2 == 1
        sides = [
            {"arm": item["left_arm"], "doc": item["left_doc"], "pid": item["left_pid"]},
            {"arm": item["right_arm"], "doc": item["right_doc"], "pid": item["right_pid"]},
        ]
        if swap:
            sides.reverse()
        left, right = sides
        review_rows.append({
            "pair_id": pair_id, "question": queries[item["query_id"]],
            "title_left": sources[left["doc"]].title or "",
            "text_left": passages[left["pid"]].text,
            "title_right": sources[right["doc"]].title or "",
            "text_right": passages[right["pid"]].text,
            "relevance_left_0_1_2": "", "relevance_right_0_1_2": "",
            "patient_match_left_0_1": "", "patient_match_right_0_1": "",
            "numeric_support_left_0_1": "", "numeric_support_right_0_1": "",
            "contradiction_left_0_1": "", "contradiction_right_0_1": "",
            "pair_preference_left_right_tie": "", "reviewer_notes": "",
        })
        hidden_rows.append({
            "pair_id": pair_id, "stratum": item["stratum"], "query_id": item["query_id"],
            "left": left, "right": right, "orientation_swapped": swap,
            "sampling_metadata": item["metadata"],
        })
    review_rows.sort(key=lambda row: stable_key("packet-order", row["pair_id"]))
    hidden_rows.sort(key=lambda row: stable_key("packet-order", row["pair_id"]))

    b1_counts = Counter()
    for row in b1_plan:
        b1_counts["pairs"] += 1
        b1_counts["substitutions"] += bool(row["substituted"])
        b1_counts["scores_valid"] += row["cached_score"] is not None
        b1_counts["scores_missing"] += row["cached_score"] is None
        if row["substituted"] and row["cached_score"] is not None:
            if row["cached_score"] > row["a0_cached_score"]:
                b1_counts["substitution_score_higher"] += 1
            elif row["cached_score"] < row["a0_cached_score"]:
                b1_counts["substitution_score_lower"] += 1
            else:
                b1_counts["substitution_score_equal"] += 1

    top60_report = {
        "format": "medrag-v4-b0-b1-top60-impact-v1", "status": "BLOCKED_BY_EVALUATION",
        "scope": "six frozen pilot queries; cached artifacts only",
        "query_ids": query_ids,
        "observed": {
            "shared_parent_alternative_passages": len(alternative_rows),
            "locked_score_higher_hypotheses": 57,
            "locked_score_higher_on_a0_top60_parents": sum(row["a0_top60"] for row in locked),
            "b0_new_parents": sum(row["new_b0_parents"] for row in per_query),
            "a0_only_parents": sum(row["a0_only_parents"] for row in per_query),
            "a0_only_parents_in_baseline_top60": sum(
                row["a0_only_parents_in_baseline_top60"] for row in per_query),
        },
        "b1_budget_matched": {
            "policy": "keep A0 200-parent set; use dense representative only if the exact pair survived frozen B0 top-200; otherwise keep A0 passage",
            "pairs": b1_counts["pairs"], "unique_parents": 1200,
            "passage_substitutions": b1_counts["substitutions"],
            "selected_scores_valid": b1_counts["scores_valid"],
            "selected_scores_missing": b1_counts["scores_missing"],
            "substitution_score_higher": b1_counts["substitution_score_higher"],
            "substitution_score_lower": b1_counts["substitution_score_lower"],
            "substitution_score_equal": b1_counts["substitution_score_equal"],
            "exact_top60_status": "BLOCKED_MISSING_SCORES" if b1_counts["scores_missing"] else "READY",
            "top60_parent_promotions": sum(len(row["b1_top60_parent_promotions"]) for row in per_query),
            "top60_parent_demotions": sum(len(row["b1_top60_parent_demotions"]) for row in per_query),
            "top60_changed_passages": sum(row["b1_top60_changed_passages"] for row in per_query),
            "warning": "score direction is not an independent relevance judgment",
        },
        "pure_b0_top60_status": "BLOCKED_MISSING_101_EXACT_SCORES",
        "per_query": per_query,
        "cached_score_counterfactual": {
            "status": "DIAGNOSTIC_ONLY_NOT_A_RUNTIME_POLICY",
            "method": "keep A0 parents; apply only strictly higher valid cached alternative scores; rerank with frozen A0 rank as tie-break",
            "queries": cached_counterfactual,
            "aggregate_parent_promotions": sum(len(row["promoted_parents"]) for row in cached_counterfactual),
            "aggregate_parent_demotions": sum(len(row["demoted_parents"]) for row in cached_counterfactual),
            "aggregate_top60_changed_passages": sum(
                len(row["top60_parents_with_changed_passage"]) for row in cached_counterfactual),
            "limitations": [
                "uses cached reranker score as a sensitivity analysis, not relevance truth",
                "does not impute missing scores",
                "is not the label-free B1 selection rule and must not be promoted",
            ],
        },
        "decision": "BLOCKED_BY_EVALUATION",
        "decision_basis": "B1 preserves breadth and is fully cached, but no independent paired judgments exist yet",
        "execution": {"retrieval_calls": 0, "model_calls": 0, "gpu_used": False,
                      "full_index_loaded": False, "targeted_corpus_rows": len(all_pairs)},
    }

    missing_shards = prior_readiness["missing_score_manifest"]["shards_sha256"]
    for qid, expected_hash in missing_shards.items():
        if sha(B0_NAMESPACE / "missing_pairs" / f"{qid}.json") != expected_hash:
            raise ValueError(f"B0 missing-score shard hash mismatch: {qid}")
    missing_manifest = {
        "format": "medrag-v4-b1-missing-score-reference-v1",
        "status": "PREPARED_NOT_EXECUTED", "total_missing_b0_scores": 101,
        "score_contract_fingerprint": CONTRACT_FINGERPRINT,
        "source_readiness": str(B0_READINESS.relative_to(ROOT)),
        "source_readiness_sha256": sha(B0_READINESS),
        "source_shard_directory": str((B0_NAMESPACE / "missing_pairs").relative_to(ROOT)),
        "source_shards_sha256": missing_shards,
        "prepared_gpu_job": str((B0_NAMESPACE / "gpu_scoring_job.json").relative_to(ROOT)),
        "prepared_gpu_job_sha256": sha(B0_NAMESPACE / "gpu_scoring_job.json"),
        "executed": False,
        "note": "References all 101 B0 missing pairs. The breadth-preserving B1 intersection selects none of them; pure B0 still requires all 101.",
    }

    output.mkdir(parents=True)
    atomic_json(output / "alternative_analysis.json", {
        "format": "medrag-v4-b0-shared-parent-alternatives-v1",
        "count": len(alternative_rows), "rows": alternative_rows})
    atomic_json(output / "b1_pair_plan.json", {
        "format": "medrag-v4-b1-budget-matched-pair-plan-v1",
        "policy_frozen_for_pilot": True, "pairs_per_query": 200,
        "selection_uses_labels": False, "historical_positive_features": False,
        "holdout_label_features": False, "rows": b1_plan})
    atomic_json(output / "review_item_map.json", {
        "format": "medrag-v4-b1-blinded-review-hidden-map-v1",
        "warning": "Do not provide this file to reviewers before adjudication is locked.",
        "rows": hidden_rows})
    atomic_json(output / "missing_score_manifest.json", missing_manifest)
    atomic_json(top60_json, top60_report)
    review_fields = list(review_rows[0])
    atomic_csv(review_csv, review_fields, review_rows)
    review_manifest = {
        "format": "medrag-v4-b1-blinded-review-packet-v1", "status": "READY_FOR_REVIEW",
        "packet": str(review_csv.relative_to(ROOT)), "packet_sha256": sha(review_csv),
        "row_count": len(review_rows),
        "strata": dict(Counter(row["stratum"] for row in hidden_rows)),
        "orientation": dict(Counter("swapped" if row["orientation_swapped"] else "not_swapped"
                                    for row in hidden_rows)),
        "blinding": {"scores_exposed": False, "arm_origin_exposed": False,
                     "document_ids_exposed": False, "passage_ids_exposed": False,
                     "ranks_exposed": False},
        "sampling_locked_before_labels": True,
        "independent_labels_available_at_creation": False,
    }
    atomic_json(output / "review_packet_manifest.json", review_manifest)
    artifacts = [output / "alternative_analysis.json", output / "b1_pair_plan.json",
                 output / "review_item_map.json", output / "missing_score_manifest.json",
                 output / "review_packet_manifest.json", review_csv, top60_json]
    atomic_json(output / "manifest.json", {
        "format": "medrag-v4-b1-feasibility-audit-artifacts-v1", "status": "PASS",
        "inputs_sha256": {
            str(A0_CONFIG.relative_to(ROOT)): sha(A0_CONFIG),
            str(P2_PAIRS.relative_to(ROOT)): sha(P2_PAIRS),
            str((P4 / "candidate_comparison.json").relative_to(ROOT)): sha(P4 / "candidate_comparison.json"),
            str(P1_POOL.relative_to(ROOT)): sha(P1_POOL),
            str(B0_COMPARISON.relative_to(ROOT)): sha(B0_COMPARISON),
            str(B0_READINESS.relative_to(ROOT)): sha(B0_READINESS),
            str(SEED_CACHE.relative_to(ROOT)): sha(SEED_CACHE),
        },
        "outputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in artifacts},
        "execution": top60_report["execution"],
    })
    print(json.dumps({
        "status": "PASS", "alternatives": len(alternative_rows),
        "review_rows": len(review_rows), "review_strata": review_manifest["strata"],
        "b1_budget_matched": top60_report["b1_budget_matched"],
        "a0_only_top60": top60_report["observed"]["a0_only_parents_in_baseline_top60"],
        "decision": top60_report["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path,
                        default=ROOT / "B0_BLINDED_PAIR_REVIEW.csv")
    parser.add_argument("--top60-json", type=Path,
                        default=ROOT / "B0_TOP60_IMPACT_REPORT.json")
    args = parser.parse_args()
    main(args.output, args.review_csv, args.top60_json)
