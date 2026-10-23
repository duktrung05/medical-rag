"""Prepare locked P2 passage-pair arms from P0 traces without retrieval or inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4.audit_evidence_survival import anchor_coverage, normalize, sha
from src.medrag.adapters.validation import validate_selected
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.ranking.score_cache import ScoreCache


QUERY_IDS = [53, 130, 380, 754, 771, 1144]
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
P1 = ROOT / "outputs/v4/p1_chunk_allocation_20261021_v2"
P0 = ROOT / "outputs/v4/p0_positive_path_replay_20261021_v3"
CONFIG = ROOT / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml"
BASELINE = ROOT / "outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip"
TARGET_PARENTS = {(53, 128866), (1144, 142195)}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def passage_choice(evidence: list[dict], parent: int, dense: dict, fused: dict) -> dict:
    """Prefer the strongest dense passage; fall back to fused passage score."""
    dense_items = [e for e in evidence if (parent, e["passage_id"]) in dense]
    candidates = dense_items or evidence
    if dense_items:
        return max(candidates, key=lambda e: (
            float(dense[(parent, e["passage_id"])]["score"]),
            float(fused.get((parent, e["passage_id"]), {}).get("score", 0.0)),
            e["passage_id"]))
    return max(candidates, key=lambda e: (
        float(fused.get((parent, e["passage_id"]), {}).get("score", 0.0)),
        e["passage_id"]))


def main(output: Path, score_cache_path: Path | None = None) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(CONFIG)
    if (config["retrieval"]["rerank_candidate_limit"] != 200
            or config["selection"]["max_docs"] != 200
            or config["selection"]["max_chunks"] != 60):
        raise ValueError("P2 requires locked 200/200/60 budgets")
    policy = SelectionPolicy(**config["selection"])
    index = ROOT / config["index_namespace"]
    index_manifest = read_json(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)
    migration = read_json(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query hash changed")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    p1_validation = read_json(P1 / "validation.json")
    if p1_validation["status"] != "PASS" or not p1_validation["checks"][
            "a0_matches_p0_selected_chunks"]:
        raise ValueError("P1 A0 baseline has not been validated")
    pool = [json.loads(line) for line in (P1 / "scored_parent_pool.jsonl").open(
        encoding="utf-8")]
    pool_by_q = {qid: sorted((row for row in pool if int(row["query_id"]) == qid),
                             key=lambda row: row["fused_rank"]) for qid in QUERY_IDS}
    with zipfile.ZipFile(BASELINE) as archive:
        historical = {int(row["id"]): row for row in json.loads(
            archive.read("predictions.json"))}
    with (P0 / "positive_paths.csv").open(encoding="utf-8") as stream:
        positive_paths = {(int(row["query_id"]), int(row["doc_id"])): row
                          for row in csv.DictReader(stream)}
    cache_path = (score_cache_path or PILOT / "scores.sqlite").resolve()
    cache = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    cache_meta = dict(cache.execute("SELECT key,value FROM meta"))
    cache_contract = json.loads(cache_meta["contract"])
    ScoreCache(cache_path, cache_contract)
    if (cache_contract["model"] != config["reranker"]["model"]
            or cache_contract["revision"] != config["reranker"]["revision"]
            or cache_contract["max_length"] != config["reranker"]["max_length"]
            or cache_contract["corpus_version"] != config["corpus_version"]
            or cache_contract["index_manifest_sha256"] != sha(index / "manifest.json")):
        raise ValueError("Pinned reranker score contract differs from P2 config/index")
    pair_rows = []
    query_summaries = []
    target_context = []
    truncation = []
    cached_replays = []
    replay_chunks = []
    context_reference = {int(row["id"]): row["selected"] for row in map(
        json.loads, (ROOT / "outputs/v4/adjacent_context_replay_d200_20261021"
                    / "selected_results.jsonl").open(encoding="utf-8"))}
    try:
        for qid in QUERY_IDS:
            checkpoint = read_json(PILOT / "queries" / f"{qid}.json")
            query = queries[qid]
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            if checkpoint["query_sha256"] != query_hash:
                raise ValueError(f"Query hash mismatch: {qid}")
            parents = pool_by_q[qid]
            if (len(parents) != 200 or {int(row["fused_rank"]) for row in parents}
                    != set(range(1, 201))):
                raise ValueError(f"P1 scored parent pool invalid: {qid}")
            candidate_by_parent = {int(row["official_id"]): row for row in
                                   checkpoint["diagnostics"]["ranked_candidates"]}
            dense = {(int(hit["official_id"]), hit["passage_id"]): hit
                     for hit in checkpoint["branch_trace"]["branch_hits"]["dense"]}
            fused = {(int(hit["official_id"]), hit["passage_id"]): hit
                     for hit in checkpoint["branch_trace"]["fused"]
                     if hit["passage_id"] is not None}
            baseline_seed = {int(row["doc_id"]): row["passage_id"] for row in parents}
            choices = {}
            alternatives = []
            for row in parents:
                parent = int(row["doc_id"])
                candidates = candidate_by_parent[parent]["evidence"]
                choices[parent] = passage_choice(candidates, parent, dense, fused)
                if int(row["fused_rank"]) > 190:
                    continue
                other = [item for item in candidates
                         if item["passage_id"] != baseline_seed[parent]]
                if not other:
                    continue
                alt = passage_choice(other, parent, dense, fused)
                hit = dense.get((parent, alt["passage_id"]))
                fused_hit = fused.get((parent, alt["passage_id"]))
                alternatives.append({"parent": parent, "passage": alt,
                                     "dense_rank": int(hit["rank"]) if hit else None,
                                     "fused_score": float(fused_hit["score"]) if fused_hit else 0.0,
                                     "fused_parent_rank": int(row["fused_rank"])})
            # Allocate the 10 second pairs using retrieval evidence before rerank.
            alternatives.sort(key=lambda row: (
                row["dense_rank"] if row["dense_rank"] is not None else 10**9,
                -row["fused_score"], row["fused_parent_rank"], row["parent"]))
            second = alternatives[:10]
            if len(second) != 10:
                raise ValueError(f"Fewer than ten A2 alternative passages: {qid}")
            plans = {
                "P2-A0": [(int(row["doc_id"]), baseline_seed[int(row["doc_id"])],
                           "baseline_first") for row in parents],
                "P2-A1": [(int(row["doc_id"]), choices[int(row["doc_id"])]["passage_id"],
                           "dense_representative") for row in parents],
                "P2-A2": ([(int(row["doc_id"]), baseline_seed[int(row["doc_id"])],
                            "baseline_first") for row in parents[:190]]
                           + [(item["parent"], item["passage"]["passage_id"],
                               "second_passage") for item in second]),
            }
            baseline_scores = {int(row["doc_id"]): float(row["rerank_score"])
                               for row in parents}
            dropped = [int(row["doc_id"]) for row in parents[190:]]
            dropped_top60 = [int(row["doc_id"]) for row in parents[190:]
                             if int(row["final_doc_rank"]) <= 60]
            for arm, pairs in plans.items():
                unique_parents = {parent for parent, _, _ in pairs}
                expected = 190 if arm == "P2-A2" else 200
                if len(pairs) != 200 or len(unique_parents) != expected:
                    raise ValueError(f"Invalid pair/parent budget: {qid}/{arm}")
                if len({(parent, pid) for parent, pid, _ in pairs}) != 200:
                    raise ValueError(f"Duplicate pair identity: {qid}/{arm}")
                misses = changes = 0
                for position, (parent, passage_id, pair_role) in enumerate(pairs, 1):
                    evidence = next(item for item in candidate_by_parent[parent]["evidence"]
                                    if item["passage_id"] == passage_id)
                    source = sources[parent]
                    payload = (f"{source.title}\n{evidence['text']}" if source.title
                               else evidence["text"])
                    input_sha = hashlib.sha256(payload.encode()).hexdigest()
                    cached = cache.execute(
                        "SELECT score,input_sha256 FROM scores WHERE query_sha256=? "
                        "AND official_id=? AND passage_id=?",
                        (query_hash, parent, passage_id)).fetchall()
                    if len(cached) > 1 or (cached and cached[0][1] != input_sha):
                        raise ValueError(f"Score cache input mismatch: {qid}/{parent}/{passage_id}")
                    if arm == "P2-A0" and (not cached or cached[0][0] !=
                                           baseline_scores[parent]):
                        raise ValueError(f"A0 cache parity failed: {qid}/{parent}")
                    misses += not bool(cached)
                    changes += pair_role == "dense_representative" and passage_id != baseline_seed[parent]
                    dense_hit = dense.get((parent, passage_id))
                    fused_hit = fused.get((parent, passage_id))
                    pair_rows.append({
                        "query_id": qid, "arm": arm, "pair_order": position,
                        "doc_id": parent, "passage_id": passage_id,
                        "pair_role": pair_role,
                        "fused_parent_rank": next(int(row["fused_rank"]) for row in parents
                                                   if int(row["doc_id"]) == parent),
                        "dense_rank": int(dense_hit["rank"]) if dense_hit else None,
                        "dense_similarity": float(dense_hit["score"]) if dense_hit else None,
                        "fused_passage_score": float(fused_hit["score"]) if fused_hit else None,
                        "cache_hit": bool(cached),
                        "cached_rerank_score": float(cached[0][0]) if cached else None,
                        "input_sha256": input_sha,
                    })
                query_summaries.append({
                    "query_id": qid, "arm": arm, "rerank_pairs": 200,
                    "scored_parents": len(unique_parents),
                    "changed_representatives_vs_A0": changes,
                    "cached_pairs": 200 - misses, "missing_scores": misses,
                    "dropped_parents": dropped if arm == "P2-A2" else [],
                    "dropped_A0_top60_parents": dropped_top60 if arm == "P2-A2" else [],
                })
                if misses == 0:
                    # Replay only complete score sets. This is real cached rerank ordering,
                    # not a proxy score or a label-driven passage choice.
                    scored_by_parent: dict[int, list[dict]] = {}
                    for row in pair_rows:
                        if row["query_id"] != qid or row["arm"] != arm:
                            continue
                        if not row["cache_hit"]:
                            raise ValueError("Incomplete cached pair reached replay")
                        parent = int(row["doc_id"])
                        passage_id = row["passage_id"]
                        evidence = next(item for item in candidate_by_parent[parent]["evidence"]
                                        if item["passage_id"] == passage_id)
                        scored_by_parent.setdefault(parent, []).append({
                            **evidence, "score_status": "reranked",
                            "rerank_score": float(row["cached_rerank_score"]),
                        })
                    candidates = []
                    for parent, evidence in scored_by_parent.items():
                        evidence.sort(key=lambda item: (-item["rerank_score"], item["passage_id"]))
                        original = candidate_by_parent[parent]
                        candidates.append({
                            "official_id": parent,
                            "score": float(original["score"]),
                            "ranks": dict(original["ranks"]),
                            "evidence": evidence,
                            "rerank_score": float(evidence[0]["rerank_score"]),
                        })
                    candidates.sort(key=lambda item: (-item["rerank_score"],
                                                      -item["score"], item["official_id"]))
                    selected, _ = select_results(candidates, policy,
                                                 reranker_enabled=True, limit=200)
                    contextual, _ = apply_chunk_context(selected, policy, sources, passages)
                    validate_selected([qid], {qid: contextual}, store, policy,
                                      result_limit=200, reranker_enabled=True,
                                      official_ids={int(row["doc_id"]) for row in parents})
                    if len(contextual) != expected or sum(
                            bool(doc["evidence"]) for doc in contextual) != 60:
                        raise ValueError(f"Replay output cap mismatch: {qid}/{arm}")
                    if arm == "P2-A0" and contextual != context_reference[qid]:
                        raise ValueError(f"P2-A0 differs from locked P0 output: {qid}")
                    result_by_parent = {int(doc["official_id"]): doc for doc in contextual}
                    positive_values = []
                    for (anchor_qid, anchor_parent), path in positive_paths.items():
                        if anchor_qid != qid:
                            continue
                        anchor = normalize(historical[qid]["relevant_chunks"][int(
                            path["baseline_chunk_order"])]["chunk_text"])
                        texts = [item["text"] for item in result_by_parent.get(
                            anchor_parent, {}).get("evidence", [])]
                        positive_values.append({
                            "doc_id": anchor_parent,
                            "exact_context": any(anchor in normalize(text) for text in texts),
                            "lcs_proxy": anchor_coverage(anchor, texts),
                            "selected_parent": anchor_parent in result_by_parent,
                            "exported_chunk": bool(texts),
                        })
                    cached_replays.append({
                        "query_id": qid, "arm": arm,
                        "scored_parents": expected, "rerank_pairs": 200,
                        "relevant_docs": len(contextual), "chunks": 60,
                        "historical_anchors": len(positive_values),
                        "exact_context_anchors": sum(item["exact_context"]
                                                     for item in positive_values),
                        "mean_lcs_proxy": sum(item["lcs_proxy"] for item in positive_values)
                        / len(positive_values),
                        "positive_details": positive_values,
                    })
                    for document in contextual:
                        if not document["evidence"]:
                            continue
                        evidence = document["evidence"][0]
                        replay_chunks.append({
                            "query_id": qid, "arm": arm,
                            "doc_id": int(document["official_id"]),
                            "seed_passage_id": evidence.get("seed_passage_id",
                                                            evidence["passage_id"]),
                            "rerank_score": evidence["rerank_score"],
                            "context_text": evidence["text"],
                        })

            token_rows = checkpoint["diagnostics"]["reranker_tokens"]
            truncation_row = {"query_id": qid, "rerank_pairs": 200,
                              "max_length": token_rows["max_length"],
                              "truncated_pairs": token_rows["truncated_pairs"]}
            if qid == 130:
                target = 2968260
                pair_position = next(position for position, row in enumerate(
                    checkpoint["pair_trace"], 1) if int(row["official_id"]) == target)
                target_text = next(item["text"] for item in
                                   candidate_by_parent[target]["evidence"]
                                   if item["passage_id"] == baseline_seed[target])
                phrases = ["小儿总是喜欢", "新生儿泪囊炎", "不是", "可以滴眼药水", "用药 药品库"]
                truncation_row["target_130_2968260"] = {
                    "pair_tokens": token_rows["pair_lengths"][pair_position - 1],
                    "passage_characters": len(target_text),
                    "phrase_start_character_offsets": {
                        phrase: target_text.find(phrase) for phrase in phrases},
                    "exact_token_cutoff_known": False,
                    "reason": "Pinned reranker tokenizer snapshot is unavailable locally; character positions cannot establish the token boundary",
                }
            truncation.append(truncation_row)
            for parent in [doc for target_q, doc in TARGET_PARENTS if target_q == qid]:
                positive = positive_paths[(qid, parent)]
                anchor = normalize(historical[qid]["relevant_chunks"][int(
                    positive["baseline_chunk_order"])]["chunk_text"])
                current_score = baseline_scores[parent]
                for evidence in candidate_by_parent[parent]["evidence"]:
                    passage_id = evidence["passage_id"]
                    doc = {"official_id": parent, "evidence": [{
                        **evidence, "score_status": "reranked", "rerank_score": 0.0}]}
                    contextual, _ = apply_chunk_context([doc], policy, sources, passages)
                    context = normalize(contextual[0]["evidence"][0]["text"])
                    source = sources[parent]
                    payload = f"{source.title}\n{evidence['text']}" if source.title else evidence["text"]
                    cached = cache.execute("SELECT score FROM scores WHERE query_sha256=? "
                                           "AND official_id=? AND passage_id=? AND input_sha256=?",
                                           (query_hash, parent, passage_id,
                                            hashlib.sha256(payload.encode()).hexdigest())).fetchone()
                    if cached:
                        counterfactual_scores = dict(baseline_scores)
                        counterfactual_scores[parent] = float(cached[0])
                        rank = sorted(counterfactual_scores,
                                      key=lambda doc_id: (-counterfactual_scores[doc_id],
                                                          doc_id)).index(parent) + 1
                    else:
                        rank = None
                    target_context.append({
                        "query_id": qid, "doc_id": parent,
                        "passage_id": passage_id,
                        "is_A0_seed": passage_id == baseline_seed[parent],
                        "chosen_by_A1": passage_id == choices[parent]["passage_id"],
                        "chosen_for_A2_second_pair": any(
                            item["parent"] == parent and item["passage"]["passage_id"] == passage_id
                            for item in second),
                        "dense_rank": (int(dense[(parent, passage_id)]["rank"])
                                       if (parent, passage_id) in dense else None),
                        "fused_passage_score": (float(fused[(parent, passage_id)]["score"])
                                                if (parent, passage_id) in fused else None),
                        "cached_rerank_score": float(cached[0]) if cached else None,
                        "context_contains_historical_anchor": anchor in context,
                        "context_lcs_proxy": anchor_coverage(anchor, [context]),
                        "counterfactual_parent_rank_if_sole_swap": rank,
                    })
    finally:
        cache.close()
    output.mkdir(parents=True)
    with (output / "pair_allocation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    dump_json(output / "arm_pair_summary.json", query_summaries)
    dump_json(output / "target_passage_contexts.json", target_context)
    dump_json(output / "truncation.json", truncation)
    dump_json(output / "cached_complete_replay.json", cached_replays)
    with (output / "replay_selected_chunks.jsonl").open("w", encoding="utf-8") as stream:
        for row in replay_chunks:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    misses = [row for row in pair_rows if not row["cache_hit"]]
    dump_json(output / "missing_scores.json", misses)
    all_scores_present = not misses
    validation = {
        "status": "PASS_FULL_REPLAY" if all_scores_present else "PASS_PAIR_PLAN_ONLY",
        "a0_cache_and_pair_parity": True,
        "all_arms_exactly_200_pairs_per_query": True,
        "a0_a1_200_scored_parents_per_query": True,
        "a2_190_parents_plus_10_second_pairs_per_query": True,
        "all_cached_scores_match_exact_query_parent_passage_input_hash": True,
        "pinned_reranker_model_revision_and_index_contract_verified": True,
        "retrieval_calls": 0,
        "model_calls": 0,
        "full_A1_A2_output_metrics_available": all_scores_present,
        "complete_cached_query_arm_replays": len(cached_replays),
        "reason": ("All pair scores available from the pinned P2 score cache"
                   if all_scores_present else "Some selected alternate passages lack cached scores"),
    }
    dump_json(output / "validation.json", validation)
    names = ["pair_allocation.csv", "arm_pair_summary.json", "target_passage_contexts.json",
             "truncation.json", "cached_complete_replay.json", "replay_selected_chunks.jsonl",
             "missing_scores.json", "validation.json"]
    manifest = {"format": "v4-p2-passage-pair-audit-v1", "status": "pair_plan_complete",
                "query_ids": QUERY_IDS, "model_calls": 0, "retrieval_calls": 0,
                "score_cache_path": str(cache_path),
                "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                 [P1 / "scored_parent_pool.jsonl", P1 / "validation.json",
                                  P0 / "positive_paths.csv", cache_path, CONFIG,
                                  BASELINE, query_path, index / "manifest.json",
                                  *[PILOT / "queries" / f"{qid}.json" for qid in QUERY_IDS]]},
                "artifact_sha256": {name: sha(output / name) for name in names},
                "implementation_sha256": sha(Path(__file__))}
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "arm_totals": {arm: {"pairs": sum(row["rerank_pairs"] for row in query_summaries
                                                            if row["arm"] == arm),
                                           "cache_misses": sum(row["missing_scores"] for row in query_summaries
                                                               if row["arm"] == arm)}
                                     for arm in ("P2-A0", "P2-A1", "P2-A2")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path,
                        help="Pinned-contract cache containing P2 alternate pair scores")
    args = parser.parse_args()
    main(args.output, args.score_cache)
