"""Prepare exact A0-vs-B1 final-output deltas and a blinded review packet.

Consumes frozen candidate traces and cached pair scores only. No retrieval or model
inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.medrag.adapters.validation import validate_selected
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results


QUERY_IDS = [53, 130, 380, 754, 771, 1144]
CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
CONFIG_SHA256 = "bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03"
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
P1_POOL = ROOT / "outputs/v4/p1_chunk_allocation_20261021_v2/scored_parent_pool.jsonl"
A0_REFERENCE = ROOT / "outputs/v4/adjacent_context_replay_d200_20261021/selected_results.jsonl"
B1_PLAN = ROOT / "outputs/v4/b1_feasibility_audit_20261022/b1_pair_plan.json"
B1_TOP60 = ROOT / "B0_TOP60_IMPACT_REPORT.json"
MIGRATION = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_id(*parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def grounded_evidence(document: dict, source) -> dict:
    evidence = document["evidence"][0]
    seed = evidence["seed_evidence"]
    checks = {
        "source_sha256_matches": evidence["source_sha256"] == source.text_sha256,
        "span_matches_source": source.text[evidence["start"]:evidence["end"]] == evidence["text"],
        "seed_span_matches_source": source.text[seed["start"]:seed["end"]] == seed["text"],
        "seed_id_matches": seed["passage_id"] == evidence["seed_passage_id"],
        "context_has_registered_passages": bool(evidence["context_passage_ids"]),
    }
    return {"valid": all(checks.values()), "checks": checks,
            "source_sha256": source.text_sha256, "source_url": source.url,
            "source_title": source.title, "context_start": evidence["start"],
            "context_end": evidence["end"],
            "context_passage_ids": evidence["context_passage_ids"]}


def system_row(document: dict, rank: int, source) -> dict:
    evidence = document["evidence"][0] if document["evidence"] else document["document_evidence"]
    seed = evidence["seed_evidence"] if "seed_evidence" in evidence else evidence
    grounding = grounded_evidence(document, source) if document["evidence"] else None
    return {
        "in_top60": bool(document["evidence"]), "rank": rank,
        "doc_id": int(document["official_id"]),
        "seed_passage_id": evidence.get("seed_passage_id", evidence["passage_id"]),
        "context_passage_id": evidence["passage_id"] if document["evidence"] else None,
        "seed_text": seed["text"], "seed_start": seed["start"], "seed_end": seed["end"],
        "seed_source_grounded": source.text[seed["start"]:seed["end"]] == seed["text"],
        "reranker_score": float(document["rerank_score"]),
        "source_title": source.title, "source_url": source.url,
        "context_text": evidence["text"] if document["evidence"] else None,
        "grounding": grounding,
    }


def main(output: Path, delta_path: Path, review_path: Path) -> None:
    output, delta_path, review_path = output.resolve(), delta_path.resolve(), review_path.resolve()
    if output.exists() or delta_path.exists() or review_path.exists():
        raise FileExistsError("Final-output evaluation outputs must be fresh")
    if sha(CONFIG) != CONFIG_SHA256:
        raise ValueError("Frozen A0 config hash mismatch")
    config = load_config(CONFIG)
    if (config["retrieval"]["rerank_candidate_limit"] != 200
            or config["selection"]["max_docs"] != 200
            or config["selection"]["max_chunks"] != 60):
        raise ValueError("Expected frozen 200/200/60 contract")
    policy = SelectionPolicy(**config["selection"])
    index = ROOT / config["index_namespace"]
    index_manifest = read(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)

    migration = read(MIGRATION)
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query hash mismatch")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist() if int(row["id"]) in QUERY_IDS}

    pool_by_q: dict[int, list[dict]] = defaultdict(list)
    with P1_POOL.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["query_id"]) in QUERY_IDS:
                pool_by_q[int(row["query_id"])].append(row)
    for qid in QUERY_IDS:
        pool_by_q[qid].sort(key=lambda row: int(row["fused_rank"]))
        if len(pool_by_q[qid]) != 200:
            raise ValueError(f"A0 pool is not 200 parents: {qid}")

    b1_plan_raw = read(B1_PLAN)
    if (b1_plan_raw["pairs_per_query"] != 200 or not b1_plan_raw["policy_frozen_for_pilot"]
            or b1_plan_raw["selection_uses_labels"]
            or b1_plan_raw["historical_positive_features"]
            or b1_plan_raw["holdout_label_features"]):
        raise ValueError("B1 plan policy contract mismatch")
    b1_by_q: dict[int, list[dict]] = defaultdict(list)
    for row in b1_plan_raw["rows"]:
        b1_by_q[int(row["query_id"])].append(row)
    top60_prior = read(B1_TOP60)["b1_budget_matched"]
    if (top60_prior["passage_substitutions"] != 37
            or top60_prior["top60_changed_passages"] != 18
            or top60_prior["top60_parent_promotions"] != 7
            or top60_prior["top60_parent_demotions"] != 7):
        raise ValueError("Prior B1 top-60 summary mismatch")

    a0_reference = {int(row["id"]): row["selected"] for row in map(
        json.loads, A0_REFERENCE.open(encoding="utf-8")) if int(row["id"]) in QUERY_IDS}
    replay: dict[int, dict[str, list[dict]]] = {}
    context_diagnostics = []
    for qid in QUERY_IDS:
        checkpoint = read(PILOT / "queries" / f"{qid}.json")
        if checkpoint["query_sha256"] != hashlib.sha256(queries[qid].encode()).hexdigest():
            raise ValueError(f"Query/trace mismatch: {qid}")
        candidate_by_parent = {int(row["official_id"]): row
                               for row in checkpoint["diagnostics"]["ranked_candidates"]}
        if len(b1_by_q[qid]) != 200 or len({int(row["doc_id"]) for row in b1_by_q[qid]}) != 200:
            raise ValueError(f"B1 pair/parent budget mismatch: {qid}")
        selected_pair = {int(row["doc_id"]): row for row in b1_by_q[qid]}
        systems = {}
        for arm in ("A0", "B1"):
            candidates = []
            for parent_row in pool_by_q[qid]:
                doc = int(parent_row["doc_id"])
                candidate = candidate_by_parent[doc]
                if arm == "A0":
                    passage_id = parent_row["passage_id"]
                    score = float(parent_row["rerank_score"])
                else:
                    passage_id = selected_pair[doc]["selected_passage_id"]
                    score = float(selected_pair[doc]["cached_score"])
                evidence = next(item for item in candidate["evidence"]
                                if item["passage_id"] == passage_id)
                candidates.append({
                    "official_id": doc, "score": float(candidate["score"]),
                    "ranks": dict(candidate["ranks"]),
                    "evidence": [{**evidence, "score_status": "reranked",
                                  "rerank_score": score}],
                    "rerank_score": score,
                })
            candidates.sort(key=lambda item: (-item["rerank_score"], -item["score"],
                                              item["official_id"]))
            selected, selection_diag = select_results(
                candidates, policy, reranker_enabled=True, limit=200)
            contextual, context_diag = apply_chunk_context(selected, policy, sources, passages)
            validate_selected([qid], {qid: contextual}, store, policy, result_limit=200,
                              reranker_enabled=True,
                              official_ids={int(row["doc_id"]) for row in pool_by_q[qid]})
            if len(contextual) != 200 or sum(bool(doc["evidence"]) for doc in contextual) != 60:
                raise ValueError(f"Final output cap mismatch: {qid}/{arm}")
            if arm == "A0" and contextual != a0_reference[qid]:
                raise ValueError(f"A0 reconstruction differs from frozen output: {qid}")
            systems[arm] = contextual
            context_diagnostics.append({"query_id": qid, "arm": arm,
                                        "selection": selection_diag, "context": context_diag})
        replay[qid] = systems

    delta_rows = []
    rank_only_rows = []
    review_pairs_internal = []
    per_query = []
    for qid in QUERY_IDS:
        a_docs = {int(doc["official_id"]): (rank, doc)
                  for rank, doc in enumerate(replay[qid]["A0"], 1)}
        b_docs = {int(doc["official_id"]): (rank, doc)
                  for rank, doc in enumerate(replay[qid]["B1"], 1)}
        a_top = {doc for doc, (_, row) in a_docs.items() if row["evidence"]}
        b_top = {doc for doc, (_, row) in b_docs.items() if row["evidence"]}
        promoted, demoted = sorted(b_top - a_top), sorted(a_top - b_top)
        content_rows = []
        for doc in sorted(a_top | b_top):
            arank, adoc = a_docs[doc]
            brank, bdoc = b_docs[doc]
            arow = system_row(adoc, arank, sources[doc])
            brow = system_row(bdoc, brank, sources[doc])
            seed_changed = arow["seed_passage_id"] != brow["seed_passage_id"]
            context_changed = (arow["context_text"] != brow["context_text"]
                               if arow["in_top60"] and brow["in_top60"] else None)
            if doc in promoted:
                change_type = "DOCUMENT_PROMOTED_IN_B1"
            elif doc in demoted:
                change_type = "DOCUMENT_DEMOTED_FROM_B1"
            elif seed_changed or context_changed:
                change_type = "PASSAGE_CONTENT_CHANGED"
            elif arank != brank:
                change_type = "RANK_ONLY"
            else:
                continue
            opaque = "DELTA-" + stable_id("final-delta-v1", qid, doc, change_type)[:16].upper()
            item = {
                "query_id": qid, "opaque_delta_id": opaque, "change_type": change_type,
                "a0": arow, "b1": brow,
                "rank_delta_b1_minus_a0": brank - arank,
                "score_delta_b1_minus_a0": brow["reranker_score"] - arow["reranker_score"],
                "seed_passage_changed": seed_changed,
                "exported_context_changed": context_changed,
            }
            if change_type == "RANK_ONLY":
                rank_only_rows.append(item)
            else:
                delta_rows.append(item)
            if change_type == "PASSAGE_CONTENT_CHANGED":
                content_rows.append(item)
                review_pairs_internal.append({
                    "pair_type": "SAME_PARENT_PASSAGE_CHANGE", "query_id": qid,
                    "a0_doc": doc, "b1_doc": doc, "a0": arow, "b1": brow,
                })

        promoted_rows = sorted((row for row in delta_rows if row["query_id"] == qid
                                and row["change_type"] == "DOCUMENT_PROMOTED_IN_B1"),
                               key=lambda row: (row["b1"]["rank"], row["b1"]["doc_id"]))
        demoted_rows = sorted((row for row in delta_rows if row["query_id"] == qid
                               and row["change_type"] == "DOCUMENT_DEMOTED_FROM_B1"),
                              key=lambda row: (row["a0"]["rank"], row["a0"]["doc_id"]))
        if len(promoted_rows) != len(demoted_rows):
            raise ValueError(f"Unbalanced top60 membership delta: {qid}")
        for demoted_row, promoted_row in zip(demoted_rows, promoted_rows, strict=True):
            review_pairs_internal.append({
                "pair_type": "TOP60_MEMBERSHIP_SWAP", "query_id": qid,
                "a0_doc": demoted_row["a0"]["doc_id"],
                "b1_doc": promoted_row["b1"]["doc_id"],
                "a0": demoted_row["a0"], "b1": promoted_row["b1"],
            })
        per_query.append({
            "query_id": qid, "passage_content_changes_in_shared_top60": len(content_rows),
            "document_promotions": len(promoted), "document_demotions": len(demoted),
            "rank_only_changes": sum(row["query_id"] == qid for row in rank_only_rows),
            "review_pairs": len(content_rows) + len(promoted),
        })

    if (sum(row["document_promotions"] for row in per_query) != 7
            or sum(row["document_demotions"] for row in per_query) != 7):
        raise ValueError("B1 membership delta does not reproduce 7/7")
    if not all(row["a0"]["grounding"] is None or row["a0"]["grounding"]["valid"]
               for row in delta_rows + rank_only_rows):
        raise ValueError("A0 source grounding failure")
    if not all(row["b1"]["grounding"] is None or row["b1"]["grounding"]["valid"]
               for row in delta_rows + rank_only_rows):
        raise ValueError("B1 source grounding failure")

    review_rows = []
    hidden_map = []
    for item in review_pairs_internal:
        identity = stable_id("b1-final-output-review-v1", item["pair_type"], item["query_id"],
                             item["a0_doc"], item["a0"]["seed_passage_id"],
                             item["b1_doc"], item["b1"]["seed_passage_id"])
        review_id = "REVIEW-" + identity[:16].upper()
        sides = [("A0", item["a0"]), ("B1", item["b1"])]
        swapped = int(stable_id("orientation", identity), 16) % 2 == 1
        if swapped:
            sides.reverse()
        left_arm, left = sides[0]
        right_arm, right = sides[1]
        review_rows.append({
            "review_id": review_id, "question": queries[item["query_id"]],
            "title_left": left["source_title"], "evidence_left": left["context_text"],
            "title_right": right["source_title"], "evidence_right": right["context_text"],
            "direct_answer_left_0_1_2": "", "direct_answer_right_0_1_2": "",
            "clinical_accuracy_left_0_1_2": "", "clinical_accuracy_right_0_1_2": "",
            "patient_consistency_left_yes_no_uncertain": "",
            "patient_consistency_right_yes_no_uncertain": "",
            "numeric_consistency_left_yes_no_na_uncertain": "",
            "numeric_consistency_right_yes_no_na_uncertain": "",
            "negation_correct_left_yes_no_na_uncertain": "",
            "negation_correct_right_yes_no_na_uncertain": "",
            "context_completeness_left_0_1_2": "",
            "context_completeness_right_0_1_2": "",
            "overall_preference_left_right_equal_uncertain": "",
            "reviewer_notes": "", "reviewer_id": "", "review_timestamp_utc": "",
        })
        hidden_map.append({
            "review_id": review_id, "pair_type": item["pair_type"],
            "query_id": item["query_id"], "orientation_swapped": swapped,
            "left_arm": left_arm, "right_arm": right_arm,
            "a0": item["a0"], "b1": item["b1"],
        })
    order = {row["review_id"]: stable_id("packet-order", row["review_id"])
             for row in review_rows}
    review_rows.sort(key=lambda row: order[row["review_id"]])
    hidden_map.sort(key=lambda row: order[row["review_id"]])
    review_link = {}
    for row in hidden_map:
        if row["pair_type"] == "SAME_PARENT_PASSAGE_CHANGE":
            review_link[(row["query_id"], "PASSAGE_CONTENT_CHANGED",
                         row["a0"]["doc_id"])] = row["review_id"]
        else:
            review_link[(row["query_id"], "DOCUMENT_DEMOTED_FROM_B1",
                         row["a0"]["doc_id"])] = row["review_id"]
            review_link[(row["query_id"], "DOCUMENT_PROMOTED_IN_B1",
                         row["b1"]["doc_id"])] = row["review_id"]
    for row in delta_rows:
        doc = (row["b1"]["doc_id"] if row["change_type"] == "DOCUMENT_PROMOTED_IN_B1"
               else row["a0"]["doc_id"])
        row["opaque_review_id"] = review_link[(row["query_id"], row["change_type"], doc)]
    for row in rank_only_rows:
        row["opaque_review_id"] = None
        row["review_exclusion_reason"] = "identical evidence content; relevance cannot differ"

    delta = {
        "format": "medrag-v4-b1-final-output-delta-v1", "status": "PASS",
        "scope": "six development queries; exact cached final-output replay",
        "query_ids": QUERY_IDS,
        "summary": {
            "shared_top60_passage_content_changes": sum(
                row["passage_content_changes_in_shared_top60"] for row in per_query),
            "document_promotions": 7, "document_demotions": 7,
            "rank_only_changes": len(rank_only_rows),
            "review_pairs": len(review_rows),
            "seed_changes_on_b1_top60": sum(
                row["change_type"] in {"PASSAGE_CONTENT_CHANGED", "DOCUMENT_PROMOTED_IN_B1"}
                and row["b1"]["seed_passage_id"] != row["a0"]["seed_passage_id"]
                for row in delta_rows),
            "all_exported_contexts_source_grounded": True,
        },
        "per_query": per_query,
        "content_or_membership_deltas": delta_rows,
        "rank_only_deltas": rank_only_rows,
        "review_pairing": {
            "same_parent": "one A0/B1 pair per shared top60 parent with changed exported context",
            "membership": "within-query deterministic zip of A0-rank-sorted demotions and B1-rank-sorted promotions",
        },
        "execution": {"retrieval_calls": 0, "model_calls": 0, "gpu_used": False,
                      "full_index_loaded": False},
    }

    output.mkdir(parents=True)
    write_json(delta_path, delta)
    write_csv(review_path, review_rows)
    write_json(output / "review_item_map.json", {
        "format": "medrag-v4-b1-final-output-hidden-map-v1",
        "warning": "Do not expose to reviewers before annotations are locked.",
        "rows": hidden_map})
    write_json(output / "context_diagnostics.json", context_diagnostics)
    packet_manifest = {
        "format": "medrag-v4-b1-final-output-review-packet-v1",
        "status": "READY_UNREVIEWED", "review_rows": len(review_rows),
        "pair_types": dict(Counter(row["pair_type"] for row in hidden_map)),
        "orientation": dict(Counter("swapped" if row["orientation_swapped"] else "not_swapped"
                                    for row in hidden_map)),
        "blinding": {"arm_labels": False, "scores": False, "ranks": False,
                     "internal_ids": False, "historical_positive_flags": False},
        "rubric": {
            "direct_answer_0_1_2": "0 none/wrong; 1 partial; 2 direct answer",
            "clinical_accuracy_0_1_2": "0 unsafe/wrong; 1 incomplete/unclear; 2 clinically supported",
            "patient_consistency": "yes/no/uncertain",
            "numeric_consistency": "yes/no/na/uncertain",
            "negation_correct": "yes/no/na/uncertain",
            "context_completeness_0_1_2": "0 insufficient; 1 partial; 2 self-contained",
            "overall_preference": "left/right/equal/uncertain",
        },
        "instructions": [
            "Review only the CSV; do not open review_item_map.json before locking labels.",
            "Judge direct support for the question, not topical similarity alone.",
            "Check patient/population, numbers/units, and negation explicitly.",
            "Use uncertain rather than forcing a preference when evidence is insufficient.",
        ],
        "packet_sha256_before_review": sha(review_path),
        "sampling_locked_before_labels": True,
    }
    write_json(output / "review_packet_manifest.json", packet_manifest)
    artifact_paths = [delta_path, review_path, output / "review_item_map.json",
                      output / "context_diagnostics.json", output / "review_packet_manifest.json"]
    write_json(output / "manifest.json", {
        "format": "medrag-v4-b1-final-output-evaluation-preparation-v1", "status": "PASS",
        "inputs_sha256": {
            str(CONFIG.relative_to(ROOT)): sha(CONFIG),
            str(P1_POOL.relative_to(ROOT)): sha(P1_POOL),
            str(A0_REFERENCE.relative_to(ROOT)): sha(A0_REFERENCE),
            str(B1_PLAN.relative_to(ROOT)): sha(B1_PLAN),
            str(B1_TOP60.relative_to(ROOT)): sha(B1_TOP60),
            str(query_path.relative_to(ROOT)): sha(query_path),
        },
        "outputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in artifact_paths},
        "implementation_sha256": sha(Path(__file__)),
        "execution": delta["execution"],
    })
    print(json.dumps({"status": "PASS", "summary": delta["summary"],
                      "per_query": per_query}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=Path, default=ROOT / "B1_FINAL_OUTPUT_DELTA.json")
    parser.add_argument("--review", type=Path, default=ROOT / "B1_FINAL_OUTPUT_REVIEW.csv")
    args = parser.parse_args()
    main(args.output, args.delta, args.review)
