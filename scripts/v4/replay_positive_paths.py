"""Replay each historical positive through A0 and parent-first V4 stages on CPU.

The historical positive chunks are evaluation anchors, not official gold. This
script never calls retrieval or a model; it only verifies and replays locked
artifacts, then computes exact-anchor oracle ceilings for allocation diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4.audit_evidence_survival import anchor_coverage, normalize, sha
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.context import apply_chunk_context, derive_adjacent_evidence
from src.medrag.evidence.selector import SelectionPolicy, select_results


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty positive-path table")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parent_branch_hits(branch: list[dict], parent: int) -> list[dict]:
    return [item for item in branch if int(item["official_id"]) == parent]


def exact(anchor: str, texts: list[str]) -> bool:
    return any(anchor in normalize(text) for text in texts)


def safe_context(seed: dict, source, passages: list, policy: SelectionPolicy) -> dict | None:
    try:
        return derive_adjacent_evidence(seed, source, passages, policy)
    except ValueError:
        return None


def context_exact(anchor: str, seeds: list[dict], source, passages: list,
                  policy: SelectionPolicy) -> tuple[bool, list[str]]:
    matches = []
    for seed in seeds:
        derived = safe_context(seed, source, passages, policy)
        if derived is not None and anchor in normalize(derived["text"]):
            matches.append(str(seed["passage_id"]))
    return bool(matches), matches


def first_failure(*, retrieved: bool, fused: bool, scored: bool, selected: bool,
                  exported: bool, context_hit: bool) -> str:
    if not retrieved:
        return "NOT_RETRIEVED"
    if not fused:
        return "DROPPED_AT_FUSION"
    if not scored:
        return "FUSED_NOT_RERANKED"
    if not selected:
        return "RERANKED_NOT_SELECTED_PARENT"
    if not exported:
        return "SCORED_SELECTED_CUT_BY_TOP60"
    if not context_hit:
        return "EXPORTED_CONTEXT_MISS"
    return "EXPORTED_CONTEXT_EXACT"


def main(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)

    pilot = root / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
    smoke = root / "outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550"
    audit = root / "outputs/v4/evidence_survival_audit_20261021_v2"
    context_replay = root / "outputs/v4/adjacent_context_replay_d200_20261021"
    baseline_zip = root / "outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip"
    config_path = root / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml"
    pilot_summary = read_json(pilot / "summary.json")
    pilot_lock = read_json(pilot / "run_lock.json")
    smoke_lock = read_json(smoke / "smoke_lock.json")
    context_manifest = read_json(context_replay / "manifest.json")
    config = load_config(config_path)
    query_ids = [int(value) for value in pilot_summary["query_ids"]]
    if pilot_summary.get("status") != "PILOT_COMPLETE":
        raise ValueError("Parent-first pilot is incomplete")
    if query_ids != pilot_lock["query_ids"]:
        raise ValueError("Pilot query lock differs from completed summary")
    if sha(smoke / "smoke_lock.json") != pilot_lock["a0_smoke_lock_sha256"]:
        raise ValueError("A0 smoke lock differs from parent-first pilot lock")
    if sha(baseline_zip) != pilot_lock["baseline_submission_sha256"]:
        raise ValueError("Historical positive ZIP differs from pilot lock")
    if context_manifest.get("status") != "ADJACENT_CONTEXT_REPLAY_PASS":
        raise ValueError("Adjacent-context replay is incomplete")
    if context_manifest["config_sha256"] != sha(config_path):
        raise ValueError("Adjacent-context replay config differs from current config")

    index = root / config["index_namespace"]
    index_manifest = read_json(index / "manifest.json")
    if sha(index / "manifest.json") != pilot_lock["full_index_manifest_sha256"]:
        raise ValueError("Full index differs from pilot lock")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)
    context_policy = SelectionPolicy(**config["selection"])
    if (context_policy.max_docs != 200 or context_policy.max_chunks != 60
            or context_policy.document_selection != "independent"
            or context_policy.chunk_context != "adjacent"):
        raise ValueError("Unexpected parent-first/context policy")

    with baseline_zip.open("rb") as stream:
        baseline_zip_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(baseline_zip) as archive:
        baseline = {int(row["id"]): row
                    for row in json.loads(archive.read("predictions.json"))}
    with (audit / "historical_positive_stages.csv").open(encoding="utf-8") as stream:
        anchors = [row for row in csv.DictReader(stream)
                   if int(row["query_id"]) in set(query_ids)]
    if len(anchors) != pilot_summary["historical_positive_chunks"]:
        raise ValueError("Historical positive coverage differs from pilot")
    identities = [(int(row["query_id"]), int(row["doc_id"])) for row in anchors]
    if len(identities) != len(set(identities)):
        raise ValueError("P0 replay expects one historical positive per query/document")

    context_selected_rows = {
        int(row["id"]): row["selected"]
        for row in map(json.loads, (context_replay / "selected_results.jsonl").open(
            encoding="utf-8"))
    }
    if set(context_selected_rows) != set(query_ids):
        raise ValueError("Context replay query coverage differs from pilot")

    compact_rows: list[dict] = []
    detailed_rows: list[dict] = []
    per_query: list[dict] = []
    input_paths = [
        pilot / "run_lock.json", pilot / "summary.json", smoke / "smoke_lock.json",
        audit / "historical_positive_stages.csv", context_replay / "manifest.json",
        context_replay / "selected_results.jsonl", baseline_zip, config_path,
        index / "manifest.json",
    ]
    input_paths += [pilot / "queries" / f"{qid}.json" for qid in query_ids]
    input_paths += [smoke / "smoke_queries" / f"{qid}.json" for qid in query_ids]
    input_hashes_before = {str(path.relative_to(root)): sha(path) for path in input_paths}

    for qid in query_ids:
        current = read_json(pilot / "queries" / f"{qid}.json")
        a0 = read_json(smoke / "smoke_queries" / f"{qid}.json")
        if (current["status"] != "complete" or a0["status"] != "complete"
                or current["query_sha256"] != smoke_lock["query_sha256"][str(qid)]
                or a0["query_sha256"] != current["query_sha256"]):
            raise ValueError(f"Checkpoint identity/status mismatch: query={qid}")

        replay_a0, _ = select_results(
            a0["trace"]["ranked_candidates"], SelectionPolicy(),
            reranker_enabled=True, limit=200)
        if replay_a0 != a0["selected"]:
            raise ValueError(f"A0 baseline replay mismatch: query={qid}")
        replay_current, selection_trace = select_results(
            current["diagnostics"]["ranked_candidates"], context_policy,
            reranker_enabled=True, limit=200)
        if replay_current != current["selection"]:
            raise ValueError(f"Parent-first selection replay mismatch: query={qid}")
        replay_context, context_trace = apply_chunk_context(
            replay_current, context_policy, sources, passages)
        if replay_context != context_selected_rows[qid]:
            raise ValueError(f"Adjacent-context replay mismatch: query={qid}")

        a0_candidates = {int(doc["official_id"]): doc
                         for doc in a0["trace"]["ranked_candidates"]}
        current_candidates = {int(doc["official_id"]): doc
                              for doc in current["diagnostics"]["ranked_candidates"]}
        current_pre_order = sorted(
            current["diagnostics"]["ranked_candidates"],
            key=lambda doc: (-float(doc["score"]), int(doc["official_id"])))
        current_final_order = {int(doc["official_id"]): position
                               for position, doc in enumerate(
                                   current["diagnostics"]["ranked_candidates"], 1)}
        pre_order = {int(doc["official_id"]): position
                     for position, doc in enumerate(current_pre_order, 1)}
        a0_selected = {int(doc["official_id"]): doc for doc in a0["selected"]}
        current_selected = {int(doc["official_id"]): doc for doc in replay_current}
        context_selected = {int(doc["official_id"]): doc for doc in replay_context}
        pair_positions: dict[int, list[dict]] = defaultdict(list)
        for position, pair in enumerate(current["pair_trace"], 1):
            pair_positions[int(pair["official_id"])].append({"position": position, **pair})

        query_rows = [row for row in anchors if int(row["query_id"]) == qid]
        query_stage_counts = Counter()
        for anchor_row in query_rows:
            parent = int(anchor_row["doc_id"])
            chunk_order = int(anchor_row["baseline_chunk_order"])
            old_chunk = baseline[qid]["relevant_chunks"][chunk_order]
            if int(old_chunk["doc_id"]) != parent:
                raise ValueError("Historical positive parent differs from baseline ZIP")
            old_text = str(old_chunk["chunk_text"])
            old = normalize(old_text)
            source = store.source(parent)
            registered = store.passages_for_parent(parent)

            branches = current["branch_trace"]["branch_hits"]
            doc_hits = parent_branch_hits(branches["document_bm25"], parent)
            passage_hits = parent_branch_hits(branches["passage_bm25"], parent)
            dense_hits = parent_branch_hits(branches["dense"], parent)
            retrieved = bool(doc_hits or passage_hits or dense_hits)
            fused_items = [item for item in current["branch_trace"]["fused"]
                           if int(item["official_id"]) == parent]
            fused = parent in current_candidates

            a0_doc = a0_candidates.get(parent)
            a0_scored = [] if a0_doc is None else [
                item for item in a0_doc["evidence"] if "rerank_score" in item]
            a0_export = a0_selected.get(parent, {}).get("evidence", [])
            a0_context_hit, a0_context_seed_ids = context_exact(
                old, a0_export, source, registered, context_policy)

            candidate_doc = current_candidates.get(parent)
            candidate_passages = [] if candidate_doc is None else candidate_doc["evidence"]
            scored_passages = [item for item in candidate_passages
                               if item.get("score_status") == "reranked"]
            selected_doc = current_selected.get(parent)
            context_doc = context_selected.get(parent)
            selected_parent = selected_doc is not None
            exported = bool(selected_doc and selected_doc["evidence"])
            context_items = [] if context_doc is None else context_doc["evidence"]
            context_hit = exact(old, [item["text"] for item in context_items])
            selected_anchor = [] if selected_doc is None else [selected_doc["document_evidence"]]

            scored_context_hit, scored_context_seed_ids = context_exact(
                old, scored_passages, source, registered, context_policy)
            candidate_context_hit, candidate_context_seed_ids = context_exact(
                old, candidate_passages, source, registered, context_policy)
            registered_seeds = [{
                "passage_id": item.passage_id,
                "text": item.text,
                "start": item.source_start,
                "end": item.source_end,
                "score_status": "retrieval_only",
            } for item in registered]
            registered_context_hit, registered_context_seed_ids = context_exact(
                old, registered_seeds, source, registered, context_policy)
            anchor_context_hit, anchor_context_seed_ids = context_exact(
                old, selected_anchor, source, registered, context_policy)

            failure = first_failure(
                retrieved=retrieved, fused=fused, scored=bool(scored_passages),
                selected=selected_parent, exported=exported, context_hit=context_hit)
            query_stage_counts[failure] += 1
            if context_hit:
                rescue_tier = "ALREADY_EXPORTED_CONTEXT_EXACT"
            elif anchor_context_hit:
                rescue_tier = "CHUNK_ALLOCATION_ONLY"
            elif candidate_context_hit:
                rescue_tier = "RERANK_PASSAGE_ALLOCATION"
            elif registered_context_hit:
                rescue_tier = "RETRIEVAL_FUSION_AND_ALLOCATION"
            else:
                rescue_tier = "NOT_EXACTLY_RECOVERABLE_FROM_REGISTERED_SOURCE"

            selected_without_score = selected_parent and not scored_passages
            scored_cut_top60 = selected_parent and bool(scored_passages) and not exported
            compact = {
                "query_id": qid,
                "doc_id": parent,
                "baseline_chunk_order": chunk_order,
                "baseline_best_overlap": anchor_row["baseline_best_overlap"],
                "source_contains_old_anchor": old in normalize(source.text),
                "registered_passage_contains_old_anchor": exact(
                    old, [item.text for item in registered]),
                "document_bm25_hit": bool(doc_hits),
                "document_bm25_best_rank": min((int(item["rank"]) for item in doc_hits), default=""),
                "passage_bm25_hit": bool(passage_hits),
                "passage_bm25_best_rank": min((int(item["rank"]) for item in passage_hits), default=""),
                "dense_hit": bool(dense_hits),
                "dense_best_rank": min((int(item["rank"]) for item in dense_hits), default=""),
                "retrieved_any_branch": retrieved,
                "fused_parent": fused,
                "fused_first_item_position": min(
                    (position for position, item in enumerate(
                        current["branch_trace"]["fused"], 1)
                     if int(item["official_id"]) == parent), default=""),
                "pre_rerank_parent_rank": pre_order.get(parent, ""),
                "admitted_parent_first_200": pre_order.get(parent, 10**9) <= 200,
                "a0_scored_passages": len(a0_scored),
                "a0_scored_passage_exact": exact(
                    old, [item["text"] for item in a0_scored]),
                "a0_selected_parent": parent in a0_selected,
                "a0_exported_chunk": bool(a0_export),
                "a0_export_exact": exact(old, [item["text"] for item in a0_export]),
                "a0_export_context_exact": a0_context_hit,
                "parent_first_pair_position": min(
                    (item["position"] for item in pair_positions.get(parent, [])), default=""),
                "parent_first_scored_passages": len(scored_passages),
                "parent_first_scored_passage_exact": exact(
                    old, [item["text"] for item in scored_passages]),
                "parent_first_scored_context_exact": scored_context_hit,
                "selected_parent_d200": selected_parent,
                "selected_parent_rank": current_final_order.get(parent, ""),
                "selected_parent_without_reranked_passage": selected_without_score,
                "scored_selected_but_cut_by_top60": scored_cut_top60,
                "exported_chunk": exported,
                "exported_seed_exact": exact(
                    old, [] if selected_doc is None else
                    [item["text"] for item in selected_doc["evidence"]]),
                "context_exact": context_hit,
                "context_whitespace_lcs_over_old_anchor": anchor_coverage(
                    old, [item["text"] for item in context_items]),
                "first_failure": failure,
                "minimum_rescue_tier": rescue_tier,
            }
            compact_rows.append(compact)
            detailed_rows.append({
                "identity": {
                    "query_id": qid,
                    "doc_id": parent,
                    "baseline_chunk_order": chunk_order,
                    "baseline_chunk_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
                    "baseline_best_overlap": float(anchor_row["baseline_best_overlap"]),
                },
                "grounding": {
                    "source_contains_old_anchor": compact["source_contains_old_anchor"],
                    "registered_passage_contains_old_anchor": compact[
                        "registered_passage_contains_old_anchor"],
                    "registered_context_contains_old_anchor": registered_context_hit,
                    "registered_context_seed_ids": registered_context_seed_ids,
                },
                "retrieved": {
                    "document_bm25": doc_hits,
                    "passage_bm25": passage_hits,
                    "dense": dense_hits,
                    "any_branch": retrieved,
                },
                "fused": {
                    "present": fused,
                    "items": fused_items,
                    "pre_rerank_parent_rank": pre_order.get(parent),
                },
                "a0_baseline": {
                    "scored_passages": a0_scored,
                    "selected_parent": parent in a0_selected,
                    "exported_passages": a0_export,
                    "export_context_exact": a0_context_hit,
                    "export_context_seed_ids": a0_context_seed_ids,
                },
                "parent_first": {
                    "pair_trace": pair_positions.get(parent, []),
                    "scored_passages": scored_passages,
                    "scored_context_exact": scored_context_hit,
                    "scored_context_seed_ids": scored_context_seed_ids,
                    "candidate_context_exact": candidate_context_hit,
                    "candidate_context_seed_ids": candidate_context_seed_ids,
                    "selected_parent": selected_parent,
                    "selected_parent_rank": current_final_order.get(parent),
                    "document_evidence": None if selected_doc is None else
                        selected_doc["document_evidence"],
                    "exported_passages": [] if selected_doc is None else
                        selected_doc["evidence"],
                    "context_passages": context_items,
                    "context_exact": context_hit,
                },
                "diagnosis": {
                    "first_failure": failure,
                    "minimum_rescue_tier": rescue_tier,
                    "selected_parent_without_reranked_passage": selected_without_score,
                    "scored_selected_but_cut_by_top60": scored_cut_top60,
                    "selected_anchor_context_exact": anchor_context_hit,
                    "selected_anchor_context_seed_ids": anchor_context_seed_ids,
                },
            })

        per_query.append({
            "query_id": qid,
            "historical_positives": len(query_rows),
            "failure_counts": dict(sorted(query_stage_counts.items())),
            "selected_documents": len(replay_current),
            "exported_chunks": sum(len(doc["evidence"]) for doc in replay_current),
            "selection_trace": selection_trace,
            "context_trace": context_trace,
        })

    if len(compact_rows) != len(anchors) or len(detailed_rows) != len(anchors):
        raise ValueError("Positive path output lost rows")
    stage_counts = Counter(row["first_failure"] for row in compact_rows)
    rescue_counts = Counter(row["minimum_rescue_tier"] for row in compact_rows)
    expected_stage_counts = {
        "DROPPED_AT_FUSION": 2,
        "EXPORTED_CONTEXT_EXACT": 21,
        "EXPORTED_CONTEXT_MISS": 5,
        "NOT_RETRIEVED": 2,
        "SCORED_SELECTED_CUT_BY_TOP60": 7,
    }
    expected_rescue_counts = {
        "ALREADY_EXPORTED_CONTEXT_EXACT": 21,
        "CHUNK_ALLOCATION_ONLY": 5,
        "NOT_EXACTLY_RECOVERABLE_FROM_REGISTERED_SOURCE": 6,
        "RERANK_PASSAGE_ALLOCATION": 2,
        "RETRIEVAL_FUSION_AND_ALLOCATION": 3,
    }
    if dict(sorted(stage_counts.items())) != expected_stage_counts:
        raise ValueError(f"Unexpected stage counts: {stage_counts}")
    if dict(sorted(rescue_counts.items())) != expected_rescue_counts:
        raise ValueError(f"Unexpected rescue counts: {rescue_counts}")

    a0_fused_not_reranked = sum(
        row["fused_parent"] and row["a0_scored_passages"] == 0 for row in compact_rows)
    selected_without_score = sum(
        row["selected_parent_without_reranked_passage"] for row in compact_rows)
    scored_cut_top60 = sum(row["scored_selected_but_cut_by_top60"]
                           for row in compact_rows)
    if a0_fused_not_reranked != 7 or selected_without_score != 0 or scored_cut_top60 != 7:
        raise ValueError("Critical rerank/top-60 distinction differs from expected trace")

    ceiling = {
        "metric": "exact containment of historical positive chunk after adjacent context",
        "current_parent_first_d200_context": 21,
        "chunk_allocation_only_same_scored_seeds": 26,
        "chunk_allocation_only_increment": 5,
        "rerank_and_chunk_allocation_same_fused_candidates": 28,
        "rerank_allocation_increment_after_chunk_allocation": 2,
        "allocation_ceiling_same_retrieval_fusion": 28,
        "retrieval_fusion_and_allocation_ceiling_registered_context": 31,
        "registered_source_exact_ceiling": 31,
        "total_historical_positives": len(compact_rows),
        "not_rescuable_by_allocation_same_fused_pool": len(compact_rows) - 28,
        "not_exactly_recoverable_from_registered_source": 6,
        "constraints": {
            "rerank_pairs_per_query": 200,
            "max_chunks_per_query": 60,
            "candidate_pool": "locked fused400 parent-first pilot",
            "context_window": context_policy.context_window,
            "context_max_characters": context_policy.context_max_characters,
        },
        "interpretation": [
            "26 is an oracle ceiling from moving the 60 export slots among already-scored seeds.",
            "28 additionally allows the same 200 rerank slots to choose a different existing candidate passage.",
            "31 additionally requires retrieval/fusion to recover three parents; it is outside allocation-only scope.",
            "Ceilings use historical anchors, not official gold, and are not achievable policies without a selection signal.",
        ],
    }
    summary = {
        "status": "P0_POSITIVE_PATH_REPLAY_PASS",
        "scope": "six locked full-index pilot queries and 37 historical positive query/document anchors",
        "query_ids": query_ids,
        "historical_positives": len(compact_rows),
        "baseline_a0": {
            "selected_positive_parents": sum(row["a0_selected_parent"] for row in compact_rows),
            "exported_positive_chunks": sum(row["a0_exported_chunk"] for row in compact_rows),
            "export_exact": sum(row["a0_export_exact"] for row in compact_rows),
            "context_exact": sum(row["a0_export_context_exact"] for row in compact_rows),
            "fused_positive_parent_without_reranked_passage": a0_fused_not_reranked,
        },
        "parent_first_d200": {
            "selected_positive_parents": sum(row["selected_parent_d200"] for row in compact_rows),
            "exported_positive_chunks": sum(row["exported_chunk"] for row in compact_rows),
            "export_exact": sum(row["exported_seed_exact"] for row in compact_rows),
            "context_exact": sum(row["context_exact"] for row in compact_rows),
            "selected_parent_without_reranked_passage": selected_without_score,
            "scored_selected_but_cut_by_top60": scored_cut_top60,
        },
        "first_failure_counts": dict(sorted(stage_counts.items())),
        "minimum_rescue_tier_counts": dict(sorted(rescue_counts.items())),
        "allocation_ceiling": ceiling,
        "model_calls": 0,
        "official_f2": "UNMEASURED",
        "baseline_submission_sha256": baseline_zip_sha,
    }
    failure_groups = {
        code: [{"query_id": row["query_id"], "doc_id": row["doc_id"]}
               for row in compact_rows if row["first_failure"] == code]
        for code in sorted(stage_counts)
    }
    rescue_groups = {
        tier: [{"query_id": row["query_id"], "doc_id": row["doc_id"]}
               for row in compact_rows if row["minimum_rescue_tier"] == tier]
        for tier in sorted(rescue_counts)
    }

    input_hashes_after = {str(path.relative_to(root)): sha(path) for path in input_paths}
    if input_hashes_before != input_hashes_after:
        raise ValueError("Replay input changed during P0 analysis")
    output.mkdir(parents=True)
    write_csv(output / "positive_paths.csv", compact_rows)
    with (output / "positive_paths.jsonl").open("x", encoding="utf-8") as stream:
        for row in detailed_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    write_json(output / "summary.json", summary)
    write_json(output / "allocation_ceiling.json", ceiling)
    write_json(output / "failure_groups.json", {
        "first_failure": failure_groups,
        "minimum_rescue_tier": rescue_groups,
    })
    write_json(output / "per_query.json", per_query)
    write_json(output / "input_hashes.json", input_hashes_before)
    artifacts = ["positive_paths.csv", "positive_paths.jsonl", "summary.json",
                 "allocation_ceiling.json", "failure_groups.json", "per_query.json",
                 "input_hashes.json"]
    manifest = {
        "format": "v4-p0-positive-path-replay-v1",
        "status": summary["status"],
        "query_ids": query_ids,
        "row_count": len(compact_rows),
        "artifacts_sha256": {name: sha(output / name) for name in artifacts},
        "implementation_sha256": sha(Path(__file__).resolve()),
        "summary_canonical_sha256": canonical_sha(summary),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
