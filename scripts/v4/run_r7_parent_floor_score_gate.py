"""Materialize and score the bounded R7 parent-floor downstream gate.

The CPU package is label-blind and reuses the locked 30-query retrieval traces.
GPU scoring is fail-closed against the frozen S2 CUDA/FP16 runtime and starts
with cached-score anchors before any new score is accepted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Iterable
import zipfile

import pyarrow.parquet as pq

from scripts.v4.run_p5_2_split import CONFIG, ROOT, read
from src.medrag.adapters.validation import canonical_payload, validate_payload, validate_selected
from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.core.contracts import Hit
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.identity import normalize
from src.medrag.evidence.refinement import (expand_existing_context, filter_documents,
                                            select_documents_by_confidence)
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.core import fuse, fuse_with_parent_floor, tokens
from src.medrag.retrieval.disk_bm25 import DiskBM25


OUTPUT = ROOT / "outputs/v4/r7_parent_floor_score_20261025"
R7 = ROOT / "outputs/v4/r7_targeted_parent_loss_20261025"
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024"
BASE = ROOT / "outputs/v4/a0_fast_a6000_final_20261023"
RUNTIME = ROOT / "outputs/v4/p5_gpu_incremental_a6000_20261023/runtime_lock.json"
S2_ZIP = ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip"
R6_REGISTRY = ROOT / "outputs/v4/r6_stage_oracle_20261025/positive_registry.parquet"
CANARY_IDS = (610, 1159, 53)
TARGETS = {610: 305379, 1159: 514656}
ANCHOR_LIMIT = 128
LOGIT_TOLERANCE = 1e-3


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_s2() -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(S2_ZIP) as archive:
        rows = json.loads(archive.read("predictions.json"))
    return {int(row["id"]): row for row in rows}


def environment() -> tuple[dict, dict, DiskCorpusStore, DiskBM25, dict[int, str]]:
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    index_manifest = read(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", index_manifest["corpus_version"],
                            index_manifest["fingerprint"])
    passage_bm25 = DiskBM25(index / "passages.sqlite", "passage_bm25",
                            index_manifest["fingerprint"])
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    queries = {int(row["id"]): str(row["query"]) for row in
               pq.read_table(query_path, columns=["id", "query"]).to_pylist()}
    return config, index_manifest, store, passage_bm25, queries


def fused_rows(trace: dict[str, Any], config: dict) -> tuple[list, list]:
    rankings = [[Hit(**value) for value in trace["branch_hits"][name]]
                for name in ("document_bm25", "passage_bm25", "dense")]
    baseline = fuse(rankings, k=config["retrieval"]["rrf_k"],
                    limit=config["retrieval"]["fused_candidate_limit"])
    augmented = fuse_with_parent_floor(
        rankings, k=config["retrieval"]["rrf_k"],
        item_limit=config["retrieval"]["fused_candidate_limit"], parent_limit=400)
    if augmented[:len(baseline)] != baseline:
        raise ValueError("Parent-floor changed the frozen mixed-item prefix")
    return baseline, augmented


def passage_payload(source, passage) -> str:
    return f"{source.title}\n{passage.text}" if source.title else passage.text


def pair_row(qid: int, query: str, item, source, passage, seed_policy: str,
             order: int) -> dict[str, Any]:
    if passage.official_id != item.official_id or not verify(passage, source):
        raise ValueError(f"Ungrounded new seed: {qid}/{item.official_id}/{passage.passage_id}")
    payload = passage_payload(source, passage)
    identity = {
        "query_id": qid, "official_id": item.official_id,
        "passage_id": passage.passage_id, "query_sha256": digest_text(query),
        "input_sha256": digest_text(payload),
    }
    return {
        "format": "v4-r7-parent-floor-score-pair-v1", **identity,
        "pair_id": canonical_digest(identity), "query": query, "payload": payload,
        "source_sha256": source.text_sha256, "source_start": passage.source_start,
        "source_end": passage.source_end, "passage_text_sha256": digest_text(passage.text),
        "seed_policy": seed_policy, "added_parent_order": order,
        "retrieval_score": item.score, "retrieval_ranks": item.ranks,
    }


def passage_bm25_fallbacks(index: DiskBM25, query: str,
                           parent_ids: set[int]) -> dict[int, str]:
    """Return the exact nonzero parent-local top-1 BM25 passage in one DB scan."""
    if not parent_ids:
        return {}
    meta = index._check(require_complete=True)
    row_count = int(meta["rows"])
    average_length = int(meta["total_length"]) / row_count
    scores: dict[int, float] = defaultdict(float)
    identities: dict[int, tuple[int, str]] = {}
    ordered = sorted(parent_ids)
    with sqlite3.connect(f"file:{index.path}?mode=ro", uri=True) as database:
        database.row_factory = sqlite3.Row
        for term in set(tokens(query)):
            df_row = database.execute("SELECT count FROM df WHERE term=?", (term,)).fetchone()
            if df_row is None:
                continue
            frequency = int(df_row["count"])
            inverse = math.log(1 + (row_count - frequency + .5) / (frequency + .5))
            for offset in range(0, len(ordered), 800):
                batch = ordered[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = database.execute(
                    "SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id FROM postings p "
                    "JOIN docs d ON d.row_id=p.row_id WHERE p.term=? AND d.official_id IN ("
                    + placeholders + ")", [term, *batch])
                for row in rows:
                    row_id, term_frequency, length = int(row["row_id"]), int(row["tf"]), int(row["length"])
                    identities[row_id] = (int(row["official_id"]), str(row["passage_id"]))
                    scores[row_id] += inverse * term_frequency * 2.2 / (
                        term_frequency + 1.2 * (.25 + .75 * length / max(average_length, 1)))
    best: dict[int, tuple[float, str]] = {}
    for row_id, score in scores.items():
        parent, passage_id = identities[row_id]
        candidate = (-score, passage_id)
        if parent not in best or candidate < best[parent]:
            best[parent] = candidate
    return {parent: value[1] for parent, value in best.items()}


def anchor_candidates(qid: int, query: str, final: dict[str, Any],
                      sources: dict[int, Any], passages: dict[str, Any],
                      s2_doc_count: int) -> list[dict[str, Any]]:
    output = []
    for rank, document in enumerate(final["trace"]["ranked_candidates"], 1):
        source = sources[int(document["official_id"])]
        for evidence in document["evidence"]:
            score = evidence.get("rerank_score")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score)):
                raise ValueError(f"Frozen cached score is invalid: {qid}/{evidence['passage_id']}")
            passage = passages[evidence["passage_id"]]
            if (passage.official_id != document["official_id"] or not verify(passage, source)
                    or (passage.text, passage.source_start, passage.source_end) !=
                    (evidence["text"], evidence["start"], evidence["end"])):
                raise ValueError(f"Frozen anchor grounding differs: {qid}/{passage.passage_id}")
            payload = passage_payload(source, passage)
            identity = {"query_id": qid, "official_id": document["official_id"],
                        "passage_id": passage.passage_id,
                        "query_sha256": digest_text(query), "input_sha256": digest_text(payload)}
            output.append({
                "format": "v4-r7-runtime-anchor-v1", **identity,
                "anchor_id": canonical_digest(identity), "query": query, "payload": payload,
                "expected_score": float(score), "candidate_parent_rank": rank,
                "s2_document_count": s2_doc_count,
                "distance_neg4": abs(float(score) + 4.0),
                "distance_neg2": abs(float(score) + 2.0),
                "distance_chunk_cut": abs(rank - 60),
                "distance_document_cut": abs(rank - s2_doc_count),
                "threshold_checks": [-4.0, -2.0],
            })
    return output


def choose_anchors(rows: list[dict[str, Any]], limit: int = ANCHOR_LIMIT) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Anchor limit must be positive")
    selected: dict[str, dict[str, Any]] = {}
    categories = (
        ("near_neg4", "distance_neg4"), ("near_neg2", "distance_neg2"),
        ("near_chunk_cut", "distance_chunk_cut"),
        ("near_document_cut", "distance_document_cut"),
    )
    per_category = max(1, limit // len(categories))
    for kind, field in categories:
        ordered = sorted(rows, key=lambda row: (row[field], row["anchor_id"]))
        added = 0
        for row in ordered:
            if row["anchor_id"] in selected:
                continue
            selected[row["anchor_id"]] = {**row, "anchor_kind": kind}
            added += 1
            if added == per_category or len(selected) == limit:
                break
    if len(selected) < limit:
        for row in sorted(rows, key=lambda value: value["anchor_id"]):
            selected.setdefault(row["anchor_id"], {**row, "anchor_kind": "deterministic_fill"})
            if len(selected) == limit:
                break
    if len(selected) != min(limit, len({row["anchor_id"] for row in rows})):
        raise ValueError("Could not select deterministic runtime anchors")
    return list(selected.values())


def exact_batch_anchors(query_ids: list[int], queries: dict[int, str],
                        s2: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Select four intact historical batches so FP16 padding matches S2 scoring."""
    batches = []
    source_hashes = {}
    for qid in query_ids:
        pair_path, score_path = BASE / "pairs" / f"{qid}.json", BASE / "scores" / f"{qid}.json"
        pair, score = read(pair_path), read(score_path)
        source_hashes[str(pair_path.relative_to(ROOT))] = sha256_file(pair_path)
        source_hashes[str(score_path.relative_to(ROOT))] = sha256_file(score_path)
        if pair["query"] != queries[qid] or len(pair["pairs"]) != len(score["scores"]):
            raise ValueError(f"Historical anchor source identity differs: {qid}")
        final = read(REFINED / "final" / f"{qid}.json")
        ranks = {(int(document["official_id"]), evidence["passage_id"]): rank
                 for rank, document in enumerate(final["trace"]["ranked_candidates"], 1)
                 for evidence in document["evidence"]}
        for start in range(0, len(pair["pairs"]), 32):
            pair_rows, score_rows = pair["pairs"][start:start + 32], score["scores"][start:start + 32]
            if len(pair_rows) != 32:
                continue
            members = []
            for position, (left, right) in enumerate(zip(pair_rows, score_rows, strict=True)):
                if any(left[key] != right[key] for key in
                       ("official_id", "passage_id", "input_sha256", "cache_key")):
                    raise ValueError(f"Historical pair/score alignment differs: {qid}/{start}")
                if digest_text(left["payload"]) != left["input_sha256"]:
                    raise ValueError(f"Historical anchor payload changed: {qid}/{start}")
                identity = {"query_id": qid, "official_id": left["official_id"],
                            "passage_id": left["passage_id"],
                            "query_sha256": pair["query_sha256"],
                            "input_sha256": left["input_sha256"]}
                members.append({
                    "format": "v4-r7-runtime-anchor-v2", **identity,
                    "anchor_id": canonical_digest(identity), "query": pair["query"],
                    "payload": left["payload"], "expected_score": float(right["score"]),
                    "candidate_parent_rank": ranks[(left["official_id"], left["passage_id"])],
                    "s2_document_count": len(s2[qid]["relevant_docs"]),
                    "scoring_batch_id": f"initial:{qid}:{start}",
                    "scoring_batch_position": position, "scoring_batch_count": 32,
                    "source_pair_sha256": source_hashes[str(pair_path.relative_to(ROOT))],
                    "source_score_sha256": source_hashes[str(score_path.relative_to(ROOT))],
                    "threshold_checks": [-4.0, -2.0],
                })
            batches.append({
                "batch_id": f"initial:{qid}:{start}", "members": members,
                "distance_neg4": min(abs(row["expected_score"] + 4.0) for row in members),
                "distance_neg2": min(abs(row["expected_score"] + 2.0) for row in members),
                "distance_chunk_cut": min(abs(row["candidate_parent_rank"] - 60) for row in members),
                "distance_document_cut": min(abs(
                    row["candidate_parent_rank"] - row["s2_document_count"]) for row in members),
            })
    chosen, used = [], set()
    categories = (("near_neg4", "distance_neg4"), ("near_neg2", "distance_neg2"),
                  ("near_chunk_cut", "distance_chunk_cut"),
                  ("near_document_cut", "distance_document_cut"))
    for kind, metric in categories:
        candidates = sorted(batches, key=lambda row: (row[metric], row["batch_id"]))
        selected = next(row for row in candidates if row["batch_id"] not in used)
        used.add(selected["batch_id"])
        chosen.extend({**row, "anchor_kind": kind} for row in selected["members"])
    if len(chosen) != ANCHOR_LIMIT or len({row["anchor_id"] for row in chosen}) != ANCHOR_LIMIT:
        raise ValueError("Exact historical anchor batch selection differs")
    return chosen, source_hashes


def package(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    config, index_manifest, store, passage_bm25, queries = environment()
    cohort = read(R7 / "cohort_lock.json")
    trace_report = read(R7 / "trace_report.json")
    candidate_gate = read(R7 / "candidate_gate.json")
    refined_manifest = read(REFINED / "run_manifest.json")
    runtime = read(RUNTIME)
    if (cohort.get("status") != "LOCKED" or trace_report.get("status") != "PASS"
            or candidate_gate.get("status") != "POOL_GATE_PASS_SCORE_GATE_PENDING"
            or refined_manifest.get("status") != "PASS"):
        raise ValueError("Upstream R7/refined gates are not reusable")
    query_ids = [int(value) for value in cohort["query_ids"]]
    if len(query_ids) != 30 or not set(CANARY_IDS) <= set(query_ids):
        raise ValueError("Locked cohort/canary identity differs")
    expected_added = {}
    with (R7 / "candidate_parent_replay.csv").open() as stream:
        for row in csv.DictReader(stream):
            expected_added[int(row["query_id"])] = int(row["added_parent_count"])
    s2 = load_s2()
    pairs, unscorable, query_rows = [], [], []
    old_score_digests = {}
    direct_count = fallback_count = 0
    for qid in query_ids:
        trace_path = R7 / "traces" / f"{qid}.json"
        if sha256_file(trace_path) != trace_report["trace_hashes"][str(qid)]:
            raise ValueError(f"Locked trace changed: {qid}")
        final_path = REFINED / "final" / f"{qid}.json"
        if sha256_file(final_path) != refined_manifest["final_sha256"][str(qid)]:
            raise ValueError(f"Frozen refined candidate changed: {qid}")
        trace, final, query = read(trace_path), read(final_path), queries[qid]
        if final["query_sha256"] != digest_text(query):
            raise ValueError(f"Query identity differs: {qid}")
        baseline, augmented = fused_rows(trace, config)
        baseline_parents = {item.official_id for item in baseline}
        suffix = [item for item in augmented[len(baseline):]
                  if item.official_id not in baseline_parents]
        if len(suffix) != expected_added[qid] or len({item.official_id for item in suffix}) != len(suffix):
            raise ValueError(f"Added-parent replay differs: {qid}")
        frozen_parents = {int(row["official_id"]) for row in final["trace"]["ranked_candidates"]}
        if frozen_parents != baseline_parents:
            raise ValueError(f"Frozen grouped parents differ from trace prefix: {qid}")
        frozen_candidates = final["trace"]["ranked_candidates"]
        frozen_ids = {int(document["official_id"]) for document in frozen_candidates}
        frozen_passage_ids = [evidence["passage_id"] for document in frozen_candidates
                              for evidence in document["evidence"]]
        frozen_sources = store.sources_batch(frozen_ids)
        frozen_passages = store.passages_batch(frozen_passage_ids)
        old_score_rows = []
        for document in frozen_candidates:
            for evidence in document["evidence"]:
                score = evidence.get("rerank_score")
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                    raise ValueError(f"Incomplete cached S2 score: {qid}")
                source = frozen_sources[document["official_id"]]
                passage = frozen_passages[evidence["passage_id"]]
                payload = passage_payload(source, passage)
                if not verify(passage, source) or passage.text != evidence["text"]:
                    raise ValueError(f"Cached candidate source mismatch: {qid}")
                old_score_rows.append([document["official_id"], evidence["passage_id"],
                                       digest_text(payload), float(score)])
        old_score_digests[str(qid)] = canonical_digest(old_score_rows)
        valid = 0
        suffix_sources = store.sources_batch({item.official_id for item in suffix})
        direct_passages = store.passages_batch(
            [item.passage_id for item in suffix if item.passage_id is not None])
        fallback_ids = {item.official_id for item in suffix if item.passage_id is None}
        fallback_passage_ids = passage_bm25_fallbacks(passage_bm25, query, fallback_ids)
        fallback_passages = store.passages_batch(list(fallback_passage_ids.values())) \
            if fallback_passage_ids else {}
        for order, item in enumerate(suffix, 1):
            source = suffix_sources[item.official_id]
            if item.passage_id is not None:
                passage = direct_passages[item.passage_id]
                seed_policy = "first_supplemented_item_passage"
                direct_count += 1
            else:
                fallback_id = fallback_passage_ids.get(item.official_id)
                if fallback_id is None:
                    unscorable.append({"query_id": qid, "official_id": item.official_id,
                                       "reason": "document_only_parent_has_no_nonzero_passage_bm25_fallback",
                                       "added_parent_order": order})
                    continue
                passage = fallback_passages[fallback_id]
                seed_policy = "document_only_passage_bm25_parent_top1"
                fallback_count += 1
            pairs.append(pair_row(qid, query, item, source, passage, seed_policy, order))
            valid += 1
        query_rows.append({
            "query_id": qid, "query_sha256": digest_text(query),
            "trace_sha256": sha256_file(trace_path), "refined_final_sha256": sha256_file(final_path),
            "baseline_mixed_items": len(baseline), "baseline_unique_parents": len(baseline_parents),
            "added_parents": len(suffix), "scorable_pairs": valid,
            "unscorable_parents": len(suffix) - valid, "canary": qid in CANARY_IDS,
        })
    if len(pairs) > candidate_gate["added_parent_pairs_minimum"]:
        raise ValueError("Materialized pairs exceed the locked parent-floor budget")
    identities = [(row["query_id"], row["official_id"], row["passage_id"]) for row in pairs]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate materialized new pair identity")
    anchors, anchor_source_hashes = exact_batch_anchors(query_ids, queries, s2)
    canary_pairs = sum(row["query_id"] in CANARY_IDS for row in pairs)
    expected_canary = sum(expected_added[qid] for qid in CANARY_IDS)
    if canary_pairs > expected_canary:
        raise ValueError("Canary materialization exceeds 385-parent lock")

    build_dir = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    try:
        write_jsonl(build_dir / "pairs.jsonl", pairs)
        write_jsonl(build_dir / "anchors.jsonl", anchors)
        write_jsonl(build_dir / "unscorable.jsonl", unscorable)
        write_once(build_dir / "query_manifest.json", query_rows)
        runtime_contract = {
            "format": "v4-r7-parent-floor-runtime-contract-v1",
            "baseline_runtime_sha256": sha256_file(RUNTIME),
            "score_contract": runtime["score_contract"], "versions": runtime["versions"],
            "cuda": runtime["cuda"], "gpu_name": runtime["gpu_name"],
            "dtype": runtime["dtype"], "activation": runtime["activation"],
            "batch_size": runtime["batch_size"], "num_threads": runtime["num_threads"],
            "num_interop_threads": runtime["num_interop_threads"],
            "matmul_allow_tf32": runtime["matmul_allow_tf32"],
            "cudnn_allow_tf32": runtime["cudnn_allow_tf32"],
            "model_snapshot_files_sha256": runtime["model_snapshot_files_sha256"],
            "logit_absolute_tolerance": LOGIT_TOLERANCE,
            "anchor_count": len(anchors),
            "anchor_policy": "four intact historical 32-pair batches selected near -4, -2, chunk-60 and S2 document cut",
        }
        write_once(build_dir / "runtime_contract.json", runtime_contract)
        files = ("pairs.jsonl", "anchors.jsonl", "unscorable.jsonl",
                 "query_manifest.json", "runtime_contract.json")
        manifest = {
            "format": "v4-r7-parent-floor-score-package-v1", "status": "READY_FOR_GPU",
            "query_ids": query_ids, "canary_query_ids": list(CANARY_IDS),
            "target_parents": {str(key): value for key, value in TARGETS.items()},
            "counts": {"queries": len(query_ids), "added_parents": sum(expected_added.values()),
                       "scorable_pairs": len(pairs), "unscorable_parents": len(unscorable),
                       "direct_passage_seeds": direct_count, "bm25_fallback_seeds": fallback_count,
                       "canary_scorable_pairs": canary_pairs,
                       "canary_added_parents": expected_canary, "runtime_anchors": len(anchors)},
            "policies": {"mixed_item_prefix": 400, "unique_parent_floor": 400,
                         "new_seed": "first supplemented item; document-only uses passage BM25 parent top-1",
                         "merge": "all cached refined S2 candidate scores plus explicitly scored new pairs",
                         "labels_used_for_seed_selection": False},
            "old_candidate_score_digests": old_score_digests,
            "anchor_source_sha256": anchor_source_hashes,
            "files_sha256": {name: sha256_file(build_dir / name) for name in files},
            "inputs_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in (
                    R7 / "cohort_lock.json", R7 / "trace_report.json", R7 / "candidate_gate.json",
                    R7 / "candidate_parent_replay.csv", REFINED / "run_manifest.json", RUNTIME,
                    S2_ZIP, CONFIG, Path(__file__).resolve())
            },
            "official_f2": "UNMEASURED",
        }
        write_once(build_dir / "pair_manifest.json", manifest)
        build_dir.rename(output)
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    return manifest


def validate_package(output: Path) -> dict[str, Any]:
    manifest = read(output / "pair_manifest.json")
    if manifest.get("format") != "v4-r7-parent-floor-score-package-v1":
        raise ValueError("Score package format differs")
    implementation_key = "scripts/v4/run_r7_parent_floor_score_gate.py"
    if (manifest.get("inputs_sha256", {}).get(implementation_key)
            != sha256_file(Path(__file__).resolve())):
        raise ValueError("Score package worker implementation changed")
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Score package file changed: {name}")
    pairs, anchors = read_jsonl(output / "pairs.jsonl"), read_jsonl(output / "anchors.jsonl")
    if len(pairs) != manifest["counts"]["scorable_pairs"] or len(anchors) != ANCHOR_LIMIT:
        raise ValueError("Score package counts differ")
    if len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs in score package")
    if len({row["anchor_id"] for row in anchors}) != len(anchors):
        raise ValueError("Duplicate runtime anchor IDs")
    anchor_batches: dict[str, list[int]] = defaultdict(list)
    for row in anchors:
        anchor_batches[row["scoring_batch_id"]].append(int(row["scoring_batch_position"]))
    if (len(anchor_batches) != 4 or any(sorted(positions) != list(range(32))
                                       for positions in anchor_batches.values())):
        raise ValueError("Runtime anchors are not four intact historical batches")
    for row in pairs:
        identity = {key: row[key] for key in
                    ("query_id", "official_id", "passage_id", "query_sha256", "input_sha256")}
        if (row["pair_id"] != canonical_digest(identity) or row["query_sha256"] != digest_text(row["query"])
                or row["input_sha256"] != digest_text(row["payload"])):
            raise ValueError("New pair identity/input hash differs")
    return manifest


def runtime_preflight(output: Path) -> tuple[Any, dict[str, Any], Path]:
    manifest = validate_package(output)
    contract = read(output / "runtime_contract.json")
    import torch
    actual_versions = {name: importlib.metadata.version(name) for name in contract["versions"]}
    checks = {
        "versions_exact": actual_versions == contract["versions"],
        "cuda_available": bool(torch.cuda.is_available()),
        "single_visible_gpu": torch.cuda.device_count() == 1,
        "cuda_version_exact": torch.version.cuda == contract["cuda"],
    }
    if checks["cuda_available"] and checks["single_visible_gpu"]:
        checks["gpu_name_exact"] = torch.cuda.get_device_name(0) == contract["gpu_name"]
    else:
        checks["gpu_name_exact"] = False
    if not all(checks.values()):
        status = {"status": "READY_FOR_GPU", "stage": "RUNTIME_PREFLIGHT_BLOCKED",
                  "checks": checks, "actual_versions": actual_versions,
                  "actual_torch": torch.__version__, "actual_cuda": torch.version.cuda,
                  "note": "No scores produced; cached S2 logits were not mixed with another runtime."}
        write_atomic_replace(output / "status.json", status)
        raise RuntimeError(json.dumps(status, ensure_ascii=False))
    torch.set_num_threads(contract["num_threads"])
    torch.set_num_interop_threads(contract["num_interop_threads"])
    torch.backends.cuda.matmul.allow_tf32 = contract["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = contract["cudnn_allow_tf32"]
    from src.medrag.core.model_revision import require_cached_revision
    settings = contract["score_contract"]
    snapshot = require_cached_revision(settings["model"], settings["revision"])
    for name, expected in contract["model_snapshot_files_sha256"].items():
        if sha256_file(snapshot / name) != expected:
            raise ValueError(f"Pinned model snapshot changed: {name}")
    runtime = {
        "format": "v4-r7-parent-floor-score-runtime-v1",
        "package_manifest_sha256": sha256_file(output / "pair_manifest.json"),
        "baseline_runtime_sha256": contract["baseline_runtime_sha256"],
        "versions": actual_versions, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_uuid_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "score_contract": settings, "dtype": "float16", "activation": "Identity/raw logits",
        "batch_size": contract["batch_size"], "logit_absolute_tolerance": LOGIT_TOLERANCE,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    return torch, runtime, snapshot


def model_scores(model, rows: list[dict[str, Any]], torch) -> tuple[list[dict[str, Any]], float]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    exact_batches = bool(rows) and all("scoring_batch_id" in row for row in rows)
    for row in rows:
        key = row["scoring_batch_id"] if exact_batches else (int(row["query_id"]), row["query"])
        grouped[key].append(row)
    output, total = [], 0.0
    for key, local in grouped.items():
        if exact_batches:
            local.sort(key=lambda row: row["scoring_batch_position"])
            if (len(local) != 32 or [row["scoring_batch_position"] for row in local]
                    != list(range(32))):
                raise ValueError(f"Historical runtime anchor batch is incomplete: {key}")
        qid, query = int(local[0]["query_id"]), local[0]["query"]
        torch.cuda.synchronize()
        started = time.monotonic()
        values = model.score(query, [row["payload"] for row in local])
        torch.cuda.synchronize()
        total += time.monotonic() - started
        if len(values) != len(local) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"Model returned invalid score count/value: {qid}")
        for row, value in zip(local, values, strict=True):
            identity_key = "anchor_id" if "anchor_id" in row else "pair_id"
            output.append({identity_key: row[identity_key], "query_id": qid,
                           "official_id": row["official_id"], "passage_id": row["passage_id"],
                           "query_sha256": row["query_sha256"], "input_sha256": row["input_sha256"],
                           "score": float(value)})
    return output, total


def score_scope(output: Path, scope: str) -> dict[str, Any]:
    if scope not in {"canary", "cohort"}:
        raise ValueError("Unsupported scoring scope")
    torch, runtime, _ = runtime_preflight(output)
    runtime_path = output / "runtime_lock.json"
    if runtime_path.exists():
        if read(runtime_path) != runtime:
            raise ValueError("Existing scoring runtime differs")
    else:
        write_once(runtime_path, runtime)
    from src.medrag.ranking.reranker import BGEReranker
    settings = runtime["score_contract"]
    model = BGEReranker(settings["model"], revision=settings["revision"],
                        max_length=settings["max_length"], device="cuda:0")
    weight = next(model.model.model.parameters())
    if weight.device.type != "cuda" or weight.dtype != torch.float16:
        raise ValueError("Reranker is not CUDA FP16")

    parity_path = output / "anchor_parity.json"
    if not parity_path.exists():
        anchors = read_jsonl(output / "anchors.jsonl")
        scored, seconds = model_scores(model, anchors, torch)
        by_id = {row["anchor_id"]: row for row in scored}
        details, threshold_ok = [], True
        for anchor in anchors:
            actual = by_id[anchor["anchor_id"]]["score"]
            difference = abs(actual - anchor["expected_score"])
            decisions = {str(value): (actual >= value) == (anchor["expected_score"] >= value)
                         for value in anchor["threshold_checks"]}
            threshold_ok = threshold_ok and all(decisions.values())
            details.append({"anchor_id": anchor["anchor_id"], "query_id": anchor["query_id"],
                            "official_id": anchor["official_id"], "passage_id": anchor["passage_id"],
                            "expected_score": anchor["expected_score"], "actual_score": actual,
                            "absolute_difference": difference, "threshold_decisions_same": decisions})
        tolerance_ok = all(row["absolute_difference"] <= LOGIT_TOLERANCE for row in details)
        parity = {"format": "v4-r7-runtime-anchor-parity-v1",
                  "status": "PASS" if tolerance_ok and threshold_ok else "STOP_RUNTIME_PARITY",
                  "anchor_count": len(details), "tolerance": LOGIT_TOLERANCE,
                  "maximum_absolute_difference": max(row["absolute_difference"] for row in details),
                  "tolerance_ok": tolerance_ok, "threshold_decisions_unchanged": threshold_ok,
                  "model_seconds": seconds, "details": details}
        write_once(parity_path, parity)
    parity = read(parity_path)
    if parity["status"] != "PASS":
        write_atomic_replace(output / "status.json", {"status": "STOP_RUNTIME_PARITY", "scope": scope})
        return parity
    if scope == "cohort":
        report = read(output / "canary_report.json")
        if report.get("status") != "PASS_OPEN_COHORT":
            raise ValueError("Cohort scoring requires a passing downstream canary")
    pairs = read_jsonl(output / "pairs.jsonl")
    allowed = set(CANARY_IDS) if scope == "canary" else set(validate_package(output)["query_ids"])
    if scope == "cohort":
        already = {row["pair_id"] for row in read_jsonl(output / "scores" / "canary.jsonl")}
        pairs = [row for row in pairs if row["query_id"] in allowed and row["pair_id"] not in already]
    else:
        pairs = [row for row in pairs if row["query_id"] in allowed]
    scored, seconds = model_scores(model, pairs, torch)
    destination = output / "scores" / f"{scope}.jsonl"
    write_jsonl(destination, scored)
    result = {"status": f"{scope.upper()}_SCORED", "pairs": len(scored),
              "model_seconds": seconds, "pairs_per_second": len(scored) / seconds if seconds else None,
              "scores_sha256": sha256_file(destination), "runtime_sha256": sha256_file(runtime_path)}
    write_atomic_replace(output / "status.json", result)
    return result


def load_scores(output: Path, scope: str) -> dict[str, float]:
    files = [output / "scores" / "canary.jsonl"]
    if scope == "cohort":
        files.append(output / "scores" / "cohort.jsonl")
    values = {}
    for path in files:
        for row in read_jsonl(path):
            if row["pair_id"] in values or not math.isfinite(row["score"]):
                raise ValueError("Duplicate or invalid imported new score")
            values[row["pair_id"]] = float(row["score"])
    return values


def replay_query(qid: int, pair_rows: list[dict[str, Any]], values: dict[str, float],
                 store: DiskCorpusStore, policy: SelectionPolicy, s2: dict[int, dict[str, Any]]) -> dict[str, Any]:
    frozen_path = REFINED / "final" / f"{qid}.json"
    frozen = read(frozen_path)
    candidates = deepcopy(frozen["trace"]["ranked_candidates"])
    seen_parents = {int(row["official_id"]) for row in candidates}
    for pair in pair_rows:
        if pair["pair_id"] not in values or pair["official_id"] in seen_parents:
            raise ValueError(f"Missing score or duplicate augmented parent: {qid}/{pair['official_id']}")
        passage = store.passage(pair["passage_id"])
        candidates.append({
            "official_id": pair["official_id"], "score": pair["retrieval_score"],
            "ranks": pair["retrieval_ranks"], "rerank_score": values[pair["pair_id"]],
            "evidence": [{"passage_id": passage.passage_id, "text": passage.text,
                          "start": passage.source_start, "end": passage.source_end,
                          "score_status": "reranked", "rerank_score": values[pair["pair_id"]]}],
        })
        seen_parents.add(pair["official_id"])
    candidates.sort(key=lambda row: (-row.get("rerank_score", float("-inf")),
                                     -row["score"], row["official_id"]))
    selected, selection = select_results(candidates, policy, reranker_enabled=True, limit=200)
    selected, context = apply_chunk_context(selected, policy, store, store)
    validate_selected([qid], {qid: selected}, store, policy, result_limit=200,
                      reranker_enabled=True,
                      official_ids={int(row["official_id"]) for row in selected})
    gated = filter_documents(selected, -4.0)
    gated, target = select_documents_by_confidence(
        gated, threshold=-2.0, min_documents=80, max_documents=140)
    old_contexts = {int(key): value for key, value in frozen["expanded_evidence"].items()}
    old_s2_chunks = {int(row["doc_id"]): row["chunk_text"] for row in s2[qid]["relevant_chunks"]}
    preserved = 0
    for document in gated:
        if not document["evidence"]:
            continue
        doc_id = int(document["official_id"])
        if doc_id in old_s2_chunks and doc_id in old_contexts:
            expanded = old_contexts[doc_id]
            if expanded["text"] != old_s2_chunks[doc_id]:
                raise ValueError(f"Frozen S2 context lineage differs: {qid}/{doc_id}")
            document["evidence"] = [deepcopy(expanded)]
            preserved += 1
        else:
            evidence = document["evidence"][0]
            document["evidence"] = [expand_existing_context(
                evidence, store.source(doc_id), store.passages_for_parent(doc_id))]
    prediction = records([qid], {qid: gated})[0]
    validate_payload(canonical_payload([prediction]), query_ids={qid},
                     official_ids=set(prediction["relevant_docs"]), policy=policy, result_limit=200)
    return {"query_id": qid, "prediction": prediction, "ranked_candidates": candidates,
            "selected_before_s2_gate": selected, "document_target": target,
            "selection": selection, "context": context, "old_contexts_preserved": preserved}


def positive_status(qid: int, doc_id: int, evidence_text: str,
                    prediction: dict[str, Any]) -> dict[str, bool]:
    docs = set(prediction["relevant_docs"])
    contexts = [row["chunk_text"] for row in prediction["relevant_chunks"]
                if int(row["doc_id"]) == doc_id]
    return {"document": doc_id in docs, "chunk_parent": bool(contexts),
            "evidence_visible": any(normalize(evidence_text) in normalize(text) for text in contexts)}


def replay_scope(output: Path, scope: str) -> dict[str, Any]:
    if scope not in {"canary", "cohort"}:
        raise ValueError("Unsupported replay scope")
    manifest = validate_package(output)
    values = load_scores(output, scope)
    config, _, store, _, _ = environment()
    policy = SelectionPolicy(**config["selection"])
    s2 = load_s2()
    query_ids = list(CANARY_IDS) if scope == "canary" else manifest["query_ids"]
    pairs_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(output / "pairs.jsonl"):
        if row["query_id"] in query_ids:
            pairs_by_query[int(row["query_id"])].append(row)
    expected_ids = {row["pair_id"] for rows in pairs_by_query.values() for row in rows}
    if set(values) != expected_ids:
        raise ValueError("Imported score coverage differs from replay scope")
    results = {qid: replay_query(qid, pairs_by_query[qid], values, store, policy, s2)
               for qid in query_ids}
    registry = pq.read_table(R6_REGISTRY).to_pylist()
    known = [row for row in registry if int(row["query_id"]) in query_ids]
    changes = []
    for row in known:
        qid, doc_id = int(row["query_id"]), int(row["doc_id"])
        before = positive_status(qid, doc_id, row["evidence_text"], s2[qid])
        after = positive_status(qid, doc_id, row["evidence_text"], results[qid]["prediction"])
        changes.append({"query_id": qid, "doc_id": doc_id, "registry_id": row["registry_id"],
                        "label_group": row["label_group"], "before": before, "after": after,
                        "document_gain": not before["document"] and after["document"],
                        "document_loss": before["document"] and not after["document"],
                        "evidence_gain": not before["evidence_visible"] and after["evidence_visible"],
                        "evidence_loss": before["evidence_visible"] and not after["evidence_visible"]})
    target_rows = []
    for qid, doc_id in TARGETS.items():
        rows = [row for row in changes if row["query_id"] == qid and row["doc_id"] == doc_id]
        if not rows:
            raise ValueError(f"Target positive registry row missing: {qid}/{doc_id}")
        target_rows.append(rows[0])
    control_losses = [row for row in changes if row["query_id"] == 53
                      and (row["document_loss"] or row["evidence_loss"])]
    structural_pass = all(len(value["prediction"]["relevant_docs"]) <= 200
                          and len(value["prediction"]["relevant_chunks"]) <= 60
                          for value in results.values())
    canary_pass = (scope != "canary" or
                   (any(row["after"]["document"] for row in target_rows)
                    and not control_losses and structural_pass))
    status = ("PASS_OPEN_COHORT" if canary_pass else "STOP_CANARY_NO_DOWNSTREAM_GAIN") \
        if scope == "canary" else "COHORT_DIAGNOSTIC_COMPLETE"
    report = {
        "format": f"v4-r7-parent-floor-{scope}-replay-v1", "status": status,
        "query_count": len(query_ids), "new_pairs": len(values),
        "targets": target_rows, "control_53_losses": control_losses,
        "known_positive_changes": {
            "document_gains": sum(row["document_gain"] for row in changes),
            "document_losses": sum(row["document_loss"] for row in changes),
            "evidence_gains": sum(row["evidence_gain"] for row in changes),
            "evidence_losses": sum(row["evidence_loss"] for row in changes),
            "queries_with_any_loss": sorted({row["query_id"] for row in changes
                                              if row["document_loss"] or row["evidence_loss"]}),
        },
        "structural_validators": "PASS" if structural_pass else "FAIL",
        "official_f2": "UNMEASURED",
        "scope_note": "Incomplete known-positive diagnostic; not independent evidence relevance or official F2.",
        "changes": changes,
    }
    replay_dir = output / "replay" / scope
    replay_dir.mkdir(parents=True, exist_ok=False)
    for qid, result in results.items():
        write_once(replay_dir / f"{qid}.json", result)
    write_once(output / f"{scope}_report.json", report)
    write_atomic_replace(output / "status.json", {"status": status, "scope": scope})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "validate-package", "score-canary",
                                            "replay-canary", "score-cohort", "replay-cohort"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "package":
        result = package(args.output)
    elif args.command == "validate-package":
        result = validate_package(args.output)
    elif args.command == "score-canary":
        result = score_scope(args.output, "canary")
    elif args.command == "replay-canary":
        result = replay_scope(args.output, "canary")
    elif args.command == "score-cohort":
        result = score_scope(args.output, "cohort")
    else:
        result = replay_scope(args.output, "cohort")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
