"""Measure high-recall retrieval coverage for the locked R11 cohort."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import json
import resource
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.v4.run_p5_2_split import CONFIG, ROOT, read
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import containment
from src.medrag.retrieval.core import fuse
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.disk_dense import DiskDense

OUT = ROOT / "outputs/v4/r11_high_recall_coverage_20261026"
R6 = ROOT / "outputs/v4/r6_stage_oracle_20261025"
R7 = ROOT / "outputs/v4/r7_targeted_parent_loss_20261025"
EVAL = ROOT / "outputs/v4/evidence_evaluation_lock_20261025"
CAP10 = ROOT / "outputs/v4/evidence_window_cap10_full_20261025"
DEPTHS = (400, 800, 2000, 4000)
BRANCHES = ("document_bm25", "passage_bm25", "dense")
CANARY_IDS = (53, 130, 380)
BUDGET_SECONDS = 14_400


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def normalize_query(value: str) -> str:
    return " ".join(html.unescape(unicodedata.normalize("NFKC", value)).lower().split())


def index_objects():
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("scope") != "full_release":
        raise ValueError("R11 requires the complete full-release index")
    if manifest.get("fingerprint") != "8f8245e8357137b53a1a4a927d2e6532c65aa2ed60caf495997b74ca97603eb8":
        raise ValueError("R11 index fingerprint differs from the frozen full index")
    doc = DiskBM25(index / "documents.sqlite", "document_bm25", manifest["fingerprint"])
    passage = DiskBM25(index / "passages.sqlite", "passage_bm25", manifest["fingerprint"])
    from src.medrag.retrieval.bge_encoder import BGEM3Encoder

    dense_cfg = config["dense"]
    encoder = BGEM3Encoder(dense_cfg["model"], revision=dense_cfg["revision"],
                           document_max_length=dense_cfg["document_max_length"],
                           query_max_length=dense_cfg["query_max_length"], device="cpu")
    dense = DiskDense(index / "dense", encoder, manifest["dense"]["fingerprint"],
                      block_rows=32768)
    return config, index, manifest, {"document_bm25": doc, "passage_bm25": passage,
                                    "dense": dense}


def retrieval(query: str, indexes: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {
        "document_bm25": indexes["document_bm25"].search(query, 4000),
        "passage_bm25": indexes["passage_bm25"].search(query, 4000),
        "dense": indexes["dense"].search(query, 4000),
    }
    return {name: [asdict(hit) for hit in hits] for name, hits in result.items()}


def dense_search_fp32(dense: DiskDense, query: str, limit: int = 4000):
    """FP32 cosine scan over the locked FP32 matrix; identities/tie-break are unchanged."""
    import numpy as np
    from src.medrag.core.contracts import Hit

    meta = dense._check(require_complete=True)
    vector = np.asarray(dense.encoder.encode_query([query])[0], dtype=np.float32)
    query_norm = np.linalg.norm(vector)
    if vector.shape != (int(meta["dimension"]),) or not np.isfinite(vector).all() or query_norm == 0:
        raise ValueError("FP32 dense query vector is invalid")
    matrix = np.load(dense.matrix_path, mmap_mode="r")
    count = int(meta["rows"])
    candidates = []
    for start in range(0, count, dense.block_rows):
        end = min(start + dense.block_rows, count)
        block = np.asarray(matrix[start:end])
        scores = (block @ vector) / (np.linalg.norm(block, axis=1) * query_norm)
        keep = min(limit, len(scores))
        threshold = float(np.partition(scores, len(scores) - keep)[len(scores) - keep])
        local = np.flatnonzero(scores >= threshold)
        candidates.extend((float(scores[index]), start + int(index)) for index in local)
    identities = {}
    with sqlite3.connect(f"file:{dense.mapping}?mode=ro", uri=True) as db:
        for offset in range(0, len(candidates), 800):
            row_ids = [row_id for _, row_id in candidates[offset:offset + 800]]
            sql = ("SELECT row_id,official_id,passage_id FROM rows WHERE row_id IN (" +
                   ",".join("?" for _ in row_ids) + ")")
            identities.update((int(row_id), (int(parent), str(passage_id)))
                              for row_id, parent, passage_id in db.execute(sql, row_ids))
    if len(identities) != len(candidates):
        raise ValueError("FP32 dense candidate mapping is incomplete")
    ordered = sorted(candidates, key=lambda item: (-item[0], identities[item[1]][0],
                                                    identities[item[1]][1]))[:limit]
    return [Hit(identities[row_id][0], identities[row_id][1], score, "dense", rank)
            for rank, (score, row_id) in enumerate(ordered, 1)]


def retrieval_fp32(query: str, indexes: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {
        "document_bm25": indexes["document_bm25"].search(query, 4000),
        "passage_bm25": indexes["passage_bm25"].search(query, 4000),
        "dense": dense_search_fp32(indexes["dense"], query, 4000),
    }
    return {name: [asdict(hit) for hit in hits] for name, hits in result.items()}


def file_hashes() -> dict[str, str]:
    migration_path = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"
    migration = read(migration_path)
    paths = [
        migration_path,
        ROOT / "configs/v4_submission_p2_a0.yaml",
        ROOT / migration["official_queries"],
        R6 / "positive_registry.parquet",
        R6 / "first_loss_attribution.csv",
        R6 / "manifest.json",
        R7 / "cohort_lock.json",
        EVAL / "query_split_registry.json",
        CAP10 / "export/submission.zip",
        CAP10 / "scoring_receipt_user_20261026.json",
        ROOT / "outputs/v4/task4/full_index_gpu3_20261019/manifest.json",
        ROOT / "outputs/v4/task4/full_index_gpu3_20261019/lock.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def lock_cohort() -> dict[str, Any]:
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    registry = pq.read_table(R6 / "positive_registry.parquet").to_pylist()
    rows = list(csv.DictReader((R6 / "first_loss_attribution.csv").open(newline="")))
    manual_keys = {(int(row["query_id"]), int(row["doc_id"]))
                   for row in rows
                   if row["label_group"] in ("manual_positive", "historical_and_manual_positive")
                   and row["source_present"] == "True"
                   and row["source_contains_evidence"] == "True"}
    if len(manual_keys) != 87:
        raise ValueError(f"Expected 87 verified manual query-parent pairs, got {len(manual_keys)}")
    manual_registry = {}
    for row in registry:
        qid, doc = int(row["query_id"]), int(row["doc_id"])
        if (qid, doc) in manual_keys and row["label_group"] in (
                "manual_positive", "historical_and_manual_positive"):
            manual_registry[(qid, doc)] = row
    if set(manual_registry) != manual_keys:
        raise ValueError("R6 manual attribution and positive registry identities differ")

    manual_query_path = ROOT / "data/raw/vibiomir/query.parquet"
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    queries = {int(row["id"]): str(row["query"])
               for row in pq.read_table(query_path, columns=["id", "query"]).to_pylist()}
    assignments = read(EVAL / "query_split_registry.json")["assignments"]
    r7 = read(R7 / "cohort_lock.json")
    controls = [row for row in r7["rows"] if row["cohort"] == "control_retained"]
    if len(controls) != 10 or len({int(row["query_id"]) for row in controls}) != 10:
        raise ValueError("R7 must provide exactly ten query-distinct retained controls")

    source_store = DiskCorpusStore(index / "corpus.sqlite", manifest["corpus_version"],
                                   manifest["fingerprint"])
    manual = []
    for qid, doc in sorted(manual_keys):
        label = manual_registry[(qid, doc)]
        if qid not in queries or str(qid) not in assignments:
            raise ValueError(f"Manual query missing from official query/split registry: {qid}")
        source = source_store.source(doc)
        if source.text_sha256 != label["source_sha256"]:
            raise ValueError(f"R6 source SHA mismatch for {qid}/{doc}")
        matched, start, end = containment(source.text, str(label["evidence_text"]))
        if not matched or start is None or end is None:
            raise ValueError(f"Corrected source evidence round-trip failed for {qid}/{doc}")
        manual.append({
            "query_id": qid, "doc_id": doc,
            "registry_id": label["registry_id"],
            "query_sha256": digest_text(queries[qid]),
            "source_sha256": source.text_sha256,
            "evidence_text_sha256": digest_text(str(label["evidence_text"])),
            "source_start": start, "source_end": end,
            "split": assignments[str(qid)],
        })

    cap10 = read(CAP10 / "export/predictions.json")
    cap_docs = {int(row["id"]): set(map(int, row["relevant_docs"])) for row in cap10}
    if len(cap_docs) != 1200:
        raise ValueError("Cap-10 predictions must cover all 1,200 queries")
    for pair in manual:
        pair["champion_present"] = pair["doc_id"] in cap_docs[pair["query_id"]]
    manual_missing = sum(not pair["champion_present"] for pair in manual)
    if manual_missing != 11:
        raise ValueError(f"Expected 11 manual parents outside champion docs, got {manual_missing}")

    unique_qids = sorted({pair["query_id"] for pair in manual}
                         | {int(row["query_id"]) for row in controls})
    if len(unique_qids) != 94:
        raise ValueError(f"Expected 94 unique manual/control queries, got {len(unique_qids)}")
    payload = {
        "format": "v4-r11-high-recall-cohort-v1",
        "status": "LOCKED",
        "manual_pair_count": len(manual),
        "manual_query_count": len({pair["query_id"] for pair in manual}),
        "manual_champion_covered": len(manual) - manual_missing,
        "manual_champion_misses": manual_missing,
        "controls": [{"query_id": int(row["query_id"]), "doc_id": int(row["doc_id"]),
                      "registry_id": row["registry_id"],
                      "split": assignments[str(row["query_id"])]}
                     for row in controls],
        "query_ids": unique_qids,
        "manual_pairs": manual,
        "normalization": "src.medrag.evidence.identity.normalize (NFKC, HTML decode, lowercase, collapse whitespace); no synonym lexicon",
        "retrieval": {"depths": DEPTHS, "branches": BRANCHES,
                      "rrf_k": 60, "dense_device": "cpu"},
        "budget_cpu_seconds": BUDGET_SECONDS,
        "input_sha256": file_hashes(),
    }
    return payload


def write_lock() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cohort_lock.json"
    payload = lock_cohort()
    if path.exists():
        old = read(path)
        if old != payload:
            raise ValueError("Existing R11 cohort lock differs from current inputs")
        return old
    write_once(path, payload)
    return payload


def prefix_ids(rows: list[dict[str, Any]], count: int) -> list[tuple[int, str | None]]:
    return [(int(row["official_id"]), row.get("passage_id")) for row in rows[:count]]


def max_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1024 if peak < 10**9 else 1))


def load_queries() -> dict[int, str]:
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    path = ROOT / migration["official_queries"]
    return {int(row["id"]): str(row["query"])
            for row in pq.read_table(path, columns=["id", "query"]).to_pylist()}


def run_variant_pair(qid: int, query: str, indexes: dict[str, Any], *, fast: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    raw_started = time.process_time()
    retrieve = retrieval_fp32 if fast else retrieval
    raw = retrieve(query, indexes)
    raw_wall = time.perf_counter() - started
    raw_cpu = time.process_time() - raw_started
    normalized_query = normalize_query(query)
    if normalized_query == query:
        normalized = raw
        normalized_wall = 0.0
        normalized_cpu = 0.0
        normalized_reused = True
    else:
        normalized_started = time.perf_counter()
        normalized_cpu_started = time.process_time()
        normalized = retrieve(normalized_query, indexes)
        normalized_wall = time.perf_counter() - normalized_started
        normalized_cpu = time.process_time() - normalized_cpu_started
        normalized_reused = False
    return {
        "query_id": qid,
        "raw_query_sha256": digest_text(query),
        "normalized_query_sha256": digest_text(normalized_query),
        "normalized_changed": normalized_query != query,
        "normalized_reused_raw_hits": normalized_reused,
        "dense_precision": "fp32" if fast else "fp64_reference",
        "variants": {"raw": raw, "normalized": normalized},
        "timing": {"raw_wall_seconds": raw_wall, "raw_cpu_seconds": raw_cpu,
                   "normalized_wall_seconds": normalized_wall,
                   "normalized_cpu_seconds": normalized_cpu},
        "max_rss_bytes": max_rss_bytes(),
    }


def canary(*, fast: bool = False) -> dict[str, Any]:
    lock = read(OUT / "cohort_lock.json")
    queries = load_queries()
    _, _, _, indexes = index_objects()
    traces = OUT / ("traces_fast" if fast else "traces")
    traces.mkdir(parents=True, exist_ok=True)
    pilot_dir = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2/queries"
    results = []
    total_wall = 0.0
    total_cpu = 0.0
    for qid in CANARY_IDS:
        if qid not in lock["query_ids"]:
            raise ValueError(f"Canary query {qid} is absent from R11 cohort")
        trace_path = traces / f"{qid}.json"
        if trace_path.exists():
            saved = read(trace_path)
        else:
            saved = run_variant_pair(qid, queries[qid], indexes)
            write_once(trace_path, saved)
        frozen = read(pilot_dir / f"{qid}.json")["branch_trace"]
        raw = saved["variants"]["raw"]
        branches = {}
        for name, expected_limit in (("document_bm25", 400),
                                     ("passage_bm25", 200), ("dense", 200)):
            actual = prefix_ids(raw[name], expected_limit)
            expected = prefix_ids(frozen["branch_hits"][name], expected_limit)
            branches[name] = {"exact_prefix": actual == expected,
                              "actual_count": len(actual), "expected_count": len(expected)}
        frozen_fused = prefix_ids(frozen["fused"], 400)
        reference_fused = fuse(
            [[type("HitLike", (), {"official_id": int(row["official_id"]),
                                    "passage_id": row.get("passage_id"),
                                    "score": float(row["score"]), "channel": name,
                                    "rank": int(row["rank"])})()
              for row in raw[name][:expected_limit]]
             for name, expected_limit in (("document_bm25", 400),
                                          ("passage_bm25", 200), ("dense", 200))],
            k=60, limit=400)
        fused_ids = [(int(row.official_id), row.passage_id) for row in reference_fused]
        fused_ok = fused_ids == frozen_fused
        results.append({"query_id": qid, "branches": branches,
                        "s2_mixed_fusion_exact": fused_ok,
                        "elapsed": saved["timing"],
                        "normalized_changed": saved["normalized_changed"]})
        total_wall += sum(saved["timing"][key] for key in
                          ("raw_wall_seconds", "normalized_wall_seconds"))
        total_cpu += sum(saved["timing"][key] for key in
                         ("raw_cpu_seconds", "normalized_cpu_seconds"))
    parity = all(all(branch["exact_prefix"] for branch in row["branches"].values())
                 and row["s2_mixed_fusion_exact"] for row in results)
    projection_cpu = total_cpu / len(results) * len(lock["query_ids"])
    status = "PASS" if parity and projection_cpu <= BUDGET_SECONDS else (
        "STOP_PARITY" if not parity else "STOP_BUDGET")
    report = {
        "format": "v4-r11-canary-v1", "status": status,
        "parity_pass": parity, "budget_seconds": BUDGET_SECONDS,
        "canary_queries": list(CANARY_IDS), "cohort_queries": len(lock["query_ids"]),
        "canary_wall_seconds": total_wall, "canary_cpu_seconds": total_cpu,
        "projected_cohort_cpu_seconds": projection_cpu,
        "dense_precision": "fp32" if fast else "fp64_reference",
        "max_rss_bytes": max(max_rss_bytes(), *(read(traces / f"{qid}.json")["max_rss_bytes"]
                                               for qid in CANARY_IDS)),
        "results": results,
    }
    write_once(OUT / ("runtime_canary_fast.json" if fast else "runtime_canary.json"), report)
    return report


def parent_ranks(hits: list[dict[str, Any]], depth: int) -> list[int]:
    parents = []
    seen = set()
    for hit in hits[:depth]:
        parent = int(hit["official_id"])
        if parent not in seen:
            seen.add(parent)
            parents.append(parent)
    return parents


def fuse_parents(branch_orders: dict[str, list[int]], k: int = 60) -> list[dict[str, Any]]:
    rank_maps = {channel: {parent: rank for rank, parent in enumerate(order, 1)}
                 for channel, order in branch_orders.items()}
    scores: dict[int, float] = {}
    for channel, ranks in rank_maps.items():
        for parent, rank in ranks.items():
            scores[parent] = scores.get(parent, 0.0) + 1.0 / (k + rank)
    return [{"doc_id": parent, "score": score, "rank": index}
            for index, (parent, score) in enumerate(
                sorted(scores.items(), key=lambda item: (-item[1], item[0])), 1)]


def arm_orders(trace: dict[str, Any], arm: str, depth: int) -> dict[str, list[int]]:
    raw = trace["variants"]["raw"]
    normalized = trace["variants"]["normalized"]
    orders = {}
    for branch in BRANCHES:
        raw_order = parent_ranks(raw[branch], depth)
        if arm == "raw":
            orders[branch] = raw_order
        elif arm == "normalized":
            orders[branch] = parent_ranks(normalized[branch], depth)
        elif arm == "union":
            norm_order = parent_ranks(normalized[branch], depth)
            ranks = {parent: rank for rank, parent in enumerate(raw_order, 1)}
            for rank, parent in enumerate(norm_order, 1):
                ranks[parent] = min(ranks.get(parent, rank), rank)
            orders[branch] = [parent for parent, _ in
                              sorted(ranks.items(), key=lambda item: (item[1], item[0]))]
        else:
            raise ValueError(arm)
    return orders


def evaluate(*, fast: bool = False) -> dict[str, Any]:
    lock = read(OUT / "cohort_lock.json")
    canary_report = read(OUT / ("runtime_canary_fast.json" if fast else "runtime_canary.json"))
    if canary_report["status"] != "PASS":
        raise ValueError("R11 full cohort requires passing parity and budget canary")
    traces_dir = OUT / ("traces_fast" if fast else "traces")
    expected = set(map(int, lock["query_ids"]))
    traces = {int(path.stem): read(path) for path in traces_dir.glob("*.json")}
    if set(traces) != expected:
        raise ValueError(f"Missing R11 traces: {sorted(expected - set(traces))[:12]}")

    manual = lock["manual_pairs"]
    controls = lock["controls"]
    champion_docs = {int(pair["doc_id"]): bool(pair["champion_present"]) for pair in manual}
    manual_by_split: dict[str, list[dict[str, Any]]] = {}
    for pair in manual:
        manual_by_split.setdefault(pair["split"], []).append(pair)
    rows = []
    arm_depth_data: dict[tuple[str, int, int], dict[str, Any]] = {}
    for qid in sorted(expected):
        trace = traces[qid]
        manual_here = [pair for pair in manual if int(pair["query_id"]) == qid]
        controls_here = [item for item in controls if int(item["query_id"]) == qid]
        for depth in DEPTHS:
            for arm in ("raw", "normalized", "union"):
                orders = arm_orders(trace, arm, depth)
                fused_rows = fuse_parents(orders)
                fused_400 = {int(row["doc_id"]) for row in fused_rows[:400]}
                fused_2000 = {int(row["doc_id"]) for row in fused_rows[:2000]}
                raw_union = set().union(*(set(order) for order in orders.values()))
                manual_covered = [pair for pair in manual_here if int(pair["doc_id"]) in fused_2000]
                gains = [pair for pair in manual_here if not pair["champion_present"]
                         and int(pair["doc_id"]) in fused_2000]
                losses = [pair for pair in manual_here if pair["champion_present"]
                          and int(pair["doc_id"]) not in fused_2000]
                controls_kept = [item for item in controls_here
                                 if int(item["doc_id"]) in fused_2000]
                row = {
                    "query_id": qid, "split": (manual_here[0]["split"] if manual_here
                                                 else controls_here[0]["split"]),
                    "arm": arm, "depth": depth,
                    "raw_union_manual": sum(int(pair["doc_id"]) in raw_union for pair in manual_here),
                    "fused400_manual": sum(int(pair["doc_id"]) in fused_400 for pair in manual_here),
                    "fused2000_manual": len(manual_covered),
                    "manual_gains2000": len(gains), "manual_losses2000": len(losses),
                    "controls_total": len(controls_here), "controls_kept2000": len(controls_kept),
                    "unique_parents": len(fused_rows),
                    "fused400_parent_ids": sorted(fused_400),
                    "fused2000_parent_ids": sorted(fused_2000),
                    "manual_gained_doc_ids": sorted(int(pair["doc_id"]) for pair in gains),
                    "manual_lost_doc_ids": sorted(int(pair["doc_id"]) for pair in losses),
                    "control_lost_doc_ids": sorted(int(item["doc_id"]) for item in controls_here
                                                    if int(item["doc_id"]) not in fused_2000),
                }
                rows.append(row)
                arm_depth_data[(arm, depth, qid)] = {
                    "fused400": fused_400, "fused2000": fused_2000,
                    "raw_union": raw_union, "row": row,
                }
    suffix = "_fast" if fast else ""
    csv_path = OUT / f"coverage_by_query_arm_depth{suffix}.csv"
    if not csv_path.exists():
        fields = [key for key, value in rows[0].items() if not isinstance(value, (list, dict))]
        with csv_path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: value for key, value in row.items() if key in fields}
                              for row in rows])

    aggregate = []
    decisions = []
    split_assignments = read(EVAL / "query_split_registry.json")["assignments"]
    for arm in ("raw", "normalized", "union"):
        for depth in DEPTHS:
            covered = gains = losses = top400_gains = controls_total = controls_kept = 0
            retained_total = retained_kept = 0
            split_summary = {}
            for pair in manual:
                data = arm_depth_data[(arm, depth, int(pair["query_id"]))]
                parent = int(pair["doc_id"])
                present = parent in data["fused2000"]
                covered += present
                gains += (not pair["champion_present"]) and present
                losses += pair["champion_present"] and not present
                retained_total += bool(pair["champion_present"])
                retained_kept += bool(pair["champion_present"]) and present
                split = pair["split"]
                bucket = split_summary.setdefault(split, {"manual": 0, "baseline_misses": 0,
                                                           "covered": 0, "gains": 0,
                                                           "top400_gains": 0, "losses": 0})
                bucket["manual"] += 1
                bucket["baseline_misses"] += not pair["champion_present"]
                bucket["covered"] += present
                bucket["gains"] += (not pair["champion_present"]) and present
                bucket["top400_gains"] += (not pair["champion_present"] and
                                           parent in data["fused400"])
                bucket["losses"] += pair["champion_present"] and not present
                top400_gains += not pair["champion_present"] and parent in data["fused400"]
            for control in controls:
                data = arm_depth_data[(arm, depth, int(control["query_id"]))]
                controls_total += 1
                controls_kept += int(control["doc_id"]) in data["fused2000"]
            ratio = top400_gains / gains if gains else 1.0
            gate = {
                "recovers_10_of_11": gains >= 10,
                "net_coverage_85_of_87": covered >= 85,
                "all_controls": controls_kept == controls_total == 10,
                "all_retained_manual": retained_kept == retained_total == 76,
                "gains_top400_ge_80pct": ratio >= .8,
            }
            split_direction = all(
                bucket["gains"] > 0 and bucket["losses"] == 0
                for split, bucket in split_summary.items()
                if split in ("validation", "held_out") and bucket["baseline_misses"] > 0)
            gate["validation_heldout_direction"] = split_direction
            row = {"arm": arm, "depth": depth,
                   "manual_covered_fused2000": covered, "manual_denominator": len(manual),
                   "manual_gains": gains, "manual_losses": losses,
                   "gains_in_fused400": top400_gains,
                   "gains_top400_fraction": ratio,
                   "controls_kept": controls_kept, "controls_total": controls_total,
                   "retained_manual_kept": retained_kept,
                   "retained_manual_total": retained_total,
                   "split_summary": split_summary, "gate": gate,
                   "go": all(gate.values())}
            aggregate.append(row)
            dev = split_summary.get("development", {})
            decisions.append((int(dev.get("top400_gains", 0)),
                              int(dev.get("gains", 0)), -depth,
                              {"arm": arm, "depth": depth}))
    selected = max(decisions, key=lambda item: (item[0], item[1], item[2]))[3]
    selected_row = next(row for row in aggregate if row["arm"] == selected["arm"]
                        and row["depth"] == selected["depth"])
    status = "GO" if selected_row["go"] else (
        "PARTIAL_COVERAGE" if selected_row["manual_gains"] > 0 else "STOP")
    decision = {
        "format": "v4-r11-decision-v1", "status": status,
        "selected_on_development": selected,
        "selected_metrics": selected_row,
        "all_arm_depth_metrics": aggregate,
        "interpretation": "Known-positive coverage only; official F2 and independent relevance are unmeasured.",
    }
    write_once(OUT / f"coverage_by_arm_depth{suffix}.json", {"rows": aggregate})
    write_once(OUT / f"decision{suffix}.json", decision)
    return decision


def run_full(*, fast: bool = False) -> dict[str, Any]:
    lock = read(OUT / "cohort_lock.json")
    canary_report = read(OUT / ("runtime_canary_fast.json" if fast else "runtime_canary.json"))
    if canary_report["status"] != "PASS":
        raise ValueError("Full R11 retrieval requires passing canary")
    queries = load_queries()
    _, _, _, indexes = index_objects()
    trace_dir = OUT / ("traces_fast" if fast else "traces")
    trace_dir.mkdir(exist_ok=True)
    started = time.process_time()
    done = 0
    for qid in lock["query_ids"]:
        path = trace_dir / f"{qid}.json"
        if path.exists():
            saved = read(path)
            if saved.get("query_id") != qid:
                raise ValueError(f"Trace identity mismatch at {qid}")
            continue
        saved = run_variant_pair(qid, queries[qid], indexes, fast=fast)
        write_once(path, saved)
        done += 1
        elapsed = time.process_time() - started
        print(json.dumps({"completed_new": done, "query_id": qid,
                          "elapsed_cpu_seconds": elapsed,
                          "max_rss_bytes": saved["max_rss_bytes"]}), flush=True)
        if elapsed > BUDGET_SECONDS:
            raise RuntimeError("R11 retrieval exceeded the locked 4 CPU-hour budget")
    traces = {int(path.stem) for path in trace_dir.glob("*.json")}
    if traces != set(map(int, lock["query_ids"])):
        raise ValueError("R11 trace coverage does not match locked query cohort")
    return evaluate(fast=fast)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("lock", "canary", "fastcanary", "run", "fast", "evaluate", "fast-evaluate", "all"))
    args = parser.parse_args()
    if args.phase == "lock":
        print(json.dumps(write_lock(), ensure_ascii=False))
    elif args.phase == "canary":
        print(json.dumps(canary(), ensure_ascii=False, indent=2))
    elif args.phase == "fastcanary":
        print(json.dumps(canary(fast=True), ensure_ascii=False, indent=2))
    elif args.phase == "run":
        print(json.dumps(run_full(), ensure_ascii=False, indent=2))
    elif args.phase == "fast":
        print(json.dumps(run_full(fast=True), ensure_ascii=False, indent=2))
    elif args.phase == "evaluate":
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
    elif args.phase == "fast-evaluate":
        print(json.dumps(evaluate(fast=True), ensure_ascii=False, indent=2))
    else:
        write_lock()
        report = canary()
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report["status"] == "PASS":
            print(json.dumps(run_full(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
