"""Build a full-passage candidate pool for the locked 87 manual evidence pairs."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from scripts.v4.run_p5_2_split import ROOT
from scripts.v4.run_r11_high_recall_coverage import (
    CAP10, OUT as R11, fuse_parents, index_objects, load_queries,
    parent_ranks, read as read_json, retrieval, write_once,
)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import normalize

OUTPUT = ROOT / "outputs/v4/manual87_parent_recovery_20261026"
DEPTH = 4000
PARENT_CAP = 4000


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, value)


def locked_pairs(lock: dict) -> tuple[list[dict], list[dict]]:
    pairs = list(lock["manual_pairs"])
    misses = [row for row in pairs if not row["champion_present"]]
    if len(pairs) != 87 or len({int(row["query_id"]) for row in pairs}) != 87:
        raise ValueError("Expected 87 query-distinct locked manual evidence pairs")
    if len(misses) != 11 or len({int(row["query_id"]) for row in misses}) != 11:
        raise ValueError("Expected eleven query-distinct cap-10 manual parent misses")
    return pairs, misses


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = read_json(R11 / "cohort_lock.json")
    pairs, misses = locked_pairs(lock)
    retrieval_qids = sorted({int(row["query_id"]) for row in misses})
    qids = sorted({int(row["query_id"]) for row in pairs})
    queries = load_queries()
    _, index, index_manifest, indexes = index_objects()
    if index_manifest["fingerprint"] != "8f8245e8357137b53a1a4a927d2e6532c65aa2ed60caf495997b74ca97603eb8":
        raise ValueError("Parent recovery index differs from the frozen full-release index")

    OUTPUT.mkdir(parents=True)
    trace_dir = OUTPUT / "traces"
    trace_dir.mkdir()
    total_cpu = 0.0
    for offset, qid in enumerate(retrieval_qids, 1):
        started = time.process_time()
        raw = retrieval(queries[qid], indexes)
        elapsed = time.process_time() - started
        total_cpu += elapsed
        trace = {
            "format": "v4-manual87-parent-recovery-trace-v1",
            "query_id": qid,
            "query_sha256": digest(queries[qid]),
            "variant": "official raw query",
            "depth_per_branch": DEPTH,
            "dense_precision": "fp64_reference",
            "branches": raw,
            "cpu_seconds": elapsed,
        }
        write_json(trace_dir / f"{qid}.json", trace)
        print(json.dumps({"query": qid, "completed": offset, "total": len(retrieval_qids),
                          "cpu_seconds": round(elapsed, 2),
                          "elapsed_total_cpu_seconds": round(total_cpu, 2)}), flush=True)
        if total_cpu > int(lock["budget_cpu_seconds"]):
            raise RuntimeError("Targeted raw-query recovery exceeded the locked 4 CPU-hour budget")

    cap10 = read_json(CAP10 / "export/predictions.json")
    cap_docs = {int(row["id"]): list(map(int, row["relevant_docs"])) for row in cap10}
    target_by_query: dict[int, list[dict]] = defaultdict(list)
    for row in pairs:
        target_by_query[int(row["query_id"])].append(row)

    index_db = index / "corpus.sqlite"
    reports = []
    pool_hashes = {}
    for qid in qids:
        trace_path = trace_dir / f"{qid}.json"
        trace = read_json(trace_path) if trace_path.exists() else None
        branch_orders = ({
            name: parent_ranks(trace["branches"][name], DEPTH)
            for name in ("document_bm25", "passage_bm25", "dense")
        } if trace is not None else {
            "document_bm25": [], "passage_bm25": [], "dense": [],
        })
        fused = fuse_parents(branch_orders, k=60)
        fused_rank = {int(row["doc_id"]): int(row["rank"]) for row in fused}
        branch_rank = {
            name: {parent: rank for rank, parent in enumerate(order, 1)}
            for name, order in branch_orders.items()
        }
        pool_ids = list(cap_docs[qid])
        seen = set(pool_ids)
        for row in fused:
            parent = int(row["doc_id"])
            if parent not in seen:
                pool_ids.append(parent)
                seen.add(parent)
            if len(pool_ids) >= PARENT_CAP:
                break

        parent_rows = []
        passage_ids = []
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as db:
            for start in range(0, len(pool_ids), 800):
                batch = pool_ids[start:start + 800]
                marks = ",".join("?" for _ in batch)
                rows = db.execute(
                    "SELECT official_id,passage_id,source_start,source_end,source_sha256 "
                    f"FROM passages WHERE official_id IN ({marks}) "
                    "ORDER BY official_id,source_start,source_end,passage_id", batch
                ).fetchall()
                passages_by_parent: dict[int, list[dict]] = defaultdict(list)
                for doc, passage_id, lo, hi, source_sha in rows:
                    item = {"passage_id": passage_id, "start": lo, "end": hi,
                            "source_sha256": source_sha}
                    passages_by_parent[int(doc)].append(item)
                    passage_ids.append(passage_id)
                for doc in batch:
                    parent_rows.append({
                        "official_id": doc,
                        "baseline_document_rank": (cap_docs[qid].index(doc) + 1
                                                   if doc in cap_docs[qid] else None),
                        "fused_parent_rank": fused_rank.get(doc),
                        "branch_parent_ranks": {name: ranks.get(doc)
                                                for name, ranks in branch_rank.items()},
                        "passages": passages_by_parent.get(doc, []),
                    })
        if len(parent_rows) != len(pool_ids) or len(passage_ids) != len(set(passage_ids)):
            raise ValueError(f"Candidate pool identity mismatch for query {qid}")
        pool = {
            "format": "v4-manual87-full-parent-passage-pool-v1",
            "query_id": qid,
            "query_sha256": digest(queries[qid]),
            "retrieval_trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
            "candidate_policy": {
                "union": "all cap-10 documents plus unique-parent RRF over raw document BM25, passage BM25 and dense for the eleven missing-parent queries",
                "rrf_k": 60,
                "parent_cap": PARENT_CAP,
                "passage_policy": "all registered source-grounded passage identities for each candidate parent",
                "manual_target_used_for_parent_or_passage_selection": False,
                "cohort_note": "locked manual query IDs define the evaluation cohort; target parent and text are read only after pool creation for coverage measurement",
            },
            "parents": parent_rows,
            "parent_count": len(parent_rows),
            "passage_count": len(passage_ids),
        }
        pool_path = OUTPUT / "pools" / f"{qid}.json"
        write_json(pool_path, pool)
        pool_hashes[str(qid)] = sha256_file(pool_path)

        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as db:
            for target in target_by_query[qid]:
                doc = int(target["doc_id"])
                source_row = db.execute(
                    "SELECT text FROM sources WHERE official_id=?", (doc,)
                ).fetchone()
                if source_row is None:
                    raise ValueError(f"Missing source for locked target {qid}/{doc}")
                source_text = source_row[0]
                if digest(source_text) != target["source_sha256"]:
                    raise ValueError(f"Source identity changed for locked target {qid}/{doc}")
                evidence = normalize(source_text[target["source_start"]:target["source_end"]])
                if not evidence:
                    raise ValueError(f"Empty corrected evidence span {qid}/{doc}")
                entry = next((item for item in parent_rows if item["official_id"] == doc), None)
                if entry is None:
                    passage_hit = False
                else:
                    ids = [item["passage_id"] for item in entry["passages"]]
                    passage_text = {}
                    for batch_start in range(0, len(ids), 800):
                        batch = ids[batch_start:batch_start + 800]
                        marks = ",".join("?" for _ in batch)
                        passage_text.update(db.execute(
                            f"SELECT passage_id,text FROM passages WHERE passage_id IN ({marks})",
                            batch).fetchall())
                    passage_hit = any(evidence in normalize(text)
                                      for text in passage_text.values())
                ranks = {name: branch_rank[name].get(doc) for name in branch_rank}
                reports.append({
                    "query_id": qid, "doc_id": doc, "split": target["split"],
                    "registry_id": target["registry_id"],
                    "document_bm25_parent_rank": ranks["document_bm25"],
                    "passage_bm25_parent_rank": ranks["passage_bm25"],
                    "dense_parent_rank": ranks["dense"],
                    "fused_parent_rank": fused_rank.get(doc),
                    "candidate_parent_in_pool": entry is not None,
                    "all_parent_passages_in_pool": entry is not None,
                    "manual_span_in_candidate_passage": passage_hit,
                    "base_cap10_documents_preserved": set(cap_docs[qid]).issubset(seen),
                })
    summary = {
        "format": "v4-manual87-parent-recovery-manifest-v1",
        "status": "COMPLETE",
        "cohort_lock_sha256": sha256_file(R11 / "cohort_lock.json"),
        "index_fingerprint": index_manifest["fingerprint"],
        "cap10_payload_sha256": sha256_file(CAP10 / "export/predictions.json"),
        "depth_per_branch": DEPTH,
        "parent_cap": PARENT_CAP,
        "queries": len(qids),
        "queries_retrieved": len(retrieval_qids),
        "target_pairs": len(reports),
        "candidate_parents": sum(json.loads((OUTPUT / "pools" / f"{qid}.json").read_text())["parent_count"]
                                  for qid in qids),
        "candidate_passages": sum(json.loads((OUTPUT / "pools" / f"{qid}.json").read_text())["passage_count"]
                                   for qid in qids),
        "target_parent_recall": sum(row["candidate_parent_in_pool"] for row in reports),
        "target_passage_coverage": sum(row["manual_span_in_candidate_passage"] for row in reports),
        "target_denominator": len(reports),
        "base_documents_preserved": all(row["base_cap10_documents_preserved"] for row in reports),
        "total_retrieval_cpu_seconds": total_cpu,
        "raw_query_sha256": {str(qid): digest(queries[qid]) for qid in qids},
        "trace_sha256": {str(qid): sha256_file(trace_dir / f"{qid}.json")
                         for qid in retrieval_qids},
        "pool_sha256": pool_hashes,
        "target_metrics_scope": "candidate passage coverage on 87 locked manual evidence pairs; not evidence relevance or official F2",
        "inputs_sha256": {
            "cohort_lock": sha256_file(R11 / "cohort_lock.json"),
            "cap10_predictions": sha256_file(CAP10 / "export/predictions.json"),
            "index_manifest": sha256_file(index / "manifest.json"),
            "retrieval_implementation": sha256_file(ROOT / "scripts/v4/run_r11_high_recall_coverage.py"),
        },
    }
    write_json(OUTPUT / "manual87_parent_recovery_report.json", {
        "summary": summary,
        "targets": reports,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run()
