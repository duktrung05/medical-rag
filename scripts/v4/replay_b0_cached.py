"""Complete a CPU-only B0 replay from exact cached scores; refuse incomplete coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.adapters.validation import canonical_payload, validate_selected
from src.medrag.core.config import load_config
from src.medrag.core.contracts import Hit
from src.medrag.core.passage_first import passage_candidates
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.evidence.verifier import verify
from src.medrag.ranking.score_cache import ScoreCache


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
INDEX = ROOT / "outputs/v4/task4/full_index_gpu3_20261019"
TRACES = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2/queries"
P4 = ROOT / "outputs/v4/p4_b0_candidate_replay_20261021_v2/candidate_comparison.json"
MIGRATION = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, raw: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def main(readiness_path: Path, score_cache_path: Path, output: Path) -> None:
    readiness_path, score_cache_path, output = map(
        Path.resolve, (readiness_path, score_cache_path, output))
    if output.exists():
        raise FileExistsError(output)
    readiness = read(readiness_path)
    contract = readiness["score_contract"]
    cache = ScoreCache(score_cache_path, contract)
    if cache.fingerprint != readiness["score_contract_fingerprint"]:
        raise ValueError("B0 replay score contract mismatch")
    config = load_config(CONFIG)
    manifest = read(INDEX / "manifest.json")
    store = DiskCorpusStore(INDEX / "corpus.sqlite", config["corpus_version"],
                            manifest["fingerprint"])
    p4_rows = read(P4)
    query_ids = [int(row["query_id"]) for row in p4_rows]
    migration = read(MIGRATION)
    query_path = ROOT / migration["official_queries"]
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist() if int(row["id"]) in query_ids}
    all_pairs = {(int(doc), passage_id) for row in p4_rows for doc, passage_id in row["b0_pair_ids"]}
    passages = store.passages_batch(sorted({passage_id for _, passage_id in all_pairs}))
    sources = store.sources_batch({doc for doc, _ in all_pairs})
    policy = SelectionPolicy(**config["selection"])
    selected_by_id = {}
    metrics = []
    missing = []
    for row in p4_rows:
        qid, query = int(row["query_id"]), queries[int(row["query_id"])]
        trace = read(TRACES / f"{qid}.json")
        rankings = [[Hit(int(item["official_id"]), item["passage_id"], float(item["score"]),
                         channel, int(item["rank"]))
                     for item in trace["branch_trace"]["branch_hits"][channel]]
                    for channel in ("passage_bm25", "dense")]
        fused = passage_candidates(rankings, limit=400)[:200]
        expected = [(int(doc), passage_id) for doc, passage_id in row["b0_pair_ids"]]
        if [(hit.official_id, hit.passage_id) for hit in fused] != expected:
            raise ValueError(f"B0 frozen pair order differs: {qid}")
        scored = []
        with sqlite3.connect(f"file:{score_cache_path}?mode=ro", uri=True) as db:
            for hit in fused:
                passage, source = passages[hit.passage_id], sources[hit.official_id]
                if passage.official_id != hit.official_id or not verify(passage, source):
                    raise ValueError(f"Ungrounded B0 pair: {qid}/{hit.passage_id}")
                payload = f"{source.title}\n{passage.text}" if source.title else passage.text
                input_hash = hashlib.sha256(payload.encode()).hexdigest()
                key = cache._key(query, hit.official_id, hit.passage_id, payload)
                found = db.execute(
                    "SELECT score,query_sha256,official_id,passage_id,input_sha256 "
                    "FROM scores WHERE cache_key=?", (key,)).fetchone()
                expected_meta = (hashlib.sha256(query.encode()).hexdigest(), hit.official_id,
                                 hit.passage_id, input_hash)
                if found is None:
                    missing.append({"query_id": qid, "official_id": hit.official_id,
                                    "passage_id": hit.passage_id})
                    continue
                if found[1:] != expected_meta or not math.isfinite(float(found[0])):
                    raise ValueError(f"Invalid cached B0 score: {qid}/{hit.passage_id}")
                scored.append((hit, passage, float(found[0])))
        if len(scored) != 200:
            continue
        by_doc = {}
        for hit, passage, score in scored:
            evidence = {"passage_id": passage.passage_id, "text": passage.text,
                        "start": passage.source_start, "end": passage.source_end,
                        "score_status": "reranked", "rerank_score": score}
            doc = by_doc.setdefault(hit.official_id, {"official_id": hit.official_id,
                                                      "score": hit.score, "evidence": []})
            doc["score"] = max(doc["score"], hit.score)
            doc["evidence"].append(evidence)
        for doc in by_doc.values():
            doc["evidence"].sort(key=lambda item: (-item["rerank_score"], item["passage_id"]))
            doc["rerank_score"] = doc["evidence"][0]["rerank_score"]
        documents = sorted(by_doc.values(), key=lambda item: (
            -item["rerank_score"], -item["score"], item["official_id"]))
        selected, selection_diag = select_results(documents, policy, reranker_enabled=True,
                                                  limit=200)
        # Enumerate every registered neighbor from the read-only store, not only
        # passages that happened to enter the 200-pair B0 candidate pool.
        selected, context_diag = apply_chunk_context(selected, policy, sources, store)
        selected_by_id[qid] = selected
        metrics.append({"query_id": qid, "rerank_pairs": 200,
                        "scored_parents": len(by_doc), "selected_documents": len(selected),
                        "selected_chunks": sum(len(doc["evidence"]) for doc in selected),
                        "selection": selection_diag, "context": context_diag})
    if missing:
        raise ValueError(f"B0 cached replay blocked by {len(missing)} missing exact scores")
    records = validate_selected(query_ids, selected_by_id, store, policy, result_limit=200,
                                reranker_enabled=True,
                                official_ids={doc["official_id"] for rows in selected_by_id.values()
                                              for doc in rows})
    if any(row["selected_chunks"] != 60 for row in metrics):
        raise ValueError("B0 did not produce exactly 60 unique-parent chunks/query")
    output.mkdir(parents=True)
    selected_raw = b"".join((json.dumps({"id": qid, "selected": selected_by_id[qid]},
                                         ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                            for qid in query_ids)
    write_once(output / "selected_results.jsonl", selected_raw)
    write_once(output / "predictions.json", canonical_payload(records))
    report = {
        "format": "medrag-v4-b0-cached-replay-v1", "status": "PASS",
        "query_ids": query_ids, "pair_count": 1200,
        "selected_chunks": sum(row["selected_chunks"] for row in metrics),
        "source_grounding": "PASS", "submission_compatible_objects": "PASS",
        "model_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "per_query": metrics,
    }
    write_once(output / "replay_report.json", json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.readiness, args.score_cache, args.output)
