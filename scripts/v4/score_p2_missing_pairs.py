"""Score only missing P2 passage pairs with the pinned local reranker snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4.audit_evidence_survival import sha
from src.medrag.core.config import load_config
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.data.disk_store import DiskCorpusStore, SourceLookup
from src.medrag.ranking.reranker import BGEReranker
from src.medrag.ranking.score_cache import ScoreCache


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def main(audit: Path, output: Path) -> None:
    audit, output = audit.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(ROOT / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml")
    reranker_cfg = config["reranker"]
    snapshot = require_cached_revision(reranker_cfg["model"], reranker_cfg["revision"])
    missing = read_json(audit / "missing_scores.json")
    unique = {(int(row["query_id"]), int(row["doc_id"]), row["passage_id"]): row
              for row in missing}
    if len(unique) != 6:
        raise ValueError(f"Expected six unique missing P2 pairs, found {len(unique)}")
    migration = read_json(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query fingerprint changed")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    index = ROOT / config["index_namespace"]
    index_manifest = read_json(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources = SourceLookup(store)
    requests = {}
    for (qid, parent, passage_id), row in sorted(unique.items()):
        checkpoint = read_json(ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
                               / "queries" / f"{qid}.json")
        candidate = next(item for item in checkpoint["diagnostics"]["ranked_candidates"]
                         if int(item["official_id"]) == parent)
        passage = next(item for item in candidate["evidence"]
                       if item["passage_id"] == passage_id)
        title = sources[parent].title
        payload = f"{title}\n{passage['text']}" if title else passage["text"]
        if hashlib.sha256(payload.encode()).hexdigest() != row["input_sha256"]:
            raise ValueError(f"Pair input mismatch: {qid}/{parent}/{passage_id}")
        requests.setdefault(qid, []).append((parent, passage_id, payload))
    source_path = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2/scores.sqlite"
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        contract = json.loads(meta["contract"])
        for qid, rows in requests.items():
            query_hash = hashlib.sha256(queries[qid].encode()).hexdigest()
            for parent, passage_id, _ in rows:
                if db.execute("SELECT 1 FROM scores WHERE query_sha256=? AND official_id=? "
                              "AND passage_id=?", (query_hash, parent, passage_id)).fetchone():
                    raise ValueError("Pair marked missing already exists in source cache")
    if (contract["model"] != reranker_cfg["model"]
            or contract["revision"] != reranker_cfg["revision"]
            or contract["max_length"] != reranker_cfg["max_length"]
            or contract["index_manifest_sha256"] != sha(index / "manifest.json")):
        raise ValueError("P2 score contract differs from P0")
    output.mkdir(parents=True)
    target_cache = output / "scores.sqlite"
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as source_db:
        with sqlite3.connect(target_cache) as target_db:
            source_db.backup(target_db)
    score_cache = ScoreCache(target_cache, contract)
    reranker = BGEReranker(reranker_cfg["model"], revision=reranker_cfg["revision"],
                           max_length=reranker_cfg["max_length"], device="cpu")
    scored = []
    for qid, rows in sorted(requests.items()):
        values = score_cache.score(queries[qid], rows, reranker)
        for (parent, passage_id, payload), score in zip(rows, values, strict=True):
            scored.append({"query_id": qid, "doc_id": parent,
                           "passage_id": passage_id, "rerank_score": float(score),
                           "input_sha256": hashlib.sha256(payload.encode()).hexdigest()})
    with sqlite3.connect(f"file:{target_cache}?mode=ro", uri=True) as db:
        old_count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] - len(scored)
        if old_count != 4303:
            raise ValueError("Unexpected original cache size")
        for row in scored:
            result = db.execute("SELECT score,input_sha256 FROM scores WHERE query_sha256=? "
                                "AND official_id=? AND passage_id=?",
                                (hashlib.sha256(queries[row["query_id"]].encode()).hexdigest(),
                                 row["doc_id"], row["passage_id"])).fetchall()
            if result != [(row["rerank_score"], row["input_sha256"])]:
                raise ValueError("New score was not persisted exactly")
    write_json(output / "scored_pairs.json", scored)
    manifest = {"format": "v4-p2-six-missing-pair-scores-v1", "status": "complete",
                "model_calls": len(scored), "retrieval_calls": 0,
                "device": "cpu", "model": reranker_cfg["model"],
                "revision": reranker_cfg["revision"], "max_length": reranker_cfg["max_length"],
                "snapshot": str(snapshot), "source_cache_sha256": sha(source_path),
                "input_audit_sha256": sha(audit / "missing_scores.json"),
                "artifact_sha256": {name: sha(output / name) for name in
                                    ("scores.sqlite", "scored_pairs.json")}}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "new_pairs": len(scored),
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.audit, args.output)
