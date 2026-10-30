"""Create a blinded, query-disjoint judgment packet at the document swap boundary."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/v4/doc_f2_multipassage_20261026"
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024/final"
QUERY_PATH = ROOT / "outputs/vibiomir_official_verify_20261012/query.parquet"
INDEX = ROOT / "outputs/v4/task4/full_index_gpu3_20261019"
TARGETS = {"development": 180, "validation": 60, "held_out": 60}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_key(query_id: int) -> str:
    return hashlib.sha256(f"doc-f2-direct-judgment-v1:{query_id}".encode()).hexdigest()


def build(run: Path = RUN) -> dict:
    changes_path = run / "document_changes.csv"
    packet_path = run / "direct_judgment_packet.csv"
    key_path = run / "direct_judgment_key.csv"
    manifest_path = run / "direct_judgment_manifest.json"
    for path in (packet_path, key_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    changes = defaultdict(list)
    split_by_query = {}
    with changes_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            qid = int(row["query_id"])
            split_by_query[qid] = row["split"]
            changes[qid].append(row)
    selected = {}
    for split, target in TARGETS.items():
        available = [qid for qid in changes if split_by_query[qid] == split]
        if len(available) < target:
            raise ValueError(f"Not enough changed {split} queries: {len(available)} < {target}")
        selected[split] = sorted(available, key=stable_key)[:target]

    query_table = pq.read_table(QUERY_PATH, columns=["id", "query"])
    queries = {int(row["id"]): row["query"] for row in query_table.to_pylist()}
    index_manifest = read_json(INDEX / "manifest.json")
    store = DiskCorpusStore(INDEX / "corpus.sqlite", index_manifest["corpus_version"],
                            index_manifest["fingerprint"])

    packet_rows, key_rows = [], []
    for split in ("development", "validation", "held_out"):
        for qid in selected[split]:
            data = read_json(REFINED / f"{qid}.json")
            candidates = {int(row["official_id"]): row
                          for row in data["trace"]["ranked_candidates"]}
            roles = defaultdict(set)
            margins = defaultdict(list)
            for change in changes[qid]:
                victim, entrant = int(change["victim"]), int(change["entrant"])
                roles[victim].add("champion_victim")
                roles[entrant].add("challenger_entrant")
                margins[victim].append(float(change["margin"]))
                margins[entrant].append(float(change["margin"]))
            ordered_ids = sorted(roles, key=lambda doc_id: hashlib.sha256(
                f"doc-f2-blind-v1:{qid}:{doc_id}".encode()).hexdigest())
            for ordinal, doc_id in enumerate(ordered_ids, 1):
                document = candidates[doc_id]
                passages = sorted(document["evidence"],
                                  key=lambda row: (-row["rerank_score"], row["passage_id"]))[:3]
                excerpts = []
                for passage in passages:
                    text = " ".join(passage["text"].split())
                    excerpts.append(text[:1600])
                task_id = f"{split[:1].upper()}{qid:04d}-{ordinal:02d}"
                source = store.source(doc_id)
                packet_rows.append({
                    "task_id": task_id,
                    "split": split,
                    "query_id": qid,
                    "query": queries[qid],
                    "source_url": source.url,
                    "top_passage_1": excerpts[0] if excerpts else "",
                    "top_passage_2": excerpts[1] if len(excerpts) > 1 else "",
                    "top_passage_3": excerpts[2] if len(excerpts) > 2 else "",
                    "relevance_grade_0_1_2": "",
                    "reviewer_rationale": "",
                })
                key_rows.append({
                    "task_id": task_id,
                    "split": split,
                    "query_id": qid,
                    "doc_id": doc_id,
                    "roles": "+".join(sorted(roles[doc_id])),
                    "max_swap_margin": max(margins[doc_id]),
                    "original_parent_rank": next(
                        rank for rank, row in enumerate(data["trace"]["ranked_candidates"], 1)
                        if int(row["official_id"]) == doc_id),
                    "max_passage_score": float(document["rerank_score"]),
                })

    packet_fields = list(packet_rows[0])
    with packet_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=packet_fields)
        writer.writeheader()
        writer.writerows(packet_rows)
    key_fields = list(key_rows[0])
    with key_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows(key_rows)
    manifest = {
        "format": "v4-direct-document-judgment-packet-v1",
        "status": "AWAITING_DIRECT_JUDGMENTS",
        "query_counts": {split: len(qids) for split, qids in selected.items()},
        "judgment_rows": len(packet_rows),
        "selection": "deterministic SHA-256 sample from changed queries; fixed 180/60/60",
        "blinding": "packet omits doc_id and entrant/victim role; key stored separately",
        "instructions": {
            "0": "document does not answer the query",
            "1": "document is related but only partly answers the query",
            "2": "document directly answers a material part of the query",
        },
        "inputs_sha256": {
            str(changes_path.relative_to(ROOT)): sha256_file(changes_path),
            str(QUERY_PATH.relative_to(ROOT)): sha256_file(QUERY_PATH),
            str((RUN / "manifest.json").relative_to(ROOT)): sha256_file(RUN / "manifest.json"),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__).resolve()),
        },
        "packet_sha256": sha256_file(packet_path),
        "key_sha256": sha256_file(key_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                        sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    build(args.run)
