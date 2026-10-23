"""Measure B0 passage retrieval on the registered full index without reranking."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.medrag.core.config import load_config
from src.medrag.core.passage_first import passage_candidates
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.bge_encoder import BGEM3Encoder
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.disk_dense import DiskDense


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(config_path: Path, split_path: Path, output: Path, count: int,
         device: str) -> None:
    if count <= 0 or output.exists():
        raise ValueError("Use positive count and fresh output directory")
    config = load_config(config_path)
    index = ROOT / config["index_namespace"]
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["scope"] != "full_release":
        raise ValueError("Full-release index is required")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = split["development_ids"][:count]
    if len(ids) != count or split["label_status"] != "unlabeled":
        raise ValueError("Need frozen unlabeled development query IDs")
    migration = json.loads((ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
                           .read_text(encoding="utf-8"))
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query fingerprint changed")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    started = time.monotonic()
    encoder = BGEM3Encoder(config["dense"]["model"],
                           revision=config["dense"]["revision"],
                           document_max_length=config["dense"]["document_max_length"],
                           query_max_length=config["dense"]["query_max_length"],
                           device=device)
    encoder_load = time.monotonic() - started
    dense = DiskDense(index / "dense", encoder, manifest["dense"]["fingerprint"])
    bm25 = DiskBM25(index / "passages.sqlite", "passage_bm25", manifest["fingerprint"])
    store = DiskCorpusStore(index / "corpus.sqlite", manifest["corpus_version"],
                            manifest["fingerprint"])
    rows = []
    for qid in ids:
        query = queries[qid]
        started = time.monotonic()
        lexical = bm25.search(query, config["retrieval"]["passage_candidate_limit"])
        lexical_seconds = time.monotonic() - started
        started = time.monotonic()
        semantic = dense.search(query, config["retrieval"]["passage_candidate_limit"])
        dense_seconds = time.monotonic() - started
        fused = passage_candidates([lexical, semantic],
                                   rrf_k=config["retrieval"]["rrf_k"],
                                   limit=config["retrieval"]["fused_candidate_limit"])
        pairs = fused[:200]
        passages = store.passages_batch([hit.passage_id for hit in pairs])
        sources = {hit.official_id: store.source(hit.official_id) for hit in pairs}
        if any(passages[hit.passage_id].official_id != hit.official_id or not
               verify(passages[hit.passage_id], sources[hit.official_id]) for hit in pairs):
            raise ValueError(f"Ungrounded passage in query {qid}")
        rows.append({"query_id": qid, "passage_bm25_seconds": lexical_seconds,
                     "dense_seconds": dense_seconds, "candidate_pairs": len(pairs),
                     "unique_parents": len({hit.official_id for hit in pairs}),
                     "source_validated_pairs": len(pairs)})
        print(f"query={qid} bm25={lexical_seconds:.2f}s dense={dense_seconds:.2f}s", flush=True)
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps({
        "status": "retrieval_only_no_rerank", "index_manifest_sha256": sha(manifest_path),
        "query_split_sha256": sha(split_path), "config_sha256": sha(config_path),
        "device": device, "encoder_load_seconds": encoder_load,
        "index_passages": manifest["passage_count"], "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT /
                        "configs/v4_phase2_full_independent_d200_context_gpu3.yaml")
    parser.add_argument("--split", type=Path, default=ROOT /
                        "outputs/v4/p3_query_split_20261021/query_split.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.config, args.split, args.output, args.count, args.device)
