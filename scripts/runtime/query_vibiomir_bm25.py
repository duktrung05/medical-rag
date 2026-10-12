"""Try a query against a downloaded ViBioMIR BM25 index using bounded-memory reads."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bm25s
import ijson
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2ai.bm25index import tokenize


def query_index(index: Path, query: str, top_k: int) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    started = time.perf_counter()
    terms = tokenize(query)
    wanted = set(terms)
    vocab_path = index.with_name(index.name + "_vocab.json")
    with vocab_path.open("rb") as stream:
        vocab = {term: int(token_id) for term, token_id in ijson.kvitems(stream, "")
                 if term in wanted}
    token_ids = [vocab[term] for term in terms if term in vocab]
    result = {"query": query, "matched_terms": sorted(vocab), "results": []}
    if not token_ids:
        result["reason"] = "No query terms occur in the index vocabulary"
        result["elapsed_seconds"] = time.perf_counter() - started
        return result
    doc_ids = np.load(index.with_name(index.name + "_ids.npy"), mmap_mode="r", allow_pickle=False)
    retriever = bm25s.BM25.load(str(index), load_corpus=False, load_vocab=False, mmap=True)
    # Only query terms are needed; avoid loading a second full vocabulary into RAM.
    retriever.vocab_dict = vocab
    retriever.unique_token_ids_set = set(vocab.values())
    if doc_ids.ndim != 1 or len(doc_ids) != retriever.scores["num_docs"]:
        raise ValueError("Document ID mapping does not match index size")
    positions, scores = retriever.retrieve([token_ids], k=min(top_k, len(doc_ids)),
                                          n_threads=0, show_progress=False,
                                          backend_selection="numpy")
    for position, score in zip(positions[0], scores[0]):
        if score > 0:
            result["results"].append({"rank": len(result["results"]) + 1,
                                      "doc_id": int(doc_ids[position]), "score": float(score)})
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def attach_urls(result: dict, corpus: Path) -> None:
    wanted = {row["doc_id"] for row in result["results"]}
    if not wanted:
        return
    parquet = pq.ParquetFile(corpus)
    columns = parquet.schema_arrow.names
    id_column = "id" if "id" in columns else "doc_id"
    if id_column not in columns or "url" not in columns:
        raise ValueError("URL lookup requires id/doc_id and url columns")
    urls = {}
    for batch in parquet.iter_batches(batch_size=65536, columns=[id_column, "url"]):
        values = batch.to_pydict()
        for doc_id, url in zip(values[id_column], values["url"]):
            if doc_id in wanted:
                urls[doc_id] = url
        if wanted <= urls.keys():
            break
    for row in result["results"]:
        row["url"] = urls.get(row["doc_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/vibiomir/bm25idx_crawl_20261007")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/vibiomir/links_corpus.parquet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = query_index(args.index, args.query, args.top_k)
    if args.corpus.exists():
        attach_urls(result, args.corpus)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
