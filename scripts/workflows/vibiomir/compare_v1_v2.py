"""Compare V1-style first-six lexical representation against V2 on one pilot."""
from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.retrieval_v2.core import _match_query, lexical_text, sparse_search

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=ROOT / "outputs/retrieval_v2/pilot_full_docs")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    rows = pq.read_table(args.pilot / "pilot_source.parquet").to_pylist()
    groups = defaultdict(list)
    for row in rows:
        groups[row["doc_id"]].append(row)
    v1 = sqlite3.connect(":memory:")
    v1.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, content, tokenize='unicode61')")
    for doc_id, chunks in groups.items():
        chunks.sort(key=lambda r: r["chunk_index"])
        title = chunks[0]["title"] or ""
        body = " ".join(r["text"] for r in chunks if r["chunk_index"] < 6)
        v1.execute("INSERT INTO docs(doc_id,content) VALUES(?,?)", (doc_id, lexical_text(title + " " + body)))
    v2 = sqlite3.connect(args.pilot / "v2_sparse.sqlite")
    queries = pq.read_table(ROOT / "data/raw/vibiomir/query.parquet").slice(0, args.queries).to_pylist()
    output = args.pilot / "V2_V1_LEXICAL_PILOT_ABLATION.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "v1_hits", "v2_hits", "v2_only_docs",
                                                 "v1_only_docs", "shared_docs"])
        writer.writeheader()
        for row in queries:
            match = _match_query(row["query"])
            first = (v1.execute("SELECT doc_id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                                (match, args.top_k)).fetchall() if match else [])
            first_set = {int(x[0]) for x in first}
            second_set = {doc_id for doc_id, _ in sparse_search(v2, row["query"], "document", args.top_k)}
            writer.writerow({"query_id": row["id"], "v1_hits": len(first_set),
                             "v2_hits": len(second_set), "v2_only_docs": len(second_set - first_set),
                             "v1_only_docs": len(first_set - second_set),
                             "shared_docs": len(first_set & second_set)})
    v1.close()
    v2.close()
    print(output)


if __name__ == "__main__":
    main()
