"""Reciprocal-rank fusion of the lexical and dense candidate lists.

The two retrievers disagree almost completely: measured overlap between their
top-300 lists is 12%, so 88% of what BM25 finds is invisible to the dense index
and vice versa. They also split by language — BM25's top-150 is 82.6% Vietnamese
and 0.0% Chinese, because a Vietnamese query shares no tokens with Chinese
character bigrams, while the dense index is 48.7% / 44.0%. Lexical therefore
supplies precision on Vietnamese evidence (the organisers' BM25 baseline reaches
DOCS_PRECISION 0.8805 that way) and dense supplies the Chinese half that the
leaderboard shows is worth 24% of our found gold.

RRF is used rather than score averaging because BM25 scores and cosine
similarities are not on a comparable scale, and rank is all that survives that
difference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/vibiomir"


def ranked_lists(name: str) -> dict[int, list[int]]:
    table = pq.read_table(OUT / f"{name}.parquet").to_pydict()
    per_query: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for query_id, doc_id, rank in zip(table["query_id"], table["doc_id"], table["rank"]):
        per_query[query_id].append((rank, doc_id))
    return {q: [d for _, d in sorted(v)] for q, v in per_query.items()}


def fuse(dense_name: str, lexical_name: str, w_dense: float, w_lexical: float,
         rrf_k: int, top_k: int, out_name: str) -> None:
    dense = ranked_lists(dense_name)
    lexical = ranked_lists(lexical_name)
    queries = sorted(set(dense) | set(lexical))

    rows_q, rows_d, rows_s, rows_r = [], [], [], []
    contributions = {"both": 0, "dense_only": 0, "lexical_only": 0}
    for query_id in queries:
        scores: dict[int, float] = defaultdict(float)
        in_dense = set(dense.get(query_id, ()))
        in_lex = set(lexical.get(query_id, ()))
        for rank, doc_id in enumerate(dense.get(query_id, ())):
            scores[doc_id] += w_dense / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(lexical.get(query_id, ())):
            scores[doc_id] += w_lexical / (rrf_k + rank + 1)
        ordered = sorted(scores, key=lambda d: -scores[d])[:top_k]
        for rank, doc_id in enumerate(ordered):
            rows_q.append(query_id)
            rows_d.append(int(doc_id))
            rows_s.append(float(scores[doc_id]))
            rows_r.append(rank)
            if doc_id in in_dense and doc_id in in_lex:
                contributions["both"] += 1
            elif doc_id in in_dense:
                contributions["dense_only"] += 1
            else:
                contributions["lexical_only"] += 1

    out = OUT / f"{out_name}.parquet"
    pq.write_table(pa.table({
        "query_id": pa.array(rows_q, pa.int64()), "doc_id": pa.array(rows_d, pa.int64()),
        "score": pa.array(rows_s, pa.float32()), "rank": pa.array(rows_r, pa.int32()),
    }), out)
    total = sum(contributions.values())
    share = {k: f"{v / total:.1%}" for k, v in contributions.items()}
    print(f"wrote {out}: {len(rows_q):,} rows, {len(set(rows_d)):,} unique docs")
    print(f"composition of the fused list: {share}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="docidx4_candidates")
    ap.add_argument("--lexical", default="bm25idx_candidates")
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--w-lexical", type=float, default=1.0)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=400)
    ap.add_argument("--out", default="fused_candidates")
    a = ap.parse_args()
    fuse(a.dense, a.lexical, a.w_dense, a.w_lexical, a.rrf_k, a.top_k, a.out)
