"""Stage 1 candidate generation: BM25 over text mined from corpus URLs.

No crawling happens here. Roughly 30% of ViBioMIR URLs carry the article title in
their path; this stage turns that free signal into a per-query shortlist so the
crawler only fetches pages that some query might plausibly want.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import bm25s
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ai.urltext import has_cjk, strip_diacritics, url_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/vibiomir"
OUT = ROOT / "data/vibiomir"
_WORD = re.compile(r"[a-z0-9]+")

# Vietnamese function words carry no retrieval signal once diacritics are folded.
STOP = {
    "la", "co", "khong", "cua", "va", "cho", "nhu", "the", "nao", "gi", "bi",
    "duoc", "den", "tu", "voi", "khi", "ra", "vao", "len", "mot", "nhung",
    "hay", "thi", "ma", "nay", "do", "ai", "em", "toi", "minh",
    "sao", "vay", "ha", "ne", "nhe", "xin", "chao", "bac", "si", "hoi", "ve",
    "trong", "ngoai", "tren", "duoi", "hon", "rat", "qua", "cung", "se", "da",
    "dang", "con", "nen", "phai", "can", "muon", "biet", "lam", "bao",
    "nhieu", "ban", "ho", "no", "chi", "anh", "chau", "cac",
}


def tokenize(text: str) -> list[str]:
    """Fold to the diacritic-free, lowercase space that URL slugs live in."""
    folded = strip_diacritics(text).lower()
    words = [w for w in _WORD.findall(folded) if len(w) > 1 and w not in STOP]
    # Keep CJK characters as individual tokens so a-hospital titles stay matchable.
    words.extend(ch for ch in text if has_cjk(ch))
    return words


def build(min_tokens: int = 2) -> None:
    table = pq.read_table(RAW / "links_corpus.parquet").to_pydict()
    doc_ids: list[int] = []
    corpus_tokens: list[list[str]] = []
    texts: list[str] = []
    for doc_id, url in zip(table["id"], table["url"]):
        text = url_text(url)
        if not text:
            continue
        toks = tokenize(text)
        if len(toks) < min_tokens:
            continue
        doc_ids.append(doc_id)
        corpus_tokens.append(toks)
        texts.append(text)
    print(f"indexable urls: {len(doc_ids):,}", flush=True)

    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)
    OUT.mkdir(parents=True, exist_ok=True)
    retriever.save(str(OUT / "url_bm25"), corpus=None)
    pq.write_table(
        pa.table({
            "doc_id": pa.array(doc_ids, pa.int64()),
            "url_text": pa.array(texts, pa.string()),
        }),
        OUT / "url_index_meta.parquet",
    )
    print(f"saved {OUT / 'url_bm25'}", flush=True)


def retrieve(top_k: int = 60) -> None:
    meta = pq.read_table(OUT / "url_index_meta.parquet").to_pydict()
    doc_ids = np.asarray(meta["doc_id"], dtype=np.int64)
    retriever = bm25s.BM25.load(str(OUT / "url_bm25"), load_corpus=False)
    queries = pq.read_table(RAW / "query.parquet").to_pydict()
    qtok = [tokenize(q) for q in queries["query"]]
    k = min(top_k, len(doc_ids))
    idx, scores = retriever.retrieve(qtok, k=k, n_threads=16, show_progress=True)

    rows_q, rows_d, rows_s, rows_r = [], [], [], []
    for qi, qid in enumerate(queries["id"]):
        for rank in range(k):
            rows_q.append(qid)
            rows_d.append(int(doc_ids[idx[qi, rank]]))
            rows_s.append(float(scores[qi, rank]))
            rows_r.append(rank)
    pq.write_table(
        pa.table({
            "query_id": pa.array(rows_q, pa.int64()),
            "doc_id": pa.array(rows_d, pa.int64()),
            "score": pa.array(rows_s, pa.float32()),
            "rank": pa.array(rows_r, pa.int32()),
        }),
        OUT / "url_candidates.parquet",
    )
    print(f"candidates: {len(rows_q):,} rows, unique docs {len(set(rows_d)):,}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "retrieve"])
    ap.add_argument("--top-k", type=int, default=60)
    args = ap.parse_args()
    build() if args.stage == "build" else retrieve(args.top_k)
