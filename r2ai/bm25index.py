"""Lexical document retrieval over the crawled corpus.

The organisers' own BM25 baseline reaches DOCS_PRECISION 0.8805 where our dense
pipeline reaches 0.1795 at a comparable number of correct documents. A precision
that high means the gold documents share surface terms with the query almost
directly, which is exactly what a bag-of-words model captures and what a dense
model smooths away. Document-level F2 is 96.8% of our remaining gap to that
baseline, so lexical matching is the missing component rather than a refinement.

Tokenisation has to serve both halves of the corpus: Vietnamese and English are
whitespace-delimited, while Chinese is not, so CJK runs are emitted as character
bigrams — the standard substitute for a word segmenter in Chinese IR.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from array import array
from pathlib import Path

import bm25s
import numpy as np
from bm25s.tokenization import Tokenized
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.taxonomy import Taxonomy

RAW = ROOT / "data/raw/vibiomir"
OUT = ROOT / "data/vibiomir"

_CJK = re.compile(r"[㐀-鿿豈-﫿]+")
_WORD = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)
# Vietnamese function words; harmless for Chinese since they never match a bigram.
_STOP = {
    "la", "co", "khong", "cua", "va", "cho", "nhu", "the", "nao", "gi", "bi", "duoc",
    "mot", "nhung", "hay", "thi", "ma", "nay", "do", "toi", "minh", "khi", "voi",
    "trong", "cac", "nguoi", "va", "den", "tu", "ra", "vao", "len", "ve", "o",
}


def tokenize(text: str) -> list[str]:
    """Words for Latin script, character bigrams for CJK runs."""
    tokens: list[str] = []
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    lowered = _CJK.sub(" ", text).lower()
    tokens.extend(w for w in _WORD.findall(lowered) if len(w) > 1 and w not in _STOP)
    return tokens


def doc_texts(chunks_file: str, max_chunks: int) -> tuple[np.ndarray, list[str]]:
    table = pq.read_table(
        OUT / chunks_file, columns=["doc_id", "chunk_index", "title", "text"]
    ).to_pydict()
    parts: dict[int, list[tuple[int, str]]] = {}
    titles: dict[int, str] = {}
    for doc_id, idx, title, text in zip(
        table["doc_id"], table["chunk_index"], table["title"], table["text"]
    ):
        if idx >= max_chunks:
            continue
        parts.setdefault(doc_id, []).append((idx, text))
        if title and doc_id not in titles:
            titles[doc_id] = title
    doc_ids = np.array(sorted(parts), dtype=np.int64)
    texts = []
    for doc_id in doc_ids:
        key = int(doc_id)
        body = " ".join(t for _, t in sorted(parts[key]))
        texts.append(f"{titles.get(key, '')} {body}".strip())
    return doc_ids, texts


def build(chunks_file: str, max_chunks: int, out_name: str) -> None:
    doc_ids, texts = doc_texts(chunks_file, max_chunks)
    print(f"documents: {len(doc_ids):,}", flush=True)

    # Token *ids*, not token strings. Holding 3.2M documents as lists of Python
    # strings is ~1.9 billion string objects (>100 GB) and gets the build
    # OOM-killed at the very last step; array('i') per document is ~8 GB.
    vocab: dict[str, int] = {}
    corpus: list[array] = []
    for i, text in enumerate(texts):
        row = array("i")
        for token in tokenize(text):
            token_id = vocab.get(token)
            if token_id is None:
                token_id = vocab[token] = len(vocab)
            row.append(token_id)
        corpus.append(row)
        texts[i] = ""  # release the source text as we go
        if (i + 1) % 250_000 == 0:
            print(f"  tokenised {i + 1:,}/{len(texts):,}  vocab {len(vocab):,}", flush=True)
    del texts
    print(f"vocab: {len(vocab):,} terms", flush=True)

    retriever = bm25s.BM25(method="lucene")
    retriever.index(Tokenized(ids=corpus, vocab=vocab))
    del corpus
    retriever.save(str(OUT / out_name), corpus=None)
    np.save(OUT / f"{out_name}_ids.npy", doc_ids)
    with (OUT / f"{out_name}_vocab.json").open("w", encoding="utf-8") as fh:
        json.dump(vocab, fh)
    print(f"saved {out_name}", flush=True)


def search(out_name: str, top_k: int, n_threads: int,
           taxonomy_config: str | None = None,
           candidate_out: str | None = None) -> None:
    doc_ids = np.load(OUT / f"{out_name}_ids.npy")
    retriever = bm25s.BM25.load(str(OUT / out_name), load_corpus=False)
    with (OUT / f"{out_name}_vocab.json").open(encoding="utf-8") as fh:
        vocab = json.load(fh)
    queries = pq.read_table(RAW / "query.parquet").to_pydict()
    # Queries must be mapped through the same vocabulary the index was built
    # with; unknown terms are dropped since they match nothing anyway.
    taxonomy = Taxonomy.load(taxonomy_config) if taxonomy_config else None
    qids_tok = []
    expanded_queries = 0
    for query in queries["query"]:
        terms = tokenize(query)
        if taxonomy is not None:
            expansions = [token for alias in taxonomy.aliases_for(query) for token in tokenize(alias)]
            if expansions:
                expanded_queries += 1
                terms = list(dict.fromkeys([*terms, *expansions]))
        qids_tok.append([vocab[t] for t in terms if t in vocab])
    if taxonomy is not None:
        print(f"taxonomy alias expansion: {expanded_queries}/{len(queries['query'])} queries", flush=True)
    empty = sum(1 for q in qids_tok if not q)
    if empty:
        print(f"warning: {empty} queries have no in-vocabulary terms", flush=True)
    k = min(top_k, len(doc_ids))
    idx, scores = retriever.retrieve(Tokenized(ids=qids_tok, vocab=vocab), k=k,
                                     n_threads=n_threads, show_progress=False)

    rows_q, rows_d, rows_s, rows_r = [], [], [], []
    for qi, qid in enumerate(queries["id"]):
        for rank in range(k):
            rows_q.append(qid)
            rows_d.append(int(doc_ids[idx[qi, rank]]))
            rows_s.append(float(scores[qi, rank]))
            rows_r.append(rank)
    out = OUT / f"{candidate_out or out_name}_candidates.parquet"
    pq.write_table(pa.table({
        "query_id": pa.array(rows_q, pa.int64()), "doc_id": pa.array(rows_d, pa.int64()),
        "score": pa.array(rows_s, pa.float32()), "rank": pa.array(rows_r, pa.int32()),
    }), out)
    print(f"wrote {out}: {len(rows_q):,} rows, {len(set(rows_d)):,} unique docs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "search"])
    ap.add_argument("--chunks-file", default="chunks_v4.parquet")
    ap.add_argument("--max-chunks", type=int, default=6,
                    help="chunks per document to index (caps very long pages)")
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--taxonomy-config", type=str, default=None,
                    help="enable bounded query expansion using curated concept aliases")
    ap.add_argument("--candidate-out", type=str, default=None,
                    help="candidate artifact stem; preserves the loaded index and existing baseline candidates")
    ap.add_argument("--out", default="bm25idx")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.chunks_file, a.max_chunks, a.out)
    else:
        search(a.out, a.top_k, a.n_threads, a.taxonomy_config, a.candidate_out)
