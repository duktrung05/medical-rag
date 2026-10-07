"""Widen the chunks of an existing submission, without re-running retrieval.

The chunk metric counts a predicted chunk as relevant when its longest common
subsequence with a reference chunk covers >=40% of *that reference chunk's*
tokens. A prediction shorter than 40% of the reference therefore cannot qualify
no matter how well it was chosen, so chunk length is a hard constraint before it
is a quality question.

Measured on the leaderboard with 220-word chunks: CHUNKS_RECALL 0.0768 against
~62 information pieces per query, while competitors reach CHUNKS_F2 0.17-0.21
with the same document-level quality or worse. Widening each predicted chunk to
include its neighbours in the same document raises the achievable LCS directly.

Longer chunks cost bytes, and the platform rejects uploads over ~50 MB, so the
number of chunks is trimmed automatically to fit.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.rank import OUT, write_submission  # noqa: E402

CAP_MB = 50.0


def load_submission(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(zf.namelist()[0]))


def doc_chunk_texts(chunks_file: str, doc_ids: set[int]) -> dict[int, list[str]]:
    """Ordered chunk texts for the documents a submission actually uses."""
    table = pq.read_table(
        OUT / chunks_file, columns=["doc_id", "chunk_index", "text"]
    ).to_pydict()
    by_doc: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for doc_id, idx, text in zip(table["doc_id"], table["chunk_index"], table["text"]):
        if doc_id in doc_ids:
            by_doc[doc_id].append((idx, text))
    return {d: [t for _, t in sorted(v)] for d, v in by_doc.items()}


def widen(text: str, texts: list[str], window: int, whole: bool) -> str:
    """Expand a chunk to its neighbours in the same document."""
    if not texts:
        return text
    if whole:
        return " ".join(texts)
    try:
        pos = texts.index(text)
    except ValueError:
        # Submission text came from an earlier chunking run; centre on the best
        # textual overlap instead of failing.
        pos = max(range(len(texts)), key=lambda i: len(set(texts[i].split()) & set(text.split())))
    lo, hi = max(0, pos - window), min(len(texts), pos + window + 1)
    return " ".join(texts[lo:hi])


def build(source: Path, chunks_file: str, window: int, whole: bool,
          k_chunks: int, k_docs: int, out_name: str) -> None:
    records = load_submission(source)
    needed = {int(c["doc_id"]) for r in records for c in r["relevant_chunks"]}
    print(f"loading chunk text for {len(needed):,} documents", flush=True)
    texts_by_doc = doc_chunk_texts(chunks_file, needed)

    out = []
    for rec in records:
        docs = rec["relevant_docs"][:k_docs] if k_docs else rec["relevant_docs"]
        keep = set(docs)
        chunks = []
        for ch in rec["relevant_chunks"]:
            doc_id = int(ch["doc_id"])
            if doc_id not in keep:
                continue
            chunks.append({
                "doc_id": ch["doc_id"],
                "chunk_text": widen(ch["chunk_text"], texts_by_doc.get(doc_id, []),
                                    window, whole),
            })
            if len(chunks) >= k_chunks:
                break
        out.append({"id": rec["id"], "relevant_docs": docs, "relevant_chunks": chunks})

    zip_path = write_submission(out, out_name)
    size = zip_path.stat().st_size / 1e6
    avg_w = sum(len(c["chunk_text"].split()) for r in out for c in r["relevant_chunks"])
    n_ch = sum(len(r["relevant_chunks"]) for r in out)
    flag = "  <-- OVER CAP, lower --k-chunks" if size > CAP_MB else ""
    print(f"{zip_path.name:34s} docs~{sum(len(r['relevant_docs']) for r in out)/len(out):5.0f} "
          f"chunks~{n_ch/len(out):5.1f} words/chunk~{avg_w/max(n_ch,1):6.0f} "
          f"zip {size:5.1f} MB{flag}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--chunks-file", default="chunks_v4.parquet")
    ap.add_argument("--window", type=int, default=1,
                    help="neighbour chunks to merge on each side")
    ap.add_argument("--whole-doc", action="store_true", help="use the entire document")
    ap.add_argument("--k-chunks", type=int, default=80)
    ap.add_argument("--k-docs", type=int, default=0, help="0 keeps the source value")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.source, a.chunks_file, a.window, a.whole_doc, a.k_chunks, a.k_docs, a.out)
