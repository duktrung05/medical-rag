"""Stage 1: document-level dense retrieval over the whole crawled corpus.

Embedding every chunk does not scale: 4.4M documents are ~24M chunks, which is
~28 GPU-hours to encode and a 1200 x 24M score matrix that no single card holds.
One vector per document is 2.7 hours and 9 GB instead, and BGE-M3 puts Vietnamese,
Chinese and English in one space, so this stage is cross-lingual without any
translation step.

The per-query shortlist it produces is what the expensive chunk-level stage then
runs on, so only a few hundred documents per query ever get chunked and reranked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

# Run as a script (python r2ai/index.py), so the package root has to go on
# sys.path before the sibling import below.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.langbalance import LANG_NAMES, LanguageStats, doc_languages  # noqa: E402

RAW = ROOT / "data/raw/vibiomir"
OUT = ROOT / "data/vibiomir"
EMBED_MODEL = "BAAI/bge-m3"


def doc_representations(chunks_file: str, lead_chunks: int) -> tuple[np.ndarray, list[str]]:
    """One text per document: its title plus the first few chunks.

    The opening of a consumer-health article carries its topic, which is all
    stage 1 needs; anything deeper is recovered by the chunk-level stage.
    """
    table = pq.read_table(
        OUT / chunks_file, columns=["doc_id", "chunk_index", "title", "text"]
    ).to_pydict()
    lead: dict[int, list[tuple[int, str]]] = {}
    titles: dict[int, str] = {}
    for doc_id, idx, title, text in zip(
        table["doc_id"], table["chunk_index"], table["title"], table["text"]
    ):
        if idx >= lead_chunks:
            continue
        lead.setdefault(doc_id, []).append((idx, text))
        if title and doc_id not in titles:
            titles[doc_id] = title

    doc_ids = np.array(sorted(lead), dtype=np.int64)
    texts = []
    for doc_id in doc_ids:
        body = " ".join(t for _, t in sorted(lead[int(doc_id)]))
        title = titles.get(int(doc_id), "")
        texts.append(f"{title}\n{body}" if title else body)
    return doc_ids, texts


def build(chunks_file: str, lead_chunks: int, seq: int, batch: int, device: str,
          out_name: str) -> None:
    doc_ids, texts = doc_representations(chunks_file, lead_chunks)
    print(f"documents to embed: {len(doc_ids):,}", flush=True)

    model = SentenceTransformer(EMBED_MODEL, device=device,
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = seq
    vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    # fp16 halves both the file and the VRAM needed to search it; the vectors are
    # already L2-normalised so the precision loss is irrelevant for ranking.
    vecs = vecs.astype(np.float16)
    np.save(OUT / f"{out_name}_ids.npy", doc_ids)
    np.save(OUT / f"{out_name}_vecs.npy", vecs)
    print(f"saved {out_name}_vecs.npy  shape={vecs.shape}  "
          f"{vecs.nbytes / 1e9:.1f} GB", flush=True)


def _url_map() -> dict[int, str]:
    table = pq.read_table(RAW / "links_corpus.parquet").to_pydict()
    return dict(zip(table["id"], table["url"]))


def search(out_name: str, top_k: int, device: str, block: int,
           lang_balance: bool = False) -> None:
    doc_ids = np.load(OUT / f"{out_name}_ids.npy")
    vecs = np.load(OUT / f"{out_name}_vecs.npy", mmap_mode="r")
    queries = pq.read_table(RAW / "query.parquet").to_pydict()
    qids, qtexts = queries["id"], queries["query"]

    model = SentenceTransformer(EMBED_MODEL, device=device,
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512
    qvecs = model.encode(qtexts, batch_size=32, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    del model
    torch.cuda.empty_cache()
    query_t = torch.from_numpy(qvecs.astype(np.float16)).to(device)

    langs = None
    mean = std = None
    if lang_balance:
        langs = doc_languages(doc_ids, _url_map())
        names = {LANG_NAMES[k]: int((langs == k).sum()) for k in LANG_NAMES}
        print(f"index language mix: {names}", flush=True)
        # First pass collects per-language score statistics; the second pass
        # below selects on scores standardised within each language.
        stats = LanguageStats(len(qids))
        for start in range(0, len(doc_ids), block):
            stop = min(start + block, len(doc_ids))
            chunk = torch.from_numpy(np.ascontiguousarray(vecs[start:stop])).to(device)
            sims = (query_t @ chunk.T).float().cpu().numpy()
            stats.update(sims, langs[start:stop])
            del chunk
        mean, std = stats.mean_std()
        torch.cuda.empty_cache()
        print("collected language statistics", flush=True)

    # Score in blocks of documents and keep a running top-k, so peak memory is
    # set by `block` rather than by the corpus size.
    best_scores = torch.full((len(qids), top_k), -1e4, dtype=torch.float16, device=device)
    best_idx = torch.zeros((len(qids), top_k), dtype=torch.int64, device=device)
    for start in range(0, len(doc_ids), block):
        stop = min(start + block, len(doc_ids))
        chunk = torch.from_numpy(np.ascontiguousarray(vecs[start:stop])).to(device)
        sims = query_t @ chunk.T
        if lang_balance:
            block_langs = torch.from_numpy(langs[start:stop].astype(np.int64)).to(device)
            mean_t = torch.from_numpy(mean).to(device)[:, block_langs]
            std_t = torch.from_numpy(std).to(device)[:, block_langs]
            sims = ((sims.float() - mean_t) / std_t).to(sims.dtype)
            del mean_t, std_t
        k = min(top_k, stop - start)
        scores, idx = torch.topk(sims, k=k, dim=1)
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_idx = torch.cat([best_idx, idx + start], dim=1)
        best_scores, order = torch.topk(merged_scores, k=top_k, dim=1)
        best_idx = torch.gather(merged_idx, 1, order)
        del chunk, sims
        if start % (block * 10) == 0:
            print(f"  scored {stop:,}/{len(doc_ids):,}", flush=True)
    torch.cuda.empty_cache()

    idx_np = best_idx.cpu().numpy()
    score_np = best_scores.float().cpu().numpy()
    rows_q, rows_d, rows_s, rows_r = [], [], [], []
    for qi, qid in enumerate(qids):
        for rank in range(top_k):
            rows_q.append(qid)
            rows_d.append(int(doc_ids[idx_np[qi, rank]]))
            rows_s.append(float(score_np[qi, rank]))
            rows_r.append(rank)
    out = OUT / f"{out_name}_candidates.parquet"
    pq.write_table(pa.table({
        "query_id": pa.array(rows_q, pa.int64()), "doc_id": pa.array(rows_d, pa.int64()),
        "score": pa.array(rows_s, pa.float32()), "rank": pa.array(rows_r, pa.int32()),
    }), out)
    print(f"wrote {out}: {len(rows_q):,} rows, {len(set(rows_d)):,} unique docs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "search"])
    ap.add_argument("--chunks-file", default="chunks_full.parquet")
    ap.add_argument("--lead-chunks", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=400)
    ap.add_argument("--block", type=int, default=500_000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lang-balance", action="store_true",
                    help="standardise scores within each language group")
    ap.add_argument("--out", default="docidx")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.chunks_file, a.lead_chunks, a.seq, a.batch, a.device, a.out)
    else:
        search(a.out, a.top_k, a.device, a.block, a.lang_balance)
