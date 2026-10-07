"""Package a ViBioMIR submission from the cached fused candidates and embeddings.

This uses the existing candidate ranking for documents and selects the most
similar original corpus chunk for each document. It needs no model or GPU.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/vibiomir"
RAW = ROOT / "data/raw/vibiomir"
SUB = ROOT / "outputs/submissions"


def candidate_rankings(path: Path) -> dict[int, list[int]]:
    table = pq.read_table(path, columns=["query_id", "doc_id", "rank"]).to_pydict()
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for qid, doc_id, rank in zip(table["query_id"], table["doc_id"], table["rank"]):
        grouped[int(qid)].append((int(rank), int(doc_id)))
    return {qid: [doc for _, doc in sorted(rows)] for qid, rows in grouped.items()}


def chunk_doc_ids(path: Path) -> np.ndarray:
    parquet = pq.ParquetFile(path)
    batches = (
        batch.column(0).to_numpy(zero_copy_only=False)
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=["doc_id"])
    )
    return np.concatenate(list(batches))


def selected_texts(path: Path, needed: set[int]) -> dict[int, str]:
    """Read only text at selected physical row positions; preserve its exact value."""
    found: dict[int, str] = {}
    cursor = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=["text"]):
        end = cursor + batch.num_rows
        positions = [pos for pos in needed if cursor <= pos < end]
        if positions:
            column = batch.column(0)
            for pos in positions:
                found[pos] = column[pos - cursor].as_py()
                needed.remove(pos)
        cursor = end
    if needed:
        raise ValueError(f"Missing text for {len(needed)} selected chunk rows")
    return found


def write_zip(records: list[dict], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n]\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("predictions.json", payload.encode("utf-8"))
    return path.stat().st_size / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-docs", type=int, default=200)
    parser.add_argument("--max-chunks", type=int, default=120)
    parser.add_argument("--cap-mb", type=float, default=48.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    query_ids = [int(x) for x in pq.read_table(RAW / "query.parquet", columns=["id"]).column(0).to_pylist()]
    rankings = candidate_rankings(DATA / "fused_candidates.parquet")
    if set(query_ids) != set(rankings):
        raise ValueError("Candidate query IDs do not match query.parquet")
    candidate_ids = {doc for docs in rankings.values() for doc in docs}
    print(f"queries={len(query_ids)}, candidate documents={len(candidate_ids):,}", flush=True)

    chunks_path = DATA / "chunks_v4.parquet"
    all_doc_ids = chunk_doc_ids(chunks_path)
    mask = np.isin(all_doc_ids, np.fromiter(candidate_ids, dtype=np.int64))
    original_rows = np.flatnonzero(mask)
    filtered_doc_ids = all_doc_ids[mask]
    del all_doc_ids, mask
    order = np.argsort(filtered_doc_ids, kind="stable")
    sorted_doc_ids = filtered_doc_ids[order]
    del filtered_doc_ids
    print(f"candidate chunks={len(order):,}", flush=True)

    with np.load(DATA / "emb_chunks_v4_fused_candidates.npz") as cache:
        vectors = cache["chunks"]
        query_vectors = cache["queries"]
    if len(vectors) != len(original_rows) or len(query_vectors) != len(query_ids):
        raise ValueError("Cached embedding shape does not match current corpus and queries")
    print("cached embeddings loaded", flush=True)

    # Keep one evidence chunk per ranked document. The final chunk cutoff is
    # selected by actual ZIP size below, using standard deflate compression.
    chosen: dict[int, list[tuple[int, int]]] = {}
    for qi, qid in enumerate(query_ids):
        ranked_docs = rankings[qid][:args.k_docs]
        index_parts: list[np.ndarray] = []
        stops: list[int] = []
        count = 0
        for doc in ranked_docs:
            lo = int(np.searchsorted(sorted_doc_ids, doc, side="left"))
            hi = int(np.searchsorted(sorted_doc_ids, doc, side="right"))
            if lo == hi:
                raise ValueError(f"No chunks for candidate document {doc}")
            part = order[lo:hi]
            index_parts.append(part)
            count += len(part)
            stops.append(count)
        indices = np.concatenate(index_parts)
        scores = vectors[indices] @ query_vectors[qi]
        starts = [0] + stops[:-1]
        selected = [int(indices[start + np.argmax(scores[start:stop])])
                    for start, stop in zip(starts, stops)]
        chosen[qid] = [(doc, int(original_rows[idx])) for doc, idx in zip(ranked_docs, selected)]
        if (qi + 1) % 200 == 0:
            print(f"selected chunks for {qi + 1}/{len(query_ids)} queries", flush=True)
    del vectors, query_vectors

    needed = {row for rows in chosen.values() for _, row in rows[:args.max_chunks]}
    print(f"reading {len(needed):,} original chunk texts", flush=True)
    texts = selected_texts(chunks_path, needed)

    cutoff = min(args.max_chunks, args.k_docs)
    while cutoff > 0:
        records = []
        for qid in query_ids:
            rows = chosen[qid]
            records.append({
                "id": qid,
                "relevant_docs": [doc for doc, _ in rows],
                "relevant_chunks": [
                    {"doc_id": doc, "chunk_text": texts[row]}
                    for doc, row in rows[:cutoff]
                ],
            })
        size_mb = write_zip(records, args.out)
        print(f"K_docs={args.k_docs}, K_chunks={cutoff}, ZIP={size_mb:.2f} MB", flush=True)
        if size_mb <= args.cap_mb:
            return
        cutoff -= 10
    raise ValueError("Could not fit a nonempty submission under the ZIP cap")


if __name__ == "__main__":
    main()
