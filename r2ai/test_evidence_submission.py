"""Rerank evidence from a real baseline ZIP, cache scores, select and validate.

Reuse the baseline retrieval pool to isolate the effect of evidence filtering.
Every evidence text is checked against the original corpus before inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import zipfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.evidence_io import fingerprint, read_ranking_cache, write_ranking_cache
from r2ai.evidence_selector import EvidenceCandidate, QueryRanking, SelectionConfig
from r2ai.rank import RERANK_MODEL, RERANK_REVISION, cross_encoder_rerank
from r2ai.select_evidence import replay
from r2ai.validate import validate


def prepare(baseline: Path, chunks_path: Path, queries_path: Path, output: Path):
    cache = output / "prepared.jsonl.gz"
    sources = {"baseline": fingerprint(baseline), "chunks": fingerprint(chunks_path),
               "queries": fingerprint(queries_path)}
    if cache.exists():
        rankings, metadata = read_ranking_cache(cache)
        if metadata["inputs"] != sources:
            raise ValueError("Prepared pool input fingerprints changed; choose a fresh output")
        return rankings, metadata
    with zipfile.ZipFile(baseline) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError("Baseline ZIP must contain one JSON file")
        records = json.loads(archive.read(names[0]))
    queries = pq.read_table(queries_path, columns=["id", "query"]).to_pylist()
    query_texts = {row["id"]: row["query"] for row in queries}
    if (len(query_texts) != len(queries) or len(records) != len(query_texts)
            or {row["id"] for row in records} != set(query_texts)):
        raise ValueError("Baseline must cover all query IDs exactly once")
    wanted: dict[int, set[str]] = {}
    for row in records:
        for chunk in row["relevant_chunks"]:
            wanted.setdefault(chunk["doc_id"], set()).add(chunk["chunk_text"])
    needed_docs = np.asarray(list(wanted), dtype=np.int64)
    needed = sum(map(len, wanted.values()))
    found = {}
    print(f"Matching {needed:,} unique baseline chunks to original corpus", flush=True)
    parquet = pq.ParquetFile(chunks_path)
    scanned = 0
    started = time.monotonic()
    for batch in parquet.iter_batches(batch_size=100_000, columns=["doc_id", "chunk_index", "text", "title"]):
        docs = batch.column(0).to_numpy(zero_copy_only=False)
        positions = np.flatnonzero(np.isin(docs, needed_docs))
        for position in positions:
            doc = int(docs[position])
            remaining = wanted.get(doc)
            if not remaining:
                continue
            text = batch.column(2)[int(position)].as_py()
            if text in remaining:
                found[doc, text] = (int(batch.column(1)[int(position)].as_py()),
                                    batch.column(3)[int(position)].as_py() or "")
                remaining.remove(text)
        scanned += batch.num_rows
        if scanned % 1_000_000 == 0:
            print(f"Matched {len(found):,}/{needed:,}; scanned {scanned:,} corpus rows", flush=True)
        if len(found) == needed:
            break
    if len(found) != needed:
        raise ValueError(f"{needed - len(found)} baseline chunks do not match current corpus")
    rankings = []
    for row in records:
        candidates = []
        for chunk in row["relevant_chunks"]:
            index, title = found[chunk["doc_id"], chunk["chunk_text"]]
            candidates.append(EvidenceCandidate(chunk["doc_id"], index, chunk["chunk_text"], title=title))
        rankings.append(QueryRanking(row["id"], row["relevant_docs"], candidates, query_texts[row["id"]]))
    metadata = {"inputs": sources, "candidate_source": "existing baseline retrieval evidence, verified exact corpus text",
                "retrieval_scores": "unavailable in baseline ZIP; null", "reranker": {"enabled": False},
                "verified_unique_chunks": len(found), "corpus_rows_scanned": scanned,
                "prepare_seconds": time.monotonic() - started}
    write_ranking_cache(cache, rankings, metadata)
    return rankings, metadata


def rerank_blocks(rankings, source_metadata, output, device, batch, queries_per_block):
    import torch

    torch.set_num_threads(4)
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20, device)
    model = {"enabled": True, "model": RERANK_MODEL, "revision": RERANK_REVISION,
             "activation": "identity", "precision": "float16", "max_length": 512}
    runtime = {"torch": version("torch"), "sentence_transformers": version("sentence-transformers"),
               "transformers": version("transformers"), "device": device}
    scored = []
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for start in range(0, len(rankings), queries_per_block):
        block = rankings[start:start + queries_per_block]
        checkpoint = checkpoints / f"queries_{start:04d}.jsonl.gz"
        if checkpoint.exists():
            cached, manifest = read_ranking_cache(checkpoint)
            if (manifest["inputs"] != source_metadata["inputs"] or manifest["reranker"]["revision"] != RERANK_REVISION
                    or [r.query_id for r in cached] != [r.query_id for r in block]):
                raise ValueError(f"Checkpoint provenance mismatch: {checkpoint}")
            scored.extend(cached)
            print(f"Resumed {len(scored)}/{len(rankings)} queries", flush=True)
            continue
        flat_candidates = [c for row in block for c in row.candidates]
        texts = [c.chunk_text for c in flat_candidates]
        titles = [c.title for c in flat_candidates]
        shortlist = []
        cursor = 0
        for row in block:
            stop = cursor + len(row.candidates)
            # Retrieval scores were not serialized by the old baseline. Keep them unknown.
            shortlist.append((np.arange(cursor, stop), np.full(stop - cursor, np.nan)))
            cursor = stop
        started = time.monotonic()
        while True:
            try:
                pools = cross_encoder_rerank(shortlist, [r.query_text for r in block], texts, titles,
                                             device, batch, model_metadata=model)
                break
            except torch.cuda.OutOfMemoryError:
                if batch <= 1:
                    raise
                batch = max(1, batch // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Retrying with batch={batch}", flush=True)
        result = []
        for row, pool in zip(block, pools, strict=True):
            candidates = [replace(flat_candidates[int(index)], rerank_score=score)
                          for index, score in zip(pool.indices, pool.rerank_scores, strict=True)]
            result.append(QueryRanking(row.query_id, row.document_order, candidates, row.query_text))
        manifest = {**source_metadata, "reranker": dict(model), "runtime": runtime,
                    "priority": "descending authentic cross-encoder score", "batch": batch,
                    "block_seconds": time.monotonic() - started}
        write_ranking_cache(checkpoint, result, manifest)
        scored.extend(result)
        print(f"Checkpoint saved: {len(scored)}/{len(rankings)} queries; "
              f"{len(flat_candidates)/(time.monotonic()-started):.1f} pairs/s", flush=True)
    metadata = {**source_metadata, "reranker": model, "runtime": runtime,
                "priority": "descending authentic cross-encoder score", "batch": batch,
                "threshold_calibrated": False}
    return scored, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--chunks-file", type=Path, default=ROOT / "data/vibiomir/chunks_v4.parquet")
    parser.add_argument("--queries-file", type=Path, default=ROOT / "data/raw/vibiomir/query.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--queries-per-block", type=int, default=20)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--max-chunks", type=int, default=120)
    parser.add_argument("--document-mode", choices=("parents", "baseline", "threshold"), default="baseline")
    args = parser.parse_args()
    if args.queries_per_block <= 0 or args.batch <= 0:
        parser.error("queries-per-block and batch must be positive")
    config = SelectionConfig(mode="threshold", max_docs=args.max_docs, max_chunks=args.max_chunks,
                             chunks_per_doc=1, chunk_threshold=args.threshold, document_mode=args.document_mode)
    args.output.mkdir(parents=True, exist_ok=True)
    rankings, metadata = prepare(args.baseline, args.chunks_file, args.queries_file, args.output)
    if args.prepare_only:
        print(f"Prepared {len(rankings)} queries at {args.output / 'prepared.jsonl.gz'}", flush=True)
        return
    cache = args.output / "rankings.jsonl.gz"
    if cache.exists():
        rankings, metadata = read_ranking_cache(cache)
        if metadata["inputs"] != {"baseline": fingerprint(args.baseline), "chunks": fingerprint(args.chunks_file),
                                   "queries": fingerprint(args.queries_file)}:
            raise ValueError("Ranking cache input fingerprints changed")
    else:
        rankings, metadata = rerank_blocks(rankings, metadata, args.output, args.device, args.batch, args.queries_per_block)
        write_ranking_cache(cache, rankings, metadata)
    summary = replay(rankings, metadata, config, args.output / "selected")
    zip_path = args.output / "selected/submission.zip"
    errors = validate(zip_path, query_path=args.queries_file)
    (args.output / "validation.json").write_text(json.dumps({"valid": not errors, "errors": errors,
                                                             "zip": fingerprint(zip_path)}, indent=2) + "\n")
    if errors:
        raise ValueError(errors)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Validated submission: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
