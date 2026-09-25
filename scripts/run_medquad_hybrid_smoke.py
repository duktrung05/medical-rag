"""Run a time-bounded, reproducible BM25+dense hybrid smoke on MedQuAD dev."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_hybrid.yaml")
    parser.add_argument("--output-dir", default="outputs/medical_medquad_hybrid_smoke")
    parser.add_argument("--query-limit", type=int, default=32)
    parser.add_argument("--corpus-limit", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.query_limit, args.corpus_limit, args.timeout_seconds) < 1:
        parser.error("query-limit, corpus-limit, and timeout-seconds must be positive")
    if args.worker:
        run_smoke(args)
        return 0

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_args = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--config", str(Path(args.config).resolve()),
        "--output-dir", str(output),
        "--query-limit", str(args.query_limit),
        "--corpus-limit", str(args.corpus_limit),
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    started = time.perf_counter()
    try:
        result = subprocess.run(worker_args, env=env, timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        timeout_record = {
            "status": "timeout",
            "timeout_seconds": args.timeout_seconds,
            "elapsed_seconds": elapsed,
            "query_limit": args.query_limit,
            "corpus_limit": args.corpus_limit,
        }
        (output / "run_status.json").write_text(json.dumps(timeout_record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(timeout_record, indent=2))
        return 124
    return result.returncode


def run_smoke(args) -> None:
    from src.config import DenseRetrievalConfig, SparseRetrievalConfig, load_pipeline_config
    from src.data.loader import DataLoader, DocumentChunkMap, save_jsonl_records
    from src.data.schema import ChunkRecord
    from src.data.validator import validate_chunks_integrity
    from src.evaluation.evaluator import Evaluator
    from src.indexing.sparse_index import corpus_sha256
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.pipeline import RetrievalPipeline

    started = time.perf_counter()
    config = load_pipeline_config(args.config)
    chunks = DataLoader.load_chunks(config.corpus)
    queries = DataLoader.load_queries(config.corpus.parent / "queries.jsonl")
    truths = DataLoader.load_ground_truth(config.corpus.parent / "ground_truth.jsonl")
    truths_by_id = {record.id: record for record in truths}
    rng = random.Random(42)
    selected_queries = sorted(rng.sample(queries, min(args.query_limit, len(queries))), key=lambda row: row.id)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    positive_ids = {
        cid for query in selected_queries for cid in truths_by_id[query.id].relevant_chunks
        if cid in chunk_by_id
    }
    if len(positive_ids) > args.corpus_limit:
        raise ValueError(
            f"Selected queries have {len(positive_ids)} unique positives, exceeding corpus-limit={args.corpus_limit}; "
            "raise corpus-limit or lower query-limit"
        )
    distractors = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in positive_ids]
    rng.shuffle(distractors)
    selected_ids = positive_ids | set(distractors[:args.corpus_limit - len(positive_ids)])
    selected_chunks = [chunk_by_id[cid] for cid in sorted(selected_ids)]
    errors = validate_chunks_integrity(selected_chunks)
    if errors:
        raise ValueError("Subset data integrity errors: " + "; ".join(errors))

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl_records(output / "chunks.jsonl", selected_chunks)
    save_jsonl_records(output / "queries.jsonl", selected_queries)
    selected_truths = [truths_by_id[query.id] for query in selected_queries]
    save_jsonl_records(output / "ground_truth.jsonl", selected_truths)

    sparse_dir, dense_dir = output / "sparse_index", output / "dense_index"
    sparse_config = SparseRetrievalConfig(enabled=True, index_path=sparse_dir, top_k=args.corpus_limit)
    dense_config: DenseRetrievalConfig = config.retrieval.dense.model_copy(update={
        "index_path": dense_dir,
        "top_k": args.corpus_limit,
    })
    BM25Retriever.build_index(selected_chunks, sparse_dir, sparse_config)
    dense_manifest = DenseRetriever.build_index(selected_chunks, dense_config, device="cpu")

    expected_hash = corpus_sha256(selected_chunks)
    sparse = BM25Retriever(sparse_dir, expected_corpus_hash=expected_hash)
    dense = DenseRetriever(
        model_name=dense_config.model_name or "",
        index_path=dense_dir,
        revision=dense_config.revision,
        tokenizer_name=dense_config.tokenizer_name,
        tokenizer_revision=dense_config.tokenizer_revision,
        batch_size=dense_config.batch_size,
        max_length=dense_config.max_length,
        query_prefix=dense_config.query_prefix,
        passage_prefix=dense_config.passage_prefix,
        normalize_embeddings=dense_config.normalize_embeddings,
        expected_corpus_hash=expected_hash,
        index_type=dense_config.index_type,
        device="cpu",
    )
    hybrid = HybridRetriever(
        sparse, dense, rrf_k=config.fusion.rrf_k,
        sparse_top_k=args.corpus_limit, dense_top_k=args.corpus_limit,
    )
    doc_map = DocumentChunkMap.from_chunks(selected_chunks)
    pipeline = RetrievalPipeline(
        hybrid, doc_map,
        chunk_text_lookup={chunk.chunk_id: chunk.text for chunk in selected_chunks},
        chunk_context_lookup={chunk.chunk_id: (chunk.title, chunk.context) for chunk in selected_chunks},
        retrieval_top_k=min(config.fusion.top_k, args.corpus_limit),
        chunk_threshold=0.0, chunk_delta=1.0, doc_threshold=0.0, doc_delta=1.0,
        chunk_min_k=1, doc_min_k=1, max_chunk_k=20, max_doc_k=20,
    )
    predictions = []
    rankings = []
    for query in selected_queries:
        predictions.append(pipeline.run_query(query.id, query.query))
        candidate_details = pipeline.last_candidate_scores
        rankings.append({
            "id": query.id,
            "results": [
                {"rank": rank, "chunk_id": cid,
                 "rrf_score": detail["raw_retrieval_score"],
                 "bm25_rank": detail["provenance"].get("bm25_rank"),
                 "dense_rank": detail["provenance"].get("dense_rank"),
                 "sources": detail["provenance"].get("sources", ())}
                for rank, (cid, detail) in enumerate(candidate_details.items(), 1)
            ],
        })

    # Persist predictions from the end-to-end pipeline. Candidate rankings retain provenance.
    save_jsonl_records(output / "predictions.jsonl", predictions)
    save_jsonl_records(output / "rankings.jsonl", rankings)
    metrics = Evaluator(beta=2).evaluate(selected_truths, predictions).to_dict()
    duration = time.perf_counter() - started
    report = {
        "status": "complete",
        "dataset": "MedQuAD dev subset",
        "query_language": "en",
        "num_queries": len(selected_queries),
        "num_chunks": len(selected_chunks),
        "positive_chunks_in_subset": len(positive_ids),
        "device": "cpu",
        "elapsed_seconds": duration,
        "rrf_k": config.fusion.rrf_k,
        "metrics": metrics,
        "dense_index_model_revision": dense_manifest["model_revision"],
        "scope_note": "Time-bounded smoke run; metrics are for a sampled subset and are not full-dev results.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "run_status.json").write_text(json.dumps({"status": "complete", "elapsed_seconds": duration}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
