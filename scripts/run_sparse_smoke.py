"""Evaluate a sparse BM25 experiment on a prepared split."""

import argparse
import json
from pathlib import Path

from src.adapters import adapt_query
from src.config import PipelineConfig, load_pipeline_config
from src.data.loader import DataLoader, DocumentChunkMap
from src.data.schema import PredictionRecord
from src.evaluation.evaluator import Evaluator
from src.evaluation.recall import calculate_recall
from src.indexing.sparse_index import corpus_sha256
from src.retrieval.bm25 import BM25Retriever
from src.submission.validate import validate_submission_file


def _predictions_at_k(query_ids, rankings, doc_map, k):
    predictions = []
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        chunk_ids = [chunk_id for chunk_id, _ in ranking[:k]]
        doc_ids = []
        for chunk_id in chunk_ids:
            doc_id = doc_map.get_parent_doc(chunk_id)
            if doc_id is not None and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        predictions.append(
            PredictionRecord(id=query_id, relevant_chunks=chunk_ids, relevant_docs=doc_ids)
        )
    return predictions


def run_smoke(config: PipelineConfig, split_dir: Path, output_dir: Path, top_k_values: list[int]):
    if config.backend != "sparse":
        raise ValueError("Sparse smoke requires backend: sparse")
    sparse = config.retrieval.sparse
    chunks = DataLoader.load_chunks(split_dir / "chunks.jsonl")
    queries = DataLoader.load_queries(split_dir / "queries.jsonl")
    truths = DataLoader.load_ground_truth(split_dir / "ground_truth.jsonl")
    if {query.id for query in queries} != {truth.id for truth in truths}:
        raise ValueError("Query IDs and ground-truth IDs do not match")

    expected_hash = corpus_sha256(chunks)
    BM25Retriever.build_index(chunks, sparse.index_path, sparse)
    retriever = BM25Retriever(
        sparse.index_path,
        expected_corpus_hash=expected_hash,
        tokenizer=sparse.tokenizer,
        char_ngram_size=sparse.char_ngram_size,
    )
    rankings = retriever.search_batch(
        [adapt_query(query).text for query in queries], top_k=max(top_k_values)
    )

    truth_by_id = {truth.id: truth for truth in truths}
    first_positive_ranks = []
    for query, ranking in zip(queries, rankings, strict=True):
        relevant = set(truth_by_id[query.id].relevant_chunks)
        first_positive_ranks.append(
            next((rank for rank, (chunk_id, _) in enumerate(ranking, start=1) if chunk_id in relevant), None)
        )
    observed_ranks = [rank for rank in first_positive_ranks if rank is not None]

    doc_map = DocumentChunkMap.from_chunks(chunks)
    evaluator = Evaluator(beta=2.0)
    metrics = {}
    for k in top_k_values:
        predictions = _predictions_at_k([query.id for query in queries], rankings, doc_map, k)
        metrics[str(k)] = {
            "chunk_recall_at_k": sum(
                calculate_recall(truth_by_id[pred.id].relevant_chunks, pred.relevant_chunks)
                for pred in predictions
            ) / len(predictions),
            "evaluation": evaluator.evaluate(truths, predictions, doc_map=doc_map).to_dict(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_name": config.experiment_name,
        "tokenizer": sparse.tokenizer,
        "char_ngram_size": sparse.char_ngram_size,
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "corpus_sha256": expected_hash,
        "mean_first_positive_rank": (
            sum(observed_ranks) / len(observed_ranks) if observed_ranks else None
        ),
        "queries_with_positive_hit": len(observed_ranks),
        "query_count": len(queries),
        "metrics_at_k": metrics,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_submission_file(
        _write_top_k_predictions(output_dir, queries, rankings, doc_map, max(top_k_values)),
        doc_map,
        split_dir / "queries.jsonl",
    )
    if not validation.is_valid:
        validation.print_summary()
        raise ValueError("Sparse smoke predictions failed submission validation")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _write_top_k_predictions(output_dir, queries, rankings, doc_map, k):
    from src.data.loader import save_jsonl_records

    path = output_dir / f"predictions_k{k}.jsonl"
    save_jsonl_records(
        path,
        _predictions_at_k([query.id for query in queries], rankings, doc_map, k),
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-dir", default="data/medquad/dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-values", default="10,50,100")
    args = parser.parse_args()
    top_k_values = sorted({int(value) for value in args.top_k_values.split(",") if value.strip()})
    if not top_k_values or min(top_k_values) < 1:
        raise ValueError("--top-k-values must contain positive integers")
    config = load_pipeline_config(args.config)
    run_smoke(config, Path(args.split_dir).resolve(), Path(args.output_dir).resolve(), top_k_values)


if __name__ == "__main__":
    main()
