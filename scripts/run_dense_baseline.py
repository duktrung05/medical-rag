"""Benchmark the configured production dense index on a labeled medical split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.config import load_pipeline_config
from src.data.loader import DataLoader, DocumentChunkMap, save_jsonl_records
from src.data.validator import validate_chunks_integrity, validate_queries_integrity
from src.evaluation.evaluator import Evaluator
from src.indexing.sparse_index import corpus_sha256
from src.retrieval.dense import DenseRetriever
from src.retrieval.exact_dense import RankedChunk
from src.submission.validate import validate_submission_file
from scripts.run_dense_smoke import cross_language_positive_metrics, predictions_at_k


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_dense.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_pipeline_config(config_path)
    if config.backend != "dense":
        raise ValueError("Benchmark requires a dense-only pipeline config")
    split_dir = config.corpus.parent
    query_path = split_dir / "queries.jsonl"
    truth_path = split_dir / "ground_truth.jsonl"
    chunks = DataLoader.load_chunks(config.corpus)
    queries = DataLoader.load_queries(query_path)
    truths = DataLoader.load_ground_truth(truth_path)
    errors = validate_chunks_integrity(chunks) + validate_queries_integrity(queries)
    if errors:
        raise ValueError("Data integrity errors: " + "; ".join(errors))
    if {query.id for query in queries} != {truth.id for truth in truths}:
        raise ValueError("Query IDs and ground-truth IDs do not match")

    dense = config.retrieval.dense
    retriever = DenseRetriever(
        model_name=dense.model_name or "",
        index_path=dense.index_path,
        revision=dense.revision,
        tokenizer_name=dense.tokenizer_name,
        tokenizer_revision=dense.tokenizer_revision,
        batch_size=dense.batch_size,
        max_length=dense.max_length,
        query_prefix=dense.query_prefix,
        passage_prefix=dense.passage_prefix,
        normalize_embeddings=dense.normalize_embeddings,
        expected_corpus_hash=corpus_sha256(chunks),
        index_type=dense.index_type,
        device=args.device or dense.device,
    )
    top_k_values = (1, 3, 5, 10)
    max_k = min(dense.top_k, len(chunks))
    raw_rankings = retriever.search_batch([query.query for query in queries], top_k=max_k)
    rankings = [
        [RankedChunk(chunk_id, score, rank) for rank, (chunk_id, score) in enumerate(row, 1)]
        for row in raw_rankings
    ]
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl_records(
        output_dir / "rankings.jsonl",
        [
            {
                "id": query.id,
                "results": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "doc_id": chunk_lookup[item.chunk_id].doc_id,
                        "language": chunk_lookup[item.chunk_id].language,
                        "score": item.score,
                    }
                    for item in ranking
                ],
            }
            for query, ranking in zip(queries, rankings, strict=True)
        ],
    )

    doc_map = DocumentChunkMap.from_chunks(chunks)
    truths_by_id = {truth.id: truth for truth in truths}
    chunk_languages = {chunk.chunk_id: chunk.language for chunk in chunks}
    evaluator = Evaluator(beta=2.0)
    metrics = {}
    for k in top_k_values:
        predictions = predictions_at_k([query.id for query in queries], rankings, doc_map, k)
        prediction_path = output_dir / f"predictions_k{k}.jsonl"
        save_jsonl_records(prediction_path, predictions)
        validation = validate_submission_file(prediction_path, doc_map, query_path)
        if not validation.is_valid:
            raise ValueError(f"Invalid predictions at K={k}: {validation}")
        metrics[str(k)] = evaluator.evaluate(truths, predictions).to_dict()
        metrics[str(k)]["vi_query_positive_language"] = cross_language_positive_metrics(
            queries, truths_by_id, rankings, chunk_languages, k
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment_name": config.experiment_name,
        "config_path": str(config_path),
        "index_path": str(dense.index_path),
        "model_name": retriever.model_name,
        "model_revision": retriever.model_revision,
        "tokenizer_revision": retriever.tokenizer_revision_resolved,
        "device": str(retriever.device),
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "ranking_depth": max_k,
        "input_sha256": {
            "config": sha256(config_path),
            "chunks": sha256(config.corpus),
            "queries": sha256(query_path),
            "ground_truth": sha256(truth_path),
            "index_manifest": sha256(dense.index_path / "manifest.json"),
            "index_embeddings": sha256(dense.index_path / "passage_embeddings.npy"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(queries)} query rankings and VI → VI/EN/ZH metrics to {output_dir}")


if __name__ == "__main__":
    main()
