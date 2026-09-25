"""Run a reproducible E5 exact-dense retrieval smoke experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.adapters import adapt_query
from src.data.loader import DataLoader, DocumentChunkMap, save_jsonl_records
from src.data.schema import PredictionRecord
from src.data.validator import validate_chunks_integrity, validate_queries_integrity
from src.evaluation.evaluator import Evaluator
from src.retrieval.dense_encoding import encode_texts, load_encoder, resolve_device
from src.retrieval.exact_dense import RankedChunk, exact_search
from src.submission.validate import validate_submission_file


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    seed: int = 42
    model_name: str
    revision: str = "main"
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize_embeddings: bool = True
    max_length: int = Field(default=512, ge=8)
    batch_size: int = Field(default=32, ge=1)
    top_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions_at_k(
    query_ids: list[str],
    rankings: list[list[RankedChunk]],
    doc_map: DocumentChunkMap,
    k: int,
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for query_id, ranking in zip(query_ids, rankings, strict=True):
        selected = ranking[:k]
        chunks = [item.chunk_id for item in selected]
        docs: list[str] = []
        for chunk_id in chunks:
            doc_id = doc_map.get_parent_doc(chunk_id)
            if doc_id is not None and doc_id not in docs:
                docs.append(doc_id)
        predictions.append(
            PredictionRecord(id=query_id, relevant_docs=docs, relevant_chunks=chunks)
        )
    return predictions


def cross_language_positive_metrics(
    queries,
    truths_by_id,
    rankings: list[list[RankedChunk]],
    chunk_languages: dict[str, str],
    k: int,
) -> dict[str, dict]:
    """Report VI-query retrieval recall and first-positive rank by positive language."""
    language_metrics: dict[str, dict] = {}
    for language in ("vi", "en", "zh"):
        recalls = []
        first_ranks = []
        hit_queries = 0
        for query, ranking in zip(queries, rankings, strict=True):
            positives = {
                chunk_id
                for chunk_id in truths_by_id[query.id].relevant_chunks
                if chunk_languages.get(chunk_id) == language
            }
            if not positives:
                continue
            retrieved = [item.chunk_id for item in ranking[:k]]
            recalls.append(len(positives.intersection(retrieved)) / len(positives))
            first_rank = next(
                (rank for rank, item in enumerate(ranking, start=1) if item.chunk_id in positives),
                None,
            )
            if first_rank is not None:
                hit_queries += 1
                first_ranks.append(first_rank)
        language_metrics[language] = {
            "positive_query_count": len(recalls),
            "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
            "queries_with_positive_hit": hit_queries,
            "mean_first_positive_rank": (
                sum(first_ranks) / len(first_ranks) if first_ranks else None
            ),
        }
    return language_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke_dense_e5.yaml")
    parser.add_argument("--split-dir", required=True, help="Directory containing chunks/queries/ground_truth JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    started = time.time()
    config_path = Path(args.config).resolve()
    split_dir = Path(args.split_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SmokeConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    top_k_values = sorted(set(config.top_k_values))
    if not top_k_values or min(top_k_values) < 1:
        raise ValueError("top_k_values must contain positive integers.")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(args.device)

    chunk_path = split_dir / "chunks.jsonl"
    query_path = split_dir / "queries.jsonl"
    truth_path = split_dir / "ground_truth.jsonl"
    chunks = DataLoader.load_chunks(chunk_path)
    queries = DataLoader.load_queries(query_path)
    truths = DataLoader.load_ground_truth(truth_path)
    errors = validate_chunks_integrity(chunks) + validate_queries_integrity(queries)
    if errors:
        raise ValueError("Data integrity errors: " + "; ".join(errors))
    if {item.id for item in queries} != {item.id for item in truths}:
        raise ValueError("Query IDs and ground-truth IDs do not match.")

    print(json.dumps(config.model_dump(), indent=2), flush=True)
    print(f"Loading {config.model_name} on {device}...", flush=True)
    tokenizer, model, device, model_revision, tokenizer_revision = load_encoder(
        config.model_name,
        revision=config.revision,
        tokenizer_name=config.tokenizer_name,
        tokenizer_revision=config.tokenizer_revision,
        device=str(device),
    )

    passage_embeddings = encode_texts(
        [config.passage_prefix + item.text for item in chunks],
        tokenizer,
        model,
        batch_size=config.batch_size,
        max_length=config.max_length,
        normalize=config.normalize_embeddings,
        device=device,
    )
    query_embeddings = encode_texts(
        [config.query_prefix + adapt_query(item).text for item in queries],
        tokenizer,
        model,
        batch_size=config.batch_size,
        max_length=config.max_length,
        normalize=config.normalize_embeddings,
        device=device,
    )
    if not np.isfinite(passage_embeddings).all() or not np.isfinite(query_embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    np.save(output_dir / "embeddings.npy", passage_embeddings)
    (output_dir / "chunk_ids.json").write_text(
        json.dumps([item.chunk_id for item in chunks], indent=2), encoding="utf-8"
    )
    rankings = exact_search(
        query_embeddings,
        passage_embeddings,
        [item.chunk_id for item in chunks],
        top_k=max(top_k_values),
    )
    chunk_lookup = {item.chunk_id: item for item in chunks}
    ranking_rows = []
    for query, ranking in zip(queries, rankings, strict=True):
        ranking_rows.append(
            {
                "id": query.id,
                "query": query.query,
                "results": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "doc_id": chunk_lookup[item.chunk_id].doc_id,
                        "language": chunk_lookup[item.chunk_id].language,
                        "score": item.score,
                        "text_preview": chunk_lookup[item.chunk_id].text[:240],
                    }
                    for item in ranking
                ],
            }
        )
    save_jsonl_records(output_dir / "rankings.jsonl", ranking_rows)

    doc_map = DocumentChunkMap.from_chunks(chunks)
    truths_by_id = {truth.id: truth for truth in truths}
    chunk_languages = {chunk.chunk_id: chunk.language for chunk in chunks}
    evaluator = Evaluator(beta=2.0)
    metrics: dict[str, dict] = {}
    validations: dict[str, bool] = {}
    for k in top_k_values:
        predictions = predictions_at_k(
            [item.id for item in queries], rankings, doc_map, k
        )
        prediction_path = output_dir / f"predictions_k{k}.jsonl"
        save_jsonl_records(prediction_path, predictions)
        validation = validate_submission_file(prediction_path, doc_map, query_path)
        validations[str(k)] = validation.is_valid
        metrics[str(k)] = evaluator.evaluate(truths, predictions, doc_map=doc_map).to_dict()
        metrics[str(k)]["vi_query_positive_language"] = cross_language_positive_metrics(
            queries, truths_by_id, rankings, chunk_languages, k
        )

    manifest = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "model_revision": model_revision,
        "tokenizer_name": config.tokenizer_name or config.model_name,
        "tokenizer_revision": tokenizer_revision,
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "embedding_shape": list(passage_embeddings.shape),
        "input_sha256": {
            "chunks": _sha256(chunk_path),
            "queries": _sha256(query_path),
            "ground_truth": _sha256(truth_path),
        },
        "valid_predictions": validations,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Wrote smoke-test artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
