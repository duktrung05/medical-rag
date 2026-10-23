"""Bounded migration runner for a locked Phase 1/2 document and query subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from array import array
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2ai import rank
from r2ai.evidence_io import read_ranking_cache, selection_summary, write_predictions, write_ranking_cache
from r2ai.evidence_selector import EvidenceCandidate, QueryRanking, SelectionConfig, select_evidence
from r2ai.validate import validate as validate_submission
from src.medrag.data.phase2_batch import load_phase2_batch, sha256_file, write_batch
from r2ai.bm25index import tokenize as bm25_tokenize


def _read_id_list(path: Path, name: str) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{name} must be a non-empty JSON array of integer IDs")
    if any(type(value) is not int for value in data) or len(set(data)) != len(data):
        raise ValueError(f"{name} must contain unique integer IDs")
    return data


def _canonical_predictions_sha256(predictions: list[dict]) -> str:
    payload = json.dumps(predictions, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cuda_memory(device: str, *, reset_peak: bool = False) -> dict:
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"available": False}
    torch.cuda.synchronize(device)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "available": True,
        "device": device,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _document_dense_vectors(texts: list[str], titles: list[str], device: str,
                            batch_size: int, model_metadata: dict) -> np.ndarray:
    import torch

    model = rank._embedder(device)
    model.max_seq_length = 256
    first = model._first_module()
    auto_model = getattr(first, "auto_model", None)
    reported = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
    resolved = rank._resolved_model_revision(rank.EMBED_MODEL, rank.EMBED_REVISION)
    if reported is not None and reported != resolved:
        raise ValueError(f"Loaded document encoder commit {reported} differs from cached snapshot {resolved}")
    model_metadata["resolved_revision"] = resolved
    payload = [f"{title}\n{text}" if title else text for title, text in zip(titles, texts, strict=True)]
    vectors = model.encode(payload, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(vectors, dtype=np.float32)


def _rank_documents(batch, settings: dict, device: str, batch_size: int, embedding_metadata: dict):
    passages_by_doc: dict[int, list[dict]] = defaultdict(list)
    for row in batch.passages:
        passages_by_doc[int(row["doc_id"])].append(row)
    documents = sorted(batch.documents, key=lambda row: int(row["doc_id"]))
    doc_ids = [int(row["doc_id"]) for row in documents]
    titles = [row["title"] or "" for row in documents]
    bm25_limit = int(settings["bm25_document_passages"])
    dense_limit = int(settings["dense_document_passages"])
    bm25_texts, dense_texts = [], []
    for doc_id in doc_ids:
        doc_passages = sorted(passages_by_doc[doc_id], key=lambda row: int(row["passage_index"]))
        bm25_texts.append("\n".join(row["passage_text"] for row in doc_passages[:bm25_limit]))
        dense_texts.append("\n".join(row["passage_text"] for row in doc_passages[:dense_limit]))
    dense_vectors = _document_dense_vectors(dense_texts, titles, device, batch_size,
                                            embedding_metadata.setdefault("document_encoder", {}))
    qids = [int(row["id"]) for row in batch.queries]
    qtexts = [str(row["query"]) for row in batch.queries]
    query_vectors = rank.embed_queries(qtexts, device,
                                       embedding_metadata.setdefault("document_query_encoder", {}))
    import bm25s
    from bm25s.tokenization import Tokenized

    vocab: dict[str, int] = {}
    tokenized_docs = []
    for title, text in zip(titles, bm25_texts, strict=True):
        row = array("i")
        for token in bm25_tokenize(f"{title} {text}"):
            token_id = vocab.get(token)
            if token_id is None:
                token_id = vocab[token] = len(vocab)
            row.append(token_id)
        tokenized_docs.append(row)
    lexical = bm25s.BM25(method="lucene")
    lexical.index(Tokenized(ids=tokenized_docs, vocab=vocab))
    query_token_ids = [[vocab[token] for token in bm25_tokenize(query) if token in vocab]
                       for query in qtexts]
    branch_k = int(settings["document_branch_top_k"])
    if any(query_token_ids):
        lexical_idx, lexical_scores = lexical.retrieve(
            Tokenized(ids=query_token_ids, vocab=vocab), k=min(branch_k, len(doc_ids)),
            n_threads=1, show_progress=False)
    else:
        lexical_idx = np.empty((len(qids), 0), dtype=np.int64)
        lexical_scores = np.empty((len(qids), 0), dtype=np.float32)
    fused_by_query = {}
    trace = {}
    k = int(settings["rrf_k"])
    fused_k = int(settings["fused_document_top_k"])
    for query_index, (qid, query) in enumerate(zip(qids, qtexts, strict=True)):
        dense_scores = dense_vectors @ query_vectors[query_index]
        dense_order = sorted(range(len(doc_ids)), key=lambda i: (-float(dense_scores[i]), doc_ids[i]))[:branch_k]
        bm25_order = [doc_ids[int(index)] for index in lexical_idx[query_index]]
        dense_doc_order = [doc_ids[index] for index in dense_order]
        fused = defaultdict(float)
        for weight, ordered in ((float(settings["bm25_weight"]), bm25_order),
                                (float(settings["dense_weight"]), dense_doc_order)):
            for position, doc_id in enumerate(ordered, start=1):
                fused[doc_id] += weight / (k + position)
        order = sorted(fused, key=lambda doc_id: (-fused[doc_id], doc_id))[:fused_k]
        fused_by_query[qid] = set(order)
        trace[qid] = {"bm25_candidates": len(bm25_order), "dense_candidates": len(dense_doc_order),
                      "bm25_top_score": float(lexical_scores[query_index, 0]) if lexical_scores.shape[1] else None,
                      "fused_candidates": len(order), "document_order": order}
    return qids, qtexts, fused_by_query, trace


def _rank_passages(batch, allowed_by_query, qids, qtexts, settings, device, batch_size,
                   embedding_metadata: dict):
    texts = [row["passage_text"] for row in batch.passages]
    titles_by_doc = {int(row["doc_id"]): row["title"] or "" for row in batch.documents}
    titles = [titles_by_doc[int(row["doc_id"])] for row in batch.passages]
    doc_of = np.asarray([int(row["doc_id"]) for row in batch.passages], dtype=np.int64)
    chunk_indices = np.asarray([int(row["passage_index"]) for row in batch.passages], dtype=np.int32)
    passage_vectors = rank.embed_chunks(
        texts, titles, batch_size, device,
        model_metadata=embedding_metadata.setdefault("passage_encoder", {}))
    query_vectors = rank.embed_queries(
        qtexts, device, embedding_metadata.setdefault("passage_query_encoder", {}))
    shortlist = rank.dense_shortlist(passage_vectors, query_vectors, qids, doc_of,
                                     int(settings["passage_top_k"]), allowed_by_query, device)
    reranker_meta = {"enabled": True, "name": rank.RERANK_MODEL,
                     "requested_revision": settings["reranker_revision"]}
    ranked = rank.cross_encoder_rerank(shortlist, qtexts, texts, titles, device,
                                       int(settings["reranker_batch_size"]),
                                       int(settings["rerank_top"]),
                                       revision=settings["reranker_revision"],
                                       model_metadata=reranker_meta)
    provenance = {(int(row["doc_id"]), int(row["passage_index"])): row for row in batch.passages}
    rankings = []
    for qid, qtext, pool in zip(qids, qtexts, ranked, strict=True):
        document_order, seen_docs, candidates = [], set(), []
        for index, dense_score, rerank_score in zip(pool.indices, pool.retrieval_scores,
                                                     pool.rerank_scores, strict=True):
            pos = int(index)
            doc_id, passage_index = int(doc_of[pos]), int(chunk_indices[pos])
            if doc_id not in seen_docs:
                document_order.append(doc_id)
                seen_docs.add(doc_id)
            row = provenance[(doc_id, passage_index)]
            candidates.append(EvidenceCandidate(
                doc_id=doc_id, chunk_index=passage_index, chunk_text=texts[pos],
                rerank_score=rerank_score, retrieval_score=float(dense_score), title=titles[pos]))
            if row["passage_text"] != texts[pos] or row["passage_id"] != provenance[(doc_id, passage_index)]["passage_id"]:
                raise ValueError("Ranked passage does not resolve to its Phase 2 provenance row")
        rankings.append(QueryRanking(int(qid), document_order, candidates, qtext))
    return rankings, reranker_meta


def run(config_path: Path, doc_ids_path: Path, query_ids_path: Path, output: Path,
        device: str, batch_size: int, subset_manifest_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed_output_root = (ROOT / config["output_root"]).resolve()
    if output != allowed_output_root and allowed_output_root not in output.parents:
        raise ValueError(f"Output must be inside the configured namespace: {allowed_output_root}")
    docs = _read_id_list(doc_ids_path, "doc_ids")
    queries = _read_id_list(query_ids_path, "query_ids")
    if len(docs) > int(config["max_documents"]):
        raise ValueError(f"This runner is limited to {config['max_documents']} documents")
    started = time.time()
    batch = load_phase2_batch(config_path, doc_ids=docs, query_ids=queries,
                              max_documents=int(config["max_documents"]))
    stage_seconds = {"load_and_provenance_seconds": time.time() - started}
    embedding_metadata = {}
    settings = config["retrieval"]
    models = config["models"]
    expected = {"embedding_model": rank.EMBED_MODEL, "embedding_revision": rank.EMBED_REVISION,
                "reranker_model": rank.RERANK_MODEL, "reranker_revision": rank.RERANK_REVISION}
    for key, value in expected.items():
        if models.get(key) != value:
            raise ValueError(f"Configured {key} does not match the pinned r2ai model contract")
    if (int(models["passage_max_length"]) != 512 or int(models["reranker_max_length"]) != 512
            or not models["passage_title_prefix"] or not models["normalize_embeddings"]):
        raise ValueError("Runner currently supports the pinned 512-token, title-prefixed normalized passage contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Temporary run path already exists: {staging}")
    staging.mkdir()
    try:
        dataset_dir = staging / "dataset"
        write_batch(batch, dataset_dir)
        (staging / "query_subset.parquet").write_bytes((dataset_dir / "queries.parquet").read_bytes())
        shutil.copyfile(doc_ids_path, staging / "doc_ids.json")
        shutil.copyfile(query_ids_path, staging / "query_ids.json")
        if subset_manifest_path is not None:
            subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
            if subset_manifest.get("doc_ids_sha256") != sha256_file(doc_ids_path):
                raise ValueError("Locked subset manifest doc_ids hash differs from the supplied IDs")
            if subset_manifest.get("query_ids_sha256") != sha256_file(query_ids_path):
                raise ValueError("Locked subset manifest query_ids hash differs from the supplied IDs")
            shutil.copyfile(subset_manifest_path, staging / "subset_manifest.json")
        stage_start = time.time()
        _cuda_memory(device, reset_peak=True)
        qids, qtexts, allowed_by_query, doc_trace = _rank_documents(
            batch, settings, device, batch_size, embedding_metadata)
        stage_seconds["document_retrieval_seconds"] = time.time() - stage_start
        gpu_memory = {"document_retrieval": _cuda_memory(device)}
        stage_start = time.time()
        _cuda_memory(device, reset_peak=True)
        rankings, reranker_meta = _rank_passages(batch, allowed_by_query, qids, qtexts,
                                                 {**settings, **models}, device, batch_size,
                                                 embedding_metadata)
        stage_seconds["passage_retrieval_and_reranking_seconds"] = time.time() - stage_start
        gpu_memory["passage_retrieval_and_reranking"] = _cuda_memory(device)
        metadata = {
            "pipeline": "phase2_strategy_c_subset_migration_v1",
            "input_manifest_sha256": hashlib.sha256(
                (dataset_dir / "input_manifest.json").read_bytes()).hexdigest(),
            "provenance_sha256": sha256_file(dataset_dir / "passage_provenance.parquet"),
            "chunks_sha256": sha256_file(dataset_dir / "chunks.parquet"),
            "config_sha256": sha256_file(config_path),
            "doc_ids_file_sha256": sha256_file(doc_ids_path),
            "query_ids_file_sha256": sha256_file(query_ids_path),
            "subset_doc_ids_sha256": hashlib.sha256(json.dumps(sorted(docs)).encode()).hexdigest(),
            "subset_query_ids_sha256": hashlib.sha256(json.dumps(sorted(queries)).encode()).hexdigest(),
            "document_retrieval": settings,
            "models": {**models, "device": device, "batch_size": batch_size,
                       "precision": "float16 model weights; float32 cached vectors",
                       "embedding_resolved_revisions": embedding_metadata,
                       "reranker_resolved_revision": reranker_meta.get("resolved_revision")},
            "reranker": reranker_meta,
            "text_format": "title + newline + unchanged Phase 2 passage_text",
            "stage1_document_order_for_export": "first occurrence in reranked passage list",
        }
        rankings_path = staging / "rankings.jsonl.gz"
        write_ranking_cache(rankings_path, rankings, metadata)
        config_selection = SelectionConfig(**config["selection"])
        predictions, diagnostics = [], []
        for ranking in rankings:
            prediction, diagnostic = select_evidence(ranking, config_selection)
            predictions.append(prediction)
            diagnostics.append(diagnostic)
        passage_texts_by_doc: dict[int, set[str]] = defaultdict(set)
        for row in batch.passages:
            passage_texts_by_doc[int(row["doc_id"])].add(row["passage_text"])
        for prediction in predictions:
            if len(prediction["relevant_docs"]) > config_selection.max_docs:
                raise ValueError(f"Selected document cap exceeded for query {prediction['id']}")
            if len(prediction["relevant_chunks"]) > config_selection.max_chunks:
                raise ValueError(f"Selected passage cap exceeded for query {prediction['id']}")
            for evidence in prediction["relevant_chunks"]:
                if evidence["doc_id"] not in prediction["relevant_docs"]:
                    raise ValueError(f"Selected evidence has no selected parent: query {prediction['id']}")
                if evidence["chunk_text"] not in passage_texts_by_doc[evidence["doc_id"]]:
                    raise ValueError(f"Selected evidence does not match its Phase 2 parent: query {prediction['id']}")
        zip_path = staging / "submission.zip"
        write_predictions(zip_path, predictions)
        (staging / "diagnostics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in diagnostics),
            encoding="utf-8")
        (staging / "selection_config.json").write_text(
            json.dumps(asdict(config_selection), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cached_rankings, cached_metadata = read_ranking_cache(rankings_path)
        replay_predictions = [select_evidence(ranking, config_selection)[0] for ranking in cached_rankings]
        replay_sha = _canonical_predictions_sha256(replay_predictions)
        prediction_sha = _canonical_predictions_sha256(predictions)
        if replay_sha != prediction_sha or cached_metadata != metadata:
            raise ValueError("CPU ranking-cache replay does not match the in-memory selection payload")
        (staging / "cache_replay_report.json").write_text(json.dumps({
            "status": "pass", "model_inference": False,
            "canonical_predictions_sha256": replay_sha,
            "rankings_sha256": sha256_file(rankings_path),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validation_errors = validate_submission(zip_path, strict_ids=True,
                                                query_path=dataset_dir / "queries.parquet",
                                                corpus_path=Path(ROOT / config["official_corpus"]))
        validation_report = {"status": "pass" if not validation_errors else "fail",
                             "errors": validation_errors, "strict_official_ids": True,
                             "query_count": len(qids), "queries_requested": len(queries)}
        (staging / "validation_report.json").write_text(
            json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if validation_errors:
            raise ValueError("Submission validation failed: " + "; ".join(validation_errors[:5]))
        resolved_revisions = [item.get("resolved_revision") for item in embedding_metadata.values()]
        if any(revision != models["embedding_revision"] for revision in resolved_revisions):
            raise ValueError(f"Embedding model resolved revisions differ from the pinned commit: {embedding_metadata}")
        if reranker_meta.get("resolved_revision") != models["reranker_revision"]:
            raise ValueError(f"Reranker resolved revision differs from the pinned commit: {reranker_meta}")
        summary = {**selection_summary(predictions), "config": asdict(config_selection),
                   "ranking_metadata": metadata, "model_inference": True,
                   "submission_validation": "pass", "f2": "unmeasured"}
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_manifest = {
            "format": "medrag-phase2-batch-run-v1", "status": "complete",
            "scope": "explicit_subset_only", "documents": len(docs), "queries": len(queries),
            "stage_seconds": stage_seconds, "document_retrieval_trace": doc_trace,
            "input_manifest_sha256": metadata["input_manifest_sha256"],
            "rankings_sha256": sha256_file(rankings_path),
            "submission_sha256": sha256_file(zip_path),
            "canonical_predictions_sha256": prediction_sha,
            "gpu_memory": gpu_memory,
            "reranked_passages": sum(item["reranked_count"] for item in diagnostics),
            "selected_evidence_grounding": "pass",
            "cache_replay": "pass", "validation": validation_report,
            "summary": summary,
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return run_manifest
    except Exception:
        # Retain the isolated incomplete directory for diagnosis; never publish a complete manifest.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    parser.add_argument("--doc-ids", type=Path, required=True, help="JSON array of 1..1000 official doc IDs")
    parser.add_argument("--query-ids", type=Path, required=True, help="JSON array of official query IDs")
    parser.add_argument("--subset-manifest", type=Path,
                        help="Manifest emitted by lock_phase2_batch_subset.py")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.doc_ids.resolve(), args.query_ids.resolve(),
                         args.output.resolve(), args.device, args.batch_size,
                         args.subset_manifest.resolve() if args.subset_manifest else None),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
