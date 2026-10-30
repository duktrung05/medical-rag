"""Run one label-blind, checkpointed recovery candidate on a CUDA host.

The retrieval and scoring stages deliberately use the same GPU of the current
run. They do not compare logits with the former A6000 runtime. A completed
export is still subject to source, submission, ZIP and size validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from scripts.v4.prepare_end_to_end_evidence_recovery import OUTPUT, ROOT, read, write_once
from scripts.v4.run_evidence_window_allocation import fit_window
from src.medrag.adapters.validation import canonical_payload, package_zip, validate_zip
from src.medrag.core.config import load_config
from src.medrag.core.contracts import Hit
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.bge_encoder import BGEM3Encoder
from src.medrag.retrieval.disk_dense import DiskDense
from src.medrag.retrieval.sql_aggregate_bm25 import SQLAggregateDiskBM25
from src.medrag.ranking.reranker import BGEReranker
from src.medrag.evidence.selector import SelectionPolicy

QUERY_IDS = range(1, 1201)
BRANCHES = ("document_bm25", "passage_bm25", "dense")
DEPTH = 800
WINDOW = 192
STRIDE = 96
MAX_CONTEXT = 5500


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def inputs(output: Path):
    config = load_config(ROOT / "configs/v4_submission_p2_a0.yaml")
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    lock = read(output / "policy_lock.json")
    if manifest["fingerprint"] != lock["index_fingerprint"]:
        raise ValueError("Index fingerprint differs from locked recovery input")
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    if sha256_file(ROOT / migration["official_queries"]) != lock["queries_sha256"]:
        raise ValueError("Official query file differs from locked recovery input")
    cap_zip = ROOT / "outputs/v4/evidence_window_cap10_full_20261025/export/submission.zip"
    if sha256_file(cap_zip) != lock["cap10_zip_sha256"]:
        raise ValueError("Baseline ZIP differs from locked recovery input")
    from scripts.v4.run_evidence_window_allocation import load_queries
    queries = load_queries()
    with zipfile.ZipFile(cap_zip) as archive:
        if archive.namelist() != ["predictions.json"] or archive.testzip() is not None:
            raise ValueError("Baseline ZIP structure/CRC differs")
        baseline = json.loads(archive.read("predictions.json"))
    if set(queries) != set(QUERY_IDS) or {int(row["id"]) for row in baseline} != set(QUERY_IDS):
        raise ValueError("Official query or baseline coverage differs")
    return config, index, manifest, queries, {int(row["id"]): row for row in baseline}


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; this host cannot execute full retrieval/scoring")
    return torch


def runtime_record(output: Path, stage: str, torch):
    gpu = torch.cuda.get_device_properties(0)
    record = {"format": "v4-recovery-runtime-v1", "stage": stage,
              "gpu": gpu.name, "torch": torch.__version__, "cuda": torch.version.cuda,
              "policy_sha256": sha256_file(output / "policy_lock.json"),
              "implementation_sha256": sha256_file(Path(__file__).resolve())}
    write_once(output / f"runtime_{stage}.json", record)
    return record


def dense_batch_top800(dense: DiskDense, queries: dict[int, str], torch,
                       *, device: str = "cuda:0") -> dict[int, list[Hit]]:
    """Search several queries in one FP32 matrix pass, then map final row IDs."""
    vectors = np.asarray(dense.encoder.encode_query(list(queries.values())), dtype=np.float32)
    if (vectors.shape != (len(queries), int(dense.meta["dimension"]))
            or not np.isfinite(vectors).all()):
        raise ValueError("Invalid query embedding batch")
    q = torch.as_tensor(vectors.copy(), device=device)
    q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True)
    matrix = np.load(dense.matrix_path, mmap_mode="r")
    top_values = torch.empty((len(queries), 0), device=device)
    top_ids = torch.empty((len(queries), 0), device=device, dtype=torch.long)
    for start in range(0, int(dense.meta["rows"]), 8192):
        end = min(start + 8192, int(dense.meta["rows"]))
        block = torch.as_tensor(np.asarray(matrix[start:end]).copy(), device=device)
        scores = ((block @ q.T) / torch.linalg.vector_norm(block, dim=1)[:, None]).T
        keep = min(DEPTH + 64, end - start)
        block_values, block_offsets = torch.topk(scores, keep, dim=1)
        values = torch.cat((top_values, block_values), dim=1)
        ids = torch.cat((top_ids, block_offsets + start), dim=1)
        top_values, positions = torch.topk(values, min(DEPTH + 64, values.shape[1]), dim=1)
        top_ids = torch.gather(ids, 1, positions)
        del block, scores, block_values, block_offsets, values, ids, positions
    values_cpu = top_values.cpu().tolist()
    ids_cpu = top_ids.cpu().tolist()
    all_ids = sorted({int(row_id) for ids in ids_cpu for row_id in ids})
    identities = {}
    with sqlite3.connect(f"file:{dense.mapping}?mode=ro", uri=True) as db:
        for offset in range(0, len(all_ids), 800):
            ids = all_ids[offset:offset + 800]
            sql = ("SELECT row_id,official_id,passage_id FROM rows WHERE row_id IN (" +
                   ",".join("?" for _ in ids) + ")")
            identities.update((int(row_id), (int(doc), str(pid)))
                              for row_id, doc, pid in db.execute(sql, ids))
    if len(identities) != len(all_ids):
        raise ValueError("Dense mapping incomplete")
    result = {}
    for qid, values, ids in zip(queries, values_cpu, ids_cpu, strict=True):
        candidates = zip(values, ids, strict=True)
        ordered = sorted(candidates, key=lambda row: (-row[0], *identities[row[1]]))[:DEPTH]
        result[qid] = [Hit(*identities[row_id], score, "dense", rank)
                       for rank, (score, row_id) in enumerate(ordered, 1)]
    return result


def parent_fusion(branches: dict[str, list[Hit]]) -> list[dict]:
    scores = defaultdict(float)
    seeds = defaultdict(list)
    for name in BRANCHES:
        seen = set()
        for hit in branches[name]:
            if hit.official_id in seen:
                continue
            seen.add(hit.official_id)
            scores[hit.official_id] += 1 / (60 + hit.rank)
            if hit.passage_id is not None:
                seeds[hit.official_id].append((hit.rank, name, str(hit.passage_id)))
    ordered = sorted(scores, key=lambda doc: (-scores[doc], doc))[:DEPTH]
    return [{"doc_id": doc, "rrf": scores[doc],
             "seed_passage_id": min(seeds[doc])[2] if seeds[doc] else None}
            for doc in ordered]


def retrieve(output: Path, start: int, stop: int) -> None:
    torch = require_cuda()
    config, index, manifest, queries, _ = inputs(output)
    runtime_record(output, "retrieve", torch)
    fingerprint = manifest["fingerprint"]
    lexical = {
        "document_bm25": SQLAggregateDiskBM25(index / "documents.sqlite", "document_bm25", fingerprint),
        "passage_bm25": SQLAggregateDiskBM25(index / "passages.sqlite", "passage_bm25", fingerprint),
    }
    dense_config = config["dense"]
    encoder = BGEM3Encoder(dense_config["model"], revision=dense_config["revision"],
                           document_max_length=dense_config["document_max_length"],
                           query_max_length=dense_config["query_max_length"], device="cuda:0")
    dense = DiskDense(index / "dense", encoder, manifest["dense"]["fingerprint"])
    for batch_start in range(start, stop + 1, 16):
        pending = {}
        for qid in range(batch_start, min(batch_start + 16, stop + 1)):
            target = output / "retrieval" / f"{qid}.json"
            if target.exists():
                row = read(target)
                if row["query_sha256"] != digest(queries[qid]):
                    raise ValueError(f"Retrieval checkpoint query changed: {qid}")
            else:
                pending[qid] = queries[qid]
        if not pending:
            continue
        started = time.monotonic()
        dense_hits = dense_batch_top800(dense, pending, torch)
        for qid, query in pending.items():
            branches = {name: obj.search(query, DEPTH) for name, obj in lexical.items()}
            branches["dense"] = dense_hits[qid]
            parents = parent_fusion(branches)
            write_once(output / "retrieval" / f"{qid}.json",
                       {"format": "v4-recovery-retrieval-query-v1", "id": qid,
                        "query_sha256": digest(query), "parents": parents,
                        "branch_counts": {name: len(rows) for name, rows in branches.items()},
                        "batch_seconds": time.monotonic() - started})
            print(json.dumps({"stage": "retrieve", "qid": qid,
                              "parents": len(parents)}), flush=True)


def score_many(model, query: str, payloads: list[str]) -> list[float]:
    result = []
    for start in range(0, len(payloads), 16):
        values = model.score(query, payloads[start:start + 16])
        if len(values) != len(payloads[start:start + 16]) or not all(map(math.isfinite, values)):
            raise ValueError("Reranker returned invalid logits")
        result.extend(map(float, values))
    return result


def windows(tokenizer, query: str, source, passage) -> list[dict]:
    if source.text[passage.source_start:passage.source_end] != passage.text:
        raise ValueError(f"Passage source offsets differ: {passage.passage_id}")
    encoded = tokenizer(passage.text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if not offsets:
        return []
    starts = list(range(0, max(1, len(offsets) - WINDOW + 1), STRIDE))
    starts.append(max(0, len(offsets) - WINDOW))
    context = {"official_id": source.official_id, "context_id": passage.passage_id,
               "text": passage.text, "start": passage.source_start}
    result = []
    for start in sorted(set(starts)):
        end = min(start + WINDOW, len(offsets))
        result.append(fit_window(tokenizer, query, source.title, source.text,
                                 context, "middle", start, end, offsets))
    return result


def baseline_windows(tokenizer, query: str, source, text: str) -> list[dict]:
    at = source.text.find(text)
    if at < 0:
        raise ValueError(f"Baseline context is not source grounded: {source.official_id}")
    context = SimpleNamespace(passage_id="baseline", source_start=at,
                              source_end=at + len(text), text=text)
    return windows(tokenizer, query, source, context)


def source_context(source, start: int, end: int) -> str:
    width = min(MAX_CONTEXT, len(source.text))
    center = (start + end) // 2
    left = max(0, min(center - width // 2, len(source.text) - width))
    result = source.text[left:left + width]
    if source.text[start:end] not in result:
        raise ValueError("Scored window is outside exported context")
    return result


def score_query(qid: int, query: str, baseline: dict, retrieval: dict,
                store: DiskCorpusStore, tokenizer, model) -> dict:
    baseline_docs = list(map(int, baseline["relevant_docs"]))
    baseline_set = set(baseline_docs)
    baseline_chunks = {int(row["doc_id"]): row["chunk_text"]
                       for row in baseline["relevant_chunks"]}
    parent_rows = retrieval["parents"]
    seeds = []
    for item in parent_rows:
        doc = int(item["doc_id"])
        pid = item["seed_passage_id"]
        passage = store.passage(pid) if pid else store.passages_for_parent(doc)[0]
        if passage.official_id != doc:
            raise ValueError("Seed passage parent differs")
        source = store.source(doc)
        text = passage.text
        payload = f"{source.title}\n{text}" if source.title else text
        seeds.append((doc, pid or passage.passage_id, payload))
    seed_scores = score_many(model, query, [row[2] for row in seeds])
    ranked = sorted(zip(seeds, seed_scores, strict=True),
                    key=lambda row: (-row[1], row[0][0]))[:200]
    selected_pool = sorted(set(baseline_docs) | {row[0][0] for row in ranked})
    evidence = {}
    stats = {"parents": len(parent_rows), "reranked_seeds": len(seeds),
             "evidence_pool_parents": len(selected_pool), "passages": 0, "windows": 0}
    for index, doc in enumerate(selected_pool, 1):
        source = store.source(doc)
        passages = store.passages_for_parent(doc)
        stats["passages"] += len(passages)
        options = []
        for passage in passages:
            options.extend(windows(tokenizer, query, source, passage))
        if doc in baseline_chunks:
            old = baseline_chunks[doc]
            options.extend(baseline_windows(tokenizer, query, source, old))
        if not options:
            raise ValueError(f"Parent has no evidence options: {qid}/{doc}")
        stats["windows"] += len(options)
        values = score_many(model, query, [row["payload"] for row in options])
        best = min(zip(options, values, strict=True),
                   key=lambda row: (-row[1], row[0]["context_id"] != "baseline",
                                    row[0]["start"], row[0]["window_id"]))
        chosen, score = best
        context = (baseline_chunks[doc] if chosen["context_id"] == "baseline" else
                   source_context(source, chosen["start"], chosen["end"]))
        if context not in source.text:
            raise ValueError("Export context is not source contiguous")
        evidence[doc] = {"score": float(score), "context": context,
                         "window_id": chosen["window_id"], "window_start": chosen["start"],
                         "window_end": chosen["end"], "source_sha256": source.text_sha256}
        if index % 50 == 0:
            print(json.dumps({"stage": "evidence", "qid": qid,
                              "parents_done": index, "pool": len(selected_pool),
                              "windows": stats["windows"]}), flush=True)
    slot_docs = set(baseline_chunks)
    outsiders = sorted(set(evidence) - slot_docs,
                      key=lambda doc: (-evidence[doc]["score"], doc))
    victims = sorted(slot_docs, key=lambda doc: (evidence[doc]["score"], -doc))
    swaps = []
    new_docs = 0
    for entrant in outsiders:
        if len(swaps) >= 10 or not victims:
            break
        if entrant not in baseline_set and (new_docs >= 3 or len(baseline_docs) + new_docs >= 200):
            continue
        victim = victims[0]
        if evidence[entrant]["score"] <= evidence[victim]["score"]:
            break
        victims.pop(0)
        swaps.append({"entrant": entrant, "victim": victim,
                      "entrant_score": evidence[entrant]["score"],
                      "victim_score": evidence[victim]["score"]})
        slot_docs.remove(victim)
        slot_docs.add(entrant)
        new_docs += entrant not in baseline_set
    docs = baseline_docs + [row["entrant"] for row in swaps if row["entrant"] not in baseline_set]
    chunks = [{"doc_id": doc, "chunk_text": evidence[doc]["context"]}
              for doc in docs if doc in slot_docs]
    if len(chunks) != 60 or len(set(slot_docs)) != 60:
        raise ValueError(f"Chunk allocation differs: {qid}")
    return {"format": "v4-recovery-scored-query-v1", "id": qid,
            "query_sha256": digest(query), "retrieval_sha256": None,
            "prediction": {"id": qid, "relevant_docs": docs, "relevant_chunks": chunks},
            "swaps": swaps, "stats": stats,
            "selected": {str(doc): evidence[doc] for doc in slot_docs}}


def score(output: Path, start: int, stop: int) -> None:
    torch = require_cuda()
    config, index, manifest, queries, baseline = inputs(output)
    runtime_record(output, "score", torch)
    store = DiskCorpusStore(index / "corpus.sqlite", manifest["corpus_version"],
                            manifest["fingerprint"])
    settings = config["reranker"]
    snapshot = require_cached_revision(settings["model"], settings["revision"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Source offsets need a fast tokenizer")
    model = BGEReranker(settings["model"], revision=settings["revision"],
                        max_length=settings["max_length"], device="cuda:0")
    for qid in range(start, stop + 1):
        target = output / "scored" / f"{qid}.json"
        source = output / "retrieval" / f"{qid}.json"
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            row = read(target)
            if row["query_sha256"] != digest(queries[qid]) or row["retrieval_sha256"] != sha256_file(source):
                raise ValueError(f"Score checkpoint identity differs: {qid}")
            continue
        retrieval = read(source)
        if retrieval["query_sha256"] != digest(queries[qid]):
            raise ValueError(f"Retrieval query identity differs: {qid}")
        started = time.monotonic()
        row = score_query(qid, queries[qid], baseline[qid], retrieval, store, tokenizer, model)
        row["retrieval_sha256"] = sha256_file(source)
        row["seconds"] = time.monotonic() - started
        write_once(target, row)
        print(json.dumps({"stage": "score", "qid": qid, "seconds": row["seconds"],
                          "swaps": len(row["swaps"])}), flush=True)


def export(output: Path) -> dict:
    config, index, manifest, queries, baseline = inputs(output)
    store = DiskCorpusStore(index / "corpus.sqlite", manifest["corpus_version"],
                            manifest["fingerprint"])
    policy = SelectionPolicy(**config["selection"])
    predictions, per_query = [], []
    selected_by_query = {}
    for qid in QUERY_IDS:
        path = output / "scored" / f"{qid}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        row = read(path)
        if row["query_sha256"] != digest(queries[qid]) or row["retrieval_sha256"] != sha256_file(output / "retrieval" / f"{qid}.json"):
            raise ValueError(f"Scored shard identity differs: {qid}")
        prediction = row["prediction"]
        if prediction["id"] != qid or not set(baseline[qid]["relevant_docs"]).issubset(prediction["relevant_docs"]):
            raise ValueError(f"Baseline document preservation differs: {qid}")
        if len(row["swaps"]) > 10 or len(prediction["relevant_chunks"]) != 60:
            raise ValueError(f"Slot contract differs: {qid}")
        for chunk in prediction["relevant_chunks"]:
            doc = int(chunk["doc_id"])
            if chunk["chunk_text"] not in store.source(doc).text:
                raise ValueError(f"Context is not in source: {qid}/{doc}")
        predictions.append(prediction)
        selected_by_query[qid] = row["selected"]
        per_query.append({"id": qid, "score_sha256": sha256_file(path),
                          "swaps": row["swaps"], "stats": row["stats"]})
    payload = canonical_payload(predictions)
    def zipped_size(data: bytes) -> int:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            archive.writestr("predictions.json", data)
        return stream.tell()
    truncation = "5500_character_source_context"
    if zipped_size(payload) > 100_000_000:
        truncation = "scored_window_source_span"
        for prediction in predictions:
            qid = prediction["id"]
            for chunk in prediction["relevant_chunks"]:
                doc = chunk["doc_id"]
                item = selected_by_query[qid][str(doc)]
                source = store.source(doc).text
                chunk["chunk_text"] = source[item["window_start"]:item["window_end"]]
        payload = canonical_payload(predictions)
    destination = output / "export"
    destination.mkdir(parents=True, exist_ok=True)
    payload_path = destination / "predictions.json"
    if payload_path.exists() and payload_path.read_bytes() != payload:
        raise ValueError("Existing predictions differ")
    if not payload_path.exists():
        payload_path.write_bytes(payload)
    zip_path = destination / "submission.zip"
    if not zip_path.exists():
        package_zip(zip_path, payload)
    report = validate_zip(zip_path, query_ids=set(QUERY_IDS),
                          official_ids={doc for row in predictions for doc in row["relevant_docs"]},
                          policy=policy, result_limit=200, max_bytes=100_000_000)
    if report["payload_sha256"] != sha256_file(payload_path):
        raise ValueError("ZIP payload differs from predictions")
    write_once(output / "per_query.json", per_query)
    result = {"format": "v4-recovery-export-v1", "status": "READY_FOR_BTC",
              "official_f2": "UNMEASURED", "zip": report,
              "context_rule": truncation,
              "per_query_sha256": sha256_file(output / "per_query.json"),
              "changed_queries": sum(bool(row["swaps"]) for row in per_query)}
    write_once(output / "run_manifest.json", result)
    from src.medrag.data.pair_lineage import write_atomic_replace
    write_atomic_replace(output / "status.json", {
        "format": "v4-end-to-end-evidence-recovery-status-v1",
        "status": "READY_FOR_BTC", "candidate_submission_created": True,
        "official_f2": "UNMEASURED", "zip_sha256": report["zip_sha256"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("retrieve", "score", "export"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.start <= args.stop <= 1200:
        parser.error("Require 1 <= start <= stop <= 1200")
    if not (args.output / "policy_lock.json").is_file():
        raise FileNotFoundError("Run prepare_end_to_end_evidence_recovery.py first")
    if args.stage == "retrieve":
        retrieve(args.output, args.start, args.stop)
    elif args.stage == "score":
        score(args.output, args.start, args.stop)
    else:
        print(json.dumps(export(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
