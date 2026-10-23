"""Frozen P2-A0 split runner: prepare CPU pairs, score externally, finalize CPU.

The pilot-replay mode uses the six pinned branch hits to exercise fusion and all
downstream stages without loading a GPU model. A full prepare uses the same
Pipeline.prepare_rerank method with the pinned dense encoder on CPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from scripts.v4.run_full_baseline import _all_official_ids, _write_bytes_checked
from src.medrag.adapters.validation import (canonical_payload, package_zip,
                                            validate_selected, validate_zip)
from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.core.pipeline import Pipeline
from src.medrag.core.contracts import Hit
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.selector import SelectionPolicy
from src.medrag.ranking.score_cache import ScoreCache

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
P2_BASELINE = ROOT / "outputs/v4/adjacent_context_replay_d200_20261021/selected_results.jsonl"
PILOT_IDS = (53, 130, 380, 754, 771, 1144)
FROZEN_CONFIG_SHA256 = "bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_checked(path, encoded(value))


def inputs(config_path: Path, mode: str) -> tuple[dict, dict, list[dict], Path, Path]:
    if config_path.resolve() != CONFIG.resolve():
        raise ValueError("P5.2 requires the frozen P2-A0 submission config")
    if sha256_file(config_path) != FROZEN_CONFIG_SHA256:
        raise ValueError("Frozen P2-A0 submission config hash differs")
    config = load_config(config_path)
    if (config["retrieval"]["rerank_allocation"] != "parent_first"
            or config["selection"]["document_selection"] != "independent"
            or config["selection"]["chunk_context"] != "adjacent"
            or config["retrieval"]["rerank_candidate_limit"] != 200
            or config["batch"]["result_limit"] != 200
            or config["batch"]["query_count"] != 1200):
        raise ValueError("Frozen P2-A0 ranking/selection contract differs")
    index = ROOT / config["index_namespace"]
    manifest_path = index / "manifest.json"
    manifest = read(manifest_path)
    if (manifest.get("scope") != "full_release" or manifest.get("parts") != 922
            or manifest.get("status") != "complete"
            or manifest.get("source_count") != 1_073_525
            or manifest.get("passage_count") != 2_951_224
            or not isinstance(manifest.get("dense"), dict)):
        raise ValueError("P5.2 requires the complete full index")
    if any(manifest.get("reranker", {}).get(name) != config["reranker"].get(name)
           for name in ("model", "revision", "max_length")):
        raise ValueError("Index/reranker contract differs from frozen config")
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    corpus_path = ROOT / migration["official_corpus"]
    if (sha256_file(query_path) != migration["official_queries_sha256"]
            or sha256_file(corpus_path) != migration["official_corpus_sha256"]):
        raise ValueError("Official query/corpus hash mismatch")
    all_queries = sorted(pq.read_table(query_path, columns=["id", "query"]).to_pylist(),
                         key=lambda row: int(row["id"]))
    if len(all_queries) != 1200 or len({int(row["id"]) for row in all_queries}) != 1200:
        raise ValueError("Official query coverage differs")
    queries = ([row for row in all_queries if int(row["id"]) in PILOT_IDS]
               if mode != "full" else all_queries)
    if mode != "full" and tuple(int(row["id"]) for row in queries) != PILOT_IDS:
        raise ValueError("Locked pilot query IDs differ")
    contract = {"format": "medrag-v4-a0-pair-score-v1",
                "index_manifest_sha256": sha256_file(manifest_path),
                "corpus_version": config["corpus_version"],
                "model": config["reranker"]["model"],
                "revision": config["reranker"]["revision"],
                "max_length": int(config["reranker"]["max_length"]),
                "precision": "fp16", "pair_format": "query_title_newline_passage-v1"}
    lock = {"format": "medrag-v4-p5-2-split-lock-v1", "mode": mode,
            "config_sha256": sha256_file(config_path),
            "index_manifest_sha256": contract["index_manifest_sha256"],
            "official_queries_sha256": migration["official_queries_sha256"],
            "official_corpus_sha256": migration["official_corpus_sha256"],
            "query_ids": [int(row["id"]) for row in queries], "score_contract": contract,
            "implementation_sha256": {name: sha256_file(ROOT / name) for name in (
                "scripts/v4/run_p5_2_split.py", "src/medrag/core/pipeline.py",
                "src/medrag/core/disk_pipeline.py", "src/medrag/evidence/selector.py",
                "src/medrag/evidence/context.py", "src/medrag/adapters/validation.py",
                "src/medrag/ranking/score_cache.py")}}
    if mode != "full":
        lock["pilot_checkpoint_sha256"] = {str(qid): sha256_file(PILOT / "queries" / f"{qid}.json")
                                            for qid in PILOT_IDS}
    return config, lock, queries, index, corpus_path


def ensure_lock(output: Path, lock: dict, *, create: bool) -> None:
    path = output / "lock.json"
    if path.exists():
        if read(path) != lock:
            raise ValueError("Split runner checkpoint lock mismatch")
    elif create and not output.exists():
        output.mkdir(parents=True)
        write_once(path, lock)
    else:
        raise FileNotFoundError(path)


def external_lock(output: Path, mode: str) -> dict:
    """Allow GPU workers to score exported shards without the corpus or index."""
    lock = read(output / "lock.json")
    contract = lock.get("score_contract", {})
    if (lock.get("format") != "medrag-v4-p5-2-split-lock-v1"
            or lock.get("mode") != mode
            or lock.get("config_sha256") != FROZEN_CONFIG_SHA256
            or contract.get("format") != "medrag-v4-a0-pair-score-v1"
            or contract.get("precision") != "fp16"
            or contract.get("pair_format") != "query_title_newline_passage-v1"
            or not isinstance(lock.get("query_ids"), list)
            or len(lock["query_ids"]) not in (6, 1200)
            or len(set(lock["query_ids"])) != len(lock["query_ids"])):
        raise ValueError("External scoring lock differs from P2-A0 contract")
    verify_pair_manifest(output, lock)
    return lock


class ReplaySearch:
    def __init__(self, hits: list[dict], fallbacks: list[dict] | None = None):
        self.hits = [Hit(**row) for row in hits]
        self.fallbacks = {int(row["official_id"]): [Hit(**hit) for hit in row["hits"]]
                          for row in (fallbacks or [])}

    def search(self, query: str, limit: int):
        return self.hits[:limit]

    def search_parent(self, query: str, parent_id: int, limit: int):
        if parent_id not in self.fallbacks:
            raise ValueError(f"Locked fallback missing for parent {parent_id}")
        return self.fallbacks[parent_id][:limit]


def finish_pipeline(config: dict, index: Path) -> Pipeline:
    manifest = read(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            manifest["fingerprint"])
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.sources = SourceLookup(store)
    pipeline.passages = PassageLookup(store)
    pipeline.reranker = object()  # enables frozen reranked selection; no model is loaded
    pipeline.selection_policy = SelectionPolicy(**config["selection"])
    retrieval = config["retrieval"]
    for name in ("document_candidate_limit", "passage_candidate_limit", "fused_candidate_limit",
                 "rrf_k", "rerank_candidate_limit", "rerank_passages_per_document"):
        setattr(pipeline, name, int(retrieval[name]))
    pipeline.rerank_allocation = retrieval["rerank_allocation"]
    return pipeline


def replay_pipeline(config: dict, index: Path, checkpoint: dict) -> Pipeline:
    pipeline = finish_pipeline(config, index)
    branches = checkpoint["branch_trace"]["branch_hits"]
    pipeline.doc_bm25 = ReplaySearch(branches["document_bm25"])
    pipeline.passage_bm25 = ReplaySearch(branches["passage_bm25"],
                                        checkpoint["branch_trace"]["fallback_hits"])
    pipeline.dense = ReplaySearch(branches["dense"])
    return pipeline


def prepare(output: Path, config: dict, lock: dict, queries: list[dict], index: Path) -> dict:
    ensure_lock(output, lock, create=True)
    mode = lock["mode"]
    pipeline = None
    if mode != "pilot":
        runtime = copy.deepcopy(config)
        runtime["dense"]["device"] = "cpu"  # runtime placement only; config file stays frozen
        pipeline = DiskPipeline(index, runtime, reranker=object())
    cache_keys = ScoreCache(output / "key_contract.sqlite", lock["score_contract"],
                            create=not (output / "key_contract.sqlite").exists())
    prepared = 0
    for row in queries:
        qid, query = int(row["id"]), str(row["query"])
        path = output / "pairs" / f"{qid}.json"
        if path.exists():
            shard = read(path)
            validate_pair_shard(shard, lock, qid, query, cache_keys)
            continue
        if mode == "pilot":
            checkpoint = read(PILOT / "queries" / f"{qid}.json")
            if checkpoint.get("status") != "complete" or checkpoint["query_sha256"] != digest(query):
                raise ValueError(f"Locked pilot checkpoint mismatch: {qid}")
            active = replay_pipeline(config, index, checkpoint)
        else:
            active = pipeline
        candidates, pairs, diagnostics = active.prepare_rerank(query, 200)
        if len(pairs) != 200:
            raise ValueError(f"Frozen pair budget differs for {qid}: {len(pairs)}")
        if mode != "full":
            if mode == "pilot-live":
                checkpoint = read(PILOT / "queries" / f"{qid}.json")
            if diagnostics["fused_items"] != len(checkpoint["branch_trace"]["fused"]):
                raise ValueError(f"Pilot fusion count differs: {qid}")
            expected = checkpoint["pair_trace"]
            if [(doc, pid, digest(payload)) for doc, pid, payload in pairs] != [
                    (int(item["official_id"]), item["passage_id"], item["input_sha256"])
                    for item in expected]:
                raise ValueError(f"Pilot pair identities/inputs differ: {qid}")
        pair_rows = [{"official_id": doc, "passage_id": pid, "payload": payload,
                      "input_sha256": digest(payload),
                      "cache_key": cache_keys._key(query, doc, pid, payload)}
                     for doc, pid, payload in pairs]
        shard = {"format": "medrag-v4-p5-2-pairs-v1", "id": qid,
                 "lock_sha256": sha256_file(output / "lock.json"), "query": query,
                 "query_sha256": digest(query), "candidate_count": len(candidates),
                 "candidates": candidates, "diagnostics": diagnostics, "pairs": pair_rows}
        write_once(path, shard)
        prepared += 1
        print(f"prepared {qid}: {len(pair_rows)} immutable pairs", flush=True)
    manifest = {"format": "medrag-v4-p5-2-pair-manifest-v1",
                "lock_sha256": sha256_file(output / "lock.json"),
                "pair_shards_sha256": {str(int(row["id"])): sha256_file(
                    output / "pairs" / f"{int(row['id'])}.json") for row in queries}}
    write_once(output / "pair_manifest.json", manifest)
    return {"prepared_now": prepared, "pair_shards": len(queries), "mode": mode}


def verify_pair_manifest(output: Path, lock: dict) -> None:
    manifest = read(output / "pair_manifest.json")
    if (manifest.get("format") != "medrag-v4-p5-2-pair-manifest-v1"
            or manifest.get("lock_sha256") != sha256_file(output / "lock.json")
            or set(manifest.get("pair_shards_sha256", {})) != {str(qid) for qid in lock["query_ids"]}):
        raise ValueError("Pair manifest identity/coverage mismatch")
    for qid, expected in manifest["pair_shards_sha256"].items():
        if sha256_file(output / "pairs" / f"{qid}.json") != expected:
            raise ValueError(f"Immutable pair shard changed: {qid}")


def validate_pair_shard(shard: dict, lock: dict, qid: int, query: str,
                        cache: ScoreCache) -> None:
    if (shard.get("format") != "medrag-v4-p5-2-pairs-v1" or shard.get("id") != qid
            or shard.get("query") != query or shard.get("query_sha256") != digest(query)
            or shard.get("lock_sha256") != digest(encoded(lock).decode("utf-8"))
            or len(shard.get("pairs", [])) != 200
            or shard.get("diagnostics", {}).get("rerank_pairs") != 200):
        raise ValueError(f"Pair shard identity/budget mismatch: {qid}")
    seen = set()
    candidates = {doc["official_id"]: doc for doc in shard["candidates"]}
    for pair in shard["pairs"]:
        doc, pid, payload = pair["official_id"], pair["passage_id"], pair["payload"]
        if (doc, pid) in seen or doc not in candidates or not any(
                item["passage_id"] == pid for item in candidates[doc]["evidence"]):
            raise ValueError(f"Duplicate or ungrounded pair: {qid}/{doc}/{pid}")
        seen.add((doc, pid))
        if (pair["input_sha256"] != digest(payload)
                or pair["cache_key"] != cache._key(query, doc, pid, payload)):
            raise ValueError(f"Pair input/hash/key mismatch: {qid}/{doc}/{pid}")


def cache_rows(cache: ScoreCache, shard: dict) -> dict[str, float]:
    found = {}
    with sqlite3.connect(f"file:{cache.path}?mode=ro", uri=True) as db:
        for pair in shard["pairs"]:
            row = db.execute("SELECT score,query_sha256,official_id,passage_id,input_sha256 "
                             "FROM scores WHERE cache_key=?", (pair["cache_key"],)).fetchone()
            if row is None:
                continue
            expected = (shard["query_sha256"], pair["official_id"], pair["passage_id"],
                        pair["input_sha256"])
            if row[1:] != expected or not math.isfinite(float(row[0])):
                raise ValueError(f"Score cache metadata mismatch: {shard['id']}/{pair['passage_id']}")
            found[pair["cache_key"]] = float(row[0])
    return found


def validate_score_shard(scores: dict, pair_shard: dict, pair_sha: str,
                         contract_fingerprint: str) -> list[float]:
    if (scores.get("format") != "medrag-v4-p5-2-scores-v1"
            or scores.get("id") != pair_shard["id"]
            or scores.get("pair_shard_sha256") != pair_sha
            or scores.get("contract_fingerprint") != contract_fingerprint
            or len(scores.get("scores", [])) != len(pair_shard["pairs"])):
        raise ValueError(f"Score shard identity/count mismatch: {pair_shard['id']}")
    values = []
    for got, pair in zip(scores["scores"], pair_shard["pairs"], strict=True):
        if (got.get("cache_key") != pair["cache_key"]
                or got.get("official_id") != pair["official_id"]
                or got.get("passage_id") != pair["passage_id"]
                or got.get("input_sha256") != pair["input_sha256"]):
            raise ValueError(f"Score pair identity/hash mismatch: {pair_shard['id']}")
        score = got.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid imported score: {pair_shard['id']}")
        values.append(float(score))
    return values


def score(output: Path, lock: dict, queries: list[dict], *, seed_cache: Path | None,
          device: str) -> dict:
    ensure_lock(output, lock, create=False)
    verify_pair_manifest(output, lock)
    cache_path = output / "scores.sqlite"
    if not cache_path.exists():
        if seed_cache is None:
            ScoreCache(cache_path, lock["score_contract"], create=True)
        else:
            seed = ScoreCache(seed_cache, lock["score_contract"])
            with sqlite3.connect(f"file:{seed.path}?mode=ro", uri=True) as source:
                with sqlite3.connect(cache_path) as target:
                    source.backup(target)
    cache = ScoreCache(cache_path, lock["score_contract"])
    model = None
    scored_now = 0
    for row in queries:
        qid, query = int(row["id"]), str(row["query"])
        pair_path = output / "pairs" / f"{qid}.json"
        shard = read(pair_path)
        validate_pair_shard(shard, lock, qid, query, cache)
        pair_sha = sha256_file(pair_path)
        target = output / "scores" / f"{qid}.json"
        if target.exists():
            validate_score_shard(read(target), shard, pair_sha, cache.fingerprint)
            continue
        cached = cache_rows(cache, shard)
        missing = [pair for pair in shard["pairs"] if pair["cache_key"] not in cached]
        if missing:
            if model is None:
                if device == "none":
                    raise ValueError(f"{len(missing)} pairs need external GPU scoring for query {qid}")
                from src.medrag.ranking.reranker import BGEReranker
                settings = lock["score_contract"]
                model = BGEReranker(settings["model"], revision=settings["revision"],
                                    max_length=int(settings["max_length"]), device=device)
            cache.score(query, [(p["official_id"], p["passage_id"], p["payload"])
                                for p in shard["pairs"]], model)
            cached = cache_rows(cache, shard)
        result = {"format": "medrag-v4-p5-2-scores-v1", "id": qid,
                  "pair_shard_sha256": pair_sha, "contract_fingerprint": cache.fingerprint,
                  "scores": [{"cache_key": p["cache_key"], "official_id": p["official_id"],
                              "passage_id": p["passage_id"], "input_sha256": p["input_sha256"],
                              "score": cached[p["cache_key"]]} for p in shard["pairs"]]}
        validate_score_shard(result, shard, pair_sha, cache.fingerprint)
        write_once(target, result)
        scored_now += 1
        print(f"scored {qid}: {len(missing)} model calls, {len(cached)-len(missing)} cache hits", flush=True)
    return {"score_shards": len(queries), "scored_now": scored_now}


def import_score(output: Path, lock: dict, qid: int, source: Path) -> dict:
    ensure_lock(output, lock, create=False)
    verify_pair_manifest(output, lock)
    cache = ScoreCache(output / "key_contract.sqlite", lock["score_contract"])
    pair_path = output / "pairs" / f"{qid}.json"
    pair = read(pair_path)
    validate_pair_shard(pair, lock, qid, pair["query"], cache)
    incoming = read(source)
    values = validate_score_shard(incoming, pair, sha256_file(pair_path), cache.fingerprint)
    score_cache_path = output / "scores.sqlite"
    if score_cache_path.exists():
        scored_cache = ScoreCache(score_cache_path, lock["score_contract"])
        known = cache_rows(scored_cache, pair)
        if any(p["cache_key"] in known and known[p["cache_key"]] != value
               for p, value in zip(pair["pairs"], values, strict=True)):
            raise ValueError(f"Imported scores conflict with pinned cache: {qid}")
    write_once(output / "scores" / f"{qid}.json", incoming)
    return {"imported_query_id": qid, "score_count": len(incoming["scores"])}


def finalize(output: Path, config: dict, lock: dict, queries: list[dict], index: Path,
             corpus_path: Path) -> dict:
    ensure_lock(output, lock, create=False)
    verify_pair_manifest(output, lock)
    pipeline = finish_pipeline(config, index)
    cache = ScoreCache(output / "key_contract.sqlite", lock["score_contract"])
    selected_by_id = {}
    for row in queries:
        qid, query = int(row["id"]), str(row["query"])
        pair_path = output / "pairs" / f"{qid}.json"
        pair = read(pair_path)
        validate_pair_shard(pair, lock, qid, query, cache)
        score_path = output / "scores" / f"{qid}.json"
        scores = read(score_path)
        pair_sha, score_sha = sha256_file(pair_path), sha256_file(score_path)
        values = validate_score_shard(scores, pair, pair_sha, cache.fingerprint)
        final_path = output / "final" / f"{qid}.json"
        if final_path.exists():
            prior = read(final_path)
            if (prior.get("format") != "medrag-v4-p5-2-final-query-v1"
                    or prior.get("id") != qid or prior.get("status") != "complete"
                    or prior.get("query_sha256") != digest(query)
                    or prior.get("pair_shard_sha256") != pair_sha
                    or prior.get("score_shard_sha256") != score_sha
                    or prior.get("selected_sha256") != digest(encoded(prior["selected"]).decode())):
                raise ValueError(f"Final query checkpoint mismatch: {qid}")
            selected_by_id[qid] = prior["selected"]
            continue
        triples = [(p["official_id"], p["passage_id"], p["payload"])
                   for p in pair["pairs"]]
        candidates = copy.deepcopy(pair["candidates"])
        selected, trace = pipeline.finish_rerank(candidates, triples, values, 200,
                                                  pair["diagnostics"], include_candidates=True)
        replay, _ = pipeline.finish_rerank(copy.deepcopy(pair["candidates"]), triples, values,
                                           200, pair["diagnostics"])
        if replay != selected:
            raise ValueError(f"Final selection replay mismatch: {qid}")
        checkpoint = {"format": "medrag-v4-p5-2-final-query-v1", "id": qid,
                      "query_sha256": digest(query), "pair_shard_sha256": pair_sha,
                      "score_shard_sha256": score_sha,
                      "selected_sha256": digest(encoded(selected).decode()),
                      "status": "complete", "selected": selected, "trace": trace}
        write_once(final_path, checkpoint)
        selected_by_id[qid] = selected
    ids = [int(row["id"]) for row in queries]
    if lock["mode"] != "full":
        baseline = {int(row["id"]): row["selected"] for row in map(
            json.loads, P2_BASELINE.open(encoding="utf-8"))}
        if any(selected_by_id[qid] != baseline[qid] for qid in ids):
            raise ValueError("Split output differs from locked P2-A0 pilot selected objects")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            read(index / "manifest.json")["fingerprint"])
    official_ids = _all_official_ids(corpus_path)
    predictions = validate_selected(ids, selected_by_id, store, pipeline.selection_policy,
                                    result_limit=200, reranker_enabled=True,
                                    official_ids=official_ids)
    payload = canonical_payload(predictions)
    if lock["mode"] != "full":
        expected = (ROOT / "outputs/v4/p5_1_a0_submission_preflight_20261022/btc_pilot_payload.json")
        if payload != expected.read_bytes().rstrip(b"\n"):
            raise ValueError("Split BTC payload differs from locked pilot")
    selected_payload = b"".join(encoded({"id": qid, "selected": selected_by_id[qid]})
                                for qid in ids)
    _write_bytes_checked(output / "selected_results.jsonl", selected_payload)
    export = output / "export"
    export.mkdir(exist_ok=True)
    _write_bytes_checked(export / "predictions.json", payload)
    zip_path = export / "submission.zip"
    if not zip_path.exists():
        package_zip(zip_path, payload)
    zip_report = validate_zip(zip_path, query_ids=set(ids), official_ids=official_ids,
                              policy=pipeline.selection_policy, result_limit=200,
                              max_bytes=int(config["batch"]["zip_max_bytes"]))
    if zip_report["payload_sha256"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("Export ZIP payload differs")
    report = {"format": "medrag-v4-p5-2-split-result-v1", "status": "PASS",
              "mode": lock["mode"], "query_count": len(ids), "pair_count": 200 * len(ids),
              "selected_documents": sum(map(len, selected_by_id.values())),
              "selected_evidence": sum(len(doc["evidence"]) for selected in selected_by_id.values()
                                       for doc in selected),
              "exact_pilot_object_parity": lock["mode"] != "full",
              "predictions_sha256": sha256_file(export / "predictions.json"),
              "submission_sha256": sha256_file(zip_path), "zip": zip_report}
    write_once(output / "run_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score", "import", "finalize"))
    parser.add_argument("--mode", choices=("pilot", "pilot-live", "full"), default="pilot")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-cache", type=Path)
    parser.add_argument("--device", default="none", help="GPU device for scoring, e.g. cuda:0")
    parser.add_argument("--query-id", type=int)
    parser.add_argument("--score-file", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in ("prepare", "finalize"):
        config, lock, queries, index, corpus = inputs(args.config.resolve(), args.mode)
    else:
        lock = external_lock(output, args.mode)
        queries = [{"id": qid, "query": read(output / "pairs" / f"{qid}.json")["query"]}
                   for qid in lock["query_ids"]]
    if args.stage == "prepare":
        result = prepare(output, config, lock, queries, index)
    elif args.stage == "score":
        result = score(output, lock, queries, seed_cache=args.seed_cache, device=args.device)
    elif args.stage == "import":
        if args.query_id is None or args.score_file is None or args.query_id not in lock["query_ids"]:
            parser.error("import requires a locked --query-id and --score-file")
        result = import_score(output, lock, args.query_id, args.score_file)
    else:
        result = finalize(output, config, lock, queries, index, corpus)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
