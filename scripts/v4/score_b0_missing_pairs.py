"""Resumably score only the locked B0 missing-pair shards on a separate GPU job."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path

from src.medrag.ranking.reranker import BGEReranker
from src.medrag.ranking.score_cache import ScoreCache


ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def exact_cached(cache: ScoreCache, shard: dict) -> dict[str, float]:
    found = {}
    with sqlite3.connect(f"file:{cache.path}?mode=ro", uri=True) as db:
        for pair in shard["pairs"]:
            row = db.execute(
                "SELECT score,query_sha256,official_id,passage_id,input_sha256 "
                "FROM scores WHERE cache_key=?", (pair["cache_key"],)).fetchone()
            if row is None:
                continue
            expected = (pair["query_sha256"], pair["official_id"], pair["passage_id"],
                        pair["input_sha256"])
            if row[1:] != expected or not math.isfinite(float(row[0])):
                raise ValueError(f"Cached score metadata mismatch: {shard['query_id']}/"
                                 f"{pair['passage_id']}")
            found[pair["cache_key"]] = float(row[0])
    return found


def validate_shard(path: Path, expected_sha: str, fingerprint: str,
                   cache: ScoreCache) -> dict:
    if sha(path) != expected_sha:
        raise ValueError(f"Missing-pair shard hash mismatch: {path}")
    shard = read(path)
    if (shard.get("format") != "medrag-v4-b0-missing-pairs-v1"
            or shard.get("score_contract_fingerprint") != fingerprint
            or shard.get("missing_pair_count") != len(shard.get("pairs", []))):
        raise ValueError(f"Missing-pair shard contract mismatch: {path}")
    seen = set()
    for pair in shard["pairs"]:
        identity = (int(pair["official_id"]), pair["passage_id"])
        if identity in seen:
            raise ValueError(f"Duplicate B0 pair: {shard['query_id']}/{identity}")
        seen.add(identity)
        if (pair["query_id"] != shard["query_id"]
                or pair["query_sha256"] != shard["query_sha256"]
                or hashlib.sha256(pair["query"].encode()).hexdigest() != shard["query_sha256"]
                or hashlib.sha256(pair["payload"].encode()).hexdigest() != pair["input_sha256"]
                or cache._key(pair["query"], identity[0], identity[1], pair["payload"])
                != pair["cache_key"]):
            raise ValueError(f"B0 pair identity/input mismatch: {shard['query_id']}/{identity}")
    return shard


def main(readiness_path: Path, output: Path, device: str) -> None:
    readiness_path, output = readiness_path.resolve(), output.resolve()
    readiness = read(readiness_path)
    if (readiness.get("format") != "medrag-v4-b0-rerank-readiness-v1"
            or readiness.get("status") != "BLOCKED_MISSING_SCORES"
            or readiness["counts"]["missing_scores"] <= 0):
        raise ValueError("B0 readiness manifest is not a locked missing-score job")
    contract = readiness["score_contract"]
    fingerprint = readiness["score_contract_fingerprint"]
    seed = ROOT / readiness["cache_inventory"]["seed_cache_for_gpu_job"]
    if sha(seed) != readiness["cache_inventory"]["seed_cache_sha256"]:
        raise ValueError("Seed cache hash mismatch")
    lock = {
        "format": "medrag-v4-b0-missing-score-lock-v1",
        "readiness_sha256": sha(readiness_path),
        "score_contract": contract,
        "score_contract_fingerprint": fingerprint,
        "input_shards_sha256": readiness["missing_score_manifest"]["shards_sha256"],
        "seed_cache_sha256": sha(seed),
        "device": device,
    }
    lock_path = output / "lock.json"
    if output.exists():
        if not lock_path.exists() or read(lock_path) != lock:
            raise ValueError("Existing B0 GPU output lock differs")
    else:
        output.mkdir(parents=True)
        write_once(lock_path, lock)
    cache_path = output / "scores.sqlite"
    if not cache_path.exists():
        with sqlite3.connect(f"file:{seed}?mode=ro", uri=True) as source:
            with sqlite3.connect(cache_path) as target:
                source.backup(target)
    cache = ScoreCache(cache_path, contract)
    if cache.fingerprint != fingerprint:
        raise ValueError("Output score-cache contract differs")
    score_dir = output / "score_shards"
    score_dir.mkdir(exist_ok=True)
    completed_path = output / "run_manifest.json"
    if completed_path.exists():
        completed = read(completed_path)
        expected_shards = completed.get("score_shards_sha256", {})
        if (completed.get("status") != "COMPLETE"
                or completed.get("score_contract_fingerprint") != fingerprint
                or completed.get("expected_missing_pairs") != readiness["counts"]["missing_scores"]
                or completed.get("score_cache_sha256") != sha(cache_path)
                or set(expected_shards) != set(readiness["missing_score_manifest"]["shards_sha256"])
                or any(not (score_dir / f"{qid}.json").exists()
                       or sha(score_dir / f"{qid}.json") != value
                       for qid, value in expected_shards.items())):
            raise ValueError("Completed B0 score output failed resume validation")
        print(json.dumps(completed, ensure_ascii=False, indent=2))
        return
    input_dir = ROOT / readiness["missing_score_manifest"]["directory"]
    model = None
    scored_now = 0
    cache_hits = 0
    for qid_text, expected_sha in readiness["missing_score_manifest"]["shards_sha256"].items():
        qid = int(qid_text)
        path = input_dir / f"{qid}.json"
        shard = validate_shard(path, expected_sha, fingerprint, cache)
        target = score_dir / f"{qid}.json"
        if target.exists():
            prior = read(target)
            if (prior.get("format") != "medrag-v4-b0-score-shard-v1"
                    or prior.get("input_shard_sha256") != expected_sha
                    or prior.get("score_contract_fingerprint") != fingerprint
                    or len(prior.get("scores", [])) != len(shard["pairs"])):
                raise ValueError(f"Existing B0 score shard differs: {qid}")
            continue
        cached = exact_cached(cache, shard)
        missing = [pair for pair in shard["pairs"] if pair["cache_key"] not in cached]
        cache_hits += len(shard["pairs"]) - len(missing)
        if missing:
            if model is None:
                model = BGEReranker(contract["model"], revision=contract["revision"],
                                    max_length=int(contract["max_length"]), device=device)
            query = missing[0]["query"]
            if any(pair["query"] != query for pair in missing):
                raise ValueError(f"Multiple queries in one shard: {qid}")
            cache.score(query, [(pair["official_id"], pair["passage_id"], pair["payload"])
                                for pair in missing], model)
            scored_now += len(missing)
            cached = exact_cached(cache, shard)
        if len(cached) != len(shard["pairs"]):
            raise ValueError(f"B0 score shard remains incomplete: {qid}")
        result = {
            "format": "medrag-v4-b0-score-shard-v1", "query_id": qid,
            "input_shard_sha256": expected_sha,
            "score_contract_fingerprint": fingerprint,
            "model_calls": len(missing),
            "preexisting_cache_hits": len(shard["pairs"]) - len(missing),
            "scores": [{"cache_key": pair["cache_key"], "official_id": pair["official_id"],
                        "passage_id": pair["passage_id"], "input_sha256": pair["input_sha256"],
                        "score": cached[pair["cache_key"]]} for pair in shard["pairs"]],
        }
        write_once(target, result)
        print(f"scored B0 query {qid}: {len(missing)} model calls", flush=True)
    shard_rows = {qid: read(score_dir / f"{qid}.json")
                  for qid in readiness["missing_score_manifest"]["shards_sha256"]}
    report = {
        "format": "medrag-v4-b0-missing-score-result-v1", "status": "COMPLETE",
        "score_contract_fingerprint": fingerprint, "device": device,
        "expected_missing_pairs": readiness["counts"]["missing_scores"],
        "model_calls_total": sum(row["model_calls"] for row in shard_rows.values()),
        "model_calls_this_invocation": scored_now,
        "preexisting_cache_hits_total": sum(row["preexisting_cache_hits"]
                                             for row in shard_rows.values()),
        "preexisting_cache_hits_this_invocation": cache_hits,
        "score_cache_sha256": sha(cache_path),
        "score_shards_sha256": {qid: sha(score_dir / f"{qid}.json")
                                 for qid in readiness["missing_score_manifest"]["shards_sha256"]},
    }
    write_once(completed_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
                        required=True)
    args = parser.parse_args()
    main(args.readiness, args.output, args.device)
