"""Validate and seal all A0_T4_V1 full score checkpoints before finalization."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import (external_lock, read, validate_pair_shard,
                                      validate_score_shard)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.ranking.score_cache import ScoreCache


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def main(output: Path) -> dict:
    output = output.resolve()
    lock = external_lock(output, "full")
    runtime_path = output / "provenance/t4_runtime.json"
    runtime = read(runtime_path)
    if runtime.get("status") != "READY_TO_SCORE" or runtime.get("candidate") != "A0_T4_V1":
        raise ValueError("Pinned T4 runtime provenance missing")
    cache = ScoreCache(output / "key_contract.sqlite", lock["score_contract"])
    score_hashes, score_count = {}, 0
    for qid in lock["query_ids"]:
        pair_path = output / "pairs" / f"{qid}.json"
        score_path = output / "scores" / f"{qid}.json"
        pair = read(pair_path)
        validate_pair_shard(pair, lock, qid, pair["query"], cache)
        values = validate_score_shard(read(score_path), pair, sha256_file(pair_path),
                                      cache.fingerprint)
        if len(values) != 200 or any(not math.isfinite(value) for value in values):
            raise ValueError(f"Score coverage/finite check failed: {qid}")
        score_count += len(values)
        score_hashes[str(qid)] = sha256_file(score_path)
    if len(score_hashes) != 1200 or score_count != 240000:
        raise ValueError("Full A0_T4_V1 score coverage differs")
    stdout = output / "provenance/scorer_stdout.log"
    if not stdout.is_file() or not stdout.read_text(encoding="utf-8").strip():
        raise ValueError("Scorer stdout provenance missing")
    report = {
        "format": "medrag-v4-a0-t4-v1-score-manifest-v1",
        "status": "SCORES_COMPLETE",
        "candidate": "A0_T4_V1",
        "query_count": 1200, "score_count": score_count,
        "lock_sha256": sha256_file(output / "lock.json"),
        "pair_manifest_sha256": sha256_file(output / "pair_manifest.json"),
        "runtime_provenance_sha256": sha256_file(runtime_path),
        "scorer_stdout_sha256": sha256_file(stdout),
        "score_contract_fingerprint": cache.fingerprint,
        "score_shards_sha256": score_hashes,
    }
    path = output / "provenance/score_manifest.json"
    _write_bytes_checked(path, canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.output), ensure_ascii=False, indent=2))
