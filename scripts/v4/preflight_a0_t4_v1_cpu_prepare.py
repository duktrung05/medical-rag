"""CPU-only gate for immutable full A0_T4_V1 query-passage pair preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import (CONFIG, FROZEN_CONFIG_SHA256, PILOT_IDS, ROOT,
                                      inputs, read, validate_pair_shard)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.ranking.score_cache import ScoreCache

PILOT_SOURCE = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022"
DEFAULT_OUTPUT = ROOT / "outputs/v4/a0_t4_v1_release_candidate_20261022"


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def preflight(output: Path) -> dict:
    output = output.resolve()
    checks: dict[str, bool] = {}
    try:
        if sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
            raise ValueError("Frozen P2-A0 config hash mismatch")
        checks["frozen_config_hash"] = True
        config, lock, queries, index, corpus = inputs(CONFIG, "full")
        checks["full_index_and_corpus_manifests"] = True
        checks["official_query_and_corpus_hashes"] = True
        checks["full_query_coverage_1200"] = len(queries) == 1200
        if len(queries) != 1200:
            raise ValueError("Full query coverage differs from 1,200")
        checks["p2_a0_score_contract"] = (
            lock["score_contract"]["model"] == config["reranker"]["model"]
            and lock["score_contract"]["revision"] == config["reranker"]["revision"]
            and lock["score_contract"]["precision"] == "fp16"
            and lock["score_contract"]["max_length"] == 512)
        if not checks["p2_a0_score_contract"]:
            raise ValueError("Frozen score contract differs")
        checks["cpu_pair_preparation_supported"] = True  # P5.2 explicitly places dense on CPU.
        dense_snapshot = require_cached_revision(config["dense"]["model"],
                                                 config["dense"]["revision"])
        if dense_snapshot.name != config["dense"]["revision"]:
            raise ValueError("Exact dense encoder revision is not cached")
        checks["exact_dense_encoder_snapshot_cached"] = True

        pilot_lock = read(PILOT_SOURCE / "lock.json")
        if (pilot_lock.get("mode") != "pilot-live"
                or pilot_lock.get("query_ids") != list(PILOT_IDS)):
            raise ValueError("Locked P2-A0 pilot pair source differs")
        pilot_cache = ScoreCache(PILOT_SOURCE / "key_contract.sqlite",
                                 pilot_lock["score_contract"])
        pilot_manifest = read(PILOT_SOURCE / "pair_manifest.json")
        for qid in PILOT_IDS:
            pair_path = PILOT_SOURCE / "pairs" / f"{qid}.json"
            pair = read(pair_path)
            validate_pair_shard(pair, pilot_lock, qid, pair["query"], pilot_cache)
            if pilot_manifest["pair_shards_sha256"][str(qid)] != sha256_file(pair_path):
                raise ValueError(f"Pilot pair shard hash mismatch: {qid}")
        checks["locked_pilot_pair_shards"] = True
        p2b = read(ROOT / "outputs/v4/p5_3b_cpu_gpu_parity_20261022/p5_3b_parity_report.json")
        checks["pilot_semantic_parity"] = p2b["semantic_parity"]["status"] == "GO"
        if not checks["pilot_semantic_parity"]:
            raise ValueError("P5.3B pilot semantic parity is not GO")
        checks["gpu_required_for_preparation"] = False
        checks["no_scores_or_cuda_in_preflight"] = True
        report = {
            "format": "medrag-v4-a0-t4-v1-cpu-prepare-preflight-v1",
            "candidate": "A0_T4_V1", "status": "GO",
            "full_inference_started": False, "pair_preparation_started": False,
            "query_count": len(queries), "expected_pairs": 200 * len(queries),
            "config_sha256": sha256_file(CONFIG),
            "index_manifest_sha256": sha256_file(index / "manifest.json"),
            "dense_model": config["dense"]["model"],
            "dense_model_revision": config["dense"]["revision"],
            "dense_snapshot_path": str(dense_snapshot),
            "official_queries_sha256": lock["official_queries_sha256"],
            "official_corpus_sha256": lock["official_corpus_sha256"],
            "pilot_pair_manifest_sha256": sha256_file(PILOT_SOURCE / "pair_manifest.json"),
            "checks": checks,
            "command": "python -m scripts.v4.run_p5_2_split prepare --mode full --config configs/v4_submission_p2_a0.yaml --output outputs/v4/a0_t4_v1_full_20261022",
            "decision_basis": "CPU preflight excludes T4 availability and mixed-hardware exact parity; those are separately reported gates.",
        }
    except Exception as exc:
        report = {"format": "medrag-v4-a0-t4-v1-cpu-prepare-preflight-v1",
                  "candidate": "A0_T4_V1", "status": "NO-GO",
                  "full_inference_started": False, "pair_preparation_started": False,
                  "checks": checks, "reason": f"{type(exc).__name__}: {exc}"}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cpu_prepare_preflight_report.json"
    payload = canonical(report)
    if path.exists() and path.read_bytes() != payload:
        # A replay may update only a failed preflight into GO; preserve both decisions.
        history = output / "cpu_prepare_preflight_history"
        history.mkdir(exist_ok=True)
        _write_bytes_checked(history / f"preflight_{sha256_file(path)[:12]}.json", path.read_bytes())
        temporary = path.with_name(f".{path.name}.incomplete")
        temporary.write_bytes(payload)
        temporary.replace(path)
    else:
        _write_bytes_checked(path, payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = preflight(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
