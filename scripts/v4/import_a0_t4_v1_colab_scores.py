"""Import full Colab T4 scores after complete provenance and pair validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import (ROOT, external_lock, import_score, read,
                                      validate_pair_shard, validate_score_shard)
from scripts.v4.verify_p5_3a_colab_canary import sha, strict_json
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.ranking.score_cache import ScoreCache

SOURCE = ROOT / "outputs/v4/a0_t4_v1_full_20261022"
HANDOFF = ROOT / "outputs/v4/a0_t4_v1_colab_bundle_20261022/handoff.json"
DEFAULT_OUTPUT = ROOT / "outputs/v4/a0_t4_v1_release_candidate_20261022"
EXPECTED_VERSIONS = {"torch_version": "2.6.0+cu124", "torch_cuda_version": "12.4",
                     "sentence_transformers_version": "4.1.0",
                     "transformers_version": "4.51.3"}


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def expected_entries(query_ids: list[int]) -> set[str]:
    return ({f"scores/{qid}.json" for qid in query_ids}
            | {f"provenance/{qid}.json" for qid in query_ids}
            | {"score_manifest.json", "gpu_canary_report.json", "scorer_stdout.log"})


def import_scores(scores_zip: Path, source: Path, handoff_path: Path,
                  output: Path) -> dict:
    scores_zip, source, handoff_path, output = map(
        Path.resolve, (scores_zip, source, handoff_path, output))
    handoff = read(handoff_path)
    lock = external_lock(source, "full")
    query_ids = lock["query_ids"]
    expected_pairs = 200 * len(query_ids)
    if (handoff.get("format") != "medrag-v4-a0-t4-v1-colab-handoff-v1"
            or handoff.get("query_count") != 1200
            or handoff.get("pair_count") != expected_pairs
            or handoff.get("model_revision") != lock["score_contract"]["revision"]):
        raise ValueError("Full Colab handoff identity mismatch")
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    with zipfile.ZipFile(scores_zip) as bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if (set(names) != expected_entries(query_ids) or len(names) != len(set(names))
                or sum(item.file_size for item in infos) > 8_000_000_000
                or bundle.testzip() is not None):
            raise ValueError("Full T4 result ZIP coverage/CRC mismatch")
        report_raw = bundle.read("gpu_canary_report.json")
        manifest_raw = bundle.read("score_manifest.json")
        stdout = bundle.read("scorer_stdout.log")
        report, manifest = strict_json(report_raw), strict_json(manifest_raw)
        if (report.get("status") != "FRESH_GPU_SCORE_COMPLETE"
                or report.get("format") != "medrag-v4-p5-3a-colab-gpu-report-v2"
                or report.get("input_bundle_sha256") != handoff["archive_sha256"]
                or report.get("score_manifest_sha256") != sha(manifest_raw)
                or report.get("pair_manifest_sha256") != sha256_file(source / "pair_manifest.json")
                or report.get("lock_sha256") != sha256_file(source / "lock.json")
                or report.get("score_contract_fingerprint") != cache.fingerprint
                or report.get("model") != lock["score_contract"]["model"]
                or report.get("model_revision") != lock["score_contract"]["revision"]
                or report.get("max_length") != lock["score_contract"]["max_length"]
                or report.get("precision") != lock["score_contract"]["precision"]
                or "T4" not in str(report.get("gpu_name", ""))
                or any(report.get(key) != value for key, value in EXPECTED_VERSIONS.items())
                or report.get("query_ids") not in (None, query_ids)
                or report.get("query_count") != len(query_ids)
                or report.get("pair_count") != expected_pairs
                or report.get("fresh_model_pairs") != expected_pairs
                or report.get("cache_hits") != 0
                or report.get("cache_entries_before_this_session") != 0):
            raise ValueError("Fresh full T4 report identity/runtime/accounting mismatch")
        if (manifest.get("format") != "medrag-v4-p5-3a-score-manifest-v1"
                or manifest.get("input_bundle_sha256") != handoff["archive_sha256"]
                or manifest.get("model_revision") != lock["score_contract"]["revision"]
                or manifest.get("query_ids") != query_ids
                or manifest.get("pair_count") != expected_pairs
                or report.get("score_manifest_sha256") != sha(manifest_raw)):
            raise ValueError("Full score manifest identity mismatch")
        resumed = report.get("resumed_query_ids")
        if (not isinstance(resumed, list) or len(resumed) != len(set(resumed))
                or any(qid not in query_ids for qid in resumed)
                or report.get("cache_entries_after_this_session") != 200 * (len(query_ids) - len(resumed))):
            raise ValueError("Fresh inference resume/cache accounting mismatch")
        snapshot = report.get("model_snapshot_files_sha256")
        if (not isinstance(snapshot, dict)
                or not any(name.endswith((".safetensors", ".bin")) for name in snapshot)
                or any(not re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in snapshot.values())):
            raise ValueError("Pinned T4 model snapshot hashes missing")
        if (manifest.get("score_shards_sha256") != report.get("score_shards_sha256")
                or manifest.get("query_proofs_sha256") != report.get("query_proofs_sha256")):
            raise ValueError("Full score/proof hash maps differ")

        proof_sessions = set()
        score_hashes = {}
        proof_hashes = {}
        score_count = 0
        for qid in query_ids:
            score_raw = bundle.read(f"scores/{qid}.json")
            proof_raw = bundle.read(f"provenance/{qid}.json")
            pair_path = source / "pairs" / f"{qid}.json"
            pair = read(pair_path)
            validate_pair_shard(pair, lock, qid, pair["query"], cache)
            values = validate_score_shard(strict_json(score_raw), pair,
                                          sha256_file(pair_path), cache.fingerprint)
            if len(values) != 200 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"Score coverage/non-finite failure: query={qid}")
            proof = strict_json(proof_raw)
            if (proof.get("format") != "medrag-v4-p5-3a-query-proof-v1"
                    or proof.get("query_id") != qid
                    or proof.get("input_bundle_sha256") != handoff["archive_sha256"]
                    or proof.get("model_revision") != lock["score_contract"]["revision"]
                    or proof.get("gpu_name") != report["gpu_name"]
                    or proof.get("pair_shard_sha256") != sha256_file(pair_path)
                    or proof.get("score_shard_sha256") != sha(score_raw)
                    or proof.get("fresh_model_pairs") != 200
                    or proof.get("cache_hits") != 0
                    or proof.get("scorer_line") != f"scored {qid}: 200 model calls, 0 cache hits"
                    or not isinstance(proof.get("session_id"), str)
                    or not proof.get("session_id")):
                raise ValueError(f"Query GPU provenance mismatch: {qid}")
            seconds = proof.get("seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError(f"Query timing invalid: {qid}")
            for key in ("peak_vram_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib"):
                value = proof.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Query VRAM provenance invalid: {qid}/{key}")
            score_hashes[str(qid)] = sha(score_raw)
            proof_hashes[str(qid)] = sha(proof_raw)
            proof_sessions.add(proof["session_id"])
            score_count += len(values)
        if (score_count != expected_pairs
                or score_hashes != report.get("score_shards_sha256")
                or proof_hashes != report.get("query_proofs_sha256")
                or set(map(str, resumed)) - set(map(str, query_ids))):
            raise ValueError("Full score/provenance count or hashes mismatch")
        expected_stdout = "".join(
            strict_json(bundle.read(f"provenance/{qid}.json"))["scorer_line"] + "\n"
            for qid in query_ids)
        if (stdout.decode("utf-8") != expected_stdout
                or report.get("scorer_stdout_sha256") != sha(stdout)
                or report.get("session_count") != len(proof_sessions)
                or report.get("wall_seconds") != sum(
                    strict_json(bundle.read(f"provenance/{qid}.json"))["seconds"]
                    for qid in query_ids)
                or not math.isclose(report.get("pairs_per_second", 0),
                                    expected_pairs / report.get("wall_seconds", 1), rel_tol=1e-9)):
            raise ValueError("Full T4 transcript/benchmark arithmetic mismatch")

        output.mkdir(parents=True, exist_ok=True)
        provenance_dir = output / "colab_provenance"
        provenance_dir.mkdir(exist_ok=True)
        # Scores are imported only after the complete archive has passed all checks.
        for qid in query_ids:
            score_path = source / "pairs" / f"{qid}.json"
            score_temp = output / ".colab_import_score.json"
            score_temp.write_bytes(bundle.read(f"scores/{qid}.json"))
            import_score(source, lock, qid, score_temp)
            score_temp.unlink()
            proof_raw = bundle.read(f"provenance/{qid}.json")
            proof_target = provenance_dir / f"{qid}.json"
            _write_bytes_checked(proof_target, proof_raw)
        _write_bytes_checked(provenance_dir / "gpu_canary_report.json", report_raw)
        _write_bytes_checked(provenance_dir / "score_manifest.json", manifest_raw)
        _write_bytes_checked(provenance_dir / "scorer_stdout.log", stdout)
        imported_hashes = {str(qid): sha256_file(source / "scores" / f"{qid}.json")
                           for qid in query_ids}
        if imported_hashes != score_hashes:
            raise ValueError("Imported local score shards differ from verified archive")
        import_report = {
            "format": "medrag-v4-a0-t4-v1-colab-score-import-v1",
            "candidate": "A0_T4_V1", "status": "GO",
            "source_scores_zip_sha256": sha256_file(scores_zip),
            "input_bundle_sha256": handoff["archive_sha256"],
            "pair_manifest_sha256": sha256_file(source / "pair_manifest.json"),
            "score_contract_fingerprint": cache.fingerprint,
            "model_revision": lock["score_contract"]["revision"],
            "query_count": len(query_ids), "score_count": score_count,
            "fresh_model_pairs": report["fresh_model_pairs"],
            "cache_hits": report["cache_hits"],
            "proof_session_count": len(proof_sessions),
            "score_shards_sha256": imported_hashes,
            "score_provenance_dir": "colab_provenance",
            "benchmark": {key: report[key] for key in (
                "gpu_name", "wall_seconds", "pairs_per_second", "peak_vram_mib",
                "torch_peak_allocated_mib", "torch_peak_reserved_mib")},
        }
    report_path = output / "colab_score_import_report.json"
    _write_bytes_checked(report_path, canonical(import_report))
    return import_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-zip", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--handoff", type=Path,
                        default=ROOT / "outputs/v4/a0_t4_v1_colab_bundle_20261022/handoff.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = import_scores(args.scores_zip, args.source, args.handoff, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
