"""Package full immutable pair shards for external pinned Colab T4 scoring."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from pathlib import Path

from scripts.v4.prepare_p5_3a_colab_canary import (DEPENDENCIES, SCORER_MODULES,
                                                   SECRET_PATTERNS, archive_entry)
from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import ROOT, ScoreCache, external_lock, read, validate_pair_shard
from src.medrag.data.phase2_batch import sha256_file

DEFAULT_SOURCE = ROOT / "outputs/v4/a0_t4_v1_full_20261022"
DEFAULT_PREFLIGHT = ROOT / "outputs/v4/a0_t4_v1_release_candidate_20261022/cpu_prepare_preflight_report.json"
DEFAULT_OUTPUT = ROOT / "outputs/v4/a0_t4_v1_colab_bundle_20261022"
INPUT_DIR = Path("outputs/v4/a0_t4_v1_colab_input")
MAX_INPUT_BYTES = 8_000_000_000


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def main(source: Path, preflight_path: Path, output: Path) -> dict:
    source, preflight_path, output = map(Path.resolve, (source, preflight_path, output))
    preflight = read(preflight_path)
    if (preflight.get("format") != "medrag-v4-a0-t4-v1-cpu-prepare-preflight-v1"
            or preflight.get("status") != "GO"
            or preflight.get("query_count") != 1200
            or preflight.get("expected_pairs") != 240000):
        raise ValueError("CPU pair preparation preflight is not GO")
    lock = external_lock(source, "full")
    query_ids = lock["query_ids"]
    if len(query_ids) != 1200:
        raise ValueError("Full A0_T4_V1 lock does not contain 1,200 query IDs")
    manifest = read(source / "pair_manifest.json")
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    files: dict[str, bytes] = {}
    for name in ("lock.json", "pair_manifest.json", "key_contract.sqlite"):
        files[str(INPUT_DIR / name)] = (source / name).read_bytes()
    pair_hashes = {}
    pair_count = 0
    for qid in query_ids:
        path = source / "pairs" / f"{qid}.json"
        digest = sha256_file(path)
        if manifest["pair_shards_sha256"].get(str(qid)) != digest:
            raise ValueError(f"Immutable pair shard hash mismatch: {qid}")
        pair = read(path)
        validate_pair_shard(pair, lock, qid, pair["query"], cache)
        pair_hashes[str(qid)] = digest
        pair_count += len(pair["pairs"])
        files[str(INPUT_DIR / "pairs" / f"{qid}.json")] = path.read_bytes()
    if pair_count != 240000:
        raise ValueError(f"Full pair count differs: {pair_count}")
    with sqlite3.connect(source / "key_contract.sqlite") as db:
        if db.execute("SELECT count(*) FROM scores").fetchone()[0] != 0:
            raise ValueError("Input key contract contains score cache entries")
    for name in lock["implementation_sha256"]:
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != lock["implementation_sha256"][name]:
            raise ValueError(f"Frozen P2-A0 implementation hash mismatch: {name}")
    code_files = [ROOT / name for name in SCORER_MODULES]
    code_files += [ROOT / "scripts/v4/run_full_baseline.py",
                   ROOT / "scripts/v4/run_p5_2_split.py",
                   ROOT / "scripts/v4/p5_3a_colab_orchestrator.py",
                   ROOT / "configs/v4_submission_p2_a0.yaml"]
    for path in code_files:
        files[str(path.relative_to(ROOT))] = path.read_bytes()
    files["p5_3a_requirements.txt"] = DEPENDENCIES.encode("ascii")
    if any("scores" in Path(name).parts or name.endswith("scores.sqlite") for name in files):
        raise ValueError("Colab input bundle must not contain score shards or score caches")
    forbidden = {"corpus.sqlite", "documents.sqlite", "passages.sqlite", "vectors.npy"}
    if any(Path(name).name in forbidden or "dense" in Path(name).parts for name in files):
        raise ValueError("Colab bundle contains corpus or retrieval index data")
    total_bytes = sum(len(payload) for payload in files.values())
    if total_bytes > MAX_INPUT_BYTES:
        raise ValueError(f"Full Colab input bundle exceeds configured size: {total_bytes}")
    for name, payload in files.items():
        if any(re.search(pattern, payload) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Secret-pattern scan failed: {name}")

    contract = lock["score_contract"]
    input_manifest = {
        "format": "medrag-v4-p5-full-colab-input-v1",
        "source_mode": "full", "query_ids": query_ids,
        "query_count": len(query_ids), "pair_count": pair_count,
        "input_dir": str(INPUT_DIR),
        "model": contract["model"], "model_revision": contract["revision"],
        "max_length": contract["max_length"], "precision": contract["precision"],
        "pair_format": contract["pair_format"],
        "tokenizer_contract": {"implementation": "BGEReranker in src/medrag/ranking/reranker.py",
                               "class": "AutoTokenizer.from_pretrained",
                               "revision": contract["revision"], "max_length": 512,
                               "predict_truncation": "CrossEncoder max_length=512",
                               "diagnostic_truncation": False},
        "torch_cuda_wheel": "torch==2.6.0+cu124",
        "score_contract_fingerprint": cache.fingerprint,
        "pair_shards_sha256": pair_hashes,
        "cpu_preflight_sha256": sha256_file(preflight_path),
        "files_sha256": {name: hashlib.sha256(payload).hexdigest()
                         for name, payload in sorted(files.items())},
    }
    files["p5_3a_input_manifest.json"] = canonical(input_manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=True) as bundle:
        for name, payload in sorted(files.items()):
            archive_entry(name, payload, bundle)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "a0_t4_v1_colab_bundle.zip"
    _write_bytes_checked(archive_path, stream.getvalue())
    _write_bytes_checked(output / "input_manifest.json", canonical(input_manifest))
    handoff = {"format": "medrag-v4-a0-t4-v1-colab-handoff-v1",
               "archive": archive_path.name,
               "archive_sha256": sha256_file(archive_path),
               "archive_bytes": archive_path.stat().st_size,
               "input_manifest_sha256": sha256_file(output / "input_manifest.json"),
               "query_count": 1200, "pair_count": 240000,
               "query_ids_sha256": hashlib.sha256(canonical(query_ids)).hexdigest(),
               "pair_manifest_sha256": sha256_file(source / "pair_manifest.json"),
               "model": contract["model"], "model_revision": contract["revision"],
               "fresh_gpu_inference_required": True}
    _write_bytes_checked(output / "handoff.json", canonical(handoff))
    pair_report = {
        "format": "medrag-v4-a0-t4-v1-cpu-pair-plan-report-v1",
        "candidate": "A0_T4_V1", "status": "GO",
        "query_count": len(query_ids), "pair_count": pair_count,
        "config_sha256": lock["config_sha256"],
        "lock_sha256": sha256_file(source / "lock.json"),
        "pair_manifest_sha256": sha256_file(source / "pair_manifest.json"),
        "pair_shards_sha256": pair_hashes,
        "all_pair_ids_input_hashes_and_passage_grounding_validated": True,
        "score_cache_entries": 0,
        "reference_parity_gate_required": False,
        "pair_source": str(source.relative_to(ROOT)),
    }
    _write_bytes_checked(source / "cpu_pair_plan_report.json", canonical(pair_report))
    gpu_ready = {
        "format": "medrag-v4-a0-t4-v1-gpu-scoring-readiness-v1",
        "candidate": "A0_T4_V1", "status": "GO",
        "execution_status": "READY_NOT_RUN",
        "query_count": 1200, "pair_count": 240000,
        "bundle_sha256": handoff["archive_sha256"],
        "model": contract["model"], "model_revision": contract["revision"],
        "precision": contract["precision"], "scorer": "P5.3A p5_3a_colab_orchestrator.py",
        "fresh_inference_required": True, "cached_scores_allowed": False,
        "checkpoint_resume": "query score shard plus per-query provenance and hashes",
        "colab_gpu": "Tesla T4 / CUDA 12.4",
        "exact_mixed_hardware_a0_parity_required": False,
        "t4_reproducibility_reported_separately": True,
    }
    _write_bytes_checked(output / "gpu_scoring_readiness_report.json", canonical(gpu_ready))
    submission = {
        "format": "medrag-v4-a0-t4-v1-full-submission-readiness-v1",
        "candidate": "A0_T4_V1", "status": "PENDING_EXTERNAL_T4_SCORING",
        "cpu_pair_preparation": "GO", "gpu_scoring": "GO",
        "gpu_scoring_execution": "NOT_RUN",
        "cpu_import_finalize_export": "PENDING_T4_SCORE_IMPORT",
        "exact_reference_parity": "NO-GO",
        "t4_reproducibility": "UNVERIFIED",
        "official_btc_evaluator_semantics": "UNVERIFIED",
        "full_inference_started": False,
        "full_gpu_score_inference_started": False,
        "final_export_exists": False,
    }
    _write_bytes_checked(DEFAULT_PREFLIGHT.parent / "full_submission_readiness_report.json",
                         canonical(submission))
    return handoff


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(main(args.source, args.preflight, args.output),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
