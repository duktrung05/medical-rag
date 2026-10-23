"""Drive-checkpointed T4 canary using the existing P5.2 GPU score function.

This module orchestrates transfer, telemetry and resume only. It never computes
reranker scores itself; every new shard is produced by run_p5_2_split.score.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import zipfile
from importlib.metadata import version
from pathlib import Path

from scripts.v4.run_p5_2_split import (ScoreCache, external_lock,
                                      read, score, validate_pair_shard,
                                      validate_score_shard)

SCORE_LINE = re.compile(r"^scored (\d+): (\d+) model calls, (\d+) cache hits$")
RUNTIME_VERSIONS = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124",
                    "sentence-transformers": "4.1.0", "transformers": "4.51.3",
                    "huggingface-hub": "0.36.2", "tokenizers": "0.21.4"}


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing Drive checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_bundle(bundle_path: Path, expected_sha: str, workdir: Path) -> tuple[Path, dict, Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("Expected bundle SHA-256 is malformed")
    if file_digest(bundle_path) != expected_sha:
        raise ValueError("Uploaded bundle SHA-256 mismatch")
    repo = workdir / "repo"
    repo.mkdir()
    with zipfile.ZipFile(bundle_path) as bundle:
        infos = bundle.infolist()
        if (len({item.filename for item in infos}) != len(infos)
                or sum(item.file_size for item in infos) > 8_000_000_000
                or any(item.is_dir() or item.filename.startswith("/")
                       or ".." in Path(item.filename).parts
                       or ("/pairs/" in f"/{item.filename}" and item.file_size > 12_000_000)
                       for item in infos)
                or bundle.testzip() is not None):
            raise ValueError("Canary ZIP names, size or CRC invalid")
        bundle.extractall(repo)
    manifest = read(repo / "p5_3a_input_manifest.json")
    mode = manifest.get("source_mode")
    query_ids = manifest.get("query_ids")
    expected_pair_count = 200 * len(query_ids) if isinstance(query_ids, list) else None
    if (manifest.get("format") not in {"medrag-v4-p5-3a-colab-input-v1",
                                        "medrag-v4-p5-full-colab-input-v1"}
            or mode not in {"pilot-live", "full"}
            or not isinstance(query_ids, list) or not query_ids
            or len(query_ids) not in (6, 1200)
            or manifest.get("pair_count") != expected_pair_count):
        raise ValueError("Canary input manifest mismatch")
    for name, expected in manifest["files_sha256"].items():
        path = repo / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Bundle file hash mismatch: {name}")
    output = repo / manifest["input_dir"]
    if (output / "scores.sqlite").exists() or (output / "scores").exists():
        raise ValueError("Canary input contains score cache/shards")
    lock = external_lock(output, mode)
    if (lock["query_ids"] != query_ids
            or lock["score_contract"]["revision"] != manifest["model_revision"]):
        raise ValueError("Canary lock differs from manifest")
    if file_digest(repo / "configs/v4_submission_p2_a0.yaml") != lock["config_sha256"]:
        raise ValueError("Frozen A0 config differs from lock")
    for name, expected in lock["implementation_sha256"].items():
        if file_digest(repo / name) != expected:
            raise ValueError(f"Frozen scorer implementation mismatch: {name}")
    cache = ScoreCache(output / "key_contract.sqlite", lock["score_contract"])
    with sqlite3.connect(output / "key_contract.sqlite") as db:
        if db.execute("SELECT count(*) FROM scores").fetchone()[0] != 0:
            raise ValueError("Input key contract contains cached scores")
    for qid in query_ids:
        pair = read(output / "pairs" / f"{qid}.json")
        validate_pair_shard(pair, lock, qid, pair["query"], cache)
    return repo, manifest, output, lock


def check_gpu() -> tuple[object, str]:
    for name, expected in RUNTIME_VERSIONS.items():
        if version(name) != expected:
            raise ValueError(f"Runtime dependency {name} differs: {version(name)} != {expected}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one CUDA GPU is required")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name or torch.version.cuda != "12.4":
        raise ValueError(f"Colab Tesla T4/CUDA 12.4 required, got {name}/{torch.version.cuda}")
    return torch, name


def snapshot_hashes(lock: dict) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    contract = lock["score_contract"]
    snapshot = Path(snapshot_download(repo_id=contract["model"],
                                      revision=contract["revision"]))
    if snapshot.name != contract["revision"]:
        raise ValueError("Resolved model revision differs")
    hashes = {str(path.relative_to(snapshot)): file_digest(path)
              for path in sorted(snapshot.rglob("*")) if path.is_file()}
    if not any(name.endswith((".safetensors", ".bin")) for name in hashes):
        raise ValueError("Model snapshot has no weights")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return hashes


def gpu_sample() -> tuple[int, int]:
    import subprocess

    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=10)
    fields = result.stdout.strip().splitlines()[0].split(",")
    return int(fields[0].strip()), int(fields[1].strip())


def validate_checkpoint(run_dir: Path, output: Path, lock: dict, qid: int,
                        expected_bundle_sha: str, gpu_name: str) -> dict | None:
    shard_path = run_dir / "scores" / f"{qid}.json"
    proof_path = run_dir / "provenance" / f"{qid}.json"
    if not shard_path.exists() or not proof_path.exists():
        # A crashed Drive write without its companion proof is not a checkpoint.
        shard_path.unlink(missing_ok=True)
        proof_path.unlink(missing_ok=True)
        return None
    proof = read(proof_path)
    pair_path = output / "pairs" / f"{qid}.json"
    pair = read(pair_path)
    cache = ScoreCache(output / "key_contract.sqlite", lock["score_contract"])
    if (proof.get("format") != "medrag-v4-p5-3a-query-proof-v1"
            or proof.get("query_id") != qid
            or proof.get("input_bundle_sha256") != expected_bundle_sha
            or proof.get("model_revision") != lock["score_contract"]["revision"]
            or proof.get("gpu_name") != gpu_name
            or proof.get("pair_shard_sha256") != file_digest(pair_path)
            or proof.get("score_shard_sha256") != file_digest(shard_path)
            or proof.get("fresh_model_pairs") != 200
            or proof.get("cache_hits") != 0
            or not isinstance(proof.get("seconds"), (int, float))
            or not math.isfinite(proof["seconds"]) or proof["seconds"] <= 0
            or not SCORE_LINE.fullmatch(proof.get("scorer_line", ""))):
        raise ValueError(f"Drive score checkpoint provenance invalid: {qid}")
    line = SCORE_LINE.fullmatch(proof["scorer_line"])
    if tuple(map(int, line.groups())) != (qid, 200, 0):
        raise ValueError(f"Drive scorer line differs: {qid}")
    validate_score_shard(read(shard_path), pair, file_digest(pair_path), cache.fingerprint)
    return proof


class ScoreTee(io.StringIO):
    def __init__(self, target, started: float, torch):
        super().__init__()
        self.target = target
        self.started = started
        self.previous = started
        self.torch = torch
        self.events: dict[int, dict] = {}
        self.buffer = ""
        self.lock = threading.Lock()

    def write(self, value: str) -> int:
        self.target.write(value)
        self.target.flush()
        with self.lock:
            self.buffer += value
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                match = SCORE_LINE.fullmatch(line)
                if match:
                    qid, fresh, hits = map(int, match.groups())
                    completed = time.perf_counter()
                    self.events[qid] = {"line": line, "fresh": fresh, "hits": hits,
                                        "started": self.previous, "completed": completed,
                                        "torch_peak_allocated_mib":
                                            self.torch.cuda.max_memory_allocated(0) / 1024**2,
                                        "torch_peak_reserved_mib":
                                            self.torch.cuda.max_memory_reserved(0) / 1024**2}
                    self.previous = completed
        return len(value)


def checkpoint_new_scores(output: Path, run_dir: Path, lock: dict, expected_sha: str,
                          gpu_name: str, tee: ScoreTee, samples: list[tuple[float, int, int]],
                          stop: threading.Event, completed: set[int], failures: list[str]) -> None:
    cache = ScoreCache(output / "key_contract.sqlite", lock["score_contract"])
    while True:
        with tee.lock:
            pending = {qid: event.copy() for qid, event in tee.events.items()
                       if qid not in completed}
        if stop.is_set() and not pending:
            return
        for qid, event in pending.items():
            try:
                if qid not in lock["query_ids"] or event["fresh"] != 200 or event["hits"] != 0:
                    raise ValueError(f"Query {qid} used cached scores or wrong pair count")
                shard_path = output / "scores" / f"{qid}.json"
                if not shard_path.exists():
                    if stop.is_set():
                        raise ValueError(f"Scorer logged completion without shard: {qid}")
                    continue
                pair_path = output / "pairs" / f"{qid}.json"
                validate_score_shard(read(shard_path), read(pair_path), file_digest(pair_path),
                                     cache.fingerprint)
                window = [mem for when, mem, _ in samples
                          if event["started"] <= when <= event["completed"]]
                peak = max(max(window, default=gpu_sample()[0]),
                           event["torch_peak_reserved_mib"])
                proof = {"format": "medrag-v4-p5-3a-query-proof-v1",
                         "query_id": qid, "input_bundle_sha256": expected_sha,
                         "model_revision": lock["score_contract"]["revision"],
                         "gpu_name": gpu_name,
                         "pair_shard_sha256": file_digest(pair_path),
                         "score_shard_sha256": file_digest(shard_path),
                         "fresh_model_pairs": event["fresh"], "cache_hits": event["hits"],
                         "scorer_line": event["line"],
                         "seconds": event["completed"] - event["started"],
                         "peak_vram_mib": peak,
                         "torch_peak_allocated_mib": event["torch_peak_allocated_mib"],
                         "torch_peak_reserved_mib": event["torch_peak_reserved_mib"],
                         "session_id": os.environ["P5_3A_SESSION_ID"]}
                write_once(run_dir / "scores" / f"{qid}.json", shard_path.read_bytes())
                write_once(run_dir / "provenance" / f"{qid}.json", canonical(proof))
                completed.add(qid)
                print(f"Drive checkpoint saved: query {qid}", flush=True)
            except Exception as exc:
                failures.append(f"{qid}: {type(exc).__name__}: {exc}")
                stop.set()
                return
        stop.wait(0.25)


def main(bundle_path: Path, drive_root: Path, expected_sha: str) -> dict:
    bundle_path, drive_root = bundle_path.resolve(), drive_root.resolve()
    session_id = f"{int(time.time())}-{os.getpid()}"
    os.environ["P5_3A_SESSION_ID"] = session_id
    os.environ["HF_HOME"] = str(Path(tempfile.gettempdir()) / "p5_3a_hf_cache")
    # Keep model files on local Colab storage; Drive holds only durable score checkpoints.
    with tempfile.TemporaryDirectory(prefix="p5_3a_canary_") as temporary:
        repo, manifest, output, lock = check_bundle(bundle_path, expected_sha, Path(temporary))
        torch, gpu_name = check_gpu()
        model_hashes = snapshot_hashes(lock)
        run_dir = drive_root / ("canary_" + expected_sha[:12])
        run_dir.mkdir(parents=True, exist_ok=True)
        run_lock = {"format": "medrag-v4-p5-3a-drive-lock-v1",
                    "input_bundle_sha256": expected_sha,
                    "pair_manifest_sha256": file_digest(output / "pair_manifest.json"),
                    "score_contract_fingerprint":
                        ScoreCache(output / "key_contract.sqlite", lock["score_contract"]).fingerprint,
                    "model_revision": lock["score_contract"]["revision"],
                    "query_ids": list(lock["query_ids"])}
        write_once(run_dir / "run_lock.json", canonical(run_lock))
        restored = {}
        for qid in lock["query_ids"]:
            proof = validate_checkpoint(run_dir, output, lock, qid, expected_sha, gpu_name)
            if proof is not None:
                target = output / "scores" / f"{qid}.json"
                write_once(target, (run_dir / "scores" / f"{qid}.json").read_bytes())
                restored[qid] = proof
        completed_report = run_dir / "gpu_canary_report.json"
        completed_zip = run_dir / "p5_3a_colab_scores.zip"
        completed_manifest = run_dir / "score_manifest.json"
        if (len(restored) == len(lock["query_ids"]) and completed_report.is_file()
                and completed_manifest.is_file() and completed_zip.is_file()):
            prior = read(completed_report)
            if (prior.get("status") != "FRESH_GPU_SCORE_COMPLETE"
                    or prior.get("input_bundle_sha256") != expected_sha
                    or prior.get("model_revision") != lock["score_contract"]["revision"]
                    or prior.get("score_manifest_sha256") != file_digest(completed_manifest)
                    or prior.get("score_shards_sha256") != {str(qid): file_digest(
                        run_dir / "scores" / f"{qid}.json") for qid in lock["query_ids"]}
                    or prior.get("query_proofs_sha256") != {str(qid): file_digest(
                        run_dir / "provenance" / f"{qid}.json") for qid in lock["query_ids"]}):
                raise ValueError("Completed Drive result does not match checkpoints")
            with zipfile.ZipFile(completed_zip) as existing:
                if (existing.testzip() is not None
                        or existing.read("gpu_canary_report.json") != completed_report.read_bytes()
                        or existing.read("score_manifest.json") != completed_manifest.read_bytes()):
                    raise ValueError("Completed Drive ZIP differs from report")
            prior["result_zip_path"] = str(completed_zip)
            prior["result_zip_sha256"] = file_digest(completed_zip)
            print(json.dumps({key: prior[key] for key in (
                "status", "fresh_model_pairs", "cache_hits", "pairs_per_second",
                "peak_vram_mib", "result_zip_path", "result_zip_sha256")}, indent=2), flush=True)
            return prior
        if (output / "scores.sqlite").exists():
            raise ValueError("Local scorer cache is not empty before inference")
        baseline_vram, _ = gpu_sample()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        tee = ScoreTee(sys.stdout, started, torch)
        samples: list[tuple[float, int, int]] = []
        stop_monitor = threading.Event()
        stop_writer = threading.Event()
        completed = set(restored)
        failures: list[str] = []

        def monitor() -> None:
            while not stop_monitor.is_set():
                try:
                    memory, utilization = gpu_sample()
                    samples.append((time.perf_counter(), memory, utilization))
                except Exception:
                    pass
                stop_monitor.wait(0.25)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        writer_thread = threading.Thread(target=checkpoint_new_scores,
            args=(output, run_dir, lock, expected_sha, gpu_name, tee, samples,
                  stop_writer, completed, failures), daemon=True)
        monitor_thread.start()
        writer_thread.start()
        try:
            queries = [{"id": qid, "query": read(output / "pairs" / f"{qid}.json")["query"]}
                       for qid in lock["query_ids"]]
            with contextlib.redirect_stdout(tee):
                score(output, lock, queries, seed_cache=None, device="cuda:0")
            torch.cuda.synchronize(0)
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=10)
            stop_writer.set()
            writer_thread.join(timeout=30)
        if failures or completed != set(lock["query_ids"]):
            raise ValueError(f"Incomplete Drive score checkpoints: {failures}; saved={sorted(completed)}")
        proofs = {qid: validate_checkpoint(run_dir, output, lock, qid, expected_sha, gpu_name)
                  for qid in lock["query_ids"]}
        if any(proof is None for proof in proofs.values()):
            raise ValueError("One or more Drive proofs vanished")
        expected_pairs = 200 * len(lock["query_ids"])
        fresh_pairs = sum(proof["fresh_model_pairs"] for proof in proofs.values())
        cache_hits = sum(proof["cache_hits"] for proof in proofs.values())
        seconds = sum(proof["seconds"] for proof in proofs.values())
        if fresh_pairs != expected_pairs or cache_hits != 0 or seconds <= 0:
            raise ValueError("Fresh score coverage or timing differs")
        score_cache = ScoreCache(output / "scores.sqlite", lock["score_contract"])
        with sqlite3.connect(score_cache.path) as db:
            current_session_cache_rows = db.execute("SELECT count(*) FROM scores").fetchone()[0]
        if current_session_cache_rows != 200 * (len(lock["query_ids"]) - len(restored)):
            raise ValueError("Current session scored-cache coverage differs")
        scorer_lines = "".join(proofs[qid]["scorer_line"] + "\n"
                                 for qid in lock["query_ids"])
        pair_rate = fresh_pairs / seconds
        score_manifest = {
            "format": "medrag-v4-p5-3a-score-manifest-v1",
            "input_bundle_sha256": expected_sha,
            "model_revision": lock["score_contract"]["revision"],
            "query_ids": list(lock["query_ids"]),
            "pair_count": expected_pairs,
            "score_shards_sha256": {str(qid): file_digest(
                run_dir / "scores" / f"{qid}.json") for qid in lock["query_ids"]},
            "query_proofs_sha256": {str(qid): file_digest(
                run_dir / "provenance" / f"{qid}.json") for qid in lock["query_ids"]},
        }
        manifest_path = run_dir / "score_manifest.json"
        write_once(manifest_path, canonical(score_manifest))
        report = {"format": "medrag-v4-p5-3a-colab-gpu-report-v2",
                  "status": "FRESH_GPU_SCORE_COMPLETE",
                  "input_bundle_sha256": expected_sha,
                  "lock_sha256": file_digest(output / "lock.json"),
                  "pair_manifest_sha256": file_digest(output / "pair_manifest.json"),
                  "score_contract_fingerprint": run_lock["score_contract_fingerprint"],
                  "score_manifest_sha256": file_digest(manifest_path),
                  "model": lock["score_contract"]["model"],
                  "model_revision": lock["score_contract"]["revision"],
                  "model_snapshot_files_sha256": model_hashes,
                  "max_length": lock["score_contract"]["max_length"],
                  "precision": lock["score_contract"]["precision"],
                  "gpu_name": gpu_name,
                  "torch_version": version("torch"),
                  "torch_cuda_version": torch.version.cuda,
                  "sentence_transformers_version": version("sentence-transformers"),
                  "transformers_version": version("transformers"),
                  "cache_entries_before_this_session": 0,
                  "cache_entries_after_this_session": current_session_cache_rows,
                  "fresh_model_pairs": fresh_pairs, "cache_hits": cache_hits,
                  "query_count": len(lock["query_ids"]),
                  "pair_count": expected_pairs,
                  "resumed_query_ids": sorted(restored),
                  "session_count": len({proof["session_id"] for proof in proofs.values()}),
                  "query_proofs_sha256": {str(qid): file_digest(
                      run_dir / "provenance" / f"{qid}.json") for qid in lock["query_ids"]},
                  "score_shards_sha256": {str(qid): file_digest(
                      run_dir / "scores" / f"{qid}.json") for qid in lock["query_ids"]},
                  "scorer_stdout_sha256": hashlib.sha256(scorer_lines.encode()).hexdigest(),
                  "wall_seconds": seconds, "pairs_per_second": pair_rate,
                  "peak_vram_mib": max(proof["peak_vram_mib"] for proof in proofs.values()),
                  "baseline_vram_mib": baseline_vram,
                  "torch_peak_allocated_mib": max(proof["torch_peak_allocated_mib"]
                                                   for proof in proofs.values()),
                  "torch_peak_reserved_mib": max(proof["torch_peak_reserved_mib"]
                                                  for proof in proofs.values()),
                  "vram_sample_count": max(len(samples), 1),
                  "eta_240000_pairs_hours_rerank_only": 240000 / pair_rate / 3600}
        report_path = run_dir / "gpu_canary_report.json"
        if report_path.exists() and read(report_path).get("input_bundle_sha256") != expected_sha:
            raise ValueError("Existing Drive report belongs to a different bundle")
        report_temporary = report_path.with_name(f".{report_path.name}.incomplete-{session_id}")
        report_temporary.write_bytes(canonical(report))
        os.replace(report_temporary, report_path)
        result = run_dir / "p5_3a_colab_scores.zip"
        temporary_zip = result.with_name(f".{result.name}.incomplete-{session_id}")
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for qid in lock["query_ids"]:
                bundle.write(run_dir / "scores" / f"{qid}.json", f"scores/{qid}.json")
                bundle.write(run_dir / "provenance" / f"{qid}.json", f"provenance/{qid}.json")
            bundle.write(manifest_path, "score_manifest.json")
            bundle.write(report_path, "gpu_canary_report.json")
            bundle.writestr("scorer_stdout.log", scorer_lines)
        os.replace(temporary_zip, result)
        report["result_zip_sha256"] = file_digest(result)
        report["result_zip_path"] = str(result)
        print(json.dumps({key: report[key] for key in (
            "status", "fresh_model_pairs", "cache_hits", "resumed_query_ids",
            "pairs_per_second", "peak_vram_mib", "eta_240000_pairs_hours_rerank_only",
            "result_zip_path", "result_zip_sha256")}, indent=2), flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--drive-output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    main(args.bundle, args.drive_output, args.expected_sha256)
