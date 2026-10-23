"""Import fresh Colab T4 scores and audit P2-A0 parity without full-batch inference."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

from scripts.v4.run_full_baseline import (_all_official_ids, _write_bytes_checked)
from scripts.v4.run_p5_2_split import (CONFIG, PILOT_IDS, ROOT, external_lock,
                                      finish_pipeline, import_score, read,
                                      validate_pair_shard, validate_score_shard,
                                      verify_pair_manifest)
from src.medrag.adapters.validation import (canonical_payload, package_zip,
                                            validate_selected, validate_zip)
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.ranking.score_cache import ScoreCache

SOURCE = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022"
OUTPUT = ROOT / "outputs/v4/p5_3a_colab_canary_20261022"


def strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes):
    return json.loads(payload, object_pairs_hook=strict_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Non-finite JSON value: {value}")))


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish_report(path: Path, report: dict) -> None:
    """Atomically replace a prior failed-attempt summary for the same input ZIP."""
    payload = canonical(report)
    if path.exists() and path.read_bytes() == payload:
        return
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def score_bundle(path: Path, handoff: dict, source: Path) -> tuple[
        dict, dict[int, bytes], str, bytes, dict[int, bytes]]:
    expected_names = {f"scores/{qid}.json" for qid in PILOT_IDS} | {
        f"provenance/{qid}.json" for qid in PILOT_IDS} | {
        "score_manifest.json", "gpu_canary_report.json", "scorer_stdout.log"}
    with zipfile.ZipFile(path) as bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if (set(names) != expected_names or len(names) != len(expected_names)
                or any(item.file_size > 12_000_000 for item in infos)
                or bundle.testzip() is not None):
            raise ValueError("Colab result ZIP coverage/integrity mismatch")
        report = strict_json(bundle.read("gpu_canary_report.json"))
        manifest_raw = bundle.read("score_manifest.json")
        manifest = strict_json(manifest_raw)
        stdout = bundle.read("scorer_stdout.log")
        scores = {qid: bundle.read(f"scores/{qid}.json") for qid in PILOT_IDS}
        proofs_raw = {qid: bundle.read(f"provenance/{qid}.json") for qid in PILOT_IDS}
        proofs = {qid: strict_json(payload) for qid, payload in proofs_raw.items()}
    lock = external_lock(source, "pilot-live")
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    contract = lock["score_contract"]
    if (manifest.get("format") != "medrag-v4-p5-3a-score-manifest-v1"
            or manifest.get("input_bundle_sha256") != handoff["archive_sha256"]
            or manifest.get("model_revision") != contract["revision"]
            or manifest.get("query_ids") != list(PILOT_IDS)
            or manifest.get("pair_count") != 1200
            or report.get("score_manifest_sha256") != sha(manifest_raw)):
        raise ValueError("GPU score manifest identity/hash mismatch")
    expected_stdout = "".join(proofs[qid].get("scorer_line", "") + "\n" for qid in PILOT_IDS)
    if (report.get("format") != "medrag-v4-p5-3a-colab-gpu-report-v2"
            or report.get("status") != "FRESH_GPU_SCORE_COMPLETE"
            or report.get("input_bundle_sha256") != handoff["archive_sha256"]
            or report.get("lock_sha256") != sha256_file(source / "lock.json")
            or report.get("pair_manifest_sha256") != sha256_file(source / "pair_manifest.json")
            or report.get("score_contract_fingerprint") != cache.fingerprint
            or report.get("model") != contract["model"]
            or report.get("model_revision") != contract["revision"]
            or report.get("max_length") != contract["max_length"]
            or report.get("precision") != contract["precision"]
            or "T4" not in str(report.get("gpu_name", ""))
            or report.get("torch_version") != "2.6.0+cu124"
            or report.get("torch_cuda_version") != "12.4"
            or report.get("sentence_transformers_version") != "4.1.0"
            or report.get("transformers_version") != "4.51.3"
            or report.get("query_count") != 6
            or report.get("cache_entries_before_this_session") != 0
            or report.get("fresh_model_pairs") != 1200
            or report.get("cache_hits") != 0
            or report.get("scorer_stdout_sha256") != sha(stdout)
            or stdout.decode("utf-8") != expected_stdout):
        raise ValueError("Fresh pinned T4 inference provenance failed")
    resumed = report.get("resumed_query_ids")
    if (not isinstance(resumed, list) or len(set(resumed)) != len(resumed)
            or any(qid not in PILOT_IDS for qid in resumed)
            or report.get("cache_entries_after_this_session") != 200 * (6 - len(resumed))
            or report.get("session_count") != len({proof.get("session_id") for proof in proofs.values()})):
        raise ValueError("Resume/cache accounting differs from fresh query proofs")
    if (report.get("query_proofs_sha256") != {str(qid): sha(proofs_raw[qid]) for qid in PILOT_IDS}
            or report.get("score_shards_sha256") != {str(qid): sha(scores[qid]) for qid in PILOT_IDS}
            or manifest.get("query_proofs_sha256") != report.get("query_proofs_sha256")
            or manifest.get("score_shards_sha256") != report.get("score_shards_sha256")):
        raise ValueError("GPU result hash manifest mismatch")
    for qid, proof in proofs.items():
        seconds = proof.get("seconds")
        if (proof.get("format") != "medrag-v4-p5-3a-query-proof-v1"
                or proof.get("query_id") != qid
                or proof.get("input_bundle_sha256") != handoff["archive_sha256"]
                or proof.get("model_revision") != contract["revision"]
                or proof.get("gpu_name") != report["gpu_name"]
                or proof.get("pair_shard_sha256") != sha256_file(source / "pairs" / f"{qid}.json")
                or proof.get("score_shard_sha256") != sha(scores[qid])
                or proof.get("fresh_model_pairs") != 200 or proof.get("cache_hits") != 0
                or proof.get("scorer_line") != f"scored {qid}: 200 model calls, 0 cache hits"
                or not isinstance(proof.get("session_id"), str) or not proof["session_id"]
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError(f"Query {qid} fresh inference proof differs")
        for key in ("peak_vram_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib"):
            value = proof.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Query {qid} benchmark invalid: {key}")
    snapshot = report.get("model_snapshot_files_sha256")
    if (not isinstance(snapshot, dict)
            or not any(name.endswith((".safetensors", ".bin")) for name in snapshot)
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in snapshot.values())):
        raise ValueError("Pinned model snapshot hashes missing/invalid")
    for name in ("wall_seconds", "pairs_per_second", "peak_vram_mib",
                 "torch_peak_allocated_mib", "torch_peak_reserved_mib"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid GPU benchmark value: {name}")
    if (not math.isclose(report["wall_seconds"], sum(proof["seconds"] for proof in proofs.values()), rel_tol=1e-9)
            or not math.isclose(report["pairs_per_second"], 1200 / report["wall_seconds"], rel_tol=1e-9)
            or report["peak_vram_mib"] != max(proof["peak_vram_mib"] for proof in proofs.values())
            or report.get("vram_sample_count", 0) <= 0):
        raise ValueError("GPU benchmark arithmetic/samples invalid")
    for qid, payload in scores.items():
        if report["score_shards_sha256"][str(qid)] != sha(payload):
            raise ValueError(f"GPU score shard hash mismatch: {qid}")
        pair_path = source / "pairs" / f"{qid}.json"
        pair = read(pair_path)
        validate_pair_shard(pair, lock, qid, pair["query"], cache)
        validate_score_shard(strict_json(payload), pair, sha256_file(pair_path), cache.fingerprint)
    return report, scores, stdout.decode("utf-8"), manifest_raw, proofs_raw


def ranking_identity(candidates: list[dict]) -> list[tuple[int, tuple[str, ...]]]:
    return [(int(doc["official_id"]), tuple(evidence["passage_id"] for evidence in doc["evidence"]))
            for doc in candidates]


def selected_chunks(selected: list[dict]) -> list[tuple]:
    return [(int(doc["official_id"]), ev["passage_id"], ev.get("seed_passage_id"))
            for doc in selected for ev in doc["evidence"]]


def selected_contexts(selected: list[dict]) -> list[tuple]:
    return [(int(doc["official_id"]), ev["text"], ev.get("start"), ev.get("end"),
             tuple(ev.get("context_passage_ids", [])))
            for doc in selected for ev in doc["evidence"]]


def sequence_comparison(actual: list | tuple, locked: list | tuple) -> dict:
    first_difference = next((index + 1 for index, (left, right) in
                             enumerate(zip(actual, locked, strict=False)) if left != right), None)
    if first_difference is None and len(actual) != len(locked):
        first_difference = min(len(actual), len(locked)) + 1
    return {
        "exact": actual == locked,
        "actual_count": len(actual),
        "locked_count": len(locked),
        "exact_positions": sum(left == right for left, right in
                               zip(actual, locked, strict=False)),
        "overlap_count": len(set(actual) & set(locked)),
        "first_difference_rank": first_difference,
    }


def materialize_pair_workspace(source: Path, staging: Path, lock: dict) -> dict[int, dict]:
    """Restore every immutable pair shard before any score import validates coverage."""
    verify_pair_manifest(source, lock)
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    pair_manifest = read(source / "pair_manifest.json")
    (staging / "pairs").mkdir(exist_ok=True)
    restored = {}
    for qid in PILOT_IDS:
        source_path = source / "pairs" / f"{qid}.json"
        expected_sha = pair_manifest["pair_shards_sha256"][str(qid)]
        if sha256_file(source_path) != expected_sha:
            raise ValueError(f"Frozen source pair shard hash mismatch: {qid}")
        shard = read(source_path)
        validate_pair_shard(shard, lock, qid, shard["query"], cache)
        target = staging / "pairs" / f"{qid}.json"
        _write_bytes_checked(target, source_path.read_bytes())
        if sha256_file(target) != expected_sha:
            raise ValueError(f"Restored pair shard hash mismatch: {qid}")
        restored[qid] = shard
    verify_pair_manifest(staging, lock)
    if sum(len(shard["pairs"]) for shard in restored.values()) != 1200:
        raise ValueError("Restored frozen pair coverage differs from 1,200")
    return restored


def audit(source: Path, output: Path, score_path: Path) -> dict:
    source, output, score_path = source.resolve(), output.resolve(), score_path.resolve()
    lock = external_lock(source, "pilot-live")
    handoff = read(output / "handoff.json")
    if handoff["format"] != "medrag-v4-p5-3a-handoff-v1":
        raise ValueError("P5.3A handoff mismatch")
    gpu, score_payloads, stdout, score_manifest, proofs = score_bundle(
        score_path, handoff, source)
    result_sha = sha256_file(score_path)
    staging = output / ("verification_" + result_sha[:12])
    staging.mkdir(parents=True, exist_ok=True)
    for name in ("lock.json", "pair_manifest.json", "key_contract.sqlite"):
        _write_bytes_checked(staging / name, (source / name).read_bytes())
    pairs_by_id = materialize_pair_workspace(source, staging, lock)
    (staging / "received").mkdir(exist_ok=True)
    (staging / "provenance").mkdir(exist_ok=True)
    for qid in PILOT_IDS:
        name = f"{qid}.json"
        received = staging / "received" / name
        _write_bytes_checked(received, score_payloads[qid])
        _write_bytes_checked(staging / "provenance" / name, proofs[qid])
        import_score(staging, lock, qid, received)
    _write_bytes_checked(staging / "score_manifest.json", score_manifest)
    _write_bytes_checked(staging / "scorer_stdout.log", stdout.encode("utf-8"))
    _write_bytes_checked(staging / "gpu_canary_report.json", canonical(gpu))

    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    pipeline = finish_pipeline(config, index)
    score_cache = ScoreCache(staging / "key_contract.sqlite", lock["score_contract"])
    selected_by_id = {}
    comparison = {}
    per_query_score_differences = {}
    score_rows = []
    for qid in PILOT_IDS:
        pair_path = staging / "pairs" / f"{qid}.json"
        pair = pairs_by_id[qid]
        values = validate_score_shard(read(staging / "scores" / f"{qid}.json"), pair,
                                      sha256_file(pair_path), score_cache.fingerprint)
        reference = read(source / "scores" / f"{qid}.json")
        reference_values = validate_score_shard(reference, pair, sha256_file(pair_path),
                                                score_cache.fingerprint)
        triples = [(p["official_id"], p["passage_id"], p["payload"]) for p in pair["pairs"]]
        selected, trace = pipeline.finish_rerank(copy.deepcopy(pair["candidates"]), triples,
                                                  values, 200, pair["diagnostics"],
                                                  include_candidates=True)
        baseline = read(source / "final" / f"{qid}.json")
        selected_by_id[qid] = selected
        chunks = selected_chunks(selected)
        baseline_chunks = selected_chunks(baseline["selected"])
        ranked = ranking_identity(trace["ranked_candidates"])
        baseline_ranked = ranking_identity(baseline["trace"]["ranked_candidates"])
        documents = [int(doc["official_id"]) for doc in selected]
        baseline_documents = [int(doc["official_id"]) for doc in baseline["selected"]]
        contexts = selected_contexts(selected)
        baseline_contexts = selected_contexts(baseline["selected"])
        comparison[str(qid)] = {
            "ranked_candidates_exact": ranked == baseline_ranked,
            "ranked_candidate_order": sequence_comparison(ranked, baseline_ranked),
            "selected_documents_exact": documents == baseline_documents,
            "selected_document_ranking": sequence_comparison(documents, baseline_documents),
            "top_60_chunks_exact": len(chunks) == len(baseline_chunks) == 60 and
                                   chunks == baseline_chunks,
            "top_60_chunks": sequence_comparison(chunks[:60], baseline_chunks[:60]),
            "selected_chunks_exact": chunks == baseline_chunks,
            "selected_contexts_exact": contexts == baseline_contexts,
            "selected_contexts": sequence_comparison(contexts, baseline_contexts),
            "selected_full_objects_exact": selected == baseline["selected"],
        }
        query_deltas = []
        for p, old, new in zip(pair["pairs"], reference_values, values, strict=True):
            query_deltas.append(abs(new - old))
            score_rows.append((qid, p["official_id"], p["passage_id"], old, new, new - old,
                               abs(new - old)))
        per_query_score_differences[str(qid)] = {
            "pair_count": len(query_deltas),
            "exact_score_pairs": sum(delta == 0 for delta in query_deltas),
            "different_score_pairs": sum(delta != 0 for delta in query_deltas),
            "mean_absolute_difference": sum(query_deltas) / len(query_deltas),
            "max_absolute_difference": max(query_deltas),
        }
    score_csv = io.StringIO()
    writer = csv.writer(score_csv)
    writer.writerow(("query_id", "official_id", "passage_id", "locked_score", "fresh_t4_score",
                     "signed_difference", "absolute_difference"))
    writer.writerows(score_rows)
    diff_path = staging / "score_differences.csv"
    _write_bytes_checked(diff_path, score_csv.getvalue().encode("utf-8"))

    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    corpus_path = ROOT / migration["official_corpus"]
    official_ids = _all_official_ids(corpus_path)
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            read(index / "manifest.json")["fingerprint"])
    predictions = validate_selected(list(PILOT_IDS), selected_by_id, store,
                                    pipeline.selection_policy, result_limit=200,
                                    reranker_enabled=True, official_ids=official_ids)
    payload = canonical_payload(predictions)
    baseline_payload = (source / "export/predictions.json").read_bytes()
    payload_exact = payload == baseline_payload
    export = staging / "export"
    export.mkdir(exist_ok=True)
    _write_bytes_checked(export / "predictions.json", payload)
    if not (export / "submission.zip").exists():
        package_zip(export / "submission.zip", payload)
    zip_report = validate_zip(export / "submission.zip", query_ids=set(PILOT_IDS),
                              official_ids=official_ids, policy=pipeline.selection_policy,
                              result_limit=200, max_bytes=int(config["batch"]["zip_max_bytes"]))
    if zip_report["payload_sha256"] != sha(payload):
        raise ValueError("Exported ZIP payload hash mismatch")
    all_output_parity = payload_exact and all(
        all(item[key] for key in ("ranked_candidates_exact", "selected_documents_exact",
                                  "top_60_chunks_exact", "selected_chunks_exact",
                                  "selected_contexts_exact"))
        for item in comparison.values())
    max_delta = max(row[6] for row in score_rows)
    report = {"format": "medrag-v4-p5-3a-colab-verification-v1",
              "status": "GO" if all_output_parity else "NO-GO",
              "reason": "all_frozen_output_parity_gates_pass" if all_output_parity else
                        "fresh_gpu_scores_change_frozen_ranking_or_btc_payload",
              "fresh_gpu_inference_verified": True,
              "query_ids": list(PILOT_IDS), "query_count": 6, "pair_count": 1200,
              "pair_integrity": {
                  "query_ids_exact": list(PILOT_IDS) == lock["query_ids"],
                  "validated_pair_count": sum(len(pair["pairs"])
                                              for pair in pairs_by_id.values()),
                  "pair_shards_sha256": read(source / "pair_manifest.json")
                                               ["pair_shards_sha256"],
                  "passage_ids_and_input_hashes_validated": True,
              },
              "gpu_score_integrity": {
                  "validated_score_count": len(score_rows),
                  "per_query_score_count": {str(qid): len(pairs_by_id[qid]["pairs"])
                                            for qid in PILOT_IDS},
                  "score_shards_sha256": gpu["score_shards_sha256"],
              },
              "input_bundle_sha256": handoff["archive_sha256"],
              "gpu_scores_zip_sha256": result_sha,
              "lock_sha256": sha256_file(source / "lock.json"),
              "pair_manifest_sha256": sha256_file(source / "pair_manifest.json"),
              "model": gpu["model"], "model_revision": gpu["model_revision"],
              "score_contract_fingerprint": gpu["score_contract_fingerprint"],
              "gpu_name": gpu["gpu_name"],
              "score_differences": {"exact_score_pairs": sum(row[6] == 0 for row in score_rows),
                                    "different_score_pairs": sum(row[6] != 0 for row in score_rows),
                                    "mean_absolute_difference": sum(row[6] for row in score_rows) / 1200,
                                    "max_absolute_difference": max_delta,
                                    "per_query": per_query_score_differences,
                                    "csv_sha256": sha256_file(diff_path)},
              "per_query_parity": comparison,
              "btc_payload_parity": payload_exact,
              "btc_payload_sha256": sha(payload),
              "locked_btc_payload_sha256": sha(baseline_payload),
              "submission_zip_sha256": sha256_file(export / "submission.zip"),
              "benchmark": {"fresh_model_pairs": gpu["fresh_model_pairs"],
                            "cache_hits": gpu["cache_hits"],
                            "wall_seconds": gpu["wall_seconds"],
                            "pairs_per_second": gpu["pairs_per_second"],
                            "peak_vram_mib": gpu["peak_vram_mib"],
                            "torch_peak_allocated_mib": gpu["torch_peak_allocated_mib"],
                            "torch_peak_reserved_mib": gpu["torch_peak_reserved_mib"],
                            "eta_240000_pairs_hours_rerank_only":
                                240000 / gpu["pairs_per_second"] / 3600},
              "artifacts": {"score_differences_csv": str(diff_path.relative_to(output)),
                            "validated_predictions": str((export / "predictions.json").relative_to(output)),
                            "validated_submission_zip": str((export / "submission.zip").relative_to(output))}}
    publish_report(staging / "verification_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-zip", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(args.source, output, args.scores_zip)
    except Exception as exc:
        report = {"format": "medrag-v4-p5-3a-colab-verification-v1", "status": "NO-GO",
                  "fresh_gpu_inference_verified": False,
                  "scores_zip": str(args.scores_zip.resolve()),
                  "reason": f"{type(exc).__name__}: {exc}"}
    suffix = sha256_file(args.scores_zip)[:12] if args.scores_zip.exists() else "missing"
    publish_report(output / f"verification_{suffix}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
