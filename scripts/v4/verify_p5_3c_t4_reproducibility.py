"""Verify two independent fresh T4 canaries and gate the A0_T4_V1 candidate."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path

from scripts.v4.audit_p5_3b_parity import finish, pair_order
from scripts.v4.run_full_baseline import _all_official_ids, _write_bytes_checked
from scripts.v4.run_p5_2_split import (CONFIG, PILOT_IDS, ROOT, external_lock,
                                      finish_pipeline, read, validate_score_shard)
from scripts.v4.verify_p5_3a_colab_canary import score_bundle, sha, strict_json
from src.medrag.adapters.validation import canonical_payload, validate_payload, validate_selected
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.ranking.score_cache import ScoreCache

SOURCE = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022"
HANDOFF = ROOT / "outputs/v4/p5_3a_colab_canary_20261022/handoff.json"
REFERENCE_ZIP = ROOT / "outputs/v4/p5_3a_colab_import/p5_3a_colab_scores.zip"
OUTPUT = ROOT / "outputs/v4/a0_t4_v1_release_candidate_20261022"


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def publish(path: Path, value: dict) -> None:
    payload = canonical(value)
    if path.exists() and path.read_bytes() == payload:
        return
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def proof_sessions(proofs: dict[int, bytes]) -> set[str]:
    return {str(strict_json(payload)["session_id"]) for payload in proofs.values()}


def chunk_identity(selected: list[dict]) -> list[tuple[int, str, str | None]]:
    return [(int(document["official_id"]), str(item["passage_id"]),
             item.get("seed_passage_id"))
            for document in selected for item in document["evidence"]]


def audit(reference_zip: Path, repeat_zip: Path, source: Path, output: Path) -> dict:
    reference_zip, repeat_zip = reference_zip.resolve(), repeat_zip.resolve()
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    handoff = read(HANDOFF)
    lock = external_lock(source, "pilot-live")
    first_gpu, first_raw, _, _, first_proofs = score_bundle(reference_zip, handoff, source)
    second_gpu, second_raw, _, _, second_proofs = score_bundle(repeat_zip, handoff, source)
    first_sessions, second_sessions = proof_sessions(first_proofs), proof_sessions(second_proofs)
    independent_sessions = first_sessions.isdisjoint(second_sessions)
    distinct_archives = sha256_file(reference_zip) != sha256_file(repeat_zip)

    config = load_config(CONFIG)
    pipeline = finish_pipeline(config, ROOT / config["index_namespace"])
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    first_selected, second_selected = {}, {}
    rows, per_query = [], {}
    for qid in PILOT_IDS:
        pair_path = source / "pairs" / f"{qid}.json"
        pair = read(pair_path)
        first = validate_score_shard(strict_json(first_raw[qid]), pair,
                                     sha256_file(pair_path), cache.fingerprint)
        second = validate_score_shard(strict_json(second_raw[qid]), pair,
                                      sha256_file(pair_path), cache.fingerprint)
        selected_first = finish(pair, first, pipeline)
        selected_second = finish(pair, second, pipeline)
        first_selected[qid], second_selected[qid] = selected_first, selected_second
        deltas = [abs(right - left) for left, right in zip(first, second, strict=True)]
        for index, (item, left, right, delta) in enumerate(zip(
                pair["pairs"], first, second, deltas, strict=True)):
            rows.append({"query_id": qid, "pair_index": index,
                         "official_id": item["official_id"],
                         "passage_id": item["passage_id"],
                         "input_sha256": item["input_sha256"],
                         "first_t4_score": left, "repeat_t4_score": right,
                         "signed_difference": right - left,
                         "absolute_difference": delta})
        first_docs = [int(item["official_id"]) for item in selected_first]
        second_docs = [int(item["official_id"]) for item in selected_second]
        first_chunks, second_chunks = chunk_identity(selected_first), chunk_identity(selected_second)
        per_query[str(qid)] = {
            "score_pairs": len(first),
            "exact_score_pairs": sum(delta == 0 for delta in deltas),
            "different_score_pairs": sum(delta != 0 for delta in deltas),
            "mean_absolute_difference": sum(deltas) / len(deltas),
            "max_absolute_difference": max(deltas),
            "pair_ranking_exact": pair_order(pair, first) == pair_order(pair, second),
            "selected_document_order_exact": first_docs == second_docs,
            "selected_chunk_order_exact": first_chunks == second_chunks,
            "selected_full_objects_exact": selected_first == selected_second,
        }

    score_buffer = io.StringIO()
    writer = csv.DictWriter(score_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    verification = output / ("verification_" + sha256_file(repeat_zip)[:12])
    verification.mkdir(parents=True, exist_ok=True)
    _write_bytes_checked(verification / "t4_score_differences.csv",
                         score_buffer.getvalue().encode("utf-8"))

    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    official_ids = _all_official_ids(ROOT / migration["official_corpus"])
    index = ROOT / config["index_namespace"]
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            read(index / "manifest.json")["fingerprint"])
    first_predictions = validate_selected(list(PILOT_IDS), first_selected, store,
                                          pipeline.selection_policy, result_limit=200,
                                          reranker_enabled=True, official_ids=official_ids)
    second_predictions = validate_selected(list(PILOT_IDS), second_selected, store,
                                           pipeline.selection_policy, result_limit=200,
                                           reranker_enabled=True, official_ids=official_ids)
    first_payload = canonical_payload(first_predictions)
    second_payload = canonical_payload(second_predictions)
    validate_payload(second_payload, query_ids=set(PILOT_IDS), official_ids=official_ids,
                     policy=pipeline.selection_policy, result_limit=200)

    exact_scores = all(row["absolute_difference"] == 0 for row in rows)
    exact_ranking = all(row["pair_ranking_exact"] for row in per_query.values())
    exact_selected = all(row["selected_full_objects_exact"] for row in per_query.values())
    exact_payload = first_payload == second_payload
    runtime_keys = ("gpu_name", "torch_version", "torch_cuda_version",
                    "sentence_transformers_version", "transformers_version",
                    "model", "model_revision", "max_length", "precision")
    runtime_exact = all(first_gpu[key] == second_gpu[key] for key in runtime_keys)
    snapshot_exact = (first_gpu["model_snapshot_files_sha256"] ==
                      second_gpu["model_snapshot_files_sha256"])
    reproducible = all((independent_sessions, distinct_archives, runtime_exact,
                        snapshot_exact, exact_scores, exact_ranking,
                        exact_selected, exact_payload))
    numerical = {
        "score_pairs": len(rows),
        "exact_score_pairs": sum(row["absolute_difference"] == 0 for row in rows),
        "different_score_pairs": sum(row["absolute_difference"] != 0 for row in rows),
        "mean_absolute_difference": sum(row["absolute_difference"] for row in rows) / len(rows),
        "max_absolute_difference": max(row["absolute_difference"] for row in rows),
        "all_finite": all(math.isfinite(row["repeat_t4_score"]) for row in rows),
        "per_query": per_query,
    }
    report = {
        "format": "medrag-v4-p5-3c-t4-reproducibility-v1",
        "candidate": "A0_T4_V1",
        "status": "GO" if reproducible else "NO-GO",
        "criteria": "two_independent_fresh_pinned_t4_runs_with_exact_scores_ranking_selected_objects_and_btc_payload",
        "reference_t4_zip_sha256": sha256_file(reference_zip),
        "repeat_t4_zip_sha256": sha256_file(repeat_zip),
        "distinct_result_archives": distinct_archives,
        "independent_session_provenance": independent_sessions,
        "reference_session_ids": sorted(first_sessions),
        "repeat_session_ids": sorted(second_sessions),
        "runtime_contract_exact": runtime_exact,
        "model_snapshot_hashes_exact": snapshot_exact,
        "fresh_inference": {"first_pairs": first_gpu["fresh_model_pairs"],
                            "repeat_pairs": second_gpu["fresh_model_pairs"],
                            "first_cache_hits": first_gpu["cache_hits"],
                            "repeat_cache_hits": second_gpu["cache_hits"]},
        "numerical": numerical,
        "pair_ranking_exact": exact_ranking,
        "selected_objects_exact": exact_selected,
        "btc_payload_exact_between_t4_runs": exact_payload,
        "first_t4_btc_payload_sha256": sha(first_payload),
        "repeat_t4_btc_payload_sha256": sha(second_payload),
        "benchmark": {"first": {key: first_gpu[key] for key in (
                              "wall_seconds", "pairs_per_second", "peak_vram_mib")},
                      "repeat": {key: second_gpu[key] for key in (
                              "wall_seconds", "pairs_per_second", "peak_vram_mib")}},
        "score_differences_csv_sha256": sha256_file(
            verification / "t4_score_differences.csv"),
    }
    _write_bytes_checked(verification / "t4_reproducibility_report.json", canonical(report))

    contract = read(output / "btc_submission_contract_report.json")
    readiness_go = reproducible and contract["local_submission_format_status"] == "GO"
    readiness = {
        "format": "medrag-v4-p5-3c-submission-readiness-v1",
        "candidate": "A0_T4_V1",
        "status": "GO" if readiness_go else "NO-GO",
        "full_inference_gate": "GO" if readiness_go else "NO-GO",
        "exact_reference_parity": "NO-GO",
        "t4_reproducibility": report["status"],
        "local_submission_format": contract["local_submission_format_status"],
        "official_btc_evaluator_semantics": contract["official_btc_evaluator_semantics"],
        "metric_equivalence_to_a0": "UNVERIFIED",
        "release_policy": "A0_T4_V1 is independently versioned; exact mixed-hardware A0 parity is not a promotion criterion.",
        "verification_report": str((verification / "t4_reproducibility_report.json")
                                   .relative_to(output)),
    }
    _write_bytes_checked(verification / "submission_readiness_report.json", canonical(readiness))
    publish(output / "t4_reproducibility_report.json", report)
    publish(output / "submission_readiness_report.json", readiness)
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-zip", required=True, type=Path)
    parser.add_argument("--reference-zip", type=Path, default=REFERENCE_ZIP)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = audit(args.reference_zip, args.repeat_zip, args.source, args.output)
    except Exception as exc:
        result = {"format": "medrag-v4-p5-3c-submission-readiness-v1",
                  "candidate": "A0_T4_V1", "status": "NO-GO",
                  "full_inference_gate": "NO-GO",
                  "reason": f"{type(exc).__name__}: {exc}"}
        publish(args.output.resolve() / "submission_readiness_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
