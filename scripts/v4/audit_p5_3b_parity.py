"""Audit exact and order-insensitive semantic parity for the P5.3A T4 canary."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import (CONFIG, PILOT_IDS, ROOT, external_lock,
                                      finish_pipeline, read, validate_score_shard)
from scripts.v4.verify_p5_3a_colab_canary import score_bundle
from src.medrag.core.config import load_config
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.ranking.score_cache import ScoreCache

SOURCE = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022"
CANARY = ROOT / "outputs/v4/p5_3a_colab_canary_20261022"
SCORES_ZIP = ROOT / "outputs/v4/p5_3a_colab_import/p5_3a_colab_scores.zip"
OUTPUT = ROOT / "outputs/v4/p5_3b_cpu_gpu_parity_20261022"


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def without_rerank_scores(value):
    if isinstance(value, dict):
        return {key: without_rerank_scores(item) for key, item in value.items()
                if key != "rerank_score"}
    if isinstance(value, list):
        return [without_rerank_scores(item) for item in value]
    return value


def evidence_identity(item: dict) -> tuple:
    return (str(item.get("seed_passage_id", item["passage_id"])), str(item["passage_id"]))


def semantic_document_map(selected: list[dict]) -> dict[int, dict]:
    result = {}
    for document in selected:
        doc_id = int(document["official_id"])
        if doc_id in result:
            raise ValueError(f"Duplicate selected document: {doc_id}")
        value = {key: without_rerank_scores(item) for key, item in document.items()
                 if key not in {"rerank_score", "evidence"}}
        evidence = document["evidence"]
        identities = [evidence_identity(item) for item in evidence]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate selected evidence: {doc_id}")
        value["evidence_by_identity"] = {
            "|".join(identity): without_rerank_scores(item)
            for identity, item in zip(identities, evidence, strict=True)
        }
        result[doc_id] = value
    return result


def selected_chunk_ids(selected: list[dict]) -> set[tuple[int, str, str]]:
    return {(int(document["official_id"]), *evidence_identity(item))
            for document in selected for item in document["evidence"]}


def pair_order(pair: dict, scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (
        -scores[index], int(pair["pairs"][index]["official_id"]),
        str(pair["pairs"][index]["passage_id"])))


def inversion_rows(qid: int, pair: dict, baseline: list[float], gpu: list[float]) -> list[dict]:
    base_order, gpu_order = pair_order(pair, baseline), pair_order(pair, gpu)
    base_rank = {index: rank + 1 for rank, index in enumerate(base_order)}
    gpu_rank = {index: rank + 1 for rank, index in enumerate(gpu_order)}
    rows = []
    for left in range(len(baseline)):
        for right in range(left + 1, len(baseline)):
            if (base_rank[left] - base_rank[right]) * (gpu_rank[left] - gpu_rank[right]) >= 0:
                continue
            left_pair, right_pair = pair["pairs"][left], pair["pairs"][right]
            rows.append({
                "query_id": qid,
                "left_pair_index": left,
                "left_official_id": left_pair["official_id"],
                "left_passage_id": left_pair["passage_id"],
                "left_input_sha256": left_pair["input_sha256"],
                "right_pair_index": right,
                "right_official_id": right_pair["official_id"],
                "right_passage_id": right_pair["passage_id"],
                "right_input_sha256": right_pair["input_sha256"],
                "baseline_left": baseline[left], "baseline_right": baseline[right],
                "gpu_left": gpu[left], "gpu_right": gpu[right],
                "baseline_gap": baseline[left] - baseline[right],
                "gpu_gap": gpu[left] - gpu[right],
                "baseline_left_rank": base_rank[left], "baseline_right_rank": base_rank[right],
                "gpu_left_rank": gpu_rank[left], "gpu_right_rank": gpu_rank[right],
            })
    return sorted(rows, key=lambda row: (abs(row["baseline_gap"]), row["left_pair_index"],
                                          row["right_pair_index"]))


def finish(pair: dict, values: list[float], pipeline) -> list[dict]:
    triples = [(item["official_id"], item["passage_id"], item["payload"])
               for item in pair["pairs"]]
    selected, _ = pipeline.finish_rerank(copy.deepcopy(pair["candidates"]), triples, values,
                                          200, pair["diagnostics"], include_candidates=True)
    return selected


def load_fp32_probe(path: Path | None, subset_sha: str) -> dict:
    if path is None or not path.exists():
        return {"status": "NOT_RUN", "reason": "no_fp32_probe_result_supplied"}
    probe = read(path)
    if (probe.get("format") != "medrag-v4-p5-3b-fp32-probe-v1"
            or probe.get("subset_manifest_sha256") != subset_sha
            or probe.get("model_revision") !=
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"):
        raise ValueError("FP32 probe result identity mismatch")
    return probe


def audit(source: Path, canary: Path, scores_zip: Path, output: Path,
          fp32_result: Path | None) -> dict:
    source, canary = source.resolve(), canary.resolve()
    scores_zip, output = scores_zip.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = external_lock(source, "pilot-live")
    handoff = read(canary / "handoff.json")
    gpu_report, gpu_payloads, _, _, _ = score_bundle(scores_zip, handoff, source)
    config = load_config(CONFIG)
    pipeline = finish_pipeline(config, ROOT / config["index_namespace"])
    score_cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])

    score_rows, inversions, per_query = [], [], {}
    subset_queries = []
    semantic_pass = True
    exact_pass = True
    for qid in PILOT_IDS:
        pair_path = source / "pairs" / f"{qid}.json"
        pair = read(pair_path)
        baseline_payload = read(source / "scores" / f"{qid}.json")
        gpu_payload = json.loads(gpu_payloads[qid])
        baseline = validate_score_shard(baseline_payload, pair, sha256_file(pair_path),
                                        score_cache.fingerprint)
        gpu = validate_score_shard(gpu_payload, pair, sha256_file(pair_path),
                                   score_cache.fingerprint)
        for index, (item, old, new) in enumerate(zip(pair["pairs"], baseline, gpu, strict=True)):
            score_rows.append({"query_id": qid, "pair_index": index,
                               "official_id": item["official_id"],
                               "passage_id": item["passage_id"],
                               "input_sha256": item["input_sha256"],
                               "baseline_score": old, "gpu_score": new,
                               "signed_difference": new - old,
                               "absolute_difference": abs(new - old)})
        query_inversions = inversion_rows(qid, pair, baseline, gpu)
        inversions.extend(query_inversions)
        if query_inversions:
            chosen = query_inversions[0]
            subset_reason = "smallest_absolute_baseline_gap_observed_inversion"
        else:
            changed = [index for index, (left, right) in enumerate(zip(baseline, gpu))
                       if left != right]
            candidates = [(abs(baseline[left] - baseline[right]), left, right)
                          for offset, left in enumerate(changed) for right in changed[offset + 1:]
                          if pair["pairs"][left]["official_id"] !=
                             pair["pairs"][right]["official_id"]]
            if not candidates:
                raise ValueError(f"No deterministic changed-pair FP32 probe case: {qid}")
            _, left, right = min(candidates)
            chosen = {"left_pair_index": left, "right_pair_index": right,
                      "left_input_sha256": pair["pairs"][left]["input_sha256"],
                      "right_input_sha256": pair["pairs"][right]["input_sha256"]}
            subset_reason = "closest_baseline_gap_changed_pair_no_pair_level_inversion"
        subset_queries.append({
            "query_id": qid,
            "query_sha256": pair["query_sha256"],
            "pair_shard_sha256": sha256_file(pair_path),
            "pair_indices": [chosen["left_pair_index"], chosen["right_pair_index"]],
            "selection_reason": subset_reason,
            "cache_keys": [pair["pairs"][chosen["left_pair_index"]]["cache_key"],
                           pair["pairs"][chosen["right_pair_index"]]["cache_key"]],
            "input_sha256": [chosen["left_input_sha256"], chosen["right_input_sha256"]],
        })

        locked_selected = read(source / "final" / f"{qid}.json")["selected"]
        gpu_selected = finish(pair, gpu, pipeline)
        locked_docs = {int(item["official_id"]) for item in locked_selected}
        gpu_docs = {int(item["official_id"]) for item in gpu_selected}
        locked_chunks, gpu_chunks = selected_chunk_ids(locked_selected), selected_chunk_ids(gpu_selected)
        if (len(locked_selected) != 200 or len(gpu_selected) != 200
                or len(locked_docs) != 200 or len(gpu_docs) != 200
                or len(locked_chunks) != 60 or len(gpu_chunks) != 60):
            raise ValueError(f"Selected document/chunk cardinality changed: {qid}")
        locked_semantic, gpu_semantic = (semantic_document_map(locked_selected),
                                         semantic_document_map(gpu_selected))
        docs_order = [int(item["official_id"]) for item in locked_selected]
        gpu_docs_order = [int(item["official_id"]) for item in gpu_selected]
        chunks_order = [(int(document["official_id"]), *evidence_identity(item))
                        for document in locked_selected for item in document["evidence"]]
        gpu_chunks_order = [(int(document["official_id"]), *evidence_identity(item))
                            for document in gpu_selected for item in document["evidence"]]
        semantic_query = (locked_docs == gpu_docs and locked_chunks == gpu_chunks
                          and locked_semantic == gpu_semantic)
        exact_query = (baseline == gpu and locked_selected == gpu_selected)
        semantic_pass &= semantic_query
        exact_pass &= exact_query
        per_query[str(qid)] = {
            "score_pairs": 200,
            "different_scores": sum(left != right for left, right in zip(baseline, gpu)),
            "mean_absolute_difference": sum(abs(right - left)
                                            for left, right in zip(baseline, gpu)) / 200,
            "max_absolute_difference": max(abs(right - left)
                                            for left, right in zip(baseline, gpu)),
            "pairwise_rank_inversions": len(query_inversions),
            "selected_document_id_set_exact": locked_docs == gpu_docs,
            "locked_selected_document_count": len(locked_docs),
            "gpu_selected_document_count": len(gpu_docs),
            "selected_document_order_exact": docs_order == gpu_docs_order,
            "selected_chunk_id_set_exact": locked_chunks == gpu_chunks,
            "locked_selected_chunk_count": len(locked_chunks),
            "gpu_selected_chunk_count": len(gpu_chunks),
            "selected_chunk_order_exact": chunks_order == gpu_chunks_order,
            "content_offsets_text_provenance_exact_ignoring_rerank_score":
                locked_semantic == gpu_semantic,
            "semantic_parity": semantic_query,
            "exact_parity": exact_query,
        }

    score_buffer = io.StringIO()
    score_writer = csv.DictWriter(score_buffer, fieldnames=list(score_rows[0]))
    score_writer.writeheader(); score_writer.writerows(score_rows)
    _write_bytes_checked(output / "score_differences.csv", score_buffer.getvalue().encode())
    inversion_buffer = io.StringIO()
    inversion_writer = csv.DictWriter(inversion_buffer, fieldnames=list(inversions[0]))
    inversion_writer.writeheader(); inversion_writer.writerows(inversions)
    _write_bytes_checked(output / "rank_inversions.csv", inversion_buffer.getvalue().encode())

    subset = {"format": "medrag-v4-p5-3b-fp32-subset-v1",
              "selection": "smallest_absolute_baseline_gap_among_observed_inversions_per_query",
              "query_ids": list(PILOT_IDS), "pair_count": 12,
              "model": lock["score_contract"]["model"],
              "model_revision": lock["score_contract"]["revision"],
              "max_length": lock["score_contract"]["max_length"],
              "queries": subset_queries}
    subset_path = output / "fp32_subset_manifest.json"
    _write_bytes_checked(subset_path, canonical(subset))
    fp32 = load_fp32_probe(fp32_result, sha256_file(subset_path))

    locked_payload = (source / "export/predictions.json").read_bytes()
    gpu_payload = (canary / "verification_0434ce1c30a2/export/predictions.json").read_bytes()
    if sha256_file(scores_zip) != "0434ce1c30a2ab7bfe727eb10a1e7d7c7ca35d1d43afd0cde6efaabf2e036fcc":
        raise ValueError("Unexpected P5.3A GPU score ZIP")
    if len(score_rows) != 1200 or any(not math.isfinite(row["gpu_score"]) for row in score_rows):
        raise ValueError("GPU score coverage/finite check failed")

    report = {
        "format": "medrag-v4-p5-3b-numerical-semantic-parity-v1",
        "release_decision": "NO-GO",
        "release_reason": "exact_parity_failed_and_official_btc_order_semantics_unverified",
        "exact_parity": {
            "status": "GO" if exact_pass else "NO-GO",
            "criteria": "all_1200_scores_ranked_selected_objects_and_canonical_payload_exact",
            "score_exact": all(row["absolute_difference"] == 0 for row in score_rows),
            "selected_objects_exact": all(item["exact_parity"] for item in per_query.values()),
            "btc_payload_exact": locked_payload == gpu_payload,
            "locked_btc_payload_sha256": digest_bytes(locked_payload),
            "gpu_btc_payload_sha256": digest_bytes(gpu_payload),
        },
        "semantic_parity": {
            "status": "GO" if semantic_pass else "NO-GO",
            "criteria": "same_document_and_chunk_ID_sets_with_exact_content_offsets_text_and_provenance_after_removing_only_rerank_score_and_list_order",
            "all_query_document_id_sets_exact": all(item["selected_document_id_set_exact"]
                                                     for item in per_query.values()),
            "all_query_chunk_id_sets_exact": all(item["selected_chunk_id_set_exact"]
                                                  for item in per_query.values()),
            "all_content_offsets_text_provenance_exact": all(
                item["content_offsets_text_provenance_exact_ignoring_rerank_score"]
                for item in per_query.values()),
            "does_not_override_exact_no_go": True,
        },
        "numerical": {
            "score_pairs": 1200,
            "exact_score_pairs": sum(row["absolute_difference"] == 0 for row in score_rows),
            "different_score_pairs": sum(row["absolute_difference"] != 0 for row in score_rows),
            "mean_absolute_difference": sum(row["absolute_difference"] for row in score_rows) / 1200,
            "max_absolute_difference": max(row["absolute_difference"] for row in score_rows),
            "pairwise_rank_inversions": len(inversions),
            "per_query": per_query,
        },
        "inference_contract": {
            "model": lock["score_contract"]["model"],
            "revision": lock["score_contract"]["revision"],
            "pair_format": lock["score_contract"]["pair_format"],
            "max_length": 512,
            "weights_dtype": "torch.float16",
            "activation": "torch.nn.Identity_raw_logits",
            "predict_batch_size": 32,
            "padding": "dynamic_longest_in_each_batch",
            "truncation": True,
            "autocast": False,
            "tokenizer": {
                "class": "XLMRobertaTokenizer",
                "snapshot_tokenizer_json_sha256":
                    gpu_report["model_snapshot_files_sha256"]["tokenizer.json"],
                "snapshot_sentencepiece_sha256":
                    gpu_report["model_snapshot_files_sha256"]["sentencepiece.bpe.model"],
                "configured_model_max_length": 8192,
                "effective_cross_encoder_max_length": 512,
                "padding_side": "right_default",
                "truncation_side": "right_default",
            },
            "gpu_runtime": {"gpu": gpu_report["gpu_name"],
                            "torch": gpu_report["torch_version"],
                            "cuda": gpu_report["torch_cuda_version"],
                            "sentence_transformers": gpu_report["sentence_transformers_version"],
                            "transformers": gpu_report["transformers_version"]},
            "baseline_provenance": {
                "reference_is_not_pure_cpu": True,
                "pilot_gpu": "NVIDIA RTX A6000",
                "pilot_new_pairs_scored": 303,
                "pilot_cached_pairs_reused": 897,
                "cpu_missing_pairs_added_later": 6,
                "note": "The 6 CPU-filled pairs are included within the 1,200 reference pairs; earlier cache creation hardware is not fully recorded per row.",
            },
        },
        "gpu_benchmark": {
            "fresh_model_pairs": gpu_report["fresh_model_pairs"],
            "wall_seconds": gpu_report["wall_seconds"],
            "pairs_per_second": gpu_report["pairs_per_second"],
            "peak_vram_mib": gpu_report["peak_vram_mib"],
            "torch_peak_allocated_mib": gpu_report["torch_peak_allocated_mib"],
            "torch_peak_reserved_mib": gpu_report["torch_peak_reserved_mib"],
        },
        "btc_order_audit": {
            "submission_schema_uses_ordered_json_arrays": True,
            "serializer_preserves_selected_document_and_chunk_order": True,
            "local_validator_requires_selected_serialized_order_equality": True,
            "local_internal_f2_uses_membership_not_rank": True,
            "official_btc_evaluator_source_or_order_contract_available": False,
            "official_btc_score_impact": "UNVERIFIED",
            "assessment": "If official BTC computes set-based F2 over full lists, these permutations do not change F2. The repository does not prove that official BTC ignores rank, so no score-equivalence claim is made.",
            "source_sha256": {
                "serializer": sha256_file(ROOT / "src/medrag/adapters/vibiomir_submission.py"),
                "validator": sha256_file(ROOT / "src/medrag/adapters/validation.py"),
                "local_evaluator": sha256_file(ROOT / "src/evaluation/evaluator.py"),
                "precision": sha256_file(ROOT / "src/evaluation/precision.py"),
                "recall": sha256_file(ROOT / "src/evaluation/recall.py"),
            },
        },
        "fp32_probe": fp32,
        "integrity": {
            "gpu_scores_zip_sha256": sha256_file(scores_zip),
            "pair_manifest_sha256": sha256_file(source / "pair_manifest.json"),
            "score_contract_fingerprint": score_cache.fingerprint,
            "model_snapshot_files_sha256": gpu_report["model_snapshot_files_sha256"],
            "all_p5_3a_checks_reused": True,
        },
        "artifacts": {
            "score_differences_csv": "score_differences.csv",
            "rank_inversions_csv": "rank_inversions.csv",
            "fp32_subset_manifest": "fp32_subset_manifest.json",
        },
    }
    _write_bytes_checked(output / "p5_3b_parity_report.json", canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--canary", type=Path, default=CANARY)
    parser.add_argument("--scores-zip", type=Path, default=SCORES_ZIP)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fp32-result", type=Path)
    args = parser.parse_args()
    report = audit(args.source, args.canary, args.scores_zip, args.output, args.fp32_result)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["release_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
