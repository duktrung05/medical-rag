"""Prepare the isolated A0_T4_V1 release candidate and pending gate reports."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v4.run_full_baseline import _all_official_ids, _write_bytes_checked
from scripts.v4.run_p5_2_split import CONFIG, FROZEN_CONFIG_SHA256, ROOT, read
from src.medrag.adapters.validation import canonical_payload, validate_payload
from src.medrag.core.config import load_config
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.selector import SelectionPolicy

OUTPUT = ROOT / "outputs/v4/a0_t4_v1_release_candidate_20261022"
P5_3B = ROOT / "outputs/v4/p5_3b_cpu_gpu_parity_20261022/p5_3b_parity_report.json"
P5_3A = ROOT / "outputs/v4/p5_3a_colab_canary_20261022/verification_0434ce1c30a2"
REFERENCE_ZIP = ROOT / "outputs/v4/p5_3a_colab_import/p5_3a_colab_scores.zip"


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def write(path: Path, value: dict) -> None:
    _write_bytes_checked(path, canonical(value))


def contains_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(contains_key(item, target) for item in value)
    return False


def rejected(payload: bytes, **kwargs) -> bool:
    try:
        validate_payload(payload, **kwargs)
    except ValueError:
        return True
    return False


def main(output: Path = OUTPUT) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
        raise ValueError("Frozen P2-A0 config changed")
    config = load_config(CONFIG)
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    official_ids = _all_official_ids(ROOT / migration["official_corpus"])
    policy = SelectionPolicy(**config["selection"])
    payload_path = P5_3A / "export/predictions.json"
    payload = payload_path.read_bytes()
    parsed = validate_payload(payload, query_ids={53, 130, 380, 754, 771, 1144},
                              official_ids=official_ids, policy=policy, result_limit=200)
    if contains_key(parsed, "rerank_score"):
        raise ValueError("BTC payload unexpectedly exports rerank_score")
    permuted = []
    for row in parsed:
        permuted.append({"id": row["id"],
                         "relevant_docs": list(reversed(row["relevant_docs"])),
                         "relevant_chunks": list(reversed(row["relevant_chunks"]))})
    permuted_payload = canonical_payload(permuted)
    validate_payload(permuted_payload, query_ids={53, 130, 380, 754, 771, 1144},
                     official_ids=official_ids, policy=policy, result_limit=200)
    extra_top = json.loads(payload)
    extra_top[0]["rerank_score"] = 0.0
    extra_chunk = json.loads(payload)
    extra_chunk[0]["relevant_chunks"][0]["rerank_score"] = 0.0
    validation_args = {"query_ids": {53, 130, 380, 754, 771, 1144},
                       "official_ids": official_ids, "policy": policy, "result_limit": 200}
    top_rejected = rejected(canonical_payload(extra_top), **validation_args)
    chunk_rejected = rejected(canonical_payload(extra_chunk), **validation_args)
    if not top_rejected or not chunk_rejected:
        raise ValueError("Local BTC schema accepted rerank_score")

    contract = {
        "format": "medrag-v4-p5-3c-btc-submission-contract-v1",
        "local_submission_format_status": "GO",
        "payload_root": "JSON array",
        "record_keys_exact": ["id", "relevant_docs", "relevant_chunks"],
        "relevant_docs_type": "ordered JSON array of unique official integer IDs",
        "relevant_chunks_type": "ordered JSON array of {doc_id,chunk_text}",
        "rerank_score_exported": False,
        "rerank_score_top_level_rejected": top_rejected,
        "rerank_score_chunk_field_rejected": chunk_rejected,
        "local_validator_accepts_document_and_chunk_permutations": True,
        "serializer_preserves_input_document_and_evidence_order": True,
        "local_internal_f2_is_order_insensitive": True,
        "official_public_page": "https://r2ai.aiguru.com.vn/",
        "official_public_metric_description": "Macro F2 (Recall-Weighted) for document and chunk retrieval",
        "official_public_page_defines_order_sensitivity": False,
        "official_public_evaluator_download_found": False,
        "official_btc_evaluator_available_in_repository": False,
        "official_btc_evaluator_semantics": "UNVERIFIED",
        "official_btc_order_sensitivity": "UNVERIFIED",
        "assessment": "Local schema treats arrays as ordered serialization but does not reject permutations. The internal F2 implementation is membership based. The official public page names recall-weighted Macro F2 and document/chunk outputs, but neither it nor this repository supplies evaluator code or order semantics; official order sensitivity remains UNVERIFIED.",
        "validated_pilot_payload_sha256": sha256_file(payload_path),
        "permuted_payload_sha256": __import__("hashlib").sha256(permuted_payload).hexdigest(),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "src/medrag/adapters/vibiomir_submission.py",
            "src/medrag/adapters/validation.py",
            "src/evaluation/evaluator.py",
            "src/evaluation/precision.py",
            "src/evaluation/recall.py")},
    }
    write(output / "btc_submission_contract_report.json", contract)

    p5_3b = read(P5_3B)
    exact = {
        "format": "medrag-v4-p5-3c-exact-reference-parity-v1",
        "candidate": "A0_T4_V1",
        "status": p5_3b["exact_parity"]["status"],
        "criterion": "byte_and_order_exact_against_frozen_mixed_hardware_p2_a0",
        "reference_is_immutable": True,
        "candidate_does_not_overwrite_reference": True,
        "exact_score_pairs": p5_3b["numerical"]["exact_score_pairs"],
        "different_score_pairs": p5_3b["numerical"]["different_score_pairs"],
        "pairwise_rank_inversions": p5_3b["numerical"]["pairwise_rank_inversions"],
        "btc_payload_exact": p5_3b["exact_parity"]["btc_payload_exact"],
        "semantic_parity": p5_3b["semantic_parity"]["status"],
        "p5_3b_report_sha256": sha256_file(P5_3B),
    }
    write(output / "exact_reference_parity_report.json", exact)

    lock = {
        "format": "medrag-v4-a0-t4-v1-release-lock-v1",
        "candidate": "A0_T4_V1",
        "status": "PENDING_T4_REPRODUCIBILITY",
        "frozen_a0_config": str(CONFIG.relative_to(ROOT)),
        "frozen_a0_config_sha256": FROZEN_CONFIG_SHA256,
        "reference_artifacts_immutable": True,
        "reference_t4_scores_zip": str(REFERENCE_ZIP.relative_to(ROOT)),
        "reference_t4_scores_zip_sha256": sha256_file(REFERENCE_ZIP),
        "model": config["reranker"]["model"],
        "model_revision": config["reranker"]["revision"],
        "max_length": config["reranker"]["max_length"],
        "precision": "fp16",
        "pair_manifest_sha256": sha256_file(
            ROOT / "outputs/v4/p5_2_split_pilot_live_20261022/pair_manifest.json"),
        "isolated_full_output": "outputs/v4/a0_t4_v1_full_20261022",
        "prohibited_changes": ["fusion", "passage_selection", "chunk_allocation",
                               "embeddings", "reranker_weights"],
        "implementation_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/v4/run_p5_2_split.py",
            "scripts/v4/p5_3a_colab_orchestrator.py",
            "scripts/v4/verify_p5_3c_t4_reproducibility.py",
            "scripts/v4/build_p5_3c_t4_repro_notebook.py",
            "scripts/v4/capture_a0_t4_v1_runtime.py",
            "scripts/v4/seal_a0_t4_v1_full_scores.py",
            "scripts/v4/run_a0_t4_v1_full.sh",
            "notebooks/p5_3c_t4_repro.ipynb")},
    }
    write(output / "release_candidate_lock.json", lock)
    pending_repro = {
        "format": "medrag-v4-p5-3c-t4-reproducibility-v1",
        "candidate": "A0_T4_V1", "status": "UNVERIFIED",
        "reason": "second_independent_t4_result_not_imported",
        "first_t4_fresh_pairs": 1200, "first_t4_cache_hits": 0,
        "required_repeat_pairs": 1200, "required_repeat_cache_hits": 0,
        "go_requires": ["disjoint_session_provenance", "same_pinned_runtime",
                        "same_model_snapshot_hashes", "1200_exact_scores",
                        "exact_pair_ranking", "exact_selected_objects",
                        "exact_btc_payload_between_t4_runs"],
    }
    write(output / "t4_reproducibility_report.json", pending_repro)
    readiness = {
        "format": "medrag-v4-p5-3c-submission-readiness-v1",
        "candidate": "A0_T4_V1", "status": "NO-GO",
        "full_inference_gate": "NO-GO",
        "reason": "t4_reproducibility_unverified",
        "exact_reference_parity": exact["status"],
        "t4_reproducibility": "UNVERIFIED",
        "local_submission_format": contract["local_submission_format_status"],
        "official_btc_evaluator_semantics": "UNVERIFIED",
        "metric_equivalence_to_a0": "UNVERIFIED",
        "next_step": "Run notebooks/p5_3c_t4_repro.ipynb on T4 and import its ZIP with verify_p5_3c_t4_reproducibility.py.",
    }
    write(output / "submission_readiness_report.json", readiness)
    return readiness


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
