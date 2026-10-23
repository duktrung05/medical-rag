"""CPU-only preflight for the P2-A0 BTC submission configuration.

This command never constructs DiskPipeline and never calls retrieval or a model.
It replays the six locked pilot selections from cached ranked candidates and
checks their exported source-grounded chunks against the P2-A0 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
QUERY_IDS = [53, 130, 380, 754, 771, 1144]
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
P2 = ROOT / "outputs/v4/p2_passage_candidate_replay_20261021"
P2_BASELINE = ROOT / "outputs/v4/adjacent_context_replay_d200_20261021/selected_results.jsonl"

from src.medrag.adapters.validation import canonical_payload, validate_selected
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from scripts.v4.run_full_baseline import _all_official_ids
from src.medrag.ranking.score_cache import ScoreCache


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    return sha256_file(path)


def gpu_check() -> dict:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                text=True, timeout=10, check=False)
        info = {"nvidia_smi_exit_code": result.returncode,
                "nvidia_smi_stdout": result.stdout.strip(),
                "nvidia_smi_stderr": result.stderr.strip(),
                "available": result.returncode == 0 and "GPU " in result.stdout}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        result = subprocess.run([sys.executable, "-c",
            "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),'count':torch.cuda.device_count()}))"],
            capture_output=True, text=True, timeout=20, check=False)
        info["torch_cuda_exit_code"] = result.returncode
        info["torch_cuda_stdout"] = result.stdout.strip()
        info["torch_cuda_stderr"] = result.stderr.strip()
        info["torch_cuda_available"] = (result.returncode == 0 and
                                         json.loads(result.stdout).get("available") is True)
        info["available"] = info["available"] and info["torch_cuda_available"]
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        info["torch_cuda_available"] = False
        info["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
        info["available"] = False
    return info


def main(config_path: Path, output: Path) -> dict:
    config_path, output = config_path.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    policy = SelectionPolicy(**config["selection"])
    assert config["retrieval"]["rerank_allocation"] == "parent_first"
    assert config["retrieval"]["fused_candidate_limit"] == 400
    assert config["retrieval"]["rerank_candidate_limit"] == 200
    assert config["selection"]["max_docs"] == 200
    assert config["selection"]["max_chunks"] == 60
    assert config["selection"]["chunks_per_doc"] == 1
    assert config["selection"]["document_selection"] == "independent"
    assert config["selection"]["chunk_context"] == "adjacent"

    index = ROOT / config["index_namespace"]
    index_manifest_path = index / "manifest.json"
    index_manifest = read(index_manifest_path)
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)

    # Exact full selected objects provide document order, seed/context passage IDs,
    # derived chunk IDs, source offsets and context text from the P2-A0 baseline.
    baseline = {int(row["id"]): row["selected"] for row in map(
        json.loads, P2_BASELINE.open(encoding="utf-8"))}
    artifact_rows = [json.loads(line) for line in
                     (P2 / "replay_selected_chunks.jsonl").open(encoding="utf-8")
                     if json.loads(line)["arm"] == "P2-A0"]
    artifact_by_query: dict[int, list[dict]] = {qid: [] for qid in QUERY_IDS}
    for row in artifact_rows:
        artifact_by_query[int(row["query_id"])].append(row)

    selected_by_id, detail_rows = {}, []
    for qid in QUERY_IDS:
        checkpoint = read(PILOT / "queries" / f"{qid}.json")
        if checkpoint.get("status") != "complete":
            raise ValueError(f"Pilot checkpoint is incomplete: {qid}")
        raw, _ = select_results(checkpoint["diagnostics"]["ranked_candidates"],
                                policy, reranker_enabled=True, limit=200)
        selected, _ = apply_chunk_context(raw, policy, sources, passages)
        selected_by_id[qid] = selected
        expected = baseline[qid]
        if selected != expected:
            raise ValueError(f"Full document/chunk/passage/context replay differs: {qid}")
        exported = [doc for doc in selected if doc["evidence"]]
        artifacts = artifact_by_query[qid]
        if len(exported) != 60 or len(artifacts) != 60:
            raise ValueError(f"Expected exactly 60 exported chunks in query {qid}")
        for doc, artifact in zip(exported, artifacts, strict=True):
            ev = doc["evidence"][0]
            if (int(doc["official_id"]) != int(artifact["doc_id"])
                    or ev["seed_passage_id"] != artifact["seed_passage_id"]
                    or ev["text"] != artifact["context_text"]):
                raise ValueError(f"P2-A0 chunk artifact mismatch: {qid}/{doc['official_id']}")
            detail_rows.append({"query_id": qid, "doc_id": int(doc["official_id"]),
                                "chunk_id": ev["passage_id"],
                                "seed_passage_id": ev["seed_passage_id"],
                                "context_passage_ids": ev["context_passage_ids"],
                                "context_sha256": hashlib.sha256(
                                    ev["text"].encode("utf-8")).hexdigest(),
                                "context_text": ev["text"]})

    # Verify BTC payload construction and source/document grounding with the full
    # registered official ID set, rather than only validating selected IDs.
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    corpus_path = ROOT / migration["official_corpus"]
    query_path = ROOT / migration["official_queries"]
    if hash_file(corpus_path) != migration["official_corpus_sha256"]:
        raise ValueError("Official corpus hash mismatch")
    if hash_file(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query hash mismatch")
    official_ids = _all_official_ids(corpus_path)
    predictions = validate_selected(QUERY_IDS, selected_by_id, store, policy,
                                    result_limit=200, reranker_enabled=True,
                                    official_ids=official_ids)
    payload = canonical_payload(predictions)
    payload_rows = json.loads(payload)
    if (len(payload_rows) != len(QUERY_IDS)
            or any(set(row) != {"id", "relevant_docs", "relevant_chunks"}
                   for row in payload_rows)):
        raise ValueError("BTC payload schema mismatch")

    # The P2 manifest pins its cache and all replay source artifacts. Re-hash
    # each declared input/artifact and check exact A0 cached pairs are complete.
    p2_manifest = read(P2 / "manifest.json")
    p2_hash_checks = {}
    for name, expected in p2_manifest["input_sha256"].items():
        path = ROOT / name
        p2_hash_checks[name] = hash_file(path) == expected
    for name, expected in p2_manifest["artifact_sha256"].items():
        p2_hash_checks[f"P2/{name}"] = hash_file(P2 / name) == expected
    if not all(p2_hash_checks.values()):
        raise ValueError("One or more P2 source/artifact hashes differ from manifest")
    with (P2 / "pair_allocation.csv").open(encoding="utf-8", newline="") as stream:
        pairs = [row for row in csv.DictReader(stream) if row["arm"] == "P2-A0"]
    if len(pairs) != 1200 or any(row["cache_hit"] != "True" for row in pairs):
        raise ValueError("P2-A0 must have 200 cached pairs for each of six queries")
    pair_counts = {qid: sum(int(row["query_id"]) == qid for row in pairs)
                   for qid in QUERY_IDS}
    if any(value != 200 for value in pair_counts.values()):
        raise ValueError(f"P2-A0 pair counts invalid: {pair_counts}")

    # Recompute the exact cache key for every A0 pair from the pinned query,
    # source title, registered passage text and score contract.
    query_rows = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    cache_path = Path(read(P2 / "manifest.json")["score_cache_path"])
    with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as db:
        cache_meta = dict(db.execute("SELECT key,value FROM meta"))
        cache_contract = json.loads(cache_meta["contract"])
    score_cache = ScoreCache(cache_path, cache_contract)
    cache_key_failures = []
    for pair in pairs:
        qid, doc_id = int(pair["query_id"]), int(pair["doc_id"])
        pid = pair["passage_id"]
        ranked = read(PILOT / "queries" / f"{qid}.json")["diagnostics"]["ranked_candidates"]
        evidence = next(item for doc in ranked if int(doc["official_id"]) == doc_id
                        for item in doc["evidence"] if item["passage_id"] == pid)
        source = sources[doc_id]
        payload_text = f"{source.title}\n{evidence['text']}" if source.title else evidence["text"]
        query = query_rows[qid]
        key = score_cache._key(query, doc_id, pid, payload_text)
        query_sha = hashlib.sha256(query.encode()).hexdigest()
        input_sha = hashlib.sha256(payload_text.encode()).hexdigest()
        with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as db:
            cached = db.execute("SELECT cache_key,score,query_sha256,official_id,passage_id,input_sha256 "
                                "FROM scores WHERE cache_key=?", (key,)).fetchone()
        valid = (cached is not None and cached[2] == query_sha and cached[3] == doc_id
                 and cached[4] == pid and cached[5] == input_sha
                 and input_sha == pair["input_sha256"]
                 and float(cached[1]) == float(pair["cached_rerank_score"]))
        if not valid:
            cache_key_failures.append({"query_id": qid, "doc_id": doc_id,
                                      "passage_id": pid})
    if cache_key_failures:
        raise ValueError(f"Exact score cache key/metadata mismatch: {cache_key_failures[:5]}")

    gpu = gpu_check()
    checks = {
        "submission_config_matches_p2_a0_contract": True,
        "pilot_replay_matches_full_p2_a0_document_chunk_passage_context_objects": True,
        "p2_chunk_artifacts_match_doc_seed_passage_and_context_text": True,
        "p2_manifest_inputs_and_artifacts_hash_valid": all(p2_hash_checks.values()),
        "cached_pairs_exactly_200_per_query_no_missing": True,
        "score_cache_keys_and_pair_metadata_verified": not cache_key_failures,
        "source_grounding_and_btc_schema_validation": True,
        "fresh_gpu_runtime_available": gpu["available"],
        "retrieval_calls": 0,
        "model_calls": 0,
    }
    report = {
        "format": "v4-p5-1-a0-submission-preflight-v1",
        "status": "GO" if all(value is True for key, value in checks.items()
                               if key not in ("retrieval_calls", "model_calls")) else "NO-GO",
        "full_inference_started": False,
        "query_ids": QUERY_IDS,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": hash_file(config_path),
        "p2_a0_source_config": "configs/v4_phase2_full_independent_d200_context_gpu3.yaml",
        "inference_contract": {"rerank_allocation": config["retrieval"]["rerank_allocation"],
                                "fusion": config["fusion"],
                                "fused_candidate_limit": config["retrieval"]["fused_candidate_limit"],
                                "rerank_pairs": config["retrieval"]["rerank_candidate_limit"],
                                "max_docs": config["selection"]["max_docs"],
                                "max_chunks": config["selection"]["max_chunks"],
                                "chunks_per_doc": config["selection"]["chunks_per_doc"],
                                "document_selection": config["selection"]["document_selection"],
                                "chunk_context": config["selection"]["chunk_context"]},
        "checks": checks,
        "gpu_runtime": gpu,
        "pilot_replay": {"query_count": len(QUERY_IDS), "selected_documents":
                          sum(map(len, selected_by_id.values())),
                          "exported_chunks": len(detail_rows),
                          "exact_full_object_matches": 6,
                          "btc_payload_sha256": hashlib.sha256(payload).hexdigest(),
                          "p2_a0_pairs_by_query": pair_counts},
        "p2_manifest_hash_checks": p2_hash_checks,
        "cache_key_audit": {"verified_pairs": len(pairs), "failures": cache_key_failures,
                            "contract_fingerprint": score_cache.fingerprint},
        "resume_runner": {
            "path": "scripts/v4/run_full_baseline.py",
            "mode": "atomic query checkpoints under OUTPUT.building; resumes only when run lock matches",
            "cache_key_contract": "pinned model/revision/max_length/index manifest/corpus plus query, parent, passage and input SHA256",
            "shared_cache_multiworker_safe": False,
            "two_gpu_sharded_runner_ready": False,
            "full_run_authorized": False,
            "command_when_blockers_are_cleared": (
                "PYTHONPATH=outputs/pubmed_runtime_deps_20261012 .venv-phase1/bin/python "
                "scripts/v4/run_full_baseline.py --index outputs/v4/task4/full_index_gpu3_20261019 "
                "--config configs/v4_submission_p2_a0.yaml "
                "--output outputs/v4/submission_p2_a0_full_20261022"),
        },
        "blockers": (["Fresh GPU runtime unavailable in this environment."] if not gpu["available"] else [])
                    + ["Current full runner is single-process/single-GPU; no isolated two-GPU shard/resume orchestration is implemented."],
    }
    (output / "selected_chunks.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in detail_rows), encoding="utf-8")
    (output / "btc_pilot_payload.json").write_bytes(payload + b"\n")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["status"] = "NO-GO" if report["blockers"] else report["status"]
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4_submission_p2_a0.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v4/p5_1_a0_submission_preflight_20261022")
    args = parser.parse_args()
    print(json.dumps(main(args.config, args.output), ensure_ascii=False, indent=2))
