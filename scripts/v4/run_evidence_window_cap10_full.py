"""Build, score, and export the single full cap-10 evidence-window candidate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from scripts.v4 import run_evidence_window_allocation as base
from scripts.v4 import run_evidence_window_cap10_validation as cap10
from src.medrag.adapters.validation import canonical_payload, package_zip, validate_payload, validate_zip
from src.medrag.core.model_revision import require_cached_revision


ROOT = base.ROOT
PILOT = ROOT / "outputs/v4/evidence_window_allocation_20261025"
SCREEN = ROOT / "outputs/v4/evidence_window_cap10_validation_20261025"
OUTPUT = ROOT / "outputs/v4/evidence_window_cap10_full_20261025"
QUERY_IDS = list(range(1, 1201))
WORKERS = 2
ANCHOR_TOLERANCE = 1e-3


def pilot_stage_map() -> dict[int, str]:
    lock = base.read(PILOT / "lock.json")
    return {int(qid): stage for stage, qids in lock["cohort"].items() for qid in qids}


def pilot_score_path(qid: int, stage_map: dict[int, str] | None = None) -> Path | None:
    stage = (stage_map or pilot_stage_map()).get(qid)
    if stage is None:
        return None
    root = PILOT if stage == "development" else SCREEN
    return root / "scores" / stage / f"{qid}.json"


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    keep = ("window_id", "official_id", "context_id", "kind", "start", "end",
            "text_sha256", "payload", "input_tokens", "body_token_start",
            "body_token_end", "requested_body_token_start",
            "requested_body_token_end", "query_id", "query_sha256", "input_sha256")
    return {
        **{key: value for key, value in pair.items() if key != "windows"},
        "format": "v4-evidence-window-cap10-full-pairs-query-v1",
        "stage": "full",
        "role": "locked_pilot_reuse" if pair["id"] in pilot_stage_map() else "full_candidate",
        "windows": [{key: row[key] for key in keep} for row in pair["windows"]],
    }


def package_input_identity() -> dict[str, str]:
    implementation = Path(__file__).resolve()
    return {
        "s2_submission": base.sha256_file(base.S2_ZIP),
        "contexts_manifest": base.sha256_file(base.CONTEXTS / "manifest.json"),
        "refined_manifest": base.sha256_file(base.REFINED / "run_manifest.json"),
        "runtime_lock": base.sha256_file(base.RUNTIME),
        "pilot_lock": base.sha256_file(PILOT / "lock.json"),
        "pilot_manifest": base.sha256_file(PILOT / "pair_manifest.json"),
        "screen_lock": base.sha256_file(SCREEN / "lock.json"),
        "screen_held_out_report": base.sha256_file(SCREEN / "reports/held_out.json"),
        "base_implementation": base.sha256_file(Path(base.__file__).resolve()),
        "cap10_implementation": base.sha256_file(Path(cap10.__file__).resolve()),
        "implementation": base.sha256_file(implementation),
    }


def materialize_pair(qid: int, s2: dict[int, dict[str, Any]], queries: dict[int, str],
                     tokenizer, store, context_manifest: dict, stage_map: dict[int, str]) -> dict:
    if qid in stage_map:
        pair = base.read(PILOT / "pairs" / f"{qid}.json")
        if pair["query_sha256"] != base.digest_text(queries[qid]):
            raise ValueError(f"Pilot query identity differs: {qid}")
        return compact_pair(pair)
    path = base.CONTEXTS / "queries" / f"{qid}.json"
    if base.sha256_file(path) != context_manifest["query_sha256"][str(qid)]:
        raise ValueError(f"Frozen context candidate changed: {qid}")
    candidate = base.read(path)
    if candidate["query_sha256"] != base.digest_text(queries[qid]):
        raise ValueError(f"Official query/context identity differs: {qid}")
    fixed = base.choose_contexts(qid, s2, candidate)
    sources = store.sources_batch({int(row["official_id"]) for row in fixed})
    windows = []
    contexts = []
    for context in fixed:
        doc = int(context["official_id"])
        source = sources[doc]
        if (context["source_sha256"] != source.text_sha256
                or source.text[context["start"]:context["end"]] != context["text"]):
            raise ValueError(f"Fixed context grounding differs: {qid}/{doc}")
        encoded = tokenizer(context["text"], add_special_tokens=False,
                            return_offsets_mapping=True)
        offsets = [tuple(value) for value in encoded["offset_mapping"]]
        local = [base.fit_window(tokenizer, queries[qid], source.title, source.text, context,
                                 kind, start, end, offsets)
                 for kind, start, end in base.window_ranges(len(offsets))]
        windows.extend(local)
        contexts.append({key: context[key] for key in (
            "official_id", "document_rank", "context_id", "text", "text_sha256",
            "start", "end", "source_sha256", "context_policy", "baseline_chunk")}
            | {"source_title": source.title, "body_tokens": len(offsets),
               "window_ids": [row["window_id"] for row in local]})
    pair = {
        "format": "v4-evidence-window-pairs-query-v1", "id": qid,
        "stage": "full", "role": "full_candidate", "query": queries[qid],
        "query_sha256": base.digest_text(queries[qid]),
        "s2_documents": list(map(int, s2[qid]["relevant_docs"])),
        "s2_chunks": s2[qid]["relevant_chunks"], "contexts": contexts,
        "windows": [{**row, "query_id": qid,
                     "query_sha256": base.digest_text(queries[qid]),
                     "input_sha256": base.canonical_digest([queries[qid], row["payload"]])}
                    for row in windows],
    }
    return compact_pair(pair)


def package(output: Path) -> dict[str, Any]:
    inputs = package_input_identity()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "package_inprogress.json"
    progress = {"format": "v4-evidence-window-cap10-full-package-progress-v1",
                "inputs_sha256": inputs}
    if progress_path.exists():
        if base.read(progress_path) != progress:
            raise ValueError("Full package resume inputs/implementation differ")
    else:
        base.write_once(progress_path, progress)
    if (output / "lock.json").exists() or (output / "pair_manifest.json").exists():
        raise FileExistsError("Full package is already finalized")
    config, store, policy, _ = base.environment()
    if policy.max_chunks != 60 or policy.chunks_per_doc != 1:
        raise ValueError("Frozen S2 chunk policy differs")
    s2 = base.load_s2()
    queries = base.load_queries()
    if set(s2) != set(QUERY_IDS) or set(queries) != set(QUERY_IDS):
        raise ValueError("Official/S2 query coverage differs")
    context_manifest = base.read(base.CONTEXTS / "manifest.json")
    runtime = base.read(base.RUNTIME)
    snapshot = require_cached_revision(runtime["score_contract"]["model"],
                                       runtime["score_contract"]["revision"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Offset-grounded windows require a fast tokenizer")
    stage_map = pilot_stage_map()
    hashes = {}
    pair_counts = {}
    for index, qid in enumerate(QUERY_IDS, 1):
        destination = output / "pairs" / f"{qid}.json"
        if destination.exists():
            pair = base.read(destination)
            if (pair.get("format") != "v4-evidence-window-cap10-full-pairs-query-v1"
                    or pair.get("id") != qid
                    or pair.get("query_sha256") != base.digest_text(queries[qid])):
                raise ValueError(f"Full pair resume shard differs: {qid}")
        else:
            pair = materialize_pair(qid, s2, queries, tokenizer, store,
                                    context_manifest, stage_map)
            base.write_once(destination, pair)
        hashes[str(qid)] = base.sha256_file(destination)
        pair_counts[str(qid)] = len(pair["windows"])
        if index % 100 == 0:
            print(json.dumps({"packaged": index, "pairs": sum(pair_counts.values())}), flush=True)
    pilot_pairs = sum(pair_counts[str(qid)] for qid in stage_map)
    lock = {
        "format": "v4-evidence-window-cap10-full-lock-v1", "status": "LOCKED",
        "query_ids": QUERY_IDS, "new_query_ids": [qid for qid in QUERY_IDS if qid not in stage_map],
        "reused_pilot_query_ids": sorted(stage_map), "workers": WORKERS,
        "policy": {**base.read(SCREEN / "lock.json")["policy"],
                   "scope": "full 1200-query candidate"},
        "score_contract": runtime["score_contract"], "versions": runtime["versions"],
        "runtime_settings": {key: runtime[key] for key in (
            "cuda", "gpu_name", "dtype", "activation", "batch_size", "num_threads",
            "num_interop_threads", "matmul_allow_tf32", "cudnn_allow_tf32",
            "model_snapshot_files_sha256")},
        "inputs_sha256": inputs,
        "pair_counts": {"total": sum(pair_counts.values()), "pilot_reused": pilot_pairs,
                        "new": sum(pair_counts.values()) - pilot_pairs},
    }
    base.write_once(output / "lock.json", lock)
    manifest = {
        "format": "v4-evidence-window-cap10-full-pair-manifest-v1",
        "status": "READY_FOR_TWO_GPU", "lock_sha256": base.sha256_file(output / "lock.json"),
        "query_count": len(QUERY_IDS), "pair_shards_sha256": hashes,
        "pair_counts": pair_counts, "total_pairs": sum(pair_counts.values()),
    }
    base.write_once(output / "pair_manifest.json", manifest)
    base.write_once(output / "anchors.json", base.read(PILOT / "anchors.json"))
    return manifest


def validate_package(output: Path) -> tuple[dict, dict]:
    lock = base.read(output / "lock.json")
    manifest = base.read(output / "pair_manifest.json")
    if (lock.get("format") != "v4-evidence-window-cap10-full-lock-v1"
            or lock["inputs_sha256"] != package_input_identity()
            or manifest.get("lock_sha256") != base.sha256_file(output / "lock.json")
            or set(manifest["pair_shards_sha256"]) != {str(qid) for qid in QUERY_IDS}):
        raise ValueError("Full cap-10 package lineage/coverage differs")
    for qid, expected in manifest["pair_shards_sha256"].items():
        if base.sha256_file(output / "pairs" / f"{qid}.json") != expected:
            raise ValueError(f"Full pair shard changed: {qid}")
    return lock, manifest


def worker_query_ids(lock: dict, worker: int, workers: int) -> list[int]:
    if workers != lock["workers"] or worker < 0 or worker >= workers:
        raise ValueError("Worker topology differs from lock")
    return [qid for index, qid in enumerate(lock["new_query_ids"]) if index % workers == worker]


def score_anchor_parity(output: Path, worker: int, model, runtime_sha: str) -> dict:
    destination = output / f"anchor_parity_worker{worker}.json"
    if destination.exists():
        result = base.read(destination)
        if result.get("status") != "PASS" or result.get("runtime_sha256") != runtime_sha:
            raise ValueError("Existing worker anchor parity differs")
        return result
    grouped = defaultdict(list)
    for row in base.read(output / "anchors.json"):
        grouped[row["scoring_batch_id"]].append(row)
    differences = []
    threshold_ok = True
    for local in grouped.values():
        local.sort(key=lambda row: row["scoring_batch_position"])
        values = model.score(local[0]["query"], [row["payload"] for row in local])
        for row, value in zip(local, values, strict=True):
            differences.append(abs(float(value) - float(row["expected_score"])))
            threshold_ok = threshold_ok and all(
                (float(value) >= threshold) == (float(row["expected_score"]) >= threshold)
                for threshold in row["threshold_checks"])
    maximum = max(differences)
    result = {"format": "v4-evidence-window-cap10-full-anchor-parity-v1",
              "worker": worker, "runtime_sha256": runtime_sha, "anchors": len(differences),
              "maximum_absolute_difference": maximum,
              "status": "PASS" if maximum <= ANCHOR_TOLERANCE and threshold_ok else "STOP"}
    base.write_once(destination, result)
    if result["status"] != "PASS":
        raise ValueError("Worker exact-batch parity failed")
    return result


def score_worker(output: Path, worker: int, workers: int) -> dict[str, Any]:
    lock, _ = validate_package(output)
    torch, _, parent_runtime = base.runtime_preflight(PILOT)
    runtime = {"format": "v4-evidence-window-cap10-full-runtime-v1", "worker": worker,
               "workers": workers, "lock_sha256": base.sha256_file(output / "lock.json"),
               "gpu_uuid_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "parent_runtime": parent_runtime,
               "implementation_sha256": base.sha256_file(Path(__file__).resolve())}
    runtime_path = output / "runtime" / f"worker{worker}.json"
    if runtime_path.exists():
        if base.read(runtime_path) != runtime:
            raise ValueError("Worker runtime resume differs")
    else:
        base.write_once(runtime_path, runtime)
    runtime_sha = base.sha256_file(runtime_path)
    from src.medrag.ranking.reranker import BGEReranker

    settings = lock["score_contract"]
    model = BGEReranker(settings["model"], revision=settings["revision"],
                        max_length=settings["max_length"], device="cuda:0")
    if next(model.model.model.parameters()).dtype != torch.float16:
        raise ValueError("Reranker is not FP16")
    score_anchor_parity(output, worker, model, runtime_sha)
    assigned = worker_query_ids(lock, worker, workers)
    hashes = {}
    pairs = 0
    seconds = 0.0
    for index, qid in enumerate(assigned, 1):
        pair_path = output / "pairs" / f"{qid}.json"
        destination = output / "scores" / "new" / f"{qid}.json"
        pair = base.read(pair_path)
        if destination.exists():
            result = base.read(destination)
            if (result["runtime_sha256"] != runtime_sha
                    or result["pair_sha256"] != base.sha256_file(pair_path)):
                raise ValueError(f"Worker score resume differs: {qid}")
        else:
            torch.cuda.synchronize()
            started = time.monotonic()
            values = model.score(pair["query"], [row["payload"] for row in pair["windows"]])
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            if len(values) != len(pair["windows"]) or any(not math.isfinite(v) for v in values):
                raise ValueError(f"Invalid full scores: {qid}")
            result = {"format": "v4-evidence-window-cap10-full-scores-query-v1",
                      "id": qid, "worker": worker, "runtime_sha256": runtime_sha,
                      "pair_sha256": base.sha256_file(pair_path), "seconds": elapsed,
                      "scores": [{"window_id": row["window_id"],
                                  "input_sha256": row["input_sha256"], "score": float(value)}
                                 for row, value in zip(pair["windows"], values, strict=True)]}
            base.write_once(destination, result)
        hashes[str(qid)] = base.sha256_file(destination)
        pairs += len(result["scores"])
        seconds += float(result["seconds"])
        if index % 50 == 0:
            print(json.dumps({"worker": worker, "queries": index, "total": len(assigned),
                              "pairs": pairs, "model_seconds": seconds}), flush=True)
    result = {"format": "v4-evidence-window-cap10-full-worker-manifest-v1",
              "status": "SCORED", "worker": worker, "workers": workers,
              "queries": len(assigned), "pairs": pairs, "model_seconds": seconds,
              "pairs_per_second": pairs / seconds, "runtime_sha256": runtime_sha,
              "score_sha256": hashes}
    base.write_once(output / "scores" / f"worker{worker}_manifest.json", result)
    return result


def finalize_scores(output: Path) -> dict[str, Any]:
    lock, manifest = validate_package(output)
    coverage = {}
    pairs = 0
    seconds = 0.0
    workers = []
    for worker in range(lock["workers"]):
        report = base.read(output / "scores" / f"worker{worker}_manifest.json")
        expected = {str(qid) for qid in worker_query_ids(lock, worker, lock["workers"])}
        if report.get("status") != "SCORED" or set(report["score_sha256"]) != expected:
            raise ValueError(f"Worker score coverage differs: {worker}")
        if set(coverage) & expected:
            raise ValueError("Worker score overlap")
        for qid, sha in report["score_sha256"].items():
            if base.sha256_file(output / "scores/new" / f"{qid}.json") != sha:
                raise ValueError(f"Worker score changed: {qid}")
        coverage.update({qid: {"kind": "new", "sha256": sha}
                         for qid, sha in report["score_sha256"].items()})
        pairs += report["pairs"]
        seconds += report["model_seconds"]
        workers.append(report)
    stage_map = pilot_stage_map()
    for qid in sorted(stage_map):
        path = pilot_score_path(qid, stage_map)
        score = base.read(path)
        if len(score["scores"]) != int(manifest["pair_counts"][str(qid)]):
            raise ValueError(f"Pilot score/pair coverage differs: {qid}")
        coverage[str(qid)] = {"kind": "pilot_reuse", "source": str(path.relative_to(ROOT)),
                              "sha256": base.sha256_file(path)}
        pairs += len(score["scores"])
    if set(coverage) != {str(qid) for qid in QUERY_IDS} or pairs != manifest["total_pairs"]:
        raise ValueError("Final full score coverage differs")
    result = {"format": "v4-evidence-window-cap10-full-score-manifest-v1",
              "status": "SCORED", "queries": len(coverage), "pairs": pairs,
              "new_model_seconds_sum_two_gpus": seconds, "coverage": coverage,
              "worker_manifests_sha256": {
                  str(worker): base.sha256_file(output / "scores" / f"worker{worker}_manifest.json")
                  for worker in range(lock["workers"])} }
    base.write_once(output / "score_manifest.json", result)
    return result


def resolved_score(output: Path, qid: int, coverage: dict) -> dict:
    row = coverage[str(qid)]
    path = (output / "scores/new" / f"{qid}.json" if row["kind"] == "new"
            else ROOT / row["source"])
    if base.sha256_file(path) != row["sha256"]:
        raise ValueError(f"Resolved score changed: {qid}")
    return base.read(path)


def export(output: Path) -> dict[str, Any]:
    lock, _ = validate_package(output)
    score_manifest = base.read(output / "score_manifest.json")
    if score_manifest.get("status") != "SCORED":
        raise ValueError("Full scores are not finalized")
    s2 = base.load_s2()
    labels = base.group_labels()
    _, _, policy, _ = base.environment()
    manual_gain: set[str] = set()
    manual_loss: set[str] = set()
    historical_gain: set[str] = set()
    historical_loss: set[str] = set()
    predictions = []
    changed_queries = 0
    swaps_total = 0
    per_query = []
    locked_challengers = {}
    for stage in ("validation", "held_out"):
        for row in base.read(SCREEN / "predictions" / f"{stage}_challenger.json"):
            locked_challengers[int(row["id"])] = row
    for index, qid in enumerate(QUERY_IDS, 1):
        pair = base.read(output / "pairs" / f"{qid}.json")
        score = resolved_score(output, qid, score_manifest["coverage"])
        selected, _, swaps = cap10.allocate(pair, score)
        contexts = {int(row["official_id"]): row for row in pair["contexts"]}
        before = {int(row["doc_id"]): row["chunk_text"] for row in pair["s2_chunks"]}
        after = {doc: contexts[doc]["text"] for doc in selected}
        prediction = {"id": qid, "relevant_docs": pair["s2_documents"],
                      "relevant_chunks": [{"doc_id": doc, "chunk_text": after[doc]}
                                          for doc in selected]}
        if prediction["relevant_docs"] != s2[qid]["relevant_docs"]:
            raise ValueError(f"Full candidate document list differs from S2: {qid}")
        validate_payload(canonical_payload([prediction]), query_ids={qid},
                         official_ids=set(pair["s2_documents"]), policy=policy,
                         result_limit=200)
        if qid in locked_challengers and prediction != locked_challengers[qid]:
            raise ValueError(f"Full candidate does not replay locked screened output: {qid}")
        manual_before = base.parent_hits(qid, before, labels, "manual_positive")
        manual_after = base.parent_hits(qid, after, labels, "manual_positive")
        historical_before = base.parent_hits(qid, before, labels, "historical_positive")
        historical_after = base.parent_hits(qid, after, labels, "historical_positive")
        manual_gain |= manual_after - manual_before
        manual_loss |= manual_before - manual_after
        historical_gain |= historical_after - historical_before
        historical_loss |= historical_before - historical_after
        predictions.append(prediction)
        changed_queries += bool(swaps)
        swaps_total += len(swaps)
        per_query.append({"id": qid, "documents": len(prediction["relevant_docs"]),
                          "chunks": len(prediction["relevant_chunks"]), "swaps": len(swaps)})
        if index % 100 == 0:
            print(json.dumps({"exported": index, "swaps": swaps_total}), flush=True)
    payload = canonical_payload(predictions)
    export_dir = output / "export"
    export_dir.mkdir(exist_ok=True)
    predictions_path = export_dir / "predictions.json"
    if predictions_path.exists():
        if predictions_path.read_bytes() != payload:
            raise ValueError("Existing full predictions differ")
    else:
        predictions_path.write_bytes(payload)
    zip_path = export_dir / "submission.zip"
    if not zip_path.exists():
        package_zip(zip_path, payload)
    official_ids = {doc for row in predictions for doc in row["relevant_docs"]}
    zip_report = validate_zip(zip_path, query_ids=set(QUERY_IDS), official_ids=official_ids,
                              policy=policy, result_limit=200, max_bytes=100_000_000)
    if zip_report["payload_sha256"] != base.sha256_file(predictions_path):
        raise ValueError("Full ZIP payload differs from predictions")
    report = {
        "format": "v4-evidence-window-cap10-full-export-v1",
        "status": "FULL_CANDIDATE_READY_FOR_BTC", "official_f2": "UNMEASURED",
        "quality_claim": "S2 remains champion until official BTC evaluation",
        "queries": len(predictions), "documents": sum(len(row["relevant_docs"]) for row in predictions),
        "chunks": sum(len(row["relevant_chunks"]) for row in predictions),
        "changed_queries": changed_queries, "swaps": swaps_total,
        "documents_exact_s2": True,
        "manual": {"gains": len(manual_gain), "losses": len(manual_loss),
                   "gained_ids": sorted(manual_gain), "lost_ids": sorted(manual_loss)},
        "historical_preservation": {"gains": len(historical_gain),
                                    "losses": len(historical_loss),
                                    "gained_ids": sorted(historical_gain),
                                    "lost_ids": sorted(historical_loss)},
        "zip": zip_report, "predictions_sha256": base.sha256_file(predictions_path),
        "score_manifest_sha256": base.sha256_file(output / "score_manifest.json"),
        "per_query_sha256": None,
    }
    base.write_once(output / "per_query.json", per_query)
    report["per_query_sha256"] = base.sha256_file(output / "per_query.json")
    base.write_once(output / "run_manifest.json", report)
    base.write_atomic_replace(output / "status.json", {"status": report["status"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "validate-package", "score-worker",
                                            "finalize-scores", "export"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.command == "package":
        result = package(args.output)
    elif args.command == "validate-package":
        lock, manifest = validate_package(args.output)
        result = {"status": "PASS", "queries": len(lock["query_ids"]),
                  "pairs": manifest["total_pairs"], "new_queries": len(lock["new_query_ids"])}
    elif args.command == "score-worker":
        if args.worker is None:
            parser.error("--worker is required for score-worker")
        result = score_worker(args.output, args.worker, args.workers)
    elif args.command == "finalize-scores":
        result = finalize_scores(args.output)
    else:
        result = export(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
