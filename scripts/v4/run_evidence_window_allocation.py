"""Run a bounded evidence-window allocator while preserving S2 document lists."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import zipfile

import pyarrow.parquet as pq

from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.run_r7_parent_floor_score_gate import exact_batch_anchors
from src.medrag.adapters.validation import canonical_payload, validate_payload
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import normalize
from src.medrag.evidence.refinement import filter_documents, select_documents_by_confidence


OUTPUT = ROOT / "outputs/v4/evidence_window_allocation_20261025"
S2_ZIP = ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip"
CONTEXTS = ROOT / "outputs/v4/r5_s2_context_candidates_20261024"
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024"
SPLIT = ROOT / "outputs/v4/evidence_evaluation_lock_20261025/query_split_registry.json"
REGISTRY = ROOT / "outputs/v4/r6_stage_oracle_20261025/positive_registry.parquet"
RUNTIME = ROOT / "outputs/v4/p5_gpu_incremental_a6000_20261023/runtime_lock.json"
R7 = ROOT / "outputs/v4/r7_parent_floor_score_20261025_v2"
SEED = "evidence-window-allocation-20261025-v1"
STAGE_COUNTS = {"development": 30, "validation": 20, "held_out": 10}
STAGE_PAIR_LIMITS = {"development": 12_600, "validation": 8_400, "held_out": 4_200}
STAGE_REVIEW_LIMITS = {"development": 18, "validation": 12, "held_out": 6}
WINDOW_BODY_TOKENS = 192
MAX_SWAPS = 6
ANCHOR_TOLERANCE = 1e-3


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_key(qid: int) -> str:
    return digest_text(f"{SEED}:{qid}")


def load_s2() -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(S2_ZIP) as archive:
        rows = json.loads(archive.read("predictions.json"))
    return {int(row["id"]): row for row in rows}


def load_queries() -> dict[int, str]:
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    path = ROOT / migration["official_queries"]
    return {int(row["id"]): str(row["query"]) for row in
            pq.read_table(path, columns=["id", "query"]).to_pylist()}


def group_labels() -> dict[tuple[int, int, str], list[dict[str, Any]]]:
    result: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(REGISTRY).to_pylist():
        if row["label_group"] in {"manual_positive", "historical_positive"}:
            result[(int(row["query_id"]), int(row["doc_id"]), row["label_group"])].append(row)
    return result


def visible(text: str, labels: list[dict[str, Any]]) -> bool:
    normalized = normalize(text)
    return any(normalize(row["evidence_text"]) in normalized for row in labels)


def choose_cohort(s2: dict[int, dict[str, Any]], assignments: dict[str, str],
                  labels: dict[tuple[int, int, str], list[dict[str, Any]]]) -> tuple[dict, dict]:
    manual_queries: dict[str, set[int]] = defaultdict(set)
    outsider_queries: dict[str, set[int]] = defaultdict(set)
    visible_queries: dict[str, set[int]] = defaultdict(set)
    for (qid, doc, group), rows in labels.items():
        if group != "manual_positive":
            continue
        stage = assignments[str(qid)]
        manual_queries[stage].add(qid)
        docs = set(map(int, s2[qid]["relevant_docs"]))
        chunks = {int(row["doc_id"]): row["chunk_text"] for row in s2[qid]["relevant_chunks"]}
        if doc in docs and doc not in chunks:
            outsider_queries[stage].add(qid)
        if doc in chunks and visible(chunks[doc], rows):
            visible_queries[stage].add(qid)
    expected = {
        "development": {5, 65, 130, 353, 398, 566, 579, 594, 747, 910, 1150},
        "validation": {71}, "held_out": set(),
    }
    expected_manual_counts = {"development": 63, "validation": 14, "held_out": 10}
    actual_manual_counts = {stage: len(values) for stage, values in manual_queries.items()}
    if any(actual_manual_counts.get(stage, 0) != count
           for stage, count in expected_manual_counts.items()):
        raise ValueError("Manual-label query support differs from the locked plan")
    if any(outsider_queries[stage] != values for stage, values in expected.items()):
        raise ValueError("Manual outsider query identities differ from the locked plan")
    cohort, roles = {}, {}
    dev_out = sorted(outsider_queries["development"])
    dev_controls = sorted(visible_queries["development"] - set(dev_out), key=stable_key)[:10]
    dev_used = set(dev_out) | set(dev_controls)
    dev_fill = sorted([qid for qid in s2 if assignments[str(qid)] == "development"
                       and qid not in dev_used and len(s2[qid]["relevant_docs"]) > 60],
                      key=stable_key)[:9]
    cohort["development"] = dev_out + dev_controls + dev_fill
    roles.update({qid: "manual_outsider" for qid in dev_out})
    roles.update({qid: "manual_visible_control" for qid in dev_controls})
    roles.update({qid: "deterministic_fill" for qid in dev_fill})
    for stage in ("validation", "held_out"):
        manual = sorted(manual_queries[stage], key=stable_key)
        needed = STAGE_COUNTS[stage] - len(manual)
        fill = sorted([qid for qid in s2 if assignments[str(qid)] == stage
                       and qid not in set(manual) and len(s2[qid]["relevant_docs"]) > 60],
                      key=stable_key)[:needed]
        cohort[stage] = manual + fill
        roles.update({qid: "manual_query" for qid in manual})
        roles.update({qid: "deterministic_fill" for qid in fill})
    if any(len(cohort[stage]) != count or len(set(cohort[stage])) != count
           for stage, count in STAGE_COUNTS.items()):
        raise ValueError("Could not lock the planned cohort sizes")
    if (set(cohort["development"]) & set(cohort["validation"])
            or set(cohort["development"]) & set(cohort["held_out"])
            or set(cohort["validation"]) & set(cohort["held_out"])):
        raise ValueError("Cohort stages overlap")
    support = {
        stage: {"manual_queries": len(manual_queries[stage] & set(qids)),
                "manual_outsider_queries": len(outsider_queries[stage] & set(qids)),
                "manual_visible_control_queries": len(visible_queries[stage] & set(qids))}
        for stage, qids in cohort.items()
    }
    return cohort, {"roles": {str(qid): roles[qid] for qids in cohort.values() for qid in qids},
                    "support": support}


def choose_contexts(qid: int, s2: dict[int, dict[str, Any]],
                    candidate: dict[str, Any]) -> list[dict[str, Any]]:
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate["contexts"]:
        by_parent[int(row["official_id"])].append(row)
    baseline = {int(row["doc_id"]): row["chunk_text"] for row in s2[qid]["relevant_chunks"]}
    selected = []
    for rank, doc in enumerate(map(int, s2[qid]["relevant_docs"]), 1):
        rows = by_parent.get(doc, [])
        if not rows:
            raise ValueError(f"S2 document lacks context candidate: {qid}/{doc}")
        if doc in baseline:
            matches = [row for row in rows if row["text"] == baseline[doc]]
            if len(matches) != 1:
                raise ValueError(f"Exact S2 chunk context missing/duplicated: {qid}/{doc}")
            chosen, policy = matches[0], "exact_s2_chunk"
        else:
            chosen = min(rows, key=lambda row: (
                -max(float(seed["baseline_rerank_score"]) for seed in row["seed_passages"]),
                row["context_id"]))
            policy = "highest_seed_logit_then_context_id"
        selected.append({**chosen, "document_rank": rank, "context_policy": policy,
                         "baseline_chunk": doc in baseline})
    return selected


def window_ranges(token_count: int) -> list[tuple[str, int, int]]:
    if token_count <= 0:
        raise ValueError("Context tokenizer returned no body tokens")
    if token_count <= WINDOW_BODY_TOKENS:
        return [("full", 0, token_count)]
    width = WINDOW_BODY_TOKENS
    raw = [("head", 0, width), ("middle", (token_count - width) // 2,
                                (token_count - width) // 2 + width),
           ("tail", token_count - width, token_count)]
    seen, output = set(), []
    for kind, start, end in raw:
        if (start, end) not in seen:
            seen.add((start, end))
            output.append((kind, start, end))
    return output


def fit_window(tokenizer, query: str, title: str, source_text: str, context: dict[str, Any],
               kind: str, start: int, end: int, offsets: list[tuple[int, int]]) -> dict[str, Any]:
    original_start, original_end = start, end
    while True:
        char_start = context["start"] + offsets[start][0]
        char_end = context["start"] + offsets[end - 1][1]
        text = source_text[char_start:char_end]
        if not text or text != context["text"][offsets[start][0]:offsets[end - 1][1]]:
            raise ValueError("Window offsets are not grounded in source/context")
        payload = f"{title}\n{text}" if title else text
        pair_tokens = len(tokenizer(query, payload, add_special_tokens=True)["input_ids"])
        if pair_tokens <= 512:
            break
        if end - start <= 16:
            raise ValueError("Query/title leave no valid evidence body window")
        remove = min(end - start - 16, max(1, pair_tokens - 512))
        if kind == "head":
            end -= remove
        elif kind == "tail":
            start += remove
        else:
            left = remove // 2
            start += left
            end -= remove - left
    identity = {"official_id": int(context["official_id"]),
                "context_id": context["context_id"], "kind": kind,
                "start": char_start, "end": char_end, "text_sha256": digest_text(text)}
    return {**identity, "window_id": canonical_digest(identity), "text": text, "payload": payload,
            "input_tokens": pair_tokens, "body_token_start": start, "body_token_end": end,
            "requested_body_token_start": original_start,
            "requested_body_token_end": original_end}


def package(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config, store, policy, _ = environment()
    if policy.max_chunks != 60 or policy.chunks_per_doc != 1:
        raise ValueError("Frozen S2 chunk policy differs")
    s2, queries = load_s2(), load_queries()
    split = read(SPLIT)
    assignments = split["assignments"]
    labels = group_labels()
    cohort, cohort_meta = choose_cohort(s2, assignments, labels)
    context_manifest = read(CONTEXTS / "manifest.json")
    refined_manifest = read(REFINED / "run_manifest.json")
    if (context_manifest.get("status") != "PASS" or context_manifest.get("queries") != 1200
            or refined_manifest.get("status") != "PASS"):
        raise ValueError("Frozen context/refined manifests are incomplete")
    runtime = read(RUNTIME)
    snapshot = require_cached_revision(runtime["score_contract"]["model"],
                                       runtime["score_contract"]["revision"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Offset-grounded windows require a fast tokenizer")
    hashes, counts, all_ids = {}, defaultdict(int), [qid for stage in cohort.values() for qid in stage]
    for qid in all_ids:
        path = CONTEXTS / "queries" / f"{qid}.json"
        if sha256_file(path) != context_manifest["query_sha256"][str(qid)]:
            raise ValueError(f"Frozen context candidate changed: {qid}")
        candidate = read(path)
        if candidate["query_sha256"] != digest_text(queries[qid]):
            raise ValueError(f"Official query/context identity differs: {qid}")
        fixed = choose_contexts(qid, s2, candidate)
        sources = store.sources_batch({int(row["official_id"]) for row in fixed})
        windows, context_rows = [], []
        for context in fixed:
            doc = int(context["official_id"])
            source = sources[doc]
            if (context["source_sha256"] != source.text_sha256
                    or source.text[context["start"]:context["end"]] != context["text"]):
                raise ValueError(f"Fixed context grounding differs: {qid}/{doc}")
            encoded = tokenizer(context["text"], add_special_tokens=False,
                                return_offsets_mapping=True)
            offsets = [tuple(value) for value in encoded["offset_mapping"]]
            local = [fit_window(tokenizer, queries[qid], source.title, source.text, context,
                                kind, start, end, offsets)
                     for kind, start, end in window_ranges(len(offsets))]
            if len({row["window_id"] for row in local}) != len(local):
                raise ValueError(f"Duplicate evidence window identity: {qid}/{doc}")
            windows.extend(local)
            context_rows.append({key: context[key] for key in (
                "official_id", "document_rank", "context_id", "text", "text_sha256",
                "start", "end", "source_sha256", "context_policy", "baseline_chunk")}
                | {"source_title": source.title, "body_tokens": len(offsets),
                   "window_ids": [row["window_id"] for row in local]})
        shard = {"format": "v4-evidence-window-pairs-query-v1", "id": qid,
                 "stage": assignments[str(qid)], "role": cohort_meta["roles"][str(qid)],
                 "query": queries[qid], "query_sha256": digest_text(queries[qid]),
                 "s2_documents": list(map(int, s2[qid]["relevant_docs"])),
                 "s2_chunks": s2[qid]["relevant_chunks"], "contexts": context_rows,
                 "windows": [{**row, "query_id": qid,
                              "query_sha256": digest_text(queries[qid]),
                              "input_sha256": canonical_digest(
                                  [queries[qid], row["payload"]])} for row in windows]}
        shard_path = output / "pairs" / f"{qid}.json"
        write_once(shard_path, shard)
        hashes[str(qid)] = sha256_file(shard_path)
        counts[assignments[str(qid)]] += len(windows)
    anchors, anchor_sources = exact_batch_anchors(all_ids, queries, s2)
    write_once(output / "anchors.json", anchors)
    lock = {
        "format": "v4-evidence-window-allocation-lock-v1", "status": "LOCKED",
        "policy": {"candidate_parents": "exact S2 relevant_docs",
                   "baseline_context": "exact S2 chunk text",
                   "outsider_context": "highest seed logit then context_id",
                   "window_body_tokens": WINDOW_BODY_TOKENS,
                   "window_anchors": ["head", "middle", "tail"],
                   "parent_evidence_score": "max raw logit over windows",
                   "max_swaps_per_query": MAX_SWAPS,
                   "export_context": "fixed full context; scoring windows stay private"},
        "cohort": cohort, **cohort_meta,
        "budgets": {"pair_limits": STAGE_PAIR_LIMITS, "actual_pairs": dict(counts),
                    "gpu_minutes_per_stage": 15, "cpu_hours_total": 2},
        "score_contract": runtime["score_contract"], "versions": runtime["versions"],
        "runtime_settings": {key: runtime[key] for key in
                             ("cuda", "gpu_name", "dtype", "activation", "batch_size",
                              "num_threads", "num_interop_threads", "matmul_allow_tf32",
                              "cudnn_allow_tf32", "model_snapshot_files_sha256")},
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                          (S2_ZIP, SPLIT, REGISTRY, CONTEXTS / "manifest.json",
                           REFINED / "run_manifest.json", RUNTIME, Path(__file__).resolve())},
        "anchor_source_sha256": anchor_sources,
    }
    if any(counts[stage] > STAGE_PAIR_LIMITS[stage] for stage in STAGE_COUNTS):
        raise ValueError(f"Materialized pair budget exceeded: {dict(counts)}")
    write_once(output / "lock.json", lock)
    manifest = {"format": "v4-evidence-window-allocation-pair-manifest-v1",
                "status": "READY_FOR_GPU", "lock_sha256": sha256_file(output / "lock.json"),
                "pair_shards_sha256": hashes, "anchors_sha256": sha256_file(output / "anchors.json"),
                "query_count": len(all_ids), "stage_query_counts": STAGE_COUNTS,
                "stage_pair_counts": dict(counts), "total_pairs": sum(counts.values())}
    write_once(output / "pair_manifest.json", manifest)
    return manifest


def validate_package(output: Path) -> tuple[dict, dict]:
    lock, manifest = read(output / "lock.json"), read(output / "pair_manifest.json")
    if (lock.get("format") != "v4-evidence-window-allocation-lock-v1"
            or lock.get("inputs_sha256", {}).get(str(Path(__file__).resolve().relative_to(ROOT)))
            != sha256_file(Path(__file__).resolve())
            or manifest.get("lock_sha256") != sha256_file(output / "lock.json")
            or manifest.get("anchors_sha256") != sha256_file(output / "anchors.json")):
        raise ValueError("Evidence-window package lock/implementation differs")
    expected = {str(qid) for ids in lock["cohort"].values() for qid in ids}
    if set(manifest["pair_shards_sha256"]) != expected:
        raise ValueError("Evidence-window pair shard coverage differs")
    for qid, expected_hash in manifest["pair_shards_sha256"].items():
        if sha256_file(output / "pairs" / f"{qid}.json") != expected_hash:
            raise ValueError(f"Evidence-window pair shard changed: {qid}")
    return lock, manifest


def runtime_preflight(output: Path):
    lock, _ = validate_package(output)
    import torch
    actual = {name: importlib.metadata.version(name) for name in lock["versions"]}
    expected = lock["runtime_settings"]
    if (actual != lock["versions"] or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or torch.version.cuda != expected["cuda"]
            or torch.cuda.get_device_name(0) != expected["gpu_name"]):
        raise ValueError("Pinned single-A6000 runtime differs")
    torch.set_num_threads(expected["num_threads"])
    torch.set_num_interop_threads(expected["num_interop_threads"])
    torch.backends.cuda.matmul.allow_tf32 = expected["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = expected["cudnn_allow_tf32"]
    settings = lock["score_contract"]
    snapshot = require_cached_revision(settings["model"], settings["revision"])
    for name, value in expected["model_snapshot_files_sha256"].items():
        if sha256_file(snapshot / name) != value:
            raise ValueError(f"Pinned model snapshot changed: {name}")
    runtime = {"format": "v4-evidence-window-runtime-v1",
               "lock_sha256": sha256_file(output / "lock.json"), "versions": actual,
               "score_contract": settings, "gpu_name": torch.cuda.get_device_name(0),
               "gpu_uuid_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "implementation_sha256": sha256_file(Path(__file__).resolve())}
    return torch, lock, runtime


def score_anchors(output: Path, model, torch, runtime_sha: str) -> dict:
    path = output / "anchor_parity.json"
    if path.exists():
        result = read(path)
        if result["runtime_sha256"] != runtime_sha or result["status"] != "PASS":
            raise ValueError("Existing anchor parity is not reusable")
        return result
    rows = read(output / "anchors.json")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scoring_batch_id"]].append(row)
    differences, threshold_ok, started = [], True, time.monotonic()
    for batch_id, local in grouped.items():
        local.sort(key=lambda row: row["scoring_batch_position"])
        if len(local) != 32 or [row["scoring_batch_position"] for row in local] != list(range(32)):
            raise ValueError(f"Incomplete intact anchor batch: {batch_id}")
        values = model.score(local[0]["query"], [row["payload"] for row in local])
        for row, value in zip(local, values, strict=True):
            differences.append(abs(float(value) - float(row["expected_score"])))
            threshold_ok = threshold_ok and all(
                (float(value) >= threshold) == (float(row["expected_score"]) >= threshold)
                for threshold in row["threshold_checks"])
    maximum = max(differences)
    result = {"format": "v4-evidence-window-anchor-parity-v1",
              "status": "PASS" if maximum <= ANCHOR_TOLERANCE and threshold_ok else "STOP",
              "runtime_sha256": runtime_sha, "anchors": len(rows), "batches": len(grouped),
              "maximum_absolute_difference": maximum, "tolerance": ANCHOR_TOLERANCE,
              "threshold_decisions_unchanged": threshold_ok,
              "model_seconds": time.monotonic() - started}
    write_once(path, result)
    if result["status"] != "PASS":
        raise ValueError("Exact-batch runtime parity failed")
    return result


def prior_gate(output: Path, stage: str) -> None:
    if stage == "validation":
        report = read(output / "reports" / "development.json")
        if report.get("decision") != "OPEN_VALIDATION":
            raise ValueError("Development gate did not open validation")
    elif stage == "held_out":
        report = read(output / "reports" / "validation.json")
        if report.get("decision") != "OPEN_HELD_OUT":
            raise ValueError("Validation gate did not open held-out")


def score(output: Path, stage: str) -> dict[str, Any]:
    torch, lock, runtime = runtime_preflight(output)
    prior_gate(output, stage)
    runtime_path = output / "runtime_lock.json"
    if runtime_path.exists():
        if read(runtime_path) != runtime:
            raise ValueError("Existing runtime lock differs")
    else:
        write_once(runtime_path, runtime)
    runtime_sha = sha256_file(runtime_path)
    from src.medrag.ranking.reranker import BGEReranker
    settings = lock["score_contract"]
    model = BGEReranker(settings["model"], revision=settings["revision"],
                        max_length=settings["max_length"], device="cuda:0")
    if next(model.model.model.parameters()).dtype != torch.float16:
        raise ValueError("Reranker is not FP16")
    score_anchors(output, model, torch, runtime_sha)
    total_pairs = total_seconds = 0.0
    hashes = {}
    for qid in lock["cohort"][stage]:
        pair_path, destination = output / "pairs" / f"{qid}.json", output / "scores" / stage / f"{qid}.json"
        pair = read(pair_path)
        if destination.exists():
            result = read(destination)
            if (result["runtime_sha256"] != runtime_sha
                    or result["pair_sha256"] != sha256_file(pair_path)):
                raise ValueError(f"Score resume lineage differs: {qid}")
        else:
            torch.cuda.synchronize()
            started = time.monotonic()
            values = model.score(pair["query"], [row["payload"] for row in pair["windows"]])
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            if len(values) != len(pair["windows"]) or any(not math.isfinite(value) for value in values):
                raise ValueError(f"Invalid evidence-window scores: {qid}")
            result = {"format": "v4-evidence-window-scores-query-v1", "id": qid,
                      "stage": stage, "runtime_sha256": runtime_sha,
                      "pair_sha256": sha256_file(pair_path), "seconds": seconds,
                      "scores": [{"window_id": row["window_id"],
                                  "input_sha256": row["input_sha256"], "score": float(value)}
                                 for row, value in zip(pair["windows"], values, strict=True)]}
            write_once(destination, result)
        total_pairs += len(result["scores"])
        total_seconds += float(result["seconds"])
        hashes[str(qid)] = sha256_file(destination)
    if total_pairs > STAGE_PAIR_LIMITS[stage] or total_seconds > 15 * 60:
        raise ValueError(f"Stage scoring budget exceeded: {stage}")
    result = {"format": "v4-evidence-window-score-stage-v1", "status": "SCORED",
              "stage": stage, "queries": len(lock["cohort"][stage]), "pairs": int(total_pairs),
              "model_seconds": total_seconds, "pairs_per_second": total_pairs / total_seconds,
              "runtime_sha256": runtime_sha, "score_sha256": hashes}
    write_once(output / "scores" / f"{stage}_manifest.json", result)
    return result


def allocate(pair: dict, score: dict) -> tuple[list[int], dict[int, float], list[dict[str, Any]]]:
    values = {row["window_id"]: float(row["score"]) for row in score["scores"]}
    expected = {row["window_id"] for row in pair["windows"]}
    if set(values) != expected:
        raise ValueError(f"Window score coverage differs: {pair['id']}")
    contexts = {int(row["official_id"]): row for row in pair["contexts"]}
    parent_scores = {doc: max(values[window_id] for window_id in context["window_ids"])
                     for doc, context in contexts.items()}
    baseline = [int(row["doc_id"]) for row in pair["s2_chunks"]]
    baseline_set = set(baseline)
    outsiders = sorted([doc for doc in contexts if doc not in baseline_set],
                      key=lambda doc: (-parent_scores[doc], contexts[doc]["document_rank"], doc))
    victims = sorted(baseline, key=lambda doc: (parent_scores[doc],
                                                -contexts[doc]["document_rank"], -doc))
    swaps = []
    for entrant, victim in zip(outsiders, victims, strict=False):
        if len(swaps) == MAX_SWAPS or parent_scores[entrant] <= parent_scores[victim]:
            break
        swaps.append({"entrant": entrant, "victim": victim,
                      "entrant_score": parent_scores[entrant], "victim_score": parent_scores[victim]})
    selected = (baseline_set - {row["victim"] for row in swaps}) | {row["entrant"] for row in swaps}
    ordered = [int(doc) for doc in pair["s2_documents"] if int(doc) in selected]
    if len(ordered) != len(baseline) or len(ordered) != len(set(ordered)):
        raise ValueError(f"Allocation did not preserve chunk count: {pair['id']}")
    return ordered, parent_scores, swaps


def parent_hits(qid: int, chunks: dict[int, str], labels: dict, group: str) -> set[str]:
    hits = set()
    for (label_qid, doc, label_group), rows in labels.items():
        if label_qid == qid and label_group == group and doc in chunks and visible(chunks[doc], rows):
            hits.add(f"{qid}:{doc}")
    return hits


def stage_decision(stage: str, manual_gain: set[str], manual_loss: set[str],
                   historical_gain: set[str], historical_loss: set[str], structure: bool) -> tuple[str, dict]:
    manual_loss_queries = len({value.split(":", 1)[0] for value in manual_loss})
    historical_loss_queries = len({value.split(":", 1)[0] for value in historical_loss})
    if stage == "development":
        checks = {"manual_gains_at_least_3": len(manual_gain) >= 3,
                  "manual_gain_queries_at_least_3": len({v.split(":", 1)[0] for v in manual_gain}) >= 3,
                  "manual_loss_queries_at_most_1": manual_loss_queries <= 1,
                  "manual_gross_gains_exceed_losses": len(manual_gain) > len(manual_loss),
                  "historical_net_nonnegative": len(historical_gain) >= len(historical_loss),
                  "historical_loss_queries_at_most_3": historical_loss_queries <= 3,
                  "structural_validators": structure}
        return ("OPEN_VALIDATION" if all(checks.values()) else "STOP_DEVELOPMENT"), checks
    budget = 2 if stage == "validation" else 1
    checks = {"manual_net_nonnegative": len(manual_gain) >= len(manual_loss),
              "manual_visible_evidence_preserved": not manual_loss,
              "historical_net_nonnegative": len(historical_gain) >= len(historical_loss),
              f"historical_loss_queries_at_most_{budget}": historical_loss_queries <= budget,
              "structural_validators": structure}
    next_decision = "OPEN_HELD_OUT" if stage == "validation" else "SCREENED_CANDIDATE"
    stop = "STOP_VALIDATION" if stage == "validation" else "STOP_HELD_OUT"
    return (next_decision if all(checks.values()) else stop), checks


def review_packet(output: Path, stage: str, swaps: list[dict[str, Any]], queries: dict[int, str]) -> None:
    rows = []
    for swap in sorted(swaps, key=lambda row: canonical_digest(
            [SEED, stage, row["query_id"], row["entrant"], row["victim"]]))[:STAGE_REVIEW_LIMITS[stage]]:
        flip = int(canonical_digest(["blind", swap["query_id"], swap["entrant"], swap["victim"]]), 16) % 2
        contexts = [swap["victim_context"], swap["entrant_context"]]
        if flip:
            contexts.reverse()
        review_id = canonical_digest([SEED, stage, swap["query_id"], swap["entrant"], swap["victim"]])
        rows.append({"review_id": review_id, "query": queries[swap["query_id"]],
                     "context_a": contexts[0], "context_b": contexts[1]})
    write_once(output / "review" / f"{stage}.json", rows)


def evaluate(output: Path, stage: str) -> dict[str, Any]:
    lock, _ = validate_package(output)
    prior_gate(output, stage)
    score_manifest = read(output / "scores" / f"{stage}_manifest.json")
    if score_manifest.get("status") != "SCORED":
        raise ValueError("Stage scores are incomplete")
    s2, labels, queries = load_s2(), group_labels(), load_queries()
    config, _, policy, _ = environment()
    manual_gain = set()
    manual_loss = set()
    historical_gain = set()
    historical_loss = set()
    query_rows, swap_rows, baseline_predictions, challenger_predictions = [], [], [], []
    structure = True
    for qid in lock["cohort"][stage]:
        pair = read(output / "pairs" / f"{qid}.json")
        score_path = output / "scores" / stage / f"{qid}.json"
        if sha256_file(score_path) != score_manifest["score_sha256"][str(qid)]:
            raise ValueError(f"Stage score changed: {qid}")
        score = read(score_path)
        selected, parent_scores, swaps = allocate(pair, score)
        contexts = {int(row["official_id"]): row for row in pair["contexts"]}
        before = {int(row["doc_id"]): row["chunk_text"] for row in pair["s2_chunks"]}
        after = {doc: contexts[doc]["text"] for doc in selected}
        m_before, m_after = parent_hits(qid, before, labels, "manual_positive"), parent_hits(qid, after, labels, "manual_positive")
        h_before, h_after = parent_hits(qid, before, labels, "historical_positive"), parent_hits(qid, after, labels, "historical_positive")
        manual_gain |= m_after - m_before
        manual_loss |= m_before - m_after
        historical_gain |= h_after - h_before
        historical_loss |= h_before - h_after
        base = {"id": qid, "relevant_docs": pair["s2_documents"],
                "relevant_chunks": pair["s2_chunks"]}
        challenger = {"id": qid, "relevant_docs": pair["s2_documents"],
                      "relevant_chunks": [{"doc_id": doc, "chunk_text": after[doc]} for doc in selected]}
        if base["relevant_docs"] != challenger["relevant_docs"]:
            structure = False
        validate_payload(canonical_payload([challenger]), query_ids={qid},
                         official_ids=set(pair["s2_documents"]), policy=policy, result_limit=200)
        baseline_predictions.append(base)
        challenger_predictions.append(challenger)
        for row in swaps:
            swap_rows.append({"query_id": qid, **row,
                              "victim_context": contexts[row["victim"]]["text"],
                              "entrant_context": contexts[row["entrant"]]["text"]})
        query_rows.append({"query_id": qid, "role": pair["role"], "swaps": len(swaps),
                           "manual_gains": sorted(m_after - m_before),
                           "manual_losses": sorted(m_before - m_after),
                           "historical_gains": sorted(h_after - h_before),
                           "historical_losses": sorted(h_before - h_after),
                           "documents_exact": base["relevant_docs"] == challenger["relevant_docs"],
                           "chunks": len(after), "score_min": min(parent_scores.values()),
                           "score_max": max(parent_scores.values())})
    decision, checks = stage_decision(stage, manual_gain, manual_loss,
                                      historical_gain, historical_loss, structure)
    report = {"format": "v4-evidence-window-allocation-stage-report-v1", "stage": stage,
              "decision": decision, "checks": checks, "queries": len(query_rows),
              "changed_queries": sum(row["swaps"] > 0 for row in query_rows),
              "swaps": sum(row["swaps"] for row in query_rows),
              "manual": {"gains": len(manual_gain), "losses": len(manual_loss),
                         "gained_ids": sorted(manual_gain), "lost_ids": sorted(manual_loss)},
              "historical_preservation": {"gains": len(historical_gain),
                                           "losses": len(historical_loss),
                                           "gained_ids": sorted(historical_gain),
                                           "lost_ids": sorted(historical_loss)},
              "official_f2": "UNMEASURED", "query_rows": query_rows}
    write_once(output / "predictions" / f"{stage}_baseline.json", baseline_predictions)
    write_once(output / "predictions" / f"{stage}_challenger.json", challenger_predictions)
    write_once(output / "swaps" / f"{stage}.json", swap_rows)
    review_packet(output, stage, swap_rows, queries)
    write_once(output / "reports" / f"{stage}.json", report)
    write_atomic_replace(output / "status.json", {"status": decision, "stage": stage})
    return report


def docs_audit(output: Path) -> dict[str, Any]:
    if (output / "docs_cap_audit.json").exists():
        raise FileExistsError(output / "docs_cap_audit.json")
    labels, s2 = group_labels(), load_s2()
    rows = []
    for qid in (53, 610, 1159):
        replay = read(R7 / "replay" / "canary" / f"{qid}.json")
        selected = filter_documents(replay["selected_before_s2_gate"], -4.0)
        base, base_target = select_documents_by_confidence(
            selected, threshold=-2.0, min_documents=80, max_documents=140)
        challenger, target = select_documents_by_confidence(
            selected, threshold=-2.0, min_documents=80, max_documents=160)
        base_ids, new_ids = {int(row["official_id"]) for row in base}, {int(row["official_id"]) for row in challenger}
        known = {(label_qid, doc) for (label_qid, doc, _), value in labels.items()
                 if label_qid == qid and value}
        rows.append({"query_id": qid, "base_target": base_target, "cap160_target": target,
                     "added_documents": sorted(new_ids - base_ids),
                     "known_positive_gains": sorted(doc for label_qid, doc in known
                                                    if doc in new_ids - base_ids),
                     "known_positive_losses": [],
                     "unjudged_added_documents": sum(
                         not any(label_qid == qid and doc == added for label_qid, doc in known)
                         for added in new_ids - base_ids),
                     "chunks_unchanged_by_design": True})
    report = {"format": "v4-r7-document-cap-140-160-audit-v1", "status": "DIAGNOSTIC",
              "rows": rows, "official_f2": "UNMEASURED",
              "decision": "Do not promote a document cap from three enriched replay queries"}
    write_once(output / "docs_cap_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "validate-package", "score-development",
                                            "evaluate-development", "score-validation",
                                            "evaluate-validation", "score-held-out",
                                            "evaluate-held-out", "docs-audit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "package":
        result = package(args.output)
    elif args.command == "validate-package":
        lock, manifest = validate_package(args.output)
        result = {"status": "PASS", "queries": manifest["query_count"],
                  "pairs": manifest["total_pairs"], "cohort": lock["cohort"],
                  "support": lock["support"]}
    elif args.command == "docs-audit":
        result = docs_audit(args.output)
    else:
        verb, stage = args.command.split("-", 1)
        stage = stage.replace("held-out", "held_out")
        result = score(args.output, stage) if verb == "score" else evaluate(args.output, stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
