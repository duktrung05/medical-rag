"""Validate the development-selected evidence-window policy with ten swaps."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from scripts.v4 import run_evidence_window_allocation as base


ROOT = base.ROOT
PARENT = ROOT / "outputs/v4/evidence_window_allocation_20261025"
OUTPUT = ROOT / "outputs/v4/evidence_window_cap10_validation_20261025"
MAX_SWAPS = 10
STAGES = ("validation", "held_out")


def allocate(pair: dict[str, Any], score: dict[str, Any]) -> tuple[list[int], dict[int, float], list[dict]]:
    values = {row["window_id"]: float(row["score"]) for row in score["scores"]}
    expected = {row["window_id"] for row in pair["windows"]}
    if set(values) != expected:
        raise ValueError(f"Window score coverage differs: {pair['id']}")
    contexts = {int(row["official_id"]): row for row in pair["contexts"]}
    parent_scores = {
        doc: max(values[window_id] for window_id in context["window_ids"])
        for doc, context in contexts.items()
    }
    baseline = [int(row["doc_id"]) for row in pair["s2_chunks"]]
    baseline_set = set(baseline)
    outsiders = sorted(
        (doc for doc in contexts if doc not in baseline_set),
        key=lambda doc: (-parent_scores[doc], contexts[doc]["document_rank"], doc),
    )
    victims = sorted(
        baseline,
        key=lambda doc: (parent_scores[doc], -contexts[doc]["document_rank"], -doc),
    )
    swaps = []
    for entrant, victim in zip(outsiders, victims, strict=False):
        if len(swaps) == MAX_SWAPS or parent_scores[entrant] <= parent_scores[victim]:
            break
        swaps.append({
            "entrant": entrant,
            "victim": victim,
            "entrant_score": parent_scores[entrant],
            "victim_score": parent_scores[victim],
        })
    selected = (baseline_set - {row["victim"] for row in swaps}) | {
        row["entrant"] for row in swaps
    }
    ordered = [int(doc) for doc in pair["s2_documents"] if int(doc) in selected]
    if len(ordered) != len(baseline) or len(ordered) != len(set(ordered)):
        raise ValueError(f"Allocation did not preserve chunk count: {pair['id']}")
    return ordered, parent_scores, swaps


def score_inputs(stage: str) -> tuple[dict, dict]:
    parent_lock, parent_manifest = base.validate_package(PARENT)
    score_manifest = base.read(PARENT / "scores" / f"{stage}_manifest.json")
    for qid in parent_lock["cohort"][stage]:
        path = PARENT / "scores" / stage / f"{qid}.json"
        if base.sha256_file(path) != score_manifest["score_sha256"][str(qid)]:
            raise ValueError(f"Parent cached score changed: {qid}")
    return parent_lock, parent_manifest


def cached_delta(stage: str) -> dict[str, Any]:
    parent_lock, _ = score_inputs(stage)
    labels = base.group_labels()
    manual_gain: set[str] = set()
    manual_loss: set[str] = set()
    historical_gain: set[str] = set()
    historical_loss: set[str] = set()
    swaps = 0
    for qid in parent_lock["cohort"][stage]:
        pair = base.read(PARENT / "pairs" / f"{qid}.json")
        score = base.read(PARENT / "scores" / stage / f"{qid}.json")
        selected, _, local_swaps = allocate(pair, score)
        contexts = {int(row["official_id"]): row for row in pair["contexts"]}
        before = {int(row["doc_id"]): row["chunk_text"] for row in pair["s2_chunks"]}
        after = {doc: contexts[doc]["text"] for doc in selected}
        manual_before = base.parent_hits(qid, before, labels, "manual_positive")
        manual_after = base.parent_hits(qid, after, labels, "manual_positive")
        historical_before = base.parent_hits(qid, before, labels, "historical_positive")
        historical_after = base.parent_hits(qid, after, labels, "historical_positive")
        manual_gain |= manual_after - manual_before
        manual_loss |= manual_before - manual_after
        historical_gain |= historical_after - historical_before
        historical_loss |= historical_before - historical_after
        swaps += len(local_swaps)
    return {
        "stage": stage,
        "max_swaps_per_query": MAX_SWAPS,
        "swaps": swaps,
        "manual": {"gains": len(manual_gain), "losses": len(manual_loss),
                   "gained_ids": sorted(manual_gain), "lost_ids": sorted(manual_loss)},
        "historical_preservation": {
            "gains": len(historical_gain), "losses": len(historical_loss),
            "gained_ids": sorted(historical_gain), "lost_ids": sorted(historical_loss),
        },
    }


def package(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    parent_lock, parent_manifest = base.validate_package(PARENT)
    development_report = base.read(PARENT / "reports/development.json")
    if (development_report.get("decision") != "STOP_DEVELOPMENT"
            or development_report["manual"]["gains"] != 2
            or development_report["manual"]["losses"] != 0):
        raise ValueError("Parent development result differs from the selection record")
    development = cached_delta("development")
    if ((development["manual"]["gains"], development["manual"]["losses"])
            != (3, 0)
            or (development["historical_preservation"]["gains"],
                development["historical_preservation"]["losses"]) != (8, 1)):
        raise ValueError("Cap-10 cached development diagnostic differs")
    implementation = Path(__file__).resolve()
    output.mkdir(parents=True)
    lock = {
        "format": "v4-evidence-window-cap10-validation-lock-v1",
        "status": "LOCKED",
        "selection_boundary": "cap 10 selected on development; validation is first gate",
        "policy": {**parent_lock["policy"], "max_swaps_per_query": MAX_SWAPS},
        "cohort": {stage: parent_lock["cohort"][stage] for stage in STAGES},
        "support": {stage: parent_lock["support"][stage] for stage in STAGES},
        "pair_counts": {stage: parent_manifest["stage_pair_counts"][stage] for stage in STAGES},
        "development_selection_diagnostic": development,
        "gates": {
            "validation": "manual net >= 0; no manual loss; historical net >= 0; <=2 historical-loss queries",
            "held_out": "manual net >= 0; no manual loss; historical net >= 0; <=1 historical-loss query",
        },
        "inputs_sha256": {
            "parent_lock": base.sha256_file(PARENT / "lock.json"),
            "parent_manifest": base.sha256_file(PARENT / "pair_manifest.json"),
            "parent_development_scores": base.sha256_file(
                PARENT / "scores/development_manifest.json"),
            "parent_development_report": base.sha256_file(
                PARENT / "reports/development.json"),
            "parent_implementation": base.sha256_file(Path(base.__file__).resolve()),
            "implementation": base.sha256_file(implementation),
        },
    }
    base.write_once(output / "lock.json", lock)
    manifest = {
        "format": "v4-evidence-window-cap10-validation-manifest-v1",
        "status": "READY_FOR_VALIDATION_GPU",
        "lock_sha256": base.sha256_file(output / "lock.json"),
        "parent_pair_shards_sha256": {
            str(qid): parent_manifest["pair_shards_sha256"][str(qid)]
            for stage in STAGES for qid in parent_lock["cohort"][stage]
        },
    }
    base.write_once(output / "manifest.json", manifest)
    return manifest


def validate_package(output: Path) -> tuple[dict, dict, dict]:
    parent_lock, parent_manifest = base.validate_package(PARENT)
    lock = base.read(output / "lock.json")
    manifest = base.read(output / "manifest.json")
    if (lock.get("format") != "v4-evidence-window-cap10-validation-lock-v1"
            or manifest.get("lock_sha256") != base.sha256_file(output / "lock.json")
            or lock["inputs_sha256"]["implementation"]
            != base.sha256_file(Path(__file__).resolve())
            or lock["inputs_sha256"]["parent_implementation"]
            != base.sha256_file(Path(base.__file__).resolve())
            or lock["inputs_sha256"]["parent_lock"] != base.sha256_file(PARENT / "lock.json")
            or lock["inputs_sha256"]["parent_manifest"]
            != base.sha256_file(PARENT / "pair_manifest.json")):
        raise ValueError("Cap-10 validation package lineage differs")
    expected = {
        str(qid): parent_manifest["pair_shards_sha256"][str(qid)]
        for stage in STAGES for qid in parent_lock["cohort"][stage]
    }
    if manifest["parent_pair_shards_sha256"] != expected:
        raise ValueError("Cap-10 parent pair coverage differs")
    return lock, manifest, parent_lock


def prior_gate(output: Path, stage: str) -> None:
    if stage == "held_out":
        report = base.read(output / "reports/validation.json")
        if report.get("decision") != "OPEN_HELD_OUT":
            raise ValueError("Validation gate did not open held-out")


def score(output: Path, stage: str) -> dict[str, Any]:
    lock, _, parent_lock = validate_package(output)
    prior_gate(output, stage)
    torch, _, runtime = base.runtime_preflight(PARENT)
    parity = base.read(PARENT / "anchor_parity.json")
    if (parity.get("status") != "PASS" or parity.get("anchors") != 128
            or parity.get("maximum_absolute_difference") != 0.0):
        raise ValueError("Parent exact-batch parity is not reusable")
    actual_versions = {name: importlib.metadata.version(name) for name in parent_lock["versions"]}
    if actual_versions != parent_lock["versions"]:
        raise ValueError("Pinned dependency versions differ")
    runtime_record = {
        "format": "v4-evidence-window-cap10-runtime-v1",
        "policy_lock_sha256": base.sha256_file(output / "lock.json"),
        "parent_runtime": runtime,
        "parent_anchor_parity_sha256": base.sha256_file(PARENT / "anchor_parity.json"),
        "gpu_uuid_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "implementation_sha256": base.sha256_file(Path(__file__).resolve()),
    }
    runtime_path = output / "runtime_lock.json"
    if runtime_path.exists():
        if base.read(runtime_path) != runtime_record:
            raise ValueError("Existing cap-10 runtime lock differs")
    else:
        base.write_once(runtime_path, runtime_record)
    runtime_sha = base.sha256_file(runtime_path)
    from src.medrag.ranking.reranker import BGEReranker

    settings = parent_lock["score_contract"]
    model = BGEReranker(settings["model"], revision=settings["revision"],
                        max_length=settings["max_length"], device="cuda:0")
    if next(model.model.model.parameters()).dtype != torch.float16:
        raise ValueError("Reranker is not FP16")
    total_pairs = 0
    total_seconds = 0.0
    hashes = {}
    for qid in lock["cohort"][stage]:
        pair_path = PARENT / "pairs" / f"{qid}.json"
        destination = output / "scores" / stage / f"{qid}.json"
        pair = base.read(pair_path)
        if destination.exists():
            result = base.read(destination)
            if (result["runtime_sha256"] != runtime_sha
                    or result["pair_sha256"] != base.sha256_file(pair_path)):
                raise ValueError(f"Score resume lineage differs: {qid}")
        else:
            torch.cuda.synchronize()
            started = time.monotonic()
            values = model.score(pair["query"], [row["payload"] for row in pair["windows"]])
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            if len(values) != len(pair["windows"]) or any(
                    not math.isfinite(value) for value in values):
                raise ValueError(f"Invalid evidence-window scores: {qid}")
            result = {
                "format": "v4-evidence-window-cap10-scores-query-v1",
                "id": qid, "stage": stage, "runtime_sha256": runtime_sha,
                "pair_sha256": base.sha256_file(pair_path), "seconds": seconds,
                "scores": [{"window_id": row["window_id"],
                            "input_sha256": row["input_sha256"], "score": float(value)}
                           for row, value in zip(pair["windows"], values, strict=True)],
            }
            base.write_once(destination, result)
        total_pairs += len(result["scores"])
        total_seconds += float(result["seconds"])
        hashes[str(qid)] = base.sha256_file(destination)
    if total_pairs != lock["pair_counts"][stage] or total_seconds > 15 * 60:
        raise ValueError(f"Stage scoring budget/coverage differs: {stage}")
    result = {
        "format": "v4-evidence-window-cap10-score-stage-v1", "status": "SCORED",
        "stage": stage, "queries": len(lock["cohort"][stage]), "pairs": total_pairs,
        "model_seconds": total_seconds, "pairs_per_second": total_pairs / total_seconds,
        "runtime_sha256": runtime_sha, "score_sha256": hashes,
    }
    base.write_once(output / "scores" / f"{stage}_manifest.json", result)
    return result


def evaluate(output: Path, stage: str) -> dict[str, Any]:
    lock, _, _ = validate_package(output)
    prior_gate(output, stage)
    score_manifest = base.read(output / "scores" / f"{stage}_manifest.json")
    if score_manifest.get("status") != "SCORED":
        raise ValueError("Stage scores are incomplete")
    s2 = base.load_s2()
    labels = base.group_labels()
    queries = base.load_queries()
    _, _, policy, _ = base.environment()
    manual_gain: set[str] = set()
    manual_loss: set[str] = set()
    historical_gain: set[str] = set()
    historical_loss: set[str] = set()
    query_rows = []
    swap_rows = []
    baseline_predictions = []
    challenger_predictions = []
    structure = True
    for qid in lock["cohort"][stage]:
        pair = base.read(PARENT / "pairs" / f"{qid}.json")
        score_path = output / "scores" / stage / f"{qid}.json"
        if base.sha256_file(score_path) != score_manifest["score_sha256"][str(qid)]:
            raise ValueError(f"Stage score changed: {qid}")
        selected, parent_scores, swaps = allocate(pair, base.read(score_path))
        contexts = {int(row["official_id"]): row for row in pair["contexts"]}
        before = {int(row["doc_id"]): row["chunk_text"] for row in pair["s2_chunks"]}
        after = {doc: contexts[doc]["text"] for doc in selected}
        manual_before = base.parent_hits(qid, before, labels, "manual_positive")
        manual_after = base.parent_hits(qid, after, labels, "manual_positive")
        historical_before = base.parent_hits(qid, before, labels, "historical_positive")
        historical_after = base.parent_hits(qid, after, labels, "historical_positive")
        manual_gain |= manual_after - manual_before
        manual_loss |= manual_before - manual_after
        historical_gain |= historical_after - historical_before
        historical_loss |= historical_before - historical_after
        baseline = {"id": qid, "relevant_docs": pair["s2_documents"],
                    "relevant_chunks": pair["s2_chunks"]}
        challenger = {"id": qid, "relevant_docs": pair["s2_documents"],
                      "relevant_chunks": [{"doc_id": doc, "chunk_text": after[doc]}
                                          for doc in selected]}
        structure = structure and baseline["relevant_docs"] == challenger["relevant_docs"]
        base.validate_payload(base.canonical_payload([challenger]), query_ids={qid},
                              official_ids=set(pair["s2_documents"]), policy=policy,
                              result_limit=200)
        baseline_predictions.append(baseline)
        challenger_predictions.append(challenger)
        for row in swaps:
            swap_rows.append({"query_id": qid, **row,
                              "victim_context": contexts[row["victim"]]["text"],
                              "entrant_context": contexts[row["entrant"]]["text"]})
        query_rows.append({
            "query_id": qid, "role": pair["role"], "swaps": len(swaps),
            "manual_gains": sorted(manual_after - manual_before),
            "manual_losses": sorted(manual_before - manual_after),
            "historical_gains": sorted(historical_after - historical_before),
            "historical_losses": sorted(historical_before - historical_after),
            "documents_exact": baseline["relevant_docs"] == challenger["relevant_docs"],
            "chunks": len(after), "score_min": min(parent_scores.values()),
            "score_max": max(parent_scores.values()),
        })
    decision, checks = base.stage_decision(
        stage, manual_gain, manual_loss, historical_gain, historical_loss, structure)
    report = {
        "format": "v4-evidence-window-cap10-stage-report-v1", "stage": stage,
        "decision": decision, "checks": checks, "queries": len(query_rows),
        "changed_queries": sum(row["swaps"] > 0 for row in query_rows),
        "swaps": sum(row["swaps"] for row in query_rows),
        "manual": {"gains": len(manual_gain), "losses": len(manual_loss),
                   "gained_ids": sorted(manual_gain), "lost_ids": sorted(manual_loss)},
        "historical_preservation": {
            "gains": len(historical_gain), "losses": len(historical_loss),
            "gained_ids": sorted(historical_gain), "lost_ids": sorted(historical_loss)},
        "official_f2": "UNMEASURED", "query_rows": query_rows,
    }
    base.write_once(output / "predictions" / f"{stage}_baseline.json", baseline_predictions)
    base.write_once(output / "predictions" / f"{stage}_challenger.json", challenger_predictions)
    base.write_once(output / "swaps" / f"{stage}.json", swap_rows)
    base.review_packet(output, stage, swap_rows, queries)
    base.write_once(output / "reports" / f"{stage}.json", report)
    base.write_atomic_replace(output / "status.json", {"status": decision, "stage": stage})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("package", "validate-package", "score-validation",
                                            "evaluate-validation", "score-held-out",
                                            "evaluate-held-out"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "package":
        result = package(args.output)
    elif args.command == "validate-package":
        lock, manifest, _ = validate_package(args.output)
        result = {"status": "PASS", "cohort": lock["cohort"],
                  "pair_counts": lock["pair_counts"], "manifest": manifest["status"]}
    else:
        verb, stage = args.command.split("-", 1)
        stage = stage.replace("held-out", "held_out")
        result = score(args.output, stage) if verb == "score" else evaluate(args.output, stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
