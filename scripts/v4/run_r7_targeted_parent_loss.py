"""Lock and trace a bounded cohort of known-parent losses through frozen S2 retrieval."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import pyarrow.parquet as pq

from scripts.v4.run_p5_2_split import CONFIG, ROOT, read
from src.medrag.core.config import load_config
from src.medrag.core.contracts import Hit
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import containment, normalize
from src.medrag.retrieval.core import fuse, fuse_with_parent_floor
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.disk_dense import DiskDense

OUTPUT = ROOT / "outputs/v4/r7_targeted_parent_loss_20261025"
R6 = ROOT / "outputs/v4/r6_stage_oracle_20261025"
EVALUATION = ROOT / "outputs/v4/evidence_evaluation_lock_20261025"
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024"
S2 = ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip"
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
RELEASE_AUDIT = ROOT / "outputs/v4/task4/release_coverage_audit_20261016.json"
PILOT_IDS = (53, 130, 380, 754, 771, 1144)


def load_s2() -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(S2) as archive:
        rows = json.loads(archive.read("predictions.json"))
    return {int(row["id"]): row for row in rows}


def stable_key(prefix: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{prefix}:{row['query_id']}:{row['doc_id']}:{row['registry_id']}".encode()
    ).hexdigest()


def choose_cohort(rows: list[dict[str, Any]], assignments: dict[str, str]) -> list[dict[str, Any]]:
    dev = [row for row in rows if assignments[row["query_id"]] == "development"
           and row["source_present"] == "True" and row["source_contains_evidence"] == "True"]
    manual = sorted(
        [row for row in dev if row["label_group"] == "manual_positive"
         and row["fused_top400"] == "False"],
        key=lambda row: (int(row["query_id"]), int(row["doc_id"]), row["registry_id"]),
    )
    if len(manual) != 8 or len({row["query_id"] for row in manual}) != 8:
        raise ValueError("Expected eight query-distinct development manual fused misses")
    used = {row["query_id"] for row in manual}
    historical_candidates = sorted(
        [row for row in dev if row["label_group"] == "historical_positive"
         and row["fused_top400"] == "False" and row["query_id"] not in used],
        key=lambda row: stable_key("r7-historical-v1", row),
    )
    historical = []
    for row in historical_candidates:
        if row["query_id"] in used:
            continue
        historical.append(row)
        used.add(row["query_id"])
        if len(historical) == 12:
            break
    if len(historical) != 12:
        raise ValueError("Could not lock twelve query-distinct historical fused misses")

    retained = [row for row in dev if row["fused_top400"] == "True"
                and row["s2_selected_document"] == "True" and row["query_id"] not in used]
    explicit = sorted([row for row in retained if int(row["query_id"]) in PILOT_IDS],
                      key=lambda row: (PILOT_IDS.index(int(row["query_id"])), row["registry_id"]))
    controls = []
    for pool in (explicit, sorted(retained, key=lambda row: stable_key("r7-control-v1", row))):
        for row in pool:
            if row["query_id"] in used:
                continue
            controls.append(row)
            used.add(row["query_id"])
            if len(controls) == 10:
                break
        if len(controls) == 10:
            break
    if len(controls) != 10:
        raise ValueError("Could not lock ten query-distinct controls")
    return ([{**row, "cohort": "manual_miss"} for row in manual]
            + [{**row, "cohort": "historical_miss"} for row in historical]
            + [{**row, "cohort": "control_retained"} for row in controls])


def lock(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    assignments_path = EVALUATION / "query_split_registry.json"
    assignments = read(assignments_path)["assignments"]
    with (R6 / "first_loss_attribution.csv").open() as stream:
        attribution = list(csv.DictReader(stream))
    cohort = choose_cohort(attribution, assignments)
    query_path = ROOT / "data/raw/vibiomir/query.parquet"
    queries = {int(row["id"]): str(row["query"]) for row in
               pq.read_table(query_path, columns=["id", "query"]).to_pylist()}
    registry = {row["registry_id"]: row for row in
                pq.read_table(R6 / "positive_registry.parquet").to_pylist()}
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    manifest_path = index / "manifest.json"
    manifest = read(manifest_path)
    store = DiskCorpusStore(index / "corpus.sqlite", manifest["corpus_version"],
                            manifest["fingerprint"])
    s2 = load_s2()
    checked = []
    for chosen in cohort:
        qid, doc = int(chosen["query_id"]), int(chosen["doc_id"])
        value = registry[chosen["registry_id"]]
        if qid != int(value["query_id"]) or doc != int(value["doc_id"]):
            raise ValueError("Registry/attribution identity differs")
        source = store.source(doc)
        matched, start, end = containment(source.text, value["evidence_text"])
        valid = bool(matched and normalize(source.text[start:end]) == normalize(value["evidence_text"]))
        if not valid or source.text_sha256 != value["source_sha256"]:
            raise ValueError(f"Cohort label/source failed validation: {qid}/{doc}")
        checked.append({
            "cohort": chosen["cohort"], "query_id": qid, "doc_id": doc,
            "registry_id": chosen["registry_id"], "label_group": chosen["label_group"],
            "query_sha256": hashlib.sha256(queries[qid].encode()).hexdigest(),
            "source_sha256": source.text_sha256, "source_start": start, "source_end": end,
            "evidence_text_sha256": value["evidence_text_sha256"],
            "source_in_index": True, "evidence_roundtrip_valid": True,
            "r6_fused_top400": chosen["fused_top400"],
            "r6_s2_selected_document": chosen["s2_selected_document"],
            "s2_document_present": doc in {int(item) for item in s2[qid]["relevant_docs"]},
            "explicit_pilot_trace": qid in PILOT_IDS,
        })
    output.mkdir(parents=True)
    with (output / "label_source_check.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checked[0]))
        writer.writeheader()
        writer.writerows(checked)
    release = read(RELEASE_AUDIT)
    if release.get("status") != "pass" or release.get("eligible_source_count") != 1_073_525:
        raise ValueError("Existing release coverage audit is not reusable")
    counts = {name: sum(row["cohort"] == name for row in checked) for name in
              ("manual_miss", "historical_miss", "control_retained")}
    payload = {
        "format": "v4-r7-targeted-parent-loss-cohort-v1", "status": "LOCKED",
        "selection_rule": "8 all manual dev fused misses; 12 deterministic query-distinct historical dev fused misses; 10 retained controls preferring explicit traces",
        "counts": counts, "query_ids": [row["query_id"] for row in checked],
        "canary_query_ids": [qid for qid in PILOT_IDS
                             if any(row["query_id"] == qid for row in checked)][:3],
        "rows": checked,
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            assignments_path, R6 / "first_loss_attribution.csv",
            R6 / "positive_registry.parquet", query_path, manifest_path,
            RELEASE_AUDIT, S2, REFINED / "run_manifest.json")},
        "label_source_check_sha256": sha256_file(output / "label_source_check.csv"),
        "budgets": {"queries_max": 30, "new_retrieval_cpu_seconds_max": 7200,
                    "canary_queries": 3, "new_context_judgments_max": 60},
    }
    if len(payload["canary_query_ids"]) != 3:
        raise ValueError("Cohort does not contain three explicit-trace controls")
    write_once(output / "cohort_lock.json", payload)
    print(json.dumps({"status": "LOCKED", "counts": counts,
                      "canary_query_ids": payload["canary_query_ids"]}))


def query_encoder(config: dict):
    from src.medrag.retrieval.bge_encoder import BGEM3Encoder
    dense = config["dense"]
    return BGEM3Encoder(dense["model"], revision=dense["revision"],
                        document_max_length=dense["document_max_length"],
                        query_max_length=dense["query_max_length"], device="cpu")


def raw_trace(query: str, config: dict, manifest: dict, index: Path,
              doc_bm25: DiskBM25, passage_bm25: DiskBM25, dense: DiskDense) -> dict[str, Any]:
    retrieval = config["retrieval"]
    started = time.monotonic()
    branches = {
        "document_bm25": doc_bm25.search(query, retrieval["document_candidate_limit"]),
        "passage_bm25": passage_bm25.search(query, retrieval["passage_candidate_limit"]),
        "dense": dense.search(query, retrieval["passage_candidate_limit"]),
    }
    fused = fuse(list(branches.values()), k=retrieval["rrf_k"],
                 limit=retrieval["fused_candidate_limit"])
    return {"branch_hits": {name: [asdict(hit) for hit in hits]
                            for name, hits in branches.items()},
            "fused": [asdict(item) for item in fused],
            "elapsed_seconds": time.monotonic() - started,
            "index_manifest_sha256": sha256_file(index / "manifest.json")}


def identities(rows: list[dict[str, Any]]) -> list[tuple[int, str | None]]:
    return [(int(row["official_id"]), row.get("passage_id")) for row in rows]


def canary(output: Path) -> None:
    cohort = read(output / "cohort_lock.json")
    if cohort["status"] != "LOCKED":
        raise ValueError("Cohort is not locked")
    destination = output / "canary_report.json"
    if destination.exists():
        raise FileExistsError(destination)
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    queries = {int(row["id"]): str(row["query"]) for row in
               pq.read_table(query_path, columns=["id", "query"]).to_pylist()}
    doc_bm25 = DiskBM25(index / "documents.sqlite", "document_bm25", manifest["fingerprint"])
    passage_bm25 = DiskBM25(index / "passages.sqlite", "passage_bm25", manifest["fingerprint"])
    dense = DiskDense(index / "dense", query_encoder(config), manifest["dense"]["fingerprint"])
    results, total = [], 0.0
    trace_path = output / "canary_raw_branch_trace.jsonl"
    with trace_path.open("x") as stream:
        for qid in cohort["canary_query_ids"]:
            current = raw_trace(queries[qid], config, manifest, index, doc_bm25, passage_bm25, dense)
            frozen = read(PILOT / "queries" / f"{qid}.json")["branch_trace"]
            branch = {}
            for name in ("document_bm25", "passage_bm25", "dense"):
                left = identities(current["branch_hits"][name])
                right = identities(frozen["branch_hits"][name])
                branch[name] = {"identities_exact": left == right,
                                "overlap": len(set(left) & set(right)), "expected": len(right)}
            fused_left, fused_right = identities(current["fused"]), identities(frozen["fused"])
            result = {"query_id": qid, "branches": branch,
                      "fused_identities_exact": fused_left == fused_right,
                      "fused_overlap": len(set(fused_left) & set(fused_right)),
                      "fused_expected": len(fused_right),
                      "elapsed_seconds": current["elapsed_seconds"]}
            total += current["elapsed_seconds"]
            results.append(result)
            stream.write(json.dumps({"query_id": qid, **current}, ensure_ascii=False) + "\n")
    parity = all(result["fused_identities_exact"] and
                 all(value["identities_exact"] for value in result["branches"].values())
                 for result in results)
    budget_ok = total <= cohort["budgets"]["new_retrieval_cpu_seconds_max"]
    report = {"format": "v4-r7-targeted-canary-v1",
              "status": "PASS" if parity and budget_ok else "STOP_PARITY_OR_BUDGET",
              "exact_identity_parity": parity, "budget_ok": budget_ok,
              "total_retrieval_seconds": total, "results": results,
              "trace_sha256": sha256_file(trace_path),
              "note": "Exact IDs/order required before attributing fresh trace to frozen S2."}
    write_once(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def branch_parent_rank(rows: list[dict[str, Any]], parent: int) -> int | None:
    ranks = [int(row["rank"]) for row in rows if int(row["official_id"]) == parent]
    return min(ranks) if ranks else None


def fused_parent_rank(rows: list[dict[str, Any]], parent: int) -> int | None:
    ranks = [index for index, row in enumerate(rows, 1) if int(row["official_id"]) == parent]
    return min(ranks) if ranks else None


def trace(output: Path) -> None:
    destination = output / "trace_report.json"
    if destination.exists():
        raise FileExistsError(destination)
    cohort = read(output / "cohort_lock.json")
    canary_report = read(output / "canary_report.json")
    if (canary_report["status"] != "PASS" or not canary_report["exact_identity_parity"]
            or canary_report["trace_sha256"] != sha256_file(output / "canary_raw_branch_trace.jsonl")):
        raise ValueError("Full trace requires a passing exact-parity canary")
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    migration = read(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    queries = {int(row["id"]): str(row["query"]) for row in
               pq.read_table(query_path, columns=["id", "query"]).to_pylist()}
    trace_dir = output / "traces"
    trace_dir.mkdir(exist_ok=False)
    with (output / "canary_raw_branch_trace.jsonl").open() as stream:
        for line in stream:
            value = json.loads(line)
            write_once(trace_dir / f"{int(value['query_id'])}.json", value)
    doc_bm25 = DiskBM25(index / "documents.sqlite", "document_bm25", manifest["fingerprint"])
    passage_bm25 = DiskBM25(index / "passages.sqlite", "passage_bm25", manifest["fingerprint"])
    dense = DiskDense(index / "dense", query_encoder(config), manifest["dense"]["fingerprint"])
    total = float(canary_report["total_retrieval_seconds"])
    for index_number, qid in enumerate(cohort["query_ids"], 1):
        path = trace_dir / f"{qid}.json"
        if path.exists():
            continue
        if total >= cohort["budgets"]["new_retrieval_cpu_seconds_max"]:
            raise RuntimeError("Retrieval budget exhausted before full cohort")
        value = {"query_id": qid, **raw_trace(
            queries[qid], config, manifest, index, doc_bm25, passage_bm25, dense)}
        total += float(value["elapsed_seconds"])
        write_once(path, value)
        print(json.dumps({"traced": index_number, "query_id": qid,
                          "query_seconds": value["elapsed_seconds"],
                          "total_retrieval_seconds": total}), flush=True)
    expected = {int(qid) for qid in cohort["query_ids"]}
    actual = {int(path.stem) for path in trace_dir.glob("*.json")}
    if actual != expected or total > cohort["budgets"]["new_retrieval_cpu_seconds_max"]:
        raise ValueError("Full trace coverage/budget differs")

    s2 = load_s2()
    refined_manifest = read(REFINED / "run_manifest.json")
    attribution = []
    for row in cohort["rows"]:
        qid, parent = int(row["query_id"]), int(row["doc_id"])
        raw = read(trace_dir / f"{qid}.json")
        branches = raw["branch_hits"]
        raw_ranks = {name: branch_parent_rank(branches[name], parent) for name in
                     ("document_bm25", "passage_bm25", "dense")}
        raw_present = any(rank is not None for rank in raw_ranks.values())
        fused_rank = fused_parent_rank(raw["fused"], parent)
        final_path = REFINED / "final" / f"{qid}.json"
        if sha256_file(final_path) != refined_manifest["final_sha256"][str(qid)]:
            raise ValueError(f"Refined query changed: {qid}")
        final = read(final_path)
        ranked = [int(value["official_id"]) for value in final["selected"]]
        ranked_position = ranked.index(parent) + 1 if parent in ranked else None
        s2_docs = [int(value) for value in s2[qid]["relevant_docs"]]
        s2_position = s2_docs.index(parent) + 1 if parent in s2_docs else None
        if not raw_present:
            stage = "RAW_UNION_MISS"
        elif fused_rank is None:
            stage = "FUSION_TOP400_DROP"
        elif ranked_position is None:
            stage = "RANKED_PARENT_DROP"
        elif s2_position is None:
            stage = "S2_DOCUMENT_GATE_DROP"
        else:
            stage = "RETAINED_S2_DOCUMENT"
        attribution.append({
            "cohort": row["cohort"], "query_id": qid, "doc_id": parent,
            "registry_id": row["registry_id"], "label_group": row["label_group"],
            "document_bm25_rank": raw_ranks["document_bm25"],
            "passage_bm25_rank": raw_ranks["passage_bm25"],
            "dense_rank": raw_ranks["dense"], "raw_union_present": raw_present,
            "fused_item_rank": fused_rank, "ranked_parent_position": ranked_position,
            "s2_document_position": s2_position, "first_measured_loss": stage,
            "trace_sha256": sha256_file(trace_dir / f"{qid}.json"),
        })
    with (output / "parent_loss_attribution.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attribution[0]))
        writer.writeheader()
        writer.writerows(attribution)
    stage_counts: dict[str, dict[str, int]] = {}
    for cohort_name in ("manual_miss", "historical_miss", "control_retained"):
        local = [row for row in attribution if row["cohort"] == cohort_name]
        stage_counts[cohort_name] = {
            stage: sum(row["first_measured_loss"] == stage for row in local)
            for stage in sorted({row["first_measured_loss"] for row in attribution})
        }
    report = {
        "format": "v4-r7-targeted-parent-loss-trace-v1", "status": "PASS",
        "queries": len(expected), "total_retrieval_seconds": total,
        "budget_seconds": cohort["budgets"]["new_retrieval_cpu_seconds_max"],
        "stage_counts": stage_counts,
        "parent_loss_attribution_sha256": sha256_file(output / "parent_loss_attribution.csv"),
        "trace_hashes": {str(qid): sha256_file(trace_dir / f"{qid}.json")
                         for qid in sorted(expected)},
        "scope": "Known-parent diagnostic; manual and historical labels are incomplete and not official F2",
    }
    write_once(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def candidate(output: Path) -> None:
    destination = output / "candidate_gate.json"
    if destination.exists():
        raise FileExistsError(destination)
    cohort = read(output / "cohort_lock.json")
    trace_report = read(output / "trace_report.json")
    if trace_report["status"] != "PASS" or trace_report["queries"] != 30:
        raise ValueError("Candidate replay requires the complete targeted trace")
    config = load_config(CONFIG)
    index = ROOT / config["index_namespace"]
    manifest = read(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", manifest["corpus_version"],
                            manifest["fingerprint"])
    registry = {row["registry_id"]: row for row in
                pq.read_table(R6 / "positive_registry.parquet").to_pylist()}
    details = []
    for row in cohort["rows"]:
        qid, parent = int(row["query_id"]), int(row["doc_id"])
        raw = read(output / "traces" / f"{qid}.json")
        rankings = [[Hit(**value) for value in raw["branch_hits"][name]]
                    for name in ("document_bm25", "passage_bm25", "dense")]
        baseline = fuse(rankings, k=config["retrieval"]["rrf_k"],
                        limit=config["retrieval"]["fused_candidate_limit"])
        augmented = fuse_with_parent_floor(
            rankings, k=config["retrieval"]["rrf_k"],
            item_limit=config["retrieval"]["fused_candidate_limit"], parent_limit=400)
        if augmented[:len(baseline)] != baseline:
            raise ValueError("Parent floor changed the frozen mixed-item prefix")
        baseline_parents = {item.official_id for item in baseline}
        augmented_parents = {item.official_id for item in augmented}
        added = augmented_parents - baseline_parents
        recovered = parent in added
        evidence_exact = False
        recovered_item = next((item for item in augmented[len(baseline):]
                               if item.official_id == parent), None)
        if recovered_item is not None and recovered_item.passage_id is not None:
            passage = store.passage(recovered_item.passage_id)
            evidence_exact = normalize(registry[row["registry_id"]]["evidence_text"]) in normalize(passage.text)
        details.append({
            "cohort": row["cohort"], "query_id": qid, "doc_id": parent,
            "baseline_parent_count": len(baseline_parents),
            "augmented_parent_count": len(augmented_parents),
            "added_parent_count": len(added), "target_recovered": recovered,
            "target_in_augmented_pool": parent in augmented_parents,
            "recovered_passage_id": recovered_item.passage_id if recovered_item else None,
            "recovered_passage_contains_registered_evidence": evidence_exact,
            "frozen_prefix_unchanged": True,
        })
    with (output / "candidate_parent_replay.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    manual = [row for row in details if row["cohort"] == "manual_miss"]
    historical = [row for row in details if row["cohort"] == "historical_miss"]
    controls = [row for row in details if row["cohort"] == "control_retained"]
    checks = {
        "frozen_prefix_unchanged": all(row["frozen_prefix_unchanged"] for row in details),
        "manual_parent_recovered": sum(row["target_recovered"] for row in manual) >= 1,
        "recovered_manual_passage_exact": all(
            row["recovered_passage_contains_registered_evidence"]
            for row in manual if row["target_recovered"]),
        "controls_retained": all(row["target_in_augmented_pool"] for row in controls),
    }
    report = {
        "format": "v4-r7-parent-floor-candidate-gate-v1",
        "status": "POOL_GATE_PASS_SCORE_GATE_PENDING" if all(checks.values()) else "POOL_GATE_FAIL",
        "checks": checks,
        "manual_recovered": sum(row["target_recovered"] for row in manual),
        "historical_recovered": sum(row["target_recovered"] for row in historical),
        "control_losses": 0,
        "added_parent_pairs_minimum": sum(row["added_parent_count"] for row in details),
        "candidate_policy": {"mixed_item_prefix": 400, "unique_parent_floor": 400,
                             "existing_item_order_and_scores_unchanged": True},
        "candidate_parent_replay_sha256": sha256_file(output / "candidate_parent_replay.csv"),
        "score_gate": ("New parents have no frozen same-input reranker scores. Promotion requires a "
                       "small pinned-runtime score canary before downstream selection claims."),
        "official_f2": "UNMEASURED",
    }
    write_once(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def finalize(output: Path) -> None:
    destination = output / "execution_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    cohort = read(output / "cohort_lock.json")
    canary_report = read(output / "canary_report.json")
    trace_report = read(output / "trace_report.json")
    candidate_report = read(output / "candidate_gate.json")
    if (cohort["status"] != "LOCKED" or canary_report["status"] != "PASS"
            or trace_report["status"] != "PASS"
            or candidate_report["status"] != "POOL_GATE_PASS_SCORE_GATE_PENDING"):
        raise ValueError("R7 execution stages are incomplete")
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        cuda_available = False
    source_paths = (
        Path(__file__), ROOT / "src/medrag/retrieval/core.py",
        ROOT / "src/medrag/core/pipeline.py", ROOT / "src/medrag/core/disk_pipeline.py",
        ROOT / "tests/v4/test_parent_floor_fusion.py",
    )
    artifacts = (
        output / "cohort_lock.json", output / "label_source_check.csv",
        output / "canary_report.json", output / "canary_raw_branch_trace.jsonl",
        output / "trace_report.json", output / "parent_loss_attribution.csv",
        output / "candidate_gate.json", output / "candidate_parent_replay.csv",
    )
    report = {
        "format": "v4-r7-targeted-parent-loss-execution-v1",
        "status": "SCORE_GATE_PENDING" if cuda_available else "SCORE_GATE_BLOCKED_NO_PINNED_GPU",
        "cuda_available": cuda_available,
        "cohort_queries": trace_report["queries"],
        "manual_parents_recovered_at_pool": candidate_report["manual_recovered"],
        "historical_parents_recovered_at_pool": candidate_report["historical_recovered"],
        "control_losses_at_pool": candidate_report["control_losses"],
        "new_pairs_minimum_for_fair_cohort_score": candidate_report["added_parent_pairs_minimum"],
        "reranker_scores_produced": 0, "submission_created": False,
        "official_f2": "UNMEASURED",
        "decision": ("Keep opt-in parent-floor patch as a measured candidate. Do not promote or run full "
                     "inference until the 30-query cohort is scored with the pinned CUDA FP16 runtime."),
        "implementation_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                                  for path in source_paths},
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path)
                            for path in artifacts},
        "s2_zip_sha256": sha256_file(S2),
    }
    write_once(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lock", "canary", "trace", "candidate", "finalize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "lock":
        lock(args.output)
    elif args.command == "canary":
        canary(args.output)
    elif args.command == "trace":
        trace(args.output)
    elif args.command == "candidate":
        candidate(args.output)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
