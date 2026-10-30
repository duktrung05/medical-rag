"""Build one fixed-policy DOCS_F2 challenger from cached multi-passage scores."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.v4.run_full_baseline import _all_official_ids
from src.medrag.adapters.validation import (canonical_payload, package_zip,
                                            validate_payload, validate_zip)
from src.medrag.core.config import load_config
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.ranking.document_relevance import (
    FEATURE_NAMES,
    conservative_document_swaps,
    document_features,
)
from src.medrag.validation.evidence_experiment import validate_split


ROOT = Path(__file__).resolve().parents[2]
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024"
CHAMPION = ROOT / "outputs/v4/evidence_window_cap10_full_20261025/export/predictions.json"
SPLIT_REGISTRY = ROOT / "outputs/v4/evidence_evaluation_lock_20261025/query_split_registry.json"
MANUAL = ROOT / "configs/vibiomir/phase2_retrieval_qrels_v2_20261015.json"
OLD_ZIP = ROOT / "outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip"
FEEDBACK = ROOT / "outputs/scored_feedback_recheck_20261009/JOINED_SCORED_CHUNKS.csv"
CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
MIGRATION = ROOT / "configs/vibiomir/phase2_batch_migration_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/v4/doc_f2_multipassage_20261026"

POLICY = {
    "name": "positive_unlabeled_logistic_multi_passage_v1",
    "training_split": "development",
    "positive_weight": 1.0,
    "unlabeled_weight": 0.02,
    "logistic_c": 0.2,
    "max_swaps_per_query": 3,
    "minimum_log_odds_margin": 1.0,
    "minimum_max_passage_score": -4.0,
    "preserve_document_count_per_query": True,
    "preserve_all_champion_chunks": True,
    "random_state": 20261026,
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def load_positive_documents() -> tuple[dict[int, set[int]], dict[tuple[int, int], set[str]]]:
    """Load only labels that prove a document contains a matched positive chunk."""
    with zipfile.ZipFile(OLD_ZIP) as archive:
        old_predictions = {row["id"]: row for row in json.loads(archive.read("predictions.json"))}
    positives: dict[int, set[int]] = defaultdict(set)
    provenance: dict[tuple[int, int], set[str]] = defaultdict(set)
    with FEEDBACK.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scoring_status"] != "scored" or row["relevant"] != "True":
                continue
            qid = int(row["query_id"])
            chunk = old_predictions[qid]["relevant_chunks"][int(row["chunk_order"])]
            text_hash = hashlib.sha256(chunk["chunk_text"].encode()).hexdigest()
            if (chunk["doc_id"] != int(row["baseline_doc_id_int"])
                    or text_hash != row["chunk_text_sha256"]):
                raise ValueError("Historical positive does not match its frozen chunk")
            doc_id = int(chunk["doc_id"])
            positives[qid].add(doc_id)
            provenance[qid, doc_id].add("historical_exact_positive_chunk")
    for row in read_json(MANUAL)["queries"]:
        qid = int(row["query_id"])
        if not 1 <= qid <= 1200:
            continue
        doc_id = int(row["doc_id"])
        positives[qid].add(doc_id)
        provenance[qid, doc_id].add("manual_query_document_evidence")
    return positives, provenance


def load_candidates() -> dict[int, list[dict]]:
    result = {}
    for qid in range(1, 1201):
        data = read_json(REFINED / "final" / f"{qid}.json")
        rows = []
        for rank, document in enumerate(data["trace"]["ranked_candidates"], 1):
            rows.append({
                "id": int(document["official_id"]),
                "max_passage_score": float(document["rerank_score"]),
                "original_rank": rank,
                "features": document_features(document),
            })
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate candidate parent for query {qid}")
        result[qid] = rows
        if qid % 100 == 0:
            print(f"Loaded candidate features {qid}/1200", flush=True)
    return result


def fit_model(candidates, positives, assignments):
    features, targets, weights = [], [], []
    counts = Counter()
    training_queries = {qid for qid in range(1, 1201)
                        if assignments[str(qid)] == POLICY["training_split"]}
    for qid in sorted(training_queries):
        positive_ids = positives.get(qid, set())
        for row in candidates[qid]:
            target = int(row["id"] in positive_ids)
            features.append(row["features"])
            targets.append(target)
            weights.append(POLICY["positive_weight"] if target else POLICY["unlabeled_weight"])
            counts["positive_rows" if target else "unlabeled_rows"] += 1
    if not counts["positive_rows"] or not counts["unlabeled_rows"]:
        raise ValueError("Training requires both positive and unlabeled candidate rows")
    matrix = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.int8)
    sample_weights = np.asarray(weights, dtype=np.float64)
    scaler = StandardScaler().fit(matrix, sample_weight=sample_weights)
    standardized = scaler.transform(matrix)
    model = LogisticRegression(
        C=POLICY["logistic_c"], solver="liblinear", max_iter=1000,
        random_state=POLICY["random_state"],
    ).fit(standardized, target_values, sample_weight=sample_weights)
    if model.n_iter_[0] >= model.max_iter:
        raise RuntimeError("Document model failed to converge")
    model_record = {
        "feature_names": list(FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
        "training_counts": dict(counts),
        "training_query_count": len(training_queries),
    }
    return scaler, model, model_record, training_queries


def label_report(baseline, challenger, positives, provenance, assignments):
    report = {}
    rows = []
    for split in ("development", "validation", "held_out"):
        for source in ("all_direct_positives", "manual_query_document_evidence",
                       "historical_exact_positive_chunk"):
            total = base_hits = new_hits = gains = losses = 0
            gain_queries, loss_queries = set(), set()
            for qid in range(1, 1201):
                if assignments[str(qid)] != split:
                    continue
                base_ids = set(baseline[qid]["relevant_docs"])
                new_ids = set(challenger[qid]["relevant_docs"])
                for doc_id in positives.get(qid, set()):
                    sources = provenance[qid, doc_id]
                    if source != "all_direct_positives" and source not in sources:
                        continue
                    total += 1
                    before, after = doc_id in base_ids, doc_id in new_ids
                    base_hits += before
                    new_hits += after
                    if after and not before:
                        gains += 1
                        gain_queries.add(qid)
                    if before and not after:
                        losses += 1
                        loss_queries.add(qid)
            key = f"{split}/{source}"
            report[key] = {
                "labeled_documents": total,
                "baseline_hits": base_hits,
                "challenger_hits": new_hits,
                "gross_gains": gains,
                "gross_losses": losses,
                "net_change": gains - losses,
                "gain_queries": len(gain_queries),
                "loss_queries": len(loss_queries),
            }
            rows.append({"split": split, "source": source, **report[key]})
    return report, rows


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    export = output / "export"
    export.mkdir()

    split_registry = read_json(SPLIT_REGISTRY)
    assignments = split_registry["assignments"]
    positives, provenance = load_positive_documents()
    training_ids = {qid for qid in range(1, 1201) if assignments[str(qid)] == "development"}
    validate_split(split_registry, training_ids)
    champion_rows = read_json(CHAMPION)
    if len(champion_rows) != 1200 or {row["id"] for row in champion_rows} != set(range(1, 1201)):
        raise ValueError("Champion predictions do not cover 1,200 official queries")
    baseline = {row["id"]: row for row in champion_rows}

    candidates = load_candidates()
    scaler, model, model_record, _ = fit_model(candidates, positives, assignments)
    challenger = {}
    diff_rows = []
    totals = Counter()
    for qid in range(1, 1201):
        base = baseline[qid]
        candidate_rows = candidates[qid]
        matrix = np.asarray([row["features"] for row in candidate_rows], dtype=np.float64)
        values = model.decision_function(scaler.transform(matrix))
        scores = {row["id"]: float(value) for row, value in zip(candidate_rows, values, strict=True)}
        compact_candidates = [{"official_id": row["id"],
                               "rerank_score": row["max_passage_score"]}
                              for row in candidate_rows]
        chunk_parents = {chunk["doc_id"] for chunk in base["relevant_chunks"]}
        selected, replacements = conservative_document_swaps(
            base["relevant_docs"], chunk_parents, compact_candidates, scores,
            max_swaps=POLICY["max_swaps_per_query"],
            minimum_margin=POLICY["minimum_log_odds_margin"],
            minimum_passage_score=POLICY["minimum_max_passage_score"],
        )
        row = copy.deepcopy(base)
        row["relevant_docs"] = selected
        challenger[qid] = row
        totals["queries"] += 1
        totals["documents"] += len(selected)
        totals["chunks"] += len(row["relevant_chunks"])
        totals["swaps"] += len(replacements)
        totals["changed_queries"] += bool(replacements)
        for replacement in replacements:
            diff_rows.append({"query_id": qid, **replacement,
                              "split": assignments[str(qid)]})

    structure_checks = {
        "same_query_ids": set(challenger) == set(baseline),
        "same_document_count_each_query": all(
            len(challenger[qid]["relevant_docs"]) == len(baseline[qid]["relevant_docs"])
            for qid in baseline),
        "chunks_byte_equal_each_query": all(
            challenger[qid]["relevant_chunks"] == baseline[qid]["relevant_chunks"]
            for qid in baseline),
        "all_chunk_parents_retained": all(
            {chunk["doc_id"] for chunk in challenger[qid]["relevant_chunks"]}
            <= set(challenger[qid]["relevant_docs"]) for qid in challenger),
    }
    if not all(structure_checks.values()):
        raise AssertionError("Document challenger changed a protected output contract")

    label_summary, label_rows = label_report(
        baseline, challenger, positives, provenance, assignments,
    )
    validation = label_summary["validation/all_direct_positives"]
    held_out = label_summary["held_out/all_direct_positives"]
    eval_manual_losses = (
        label_summary["validation/manual_query_document_evidence"]["gross_losses"]
        + label_summary["held_out/manual_query_document_evidence"]["gross_losses"]
    )
    gate_checks = {
        "candidate_changes_documents": totals["swaps"] > 0,
        "validation_positive_net_gain": validation["net_change"] > 0,
        "held_out_nonnegative": held_out["net_change"] >= 0,
        "evaluation_manual_zero_loss": eval_manual_losses == 0,
        "structural_contracts": all(structure_checks.values()),
    }
    decision = "GO_EXPORT" if all(gate_checks.values()) else "NO_GO"

    input_paths = [REFINED / "run_manifest.json", CHAMPION, SPLIT_REGISTRY, MANUAL,
                   OLD_ZIP, FEEDBACK, CONFIG, MIGRATION, Path(__file__).resolve(),
                   ROOT / "src/medrag/ranking/document_relevance.py"]
    manifest = {
        "format": "v4-document-relevance-challenger-v1",
        "status": decision,
        "policy": POLICY,
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths},
        "model": model_record,
        "counts": dict(totals),
        "structure_checks": structure_checks,
        "label_summary": label_summary,
        "gate_checks": gate_checks,
        "label_scope": (
            "Positive-unlabeled evaluation. Manual query-document evidence and exact historical "
            "positive chunks prove positives but do not enumerate all relevant documents."
        ),
        "official_f2": "UNMEASURED",
    }
    write_json(output / "policy.json", POLICY)
    write_json(output / "model.json", model_record)
    write_json(output / "manifest.json", manifest)
    with (output / "document_changes.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["query_id", "split", "victim", "entrant", "margin"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diff_rows)
    with (output / "label_evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(label_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(label_rows)

    if decision == "GO_EXPORT":
        ordered = [challenger[qid] for qid in range(1, 1201)]
        payload = canonical_payload(ordered)
        config = load_config(CONFIG)
        policy = config["selection"]
        from src.medrag.evidence.selector import SelectionPolicy
        selection_policy = SelectionPolicy(**policy)
        migration = read_json(MIGRATION)
        official_ids = _all_official_ids(ROOT / migration["official_corpus"])
        validate_payload(payload, query_ids=set(range(1, 1201)), official_ids=official_ids,
                         policy=selection_policy, result_limit=200)
        predictions_path = export / "predictions.json"
        predictions_path.write_bytes(payload)
        package_zip(export / "submission.zip", payload)
        zip_report = validate_zip(
            export / "submission.zip", query_ids=set(range(1, 1201)), official_ids=official_ids,
            policy=selection_policy, result_limit=200, max_bytes=100_000_000,
        )
        if zip_report["payload_sha256"] != hashlib.sha256(payload).hexdigest():
            raise AssertionError("Packaged payload differs from validated predictions")
        manifest["validation"] = zip_report
        manifest["status"] = "LOCAL_EXPORT_PASS"
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": dict(totals),
                      "gate_checks": gate_checks, "validation": validation,
                      "held_out": held_out}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)
