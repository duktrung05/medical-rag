"""Reconcile blinded B1 final-output reviews with the hidden A0/B1 mapping."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = ROOT / "B1_FINAL_OUTPUT_REVIEW.csv"
DEFAULT_MAP = ROOT / "outputs/v4/b1_final_output_evaluation_20261022/review_item_map.json"
DEFAULT_DELTA = ROOT / "B1_FINAL_OUTPUT_DELTA.json"

SCORED_FIELDS = [
    "direct_answer_left_0_1_2", "direct_answer_right_0_1_2",
    "clinical_accuracy_left_0_1_2", "clinical_accuracy_right_0_1_2",
    "context_completeness_left_0_1_2", "context_completeness_right_0_1_2",
]
CATEGORICAL_FIELDS = {
    "patient_consistency_left_yes_no_uncertain": {"yes", "no", "uncertain"},
    "patient_consistency_right_yes_no_uncertain": {"yes", "no", "uncertain"},
    "numeric_consistency_left_yes_no_na_uncertain": {"yes", "no", "na", "uncertain"},
    "numeric_consistency_right_yes_no_na_uncertain": {"yes", "no", "na", "uncertain"},
    "negation_correct_left_yes_no_na_uncertain": {"yes", "no", "na", "uncertain"},
    "negation_correct_right_yes_no_na_uncertain": {"yes", "no", "na", "uncertain"},
    "overall_preference_left_right_equal_uncertain": {"left", "right", "equal", "uncertain"},
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def side_value(row: dict, mapping: dict, stem: str, arm: str) -> str:
    side = "left" if mapping["left_arm"] == arm else "right"
    suffix = {
        "direct_answer": "0_1_2", "clinical_accuracy": "0_1_2",
        "context_completeness": "0_1_2",
        "patient_consistency": "yes_no_uncertain",
        "numeric_consistency": "yes_no_na_uncertain",
        "negation_correct": "yes_no_na_uncertain",
    }[stem]
    return row[f"{stem}_{side}_{suffix}"]


def markdown(result: dict) -> str:
    counts = result["judgments"]["overall_preference"]
    lines = [
        "# B1 Final-Output Relevance Evaluation — A0 vs B1",
        "",
        f"**Decision: `{result['decision']}`**",
        "",
        "This report reconciles the blinded final-output packet against the hidden A0/B1 map. "
        "The six queries are development diagnostics, not an independent holdout.",
        "",
        "## Evaluation status",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Review pairs | {result['review_status']['total_rows']} |",
        f"| Completed valid reviews | {result['review_status']['completed_valid_rows']} |",
        f"| Unreviewed rows | {result['review_status']['unreviewed_rows']} |",
        f"| Invalid/incomplete rows | {result['review_status']['invalid_or_incomplete_rows']} |",
        f"| B1 better | {counts['B1_BETTER']} |",
        f"| A0 better | {counts['A0_BETTER']} |",
        f"| Equal | {counts['EQUAL']} |",
        f"| Uncertain | {counts['UNCERTAIN']} |",
        "",
        "Unjudged outputs remain `UNKNOWN`; they are not counted as negatives or ties.",
        "Overall relevance comparison is unavailable until valid judgments exist.",
        "",
        "## Final-output delta",
        "",
        "- 16 shared top-60 parents have changed exported passage context.",
        "- 7 A0 documents leave top-60 and 7 B1 documents enter top-60.",
        "- 181 shared outputs move rank while retaining the same evidence content.",
        "- The reviewer packet contains 23 pairs: 16 passage changes and 7 membership swaps.",
        "- Every exported context passed source-span grounding checks.",
        "",
        "## Safety regressions",
        "",
        f"Safety comparisons assessed: {result['review_status']['completed_valid_rows']}/23 pairs. "
        "Zero on an unreviewed packet does not establish absence of regressions.",
        "",
        f"- Wrong-patient regressions: {result['judgments']['safety_regressions']['wrong_patient']}.",
        f"- Wrong-number regressions: {result['judgments']['safety_regressions']['wrong_number']}.",
        f"- Negation regressions: {result['judgments']['safety_regressions']['wrong_negation']}.",
        f"- Clinical-accuracy score regressions: {result['judgments']['safety_regressions']['clinical_accuracy']}.",
        "",
        "## Query-level comparison",
        "",
        "| Query | Judged | B1 better | A0 better | Equal | Uncertain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["judgments"]["per_query"]:
        lines.append(f"| {row['query_id']} | {row['judged']} | {row['B1_BETTER']} | "
                     f"{row['A0_BETTER']} | {row['EQUAL']} | {row['UNCERTAIN']} |")
    lines += [
        "",
        "## CHUNKS F2",
        "",
        f"Status: **{result['chunks_f2']['status']}**.",
        "",
        result["chunks_f2"]["reason"],
        "",
        "## Reviewer workflow",
        "",
        "Complete `B1_FINAL_OUTPUT_REVIEW.csv` without opening the hidden map. Then run:",
        "",
        "```bash",
        "PYTHONPATH=. .venv-phase2-batch/bin/python scripts/v4/reconcile_b1_final_output_reviews.py \\",
        "  --output-json B1_FINAL_OUTPUT_EVALUATION.completed.json \\",
        "  --output-report B1_FINAL_OUTPUT_RELEVANCE_EVALUATION.completed.md \\",
        "  --manifest outputs/v4/b1_final_output_evaluation_20261022/evaluation_manifest.completed.json",
        "```",
        "",
        "## Generalization and decision rule",
        "",
        "These findings cover only six development queries. The B1 runtime policy must remain frozen; "
        "holdout labels may evaluate it but must not be used to alter it.",
        "",
        "The preregistered reconciliation gate is: require all 23 valid reviews; choose "
        "`KEEP_A0` for any patient/number/negation regression or an A0 pairwise majority; choose "
        "`GO_B1_HOLDOUT` only for a B1 pairwise majority with non-inferior aggregate direct-answer "
        "and clinical-accuracy scores and no listed safety regression. Otherwise remain "
        "`BLOCKED_BY_EVALUATION`.",
        "",
        f"Current decision: **`{result['decision']}`**. {result['decision_reason']}",
        "",
        "No automatic promotion is authorized by this report.",
    ]
    return "\n".join(lines) + "\n"


def main(packet: Path, hidden_map_path: Path, delta_path: Path,
         output_json: Path, output_report: Path, manifest_path: Path) -> None:
    packet, hidden_map_path, delta_path = packet.resolve(), hidden_map_path.resolve(), delta_path.resolve()
    output_json, output_report = output_json.resolve(), output_report.resolve()
    manifest_path = manifest_path.resolve()
    if output_json.exists() or output_report.exists() or manifest_path.exists():
        raise FileExistsError("Evaluation outputs must be fresh")
    mappings = read(hidden_map_path)["rows"]
    mapping_by_id = {row["review_id"]: row for row in mappings}
    if len(mapping_by_id) != 23:
        raise ValueError("Expected 23 unique hidden review mappings")
    delta = read(delta_path)
    if delta["summary"]["review_pairs"] != 23:
        raise ValueError("Final-output delta/review map mismatch")
    with packet.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if len(rows) != 23 or {row["review_id"] for row in rows} != set(mapping_by_id):
        raise ValueError("Review packet IDs differ from hidden mapping")
    required = set(SCORED_FIELDS) | set(CATEGORICAL_FIELDS) | {"reviewer_id"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Review packet is missing fields: {sorted(required - set(fieldnames))}")

    outcomes = Counter({key: 0 for key in ("B1_BETTER", "A0_BETTER", "EQUAL", "UNCERTAIN")})
    by_query: dict[int, Counter] = defaultdict(Counter)
    completed = []
    invalid = []
    unreviewed = []
    safety = Counter({"wrong_patient": 0, "wrong_number": 0,
                      "wrong_negation": 0, "clinical_accuracy": 0})
    score_sums = Counter()
    for row in rows:
        review_id = row["review_id"]
        values = [row[field].strip().lower() for field in SCORED_FIELDS]
        values += [row[field].strip().lower() for field in CATEGORICAL_FIELDS]
        reviewer_id = row["reviewer_id"].strip()
        if not any(values) and not reviewer_id:
            unreviewed.append(review_id)
            continue
        errors = []
        for field in SCORED_FIELDS:
            if row[field].strip() not in {"0", "1", "2"}:
                errors.append(f"{field}=invalid")
        for field, allowed in CATEGORICAL_FIELDS.items():
            if row[field].strip().lower() not in allowed:
                errors.append(f"{field}=invalid")
        if not reviewer_id:
            errors.append("reviewer_id=missing")
        if errors:
            invalid.append({"review_id": review_id, "errors": errors})
            continue
        mapping = mapping_by_id[review_id]
        preference = row["overall_preference_left_right_equal_uncertain"].strip().lower()
        if preference == "equal":
            outcome = "EQUAL"
        elif preference == "uncertain":
            outcome = "UNCERTAIN"
        else:
            preferred_arm = mapping[f"{preference}_arm"]
            outcome = "B1_BETTER" if preferred_arm == "B1" else "A0_BETTER"
        outcomes[outcome] += 1
        by_query[int(mapping["query_id"])][outcome] += 1
        by_query[int(mapping["query_id"])]["judged"] += 1

        for metric in ("direct_answer", "clinical_accuracy", "context_completeness"):
            a0 = int(side_value(row, mapping, metric, "A0"))
            b1 = int(side_value(row, mapping, metric, "B1"))
            score_sums[f"a0_{metric}"] += a0
            score_sums[f"b1_{metric}"] += b1
        a0_patient = side_value(row, mapping, "patient_consistency", "A0").lower()
        b1_patient = side_value(row, mapping, "patient_consistency", "B1").lower()
        a0_number = side_value(row, mapping, "numeric_consistency", "A0").lower()
        b1_number = side_value(row, mapping, "numeric_consistency", "B1").lower()
        a0_negation = side_value(row, mapping, "negation_correct", "A0").lower()
        b1_negation = side_value(row, mapping, "negation_correct", "B1").lower()
        safety["wrong_patient"] += b1_patient == "no" and a0_patient == "yes"
        safety["wrong_number"] += b1_number == "no" and a0_number == "yes"
        safety["wrong_negation"] += b1_negation == "no" and a0_negation == "yes"
        safety["clinical_accuracy"] += (
            int(side_value(row, mapping, "clinical_accuracy", "B1"))
            < int(side_value(row, mapping, "clinical_accuracy", "A0")))
        completed.append({"review_id": review_id, "query_id": int(mapping["query_id"]),
                          "pair_type": mapping["pair_type"], "outcome": outcome,
                          "reviewer_id": reviewer_id})

    complete_set = len(completed) == 23 and not invalid and not unreviewed
    safety_total = safety["wrong_patient"] + safety["wrong_number"] + safety["wrong_negation"]
    if not complete_set:
        decision = "BLOCKED_BY_EVALUATION"
        reason = "The blinded packet has not been completely and validly reviewed."
    elif safety_total or outcomes["A0_BETTER"] > outcomes["B1_BETTER"]:
        decision = "KEEP_A0"
        reason = "The preregistered safety/quality gate does not support advancing B1."
    elif (outcomes["B1_BETTER"] > outcomes["A0_BETTER"]
          and score_sums["b1_direct_answer"] >= score_sums["a0_direct_answer"]
          and score_sums["b1_clinical_accuracy"] >= score_sums["a0_clinical_accuracy"]):
        decision = "GO_B1_HOLDOUT"
        reason = "The completed development review passes the preregistered gate; independent holdout evaluation is next, not promotion."
    else:
        decision = "BLOCKED_BY_EVALUATION"
        reason = "Completed judgments do not separate A0 and B1 under the preregistered gate."

    per_query = []
    for qid in [53, 130, 380, 754, 771, 1144]:
        counter = by_query[qid]
        per_query.append({"query_id": qid, "judged": counter["judged"],
                          **{key: counter[key] for key in
                             ("B1_BETTER", "A0_BETTER", "EQUAL", "UNCERTAIN")}})
    result = {
        "format": "medrag-v4-b1-final-output-relevance-evaluation-v1",
        "status": "COMPLETE" if complete_set else "PENDING_REVIEW",
        "scope": "six development queries; not an independent holdout",
        "review_status": {"total_rows": 23, "completed_valid_rows": len(completed),
                          "unreviewed_rows": len(unreviewed),
                          "invalid_or_incomplete_rows": len(invalid),
                          "unreviewed_review_ids": unreviewed,
                          "invalid_rows": invalid},
        "judgments": {
            "overall_preference": dict(outcomes), "per_query": per_query,
            "metric_score_sums": dict(score_sums),
            "safety_regressions": dict(safety), "completed_rows": completed,
            "unjudged_policy": "UNKNOWN_NOT_NEGATIVE",
        },
        "chunks_f2": {
            "status": "NOT_COMPUTED_INSUFFICIENT_COMPATIBLE_LABELS",
            "value": None,
            "reason": "The paired packet judges only changed outputs and supplies no compatible complete top-60 relevance set; absolute CHUNKS F2 would be invalid.",
        },
        "generalization": {
            "development_queries": 6, "independent_holdout_evaluated": False,
            "runtime_policy_tuned_on_holdout": False,
            "warning": "Development findings cannot establish holdout or official-F2 improvement.",
        },
        "decision": decision, "decision_reason": reason,
        "automatic_promotion": False,
        "provenance": {"review_packet": str(packet), "review_packet_sha256": sha(packet),
                       "hidden_map_sha256": sha(hidden_map_path),
                       "delta_sha256": sha(delta_path)},
    }
    write_json(output_json, result)
    write_text(output_report, markdown(result))
    write_json(manifest_path, {
        "format": "medrag-v4-b1-final-output-reconciliation-v1",
        "status": result["status"], "decision": decision,
        "inputs_sha256": {str(packet): sha(packet), str(hidden_map_path): sha(hidden_map_path),
                          str(delta_path): sha(delta_path)},
        "outputs_sha256": {str(output_json): sha(output_json),
                           str(output_report): sha(output_report)},
        "implementation_sha256": sha(Path(__file__)),
        "model_calls": 0, "retrieval_calls": 0, "gpu_used": False,
    })
    print(json.dumps({"status": result["status"], "decision": decision,
                      "review_status": result["review_status"],
                      "overall_preference": dict(outcomes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--delta", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--output-json", type=Path,
                        default=ROOT / "B1_FINAL_OUTPUT_EVALUATION.json")
    parser.add_argument("--output-report", type=Path,
                        default=ROOT / "B1_FINAL_OUTPUT_RELEVANCE_EVALUATION.md")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "outputs/v4/b1_final_output_evaluation_20261022/evaluation_manifest.json")
    args = parser.parse_args()
    main(args.review, args.map, args.delta, args.output_json, args.output_report, args.manifest)
