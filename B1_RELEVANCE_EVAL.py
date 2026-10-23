"""Prepare blinded B1 review materials and reconcile independently supplied labels.

Uses frozen local artifacts only. No retrieval, GPU, or model calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "B1_FINAL_OUTPUT_REVIEW.csv"
MAPPING = ROOT / "outputs/v4/b1_final_output_evaluation_20261022/review_item_map.json"
MANIFEST = ROOT / "outputs/v4/b1_final_output_evaluation_20261022/review_packet_manifest.json"
DELTA = ROOT / "B1_FINAL_OUTPUT_DELTA.json"
MD = ROOT / "B1_REVIEW_PACKET.md"
TEMPLATE = ROOT / "B1_REVIEW_LABELS_TEMPLATE.csv"

LABEL_FIELDS = ["review_id", "reviewer_id", "review_timestamp_utc",
                "relevance_left_0_3", "relevance_right_0_3",
                "preference_left_right_equal_uncertain",
                "wrong_patient_left_yes_no_uncertain", "wrong_patient_right_yes_no_uncertain",
                "wrong_number_left_yes_no_na_uncertain", "wrong_number_right_yes_no_na_uncertain",
                "negation_left_yes_no_na_uncertain", "negation_right_yes_no_na_uncertain",
                "missing_context_left_yes_no_uncertain", "missing_context_right_yes_no_uncertain",
                "uncertainty_notes"]
CHOICES = {"preference_left_right_equal_uncertain": {"left", "right", "equal", "uncertain"}}
for stem in ("wrong_patient", "missing_context"):
    for side in ("left", "right"):
        CHOICES[f"{stem}_{side}_yes_no_uncertain"] = {"yes", "no", "uncertain"}
for stem in ("wrong_number", "negation"):
    for side in ("left", "right"):
        CHOICES[f"{stem}_{side}_yes_no_na_uncertain"] = {"yes", "no", "na", "uncertain"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate CSV columns: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Extra CSV cells: {path}")
    return names, rows


def frozen() -> tuple[list[dict[str, str]], dict[str, dict], dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    delta = json.loads(DELTA.read_text(encoding="utf-8"))
    _, rows = read_csv(PACKET)
    if digest(PACKET) != manifest["packet_sha256_before_review"]:
        raise ValueError("Frozen source packet hash mismatch")
    maps = {r["review_id"]: r for r in mapping["rows"]}
    if len(rows) != 23 or len(maps) != 23 or len({r["review_id"] for r in rows}) != 23:
        raise ValueError("Expected 23 unique pairs")
    if {r["review_id"] for r in rows} != set(maps):
        raise ValueError("Packet and private mapping IDs disagree")
    if delta["status"] != "PASS" or delta["summary"]["review_pairs"] != 23:
        raise ValueError("Frozen delta is not complete")
    if (delta["summary"]["rank_only_changes"] != 181
            or len(delta["rank_only_deltas"]) != 181
            or delta["summary"]["shared_top60_passage_content_changes"] != 16
            or delta["summary"]["document_promotions"] != 7
            or delta["summary"]["document_demotions"] != 7):
        raise ValueError("Frozen delta counts disagree")
    kinds = Counter(m["pair_type"] for m in maps.values())
    if kinds != {"SAME_PARENT_PASSAGE_CHANGE": 16, "TOP60_MEMBERSHIP_SWAP": 7}:
        raise ValueError(f"Unexpected pair types: {kinds}")
    if sum(bool(m["orientation_swapped"]) for m in maps.values()) != 11:
        raise ValueError("Blinding orientation mismatch")
    delta_ids = {r["opaque_review_id"] for r in delta["content_or_membership_deltas"]}
    if delta_ids != set(maps):
        raise ValueError("Delta and private mapping IDs disagree")
    query_texts: dict[int, str] = {}
    for row in rows:
        m = maps[row["review_id"]]
        q = int(m["query_id"])
        if not row["question"].strip():
            raise ValueError(f"Missing query text: {row['review_id']}")
        if q in query_texts and query_texts[q] != row["question"]:
            raise ValueError(f"Inconsistent query text: {q}")
        query_texts[q] = row["question"]
        if {m["left_arm"], m["right_arm"]} != {"A0", "B1"}:
            raise ValueError("Invalid private orientation")
        if bool(m["orientation_swapped"]) != (m["left_arm"] == "B1"):
            raise ValueError("Private orientation flag mismatch")
        if m["pair_type"] == "SAME_PARENT_PASSAGE_CHANGE":
            if m["a0"]["doc_id"] != m["b1"]["doc_id"] or m["a0"]["context_text"] == m["b1"]["context_text"]:
                raise ValueError("Invalid shared-parent passage pair")
        elif m["a0"]["doc_id"] == m["b1"]["doc_id"]:
            raise ValueError("Invalid top-60 membership pair")
        for side in ("left", "right"):
            evidence = m[m[f"{side}_arm"].lower()]
            if row[f"title_{side}"] != evidence["source_title"] or row[f"evidence_{side}"] != evidence["context_text"]:
                raise ValueError(f"Packet content mismatch: {row['review_id']} {side}")
            checks = evidence["grounding"]["checks"]
            if not evidence["grounding"]["valid"] or not all(checks.values()):
                raise ValueError(f"Failed source grounding: {row['review_id']} {side}")
            if len(evidence["context_text"]) != evidence["grounding"]["context_end"] - evidence["grounding"]["context_start"]:
                raise ValueError(f"Source span length mismatch: {row['review_id']} {side}")
    if len(query_texts) != 6:
        raise ValueError("Expected six distinct queries")
    for field in ("arm", "rank", "score", "doc_id", "official_id", "passage_id"):
        if any(field in name.lower() for name in rows[0]):
            raise ValueError(f"Reviewer packet leaks {field}")
    return rows, maps, delta


def write_preparation() -> None:
    rows, _, _ = frozen()
    if MD.exists() or TEMPLATE.exists():
        raise FileExistsError("Preparation outputs already exist")
    lines = ["# Blinded relevance review packet", "",
             "Review each Left/Right pair independently using only the question and passages below. "
             "Record labels in `B1_REVIEW_LABELS_TEMPLATE.csv`. Do not open the private mapping or delta files before labels are locked.", "",
             "Relevance: 0 = irrelevant or misleading; 1 = topical but does not answer; "
             "2 = partially answers with usable evidence; 3 = directly answers with sufficient evidence. "
             "Prefer the passage that better supports the question. Choose `equal` for genuinely equal quality, "
             "`uncertain` when evidence is insufficient. Safety fields refer to an error in that passage; "
             "use `na` only when numbers or negation are not applicable. "
             "Use `missing_context=yes` if necessary surrounding information is absent. "
             "Explain uncertainty in `uncertainty_notes`.", ""]
    for i, row in enumerate(rows, 1):
        lines += [f"## Pair {i}: {row['review_id']}", "", "### Question", "",
                  row["question"], ""]
        for side in ("left", "right"):
            lines += [f"### {side.title()} — {row[f'title_{side}']}", "",
                      row[f"evidence_{side}"], ""]
        lines += ["Labels: relevance Left/Right 0–3; preference Left/Right/Equal/Uncertain; "
                  "wrong patient, wrong number, negation, missing context for each side; uncertainty notes.", ""]
    MD.write_text("\n".join(lines), encoding="utf-8")
    with TEMPLATE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows({"review_id": row["review_id"]} for row in rows)
    print(f"Prepared {len(rows)} blinded pairs: {MD.name}, {TEMPLATE.name}")


def evaluate(labels_path: Path, output_path: Path) -> None:
    rows, maps, delta = frozen()
    names, labels = read_csv(labels_path)
    if names != LABEL_FIELDS or len(labels) != 23 or [r["review_id"] for r in labels] != [r["review_id"] for r in rows]:
        raise ValueError("Label template schema, row count, or order changed")
    counts = Counter()
    by_query: dict[int, Counter] = defaultdict(Counter)
    safety = Counter()
    incomplete = []
    invalid = []
    grade_sum = Counter()
    uncertain_ids = []
    for row in labels:
        rid = row["review_id"]
        filled = [row[k].strip() for k in LABEL_FIELDS if k not in {"review_id", "uncertainty_notes"}]
        if not any(filled):
            incomplete.append(rid)
            continue
        errors = []
        if not row["reviewer_id"].strip() or not row["review_timestamp_utc"].strip():
            errors.append("reviewer identity/timestamp required")
        for side in ("left", "right"):
            if row[f"relevance_{side}_0_3"].strip() not in {"0", "1", "2", "3"}:
                errors.append(f"relevance_{side}_0_3")
        for field, allowed in CHOICES.items():
            if row[field].strip().lower() not in allowed:
                errors.append(field)
        if errors:
            invalid.append({"review_id": rid, "fields": errors})
            continue
        m = maps[rid]
        q = int(m["query_id"])
        preference = row["preference_left_right_equal_uncertain"].strip().lower()
        outcome = ({"equal": "tie", "uncertain": "uncertain"}.get(preference)
                   or ("B1_win" if m[f"{preference}_arm"] == "B1" else "A0_win"))
        counts[outcome] += 1
        by_query[q][outcome] += 1
        by_query[q]["reviewed"] += 1
        if outcome == "uncertain":
            uncertain_ids.append(rid)
        for arm in ("A0", "B1"):
            side = "left" if m["left_arm"] == arm else "right"
            grade_sum[arm] += int(row[f"relevance_{side}_0_3"])
        a = "left" if m["left_arm"] == "A0" else "right"
        b = "left" if m["left_arm"] == "B1" else "right"
        for stem in ("wrong_patient", "wrong_number", "negation", "missing_context"):
            suffix = "yes_no_uncertain" if stem in {"wrong_patient", "missing_context"} else "yes_no_na_uncertain"
            va = row[f"{stem}_{a}_{suffix}"].strip().lower()
            vb = row[f"{stem}_{b}_{suffix}"].strip().lower()
            if va == "no" and vb == "yes":
                safety[f"B1_{stem}_regression"] += 1
            if va in {"uncertain", "na"} or vb in {"uncertain", "na"}:
                safety[f"{stem}_unresolved_or_na"] += 1
    result = {
        "decision": "BLOCKED_BY_EVALUATION",
        "scope": "six development queries; no independent holdout",
        "reviewed": sum(counts.values()), "unreviewed_ids": incomplete, "invalid_rows": invalid,
        "pairwise": {k: counts[k] for k in ("B1_win", "A0_win", "tie", "uncertain")},
        "uncertain_ids": uncertain_ids, "grade_sum": dict(grade_sum),
        "safety": dict(safety),
        "per_query": {str(q): {k: by_query[q][k] for k in ("reviewed", "B1_win", "A0_win", "tie", "uncertain")}
                      for q in sorted({int(m["query_id"]) for m in maps.values()})},
        "rank_only": {"changed": delta["summary"]["rank_only_changes"],
                      "official_rank_sensitivity": "UNVERIFIED"},
        "chunks_f2": None,
        "note": "No inferred labels. Pairwise labels are insufficient for complete top-60 CHUNKS F2. No automatic promotion."
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "B1_RELEVANCE_RESULTS.json")
    args = parser.parse_args()
    if args.prepare == bool(args.labels):
        parser.error("Choose exactly one of --prepare or --labels")
    if args.prepare:
        write_preparation()
    else:
        evaluate(args.labels, args.output)
