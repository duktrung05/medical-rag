"""Lock independent blinded labels before joining them to the private P2 item map."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = {"item_id", "answers_question_0_1_2", "patient_context_match_yes_no_unknown",
          "numeric_match_yes_no_na", "contradicts_query_yes_no", "source_grounded_yes_no",
          "reviewer_notes"}
CHOICES = {
    "answers_question_0_1_2": {"0", "1", "2"},
    "patient_context_match_yes_no_unknown": {"yes", "no", "unknown"},
    "numeric_match_yes_no_na": {"yes", "no", "na"},
    "contradicts_query_yes_no": {"yes", "no"},
    "source_grounded_yes_no": {"yes", "no"},
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def packet_ids(packet: Path) -> set[str]:
    manifest = read_json(packet / "manifest.json")
    if manifest["status"] != "template_only_unreviewed":
        raise ValueError("Unexpected packet status")
    for name, digest in manifest["artifacts_sha256"].items():
        if name == "item_map_for_analysis.json":
            continue  # Never open even for hashing until all labels are locked.
        if sha(packet / name) != digest:
            raise ValueError(f"Packet changed: {name}")
    ids = set()
    for line in (packet / "review_items.jsonl").open(encoding="utf-8"):
        item = json.loads(line)
        if item["item_id"] in ids:
            raise ValueError("Duplicate packet item")
        ids.add(item["item_id"])
    if len(ids) != manifest["items"]:
        raise ValueError("Packet item count changed")
    return ids


def read_labels(path: Path, allowed: set[str], *, complete: bool) -> dict[str, dict]:
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Invalid annotation columns: {path}")
        for row in reader:
            item_id = row["item_id"]
            if item_id not in allowed or item_id in labels:
                raise ValueError(f"Unknown or duplicate item_id: {item_id}")
            main = row["answers_question_0_1_2"].strip()
            if not main:
                if complete or any(row[field].strip() for field in CHOICES if field != "answers_question_0_1_2"):
                    raise ValueError(f"Incomplete annotation: {item_id}")
                continue
            for field, choices in CHOICES.items():
                if row[field].strip() not in choices:
                    raise ValueError(f"Invalid {field} for {item_id}")
            labels[item_id] = {field: row[field].strip() for field in FIELDS}
    if complete and set(labels) != allowed:
        raise ValueError(f"Expected {len(allowed)} complete labels, found {len(labels)}")
    return labels


def make_secondary(packet: Path, output: Path, count: int) -> None:
    ids = packet_ids(packet)  # Deliberately does not access the private map.
    if count <= 0 or count > len(ids) or output.exists():
        raise ValueError("Invalid secondary review size or existing output")
    assigned = set(sorted(ids, key=lambda item: hashlib.sha256(
        f"p3-secondary-v1:{item}".encode()).hexdigest())[:count])
    output.mkdir(parents=True)
    with (packet / "annotation_template.csv").open(encoding="utf-8", newline="") as source, \
            (output / "secondary_annotation_template.csv").open("w", encoding="utf-8",
                                                                     newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(row for row in reader if row["item_id"] in assigned)
    with (packet / "review_items.jsonl").open(encoding="utf-8") as source, \
            (output / "secondary_review_items.jsonl").open("w", encoding="utf-8") as target:
        for line in source:
            if json.loads(line)["item_id"] in assigned:
                target.write(line)
    write_json(output / "assignment.json", {
        "status": "assigned_before_review", "packet_manifest_sha256": sha(packet / "manifest.json"),
        "count": count, "item_ids": sorted(assigned),
        "selection": "first SHA-256(p3-secondary-v1:item_id) items",
        "template_sha256": sha(output / "secondary_annotation_template.csv"),
        "review_items_sha256": sha(output / "secondary_review_items.jsonl"),
    })


def lock(packet: Path, primary: Path, adjudicated: Path, secondary: Path,
         secondary_assignment: Path,
         output: Path) -> None:
    ids = packet_ids(packet)  # Does not open the private item map.
    primary_labels = read_labels(primary, ids, complete=True)
    final_labels = read_labels(adjudicated, ids, complete=True)
    secondary_labels = read_labels(secondary, ids, complete=False)
    assignment = read_json(secondary_assignment)
    if (assignment["status"] != "assigned_before_review" or
            assignment["packet_manifest_sha256"] != sha(packet / "manifest.json") or
            assignment["template_sha256"] != sha(
                secondary_assignment.parent / "secondary_annotation_template.csv") or
            assignment["review_items_sha256"] != sha(
                secondary_assignment.parent / "secondary_review_items.jsonl") or
            set(secondary_labels) != set(assignment["item_ids"]) or
            len(secondary_labels) != assignment["count"]):
        raise ValueError("Secondary review differs from preassigned blind subset")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    files = {"primary": primary, "adjudicated": adjudicated,
             "secondary": secondary, "secondary_assignment": secondary_assignment}
    write_json(output / "locked_labels.json", {
        "status": "locked_before_unblinding", "packet": str(packet.resolve()),
        "packet_manifest_sha256": sha(packet / "manifest.json"),
        "files": {key: {"path": str(path.resolve()), "sha256": sha(path)}
                  for key, path in files.items()},
        "counts": {"packet": len(ids), "primary": len(primary_labels),
                   "adjudicated": len(final_labels), "secondary": len(secondary_labels)},
    })


def analyze(lock_dir: Path, replay: Path) -> None:
    lock_path = lock_dir / "locked_labels.json"
    frozen = read_json(lock_path)
    if frozen["status"] != "locked_before_unblinding":
        raise ValueError("Labels have not been locked")
    packet = Path(frozen["packet"])
    if sha(packet / "manifest.json") != frozen["packet_manifest_sha256"]:
        raise ValueError("Review packet manifest changed")
    ids = packet_ids(packet)
    labels = {}
    for role, spec in frozen["files"].items():
        path = Path(spec["path"])
        if sha(path) != spec["sha256"]:
            raise ValueError(f"Locked {role} labels changed")
        if role != "secondary_assignment":
            labels[role] = read_labels(path, ids, complete=role != "secondary")
    # The private map is first opened only after all label and packet hashes pass.
    map_path = packet / "item_map_for_analysis.json"
    expected_map_sha = read_json(packet / "manifest.json")["artifacts_sha256"][map_path.name]
    if sha(map_path) != expected_map_sha:
        raise ValueError("Private map changed")
    mapping = read_json(map_path)
    by_id = {item["item_id"]: item for item in mapping}
    if set(by_id) != ids or len(mapping) != len(ids):
        raise ValueError("Private map does not match blinded packet")
    final = labels["adjudicated"]
    arm_counts = defaultdict(lambda: Counter(total=0, relevance_2=0))
    arm_query_counts = defaultdict(lambda: Counter(total=0, relevance_2=0))
    for item_id, item in by_id.items():
        for arm in item["arms"]:
            if arm.startswith("P2-"):
                arm_counts[arm]["total"] += 1
                arm_counts[arm]["relevance_2"] += final[item_id]["answers_question_0_1_2"] == "2"
                key = (arm, int(item["query_id"]))
                arm_query_counts[key]["total"] += 1
                arm_query_counts[key]["relevance_2"] += final[item_id]["answers_question_0_1_2"] == "2"
    for (arm, qid), count in arm_query_counts.items():
        if count["total"] != 60:
            raise ValueError(f"Expected 60 exported chunks for {arm}/{qid}, got {count['total']}")
    arm_metrics = {arm: {"reviewed_unique_items": count["total"],
                         "relevance_2": count["relevance_2"],
                         "evidence_precision_at_60_micro": count["relevance_2"] / count["total"],
                         "evidence_precision_at_60_macro": statistics.mean(
                             query_count["relevance_2"] / 60 for (query_arm, _), query_count
                             in arm_query_counts.items() if query_arm == arm)}
                   for arm, count in sorted(arm_counts.items())}
    paired = set(labels.get("secondary", {}))
    agreement = None
    if paired:
        same = sum(labels["primary"][item]["answers_question_0_1_2"] ==
                   labels["secondary"][item]["answers_question_0_1_2"] for item in paired)
        first = Counter(labels["primary"][item]["answers_question_0_1_2"] for item in paired)
        second = Counter(labels["secondary"][item]["answers_question_0_1_2"] for item in paired)
        expected = sum(first[str(level)] * second[str(level)] for level in range(3)) / len(paired) ** 2
        observed = same / len(paired)
        agreement = {"double_reviewed": len(paired), "exact_agreement": observed,
                     "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None}
    # Scores are joined by exact query, parent and seed passage identity.
    scores = {}
    with (replay / "pair_allocation.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["cached_rerank_score"]:
                scores[(int(row["query_id"]), int(row["doc_id"]), row["passage_id"])] = float(
                    row["cached_rerank_score"])
    distributions = defaultdict(list)
    parent_items = defaultdict(list)
    for item_id, item in by_id.items():
        level = final[item_id]["answers_question_0_1_2"]
        key = (int(item["query_id"]), int(item["doc_id"]), item["seed_passage_id"])
        if key in scores:
            distributions[level].append(scores[key])
        parent_items[key[:2]].append((item_id, level, key[2]))
    separation = {level: {"n": len(values), "mean_score": statistics.mean(values),
                          "median_score": statistics.median(values)}
                  for level, values in sorted(distributions.items()) if values}
    relevant = distributions["2"]
    other = distributions["0"] + distributions["1"]
    pairwise_auc = (sum(1 if positive > negative else .5 if positive == negative else 0
                        for positive in relevant for negative in other) / (len(relevant) * len(other))
                    if relevant and other else None)
    multi = [items for items in parent_items.values() if len({x[2] for x in items}) > 1]
    best_available = {parent: max(int(level) for _, level, _ in items)
                      for parent, items in parent_items.items()}
    within_parent = {"parents_with_multiple_reviewed_passages": len(multi),
                     "selected_at_best_reviewed_level": {},
                     "scope": "reviewed candidate passages only; not corpus-wide recall"}
    for arm in sorted(arm_metrics):
        eligible = chosen = 0
        for parent, items in parent_items.items():
            if len({x[2] for x in items}) <= 1:
                continue
            selected = [int(level) for item_id, level, _ in items if arm in by_id[item_id]["arms"]]
            if selected:
                eligible += 1
                chosen += max(selected) == best_available[parent]
        within_parent["selected_at_best_reviewed_level"][arm] = {
            "numerator": chosen, "denominator": eligible,
            "fraction": chosen / eligible if eligible else None}
    write_json(lock_dir / "p3_metrics.json", {
        "status": "pilot_reviewed", "label_lock_sha256": sha(lock_path),
        "arm_evidence_precision": arm_metrics, "reviewer_agreement": agreement,
        "per_query_evidence_precision": [
            {"arm": arm, "query_id": qid, "relevance_2": count["relevance_2"],
             "denominator": count["total"], "precision_at_60": count["relevance_2"] / 60}
            for (arm, qid), count in sorted(arm_query_counts.items())],
        "score_by_relevance": separation,
        "score_separation_auc_relevance2_vs_0_1": pairwise_auc,
        "scored_review_items": sum(len(values) for values in distributions.values()),
        "within_parent": within_parent,
        "limitations": ["Six pilot queries only", "Selected/reviewed candidate pool is not an independent recall set",
                        "Unique reviewed items may represent more than one arm; arm denominators need per-query check"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assignment = sub.add_parser("make-secondary")
    assignment.add_argument("--packet", type=Path, required=True)
    assignment.add_argument("--output", type=Path, required=True)
    assignment.add_argument("--count", type=int, default=102)
    freeze = sub.add_parser("lock")
    freeze.add_argument("--packet", type=Path, required=True)
    freeze.add_argument("--primary", type=Path, required=True)
    freeze.add_argument("--adjudicated", type=Path, required=True)
    freeze.add_argument("--secondary", type=Path, required=True)
    freeze.add_argument("--secondary-assignment", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("analyze")
    report.add_argument("--lock-dir", type=Path, required=True)
    report.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "make-secondary":
        make_secondary(args.packet, args.output, args.count)
    elif args.command == "lock":
        lock(args.packet, args.primary, args.adjudicated, args.secondary,
             args.secondary_assignment, args.output)
    else:
        analyze(args.lock_dir, args.replay)
