"""One allocation-only pilot: prepare blinded items, fit on dev, confirm on val/held-out.

Uses cached S2 contexts and a fixed CPU logistic allocator. No extraction,
retrieval, GPU inference or submission export. Labels must be direct independent
judgments of the exact review items, including both positive and negative items.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.v4.lock_evidence_evaluation import OUTPUT as EVALUATION, TRAIN
from scripts.v4.run_r8_evidence_extraction import ROOT, REFINED, R5_CONTEXTS, S2, load_queries, load_s2
from scripts.v4.refine_a0_submission import environment
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import text_sha256
from src.medrag.validation.evidence_experiment import allocation_gate, paired_changes, validate_split

OUTPUT = ROOT / "outputs/v4/s2_evidence_allocation_20261025"
FEATURES = ["seed_logit", "log_context_characters", "log_seed_support"]
PILOT_COUNTS = {"development": 60, "validation": 30, "held_out": 30}


def read(path: Path):
    return json.loads(path.read_text())


def train_ids() -> set[int]:
    return {int(json.loads(line)["query_id"]) for line in TRAIN.read_text().splitlines()}


def validate_frozen_query(rows: list[dict], frozen: dict, sources: dict, ranked: list[int]) -> bool:
    docs = [int(doc) for doc in frozen["relevant_docs"]]
    if (len(docs) != len(set(docs)) or not 60 <= len(ranked) <= 200
            or len(set(ranked)) != len(ranked)):
        raise ValueError("S2 document or ranked-parent cardinality differs")
    if not set(docs) <= set(ranked) or {row["doc_id"] for row in rows} != set(docs) or len(rows) != len(docs):
        raise ValueError("Allocation pool must preserve S2 documents and one frozen context per parent")
    baseline = {int(row["doc_id"]): row["chunk_text"] for row in frozen["relevant_chunks"]}
    if len(baseline) != 60 or len(frozen["relevant_chunks"]) != 60 or not set(baseline) <= set(docs):
        raise ValueError("S2 baseline must have 60 unique chunk parents")
    linked = {}
    for row in rows:
        source = sources[row["doc_id"]]
        start, end = row["start"], row["end"]
        if (not 0 <= start < end <= len(source.text) or source.text[start:end] != row["text"]
                or row["source_sha256"] != source.text_sha256
                or row["text_sha256"] != text_sha256(row["text"])
                or any(not math.isfinite(value) for value in row["features"])):
            raise ValueError("Frozen allocation context failed source/feature validation")
        if row["baseline"]:
            linked[row["doc_id"]] = row["text"]
    if linked != baseline:
        raise ValueError("Baseline context identity changed")
    return True


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    registry = read(EVALUATION / "query_split_registry.json")
    if sha256_file(TRAIN) != registry["r3_training_sha256"]:
        raise ValueError("Training membership changed after split lock")
    validate_split(registry, train_ids())
    audit = read(EVALUATION / "audit_report.json")
    if audit["s2_zip_sha256"] != sha256_file(S2) or audit["f03"]["corrected_roundtrip_mismatch"]:
        raise ValueError("Phase A audit differs")
    output.mkdir(parents=True)
    groups = defaultdict(list)
    for qid, part in registry["assignments"].items():
        groups[part].append(int(qid))
    pilot = {part: sorted(ids, key=lambda qid: hashlib.sha256(
        f"allocation-pilot-v1:{qid}".encode()).hexdigest())[:PILOT_COUNTS[part]]
        for part, ids in groups.items()}
    s2, queries = load_s2(), load_queries()
    _, store, _, _ = environment()
    context_manifest = read(R5_CONTEXTS / "manifest.json")
    final_manifest = read(REFINED / "run_manifest.json")
    private, file_hashes, count = {}, {}, Counter()
    for part, ids in pilot.items():
        records = []
        review_path, labels_path = output / f"review_{part}.jsonl", output / f"labels_{part}.csv"
        with review_path.open("x") as review, labels_path.open("x", newline="") as label_stream:
            writer = csv.DictWriter(label_stream, fieldnames=["item_id", "relevant", "reviewer_id", "judgment_basis"])
            writer.writeheader()
            for qid in sorted(ids):
                context_path, final_path = R5_CONTEXTS / "queries" / f"{qid}.json", REFINED / "final" / f"{qid}.json"
                if (sha256_file(context_path) != context_manifest["query_sha256"][str(qid)]
                        or sha256_file(final_path) != final_manifest["final_sha256"][str(qid)]):
                    raise ValueError("Frozen query input hash differs")
                contexts, final = read(context_path), read(final_path)
                if contexts["candidate_final_sha256"] != sha256_file(final_path):
                    raise ValueError("Context lineage differs from refined query")
                ranked = [int(row["official_id"]) for row in final["selected"]]
                docs = [int(doc) for doc in s2[qid]["relevant_docs"]]
                sources = store.sources_batch(set(docs))
                baseline = {int(row["doc_id"]): row["chunk_text"] for row in s2[qid]["relevant_chunks"]}
                by_doc = defaultdict(list)
                for row in contexts["contexts"]:
                    by_doc[int(row["official_id"])].append(row)
                rows = []
                for rank, doc in enumerate(docs):
                    options = by_doc[doc]
                    if doc in baseline:
                        options = [row for row in options if row["text"] == baseline[doc]]
                    if not options:
                        raise ValueError("Frozen context is missing")
                    # Lock context choice before judgments. Baseline texts remain exact.
                    chosen = min(options, key=lambda row: (-max(seed["baseline_rerank_score"]
                        for seed in row["seed_passages"]), row["context_id"]))
                    scores = [float(seed["baseline_rerank_score"]) for seed in chosen["seed_passages"]]
                    item = text_sha256(f"allocation-v1:{qid}:{doc}:{chosen['text_sha256']}")
                    row = {"query_id": qid, "doc_id": doc, "document_rank": rank,
                           "item_id": item, "baseline": doc in baseline,
                           "features": [max(scores), math.log1p(len(chosen["text"])), math.log1p(len(scores))],
                           **{key: chosen[key] for key in ("text", "text_sha256", "source_sha256", "start", "end")}}
                    rows.append(row)
                validate_frozen_query(rows, s2[qid], sources, ranked)
                records.append({"query_id": qid, "ranked_parents": ranked, "documents": docs,
                                "rows": rows})
                # Opaque item order; reviewers see no scores, ranks or arm identity.
                for row in sorted(rows, key=lambda value: value["item_id"]):
                    review.write(json.dumps({"item_id": row["item_id"], "query": queries[qid],
                                             "context": row["text"]}, ensure_ascii=False) + "\n")
                    writer.writerow({"item_id": row["item_id"]})
                count[part] += len(rows)
        private_path = output / f"private_{part}.json"
        write_once(private_path, records)
        private[part] = private_path.name
        file_hashes.update({path.name: sha256_file(path) for path in (private_path, review_path)})
    config = {"format": "s2-evidence-allocation-pilot-v1", "status": "AWAITING_INDEPENDENT_JUDGMENTS",
              "query_ids": pilot, "judgment_items": dict(count), "features": FEATURES,
              "private_files": private, "files_sha256": file_hashes,
              "split_registry_sha256": sha256_file(EVALUATION / "query_split_registry.json"),
              "s2_sha256": sha256_file(S2), "inputs_sha256": {
                  str(path.relative_to(ROOT)): sha256_file(path) for path in (
                      R5_CONTEXTS / "manifest.json", REFINED / "run_manifest.json", TRAIN)},
              "implementation_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                  Path(__file__), ROOT / "src/medrag/validation/evidence_experiment.py")},
              "allocator": {"type": "fixed_cpu_logistic", "iterations": 400, "step": 0.1, "l2": 0.01},
              "loss_budget": "floor(0.02 * evaluated queries), counted by gross-loss queries",
              "context_policy": "one frozen existing context per parent; exact baseline fallback",
              "allocation_domain": "S2 exported documents within unchanged ranked top-200 parents",
              "metric": "macro judged relevant chunks / 60 in this frozen pool; not official recall or F2",
              "held_out_scope": "Prospective independent judgments; historically inspected queries are not pristine",
              "judge_protocol": "Read query and entire exact context; label evidence relevance directly. Do not copy parent or historical labels.",
              "prohibited": ["new retrieval", "new extraction", "GPU inference", "R3 fine tuning", "submission export"]}
    write_once(output / "experiment_lock.json", config)
    print(json.dumps({"status": config["status"], "judgment_items": dict(count),
                      "queries": {key: len(ids) for key, ids in pilot.items()}}))


def load_experiment(output: Path) -> dict:
    decision_path = output / "decision.json"
    if decision_path.exists() and read(decision_path).get("decision") == "STOP_BEFORE_ANNOTATION":
        raise ValueError("Pilot stopped before annotation after full S2 signal audit")
    config = read(output / "experiment_lock.json")
    if (sha256_file(S2) != config["s2_sha256"]
            or sha256_file(EVALUATION / "query_split_registry.json") != config["split_registry_sha256"]):
        raise ValueError("Frozen S2 or split registry changed")
    for name, expected in config["files_sha256"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Pilot input changed: {name}")
    for name, expected in {**config["inputs_sha256"], **config["implementation_sha256"]}.items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Experiment lineage changed: {name}")
    validate_split(read(EVALUATION / "query_split_registry.json"), train_ids())
    return config


def load_judgments(path: Path, records: list[dict]) -> dict[str, bool]:
    expected = {row["item_id"] for record in records for row in record["rows"]}
    result = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            item = row["item_id"]
            if (item not in expected or item in result or row["relevant"] not in {"true", "false"}
                    or not row["reviewer_id"].strip() or row["judgment_basis"] != "direct_context_review"):
                raise ValueError("Independent exact-context judgments are incomplete, invalid or duplicated")
            result[item] = row["relevant"] == "true"
    if set(result) != expected:
        raise ValueError("Every candidate context needs an independent judgment")
    return result


def scores_and_changes(records: list[dict], model: dict, labels: dict[str, bool]) -> list[dict]:
    import numpy as np
    mean, scale = np.array(model["mean"]), np.array(model["scale"])
    weights = np.array(model["weights"])
    changes = []
    for record in records:
        rows = record["rows"]
        ranked = sorted(rows, key=lambda row: (-float(((np.array(row["features"]) - mean) / scale) @ weights),
                                               row["document_rank"], row["doc_id"]))
        chosen = ranked[:60]
        if len(chosen) != 60 or len({row["doc_id"] for row in chosen}) != 60:
            raise ValueError("Allocator violated 60 unique parents")
        base_hits = {row["item_id"] for row in rows if row["baseline"] and labels[row["item_id"]]}
        hits = {row["item_id"] for row in chosen if labels[row["item_id"]]}
        changes.append({"query_id": record["query_id"], **paired_changes(base_hits, hits),
                        "baseline_relevant": len(base_hits), "challenger_relevant": len(hits),
                        "chosen_item_ids": [row["item_id"] for row in chosen]})
    return changes


def validate_records(records: list[dict]) -> bool:
    s2 = load_s2()
    _, store, _, _ = environment()
    for record in records:
        qid = record["query_id"]
        if record["documents"] != [int(doc) for doc in s2[qid]["relevant_docs"]]:
            raise ValueError("S2 document list changed")
        final = read(REFINED / "final" / f"{qid}.json")
        if record["ranked_parents"] != [int(row["official_id"]) for row in final["selected"]]:
            raise ValueError("Ranked 200 parents changed")
        validate_frozen_query(record["rows"], s2[qid], store.sources_batch(set(record["documents"])),
                              record["ranked_parents"])
    return bool(records)


def fit(output: Path, labels_path: Path) -> None:
    import numpy as np
    if (output / "model_lock.json").exists():
        raise FileExistsError(output / "model_lock.json")
    config = load_experiment(output)
    records = read(output / config["private_files"]["development"])
    structure = validate_records(records)
    labels = load_judgments(labels_path, records)
    rows = [row for record in records for row in record["rows"]]
    x = np.array([row["features"] for row in rows], dtype=float)
    y = np.array([labels[row["item_id"]] for row in rows], dtype=float)
    if len(set(y)) != 2:
        raise ValueError("Development needs both positive and negative exact-context judgments")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    weights, intercept = np.zeros(x.shape[1]), 0.0
    policy = config["allocator"]
    for _ in range(policy["iterations"]):
        residual = 1 / (1 + np.exp(-np.clip(x @ weights + intercept, -40, 40))) - y
        weights -= policy["step"] * (x.T @ residual / len(y) + policy["l2"] * weights)
        intercept -= policy["step"] * residual.mean()
    model = {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
             "intercept": float(intercept)}
    changes = scores_and_changes(records, model, labels)
    gate = allocation_gate(changes, independent=True, structure_valid=structure, loss_budget=len(records) // 50)
    lock = {"status": "LOCKED_BEFORE_VALIDATION" if gate["decision"] == "GO" else "DEVELOPMENT_NO_GO",
            "experiment_lock_sha256": sha256_file(output / "experiment_lock.json"),
            "development_labels_sha256": sha256_file(labels_path), "model": model,
            "development_gate": gate, "validation_labels_read": False,
            "scope": "Development fit is exploratory; independent confirmatory validation is required"}
    write_once(output / "model_lock.json", lock)
    write_once(output / "development_changes.json", changes)
    print(json.dumps({"status": lock["status"], "gate": gate}))


def evaluate(output: Path, labels_path: Path, part: str) -> None:
    result_path = output / f"{part}_evaluation.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    config = load_experiment(output)
    lock = read(output / "model_lock.json")
    if (lock["status"] != "LOCKED_BEFORE_VALIDATION" or lock["validation_labels_read"] is not False
            or lock["experiment_lock_sha256"] != sha256_file(output / "experiment_lock.json")):
        raise ValueError("Development did not lock an eligible challenger")
    if part == "held_out":
        validation = read(output / "validation_evaluation.json")
        if (validation["gate"]["decision"] != "GO"
                or validation["model_lock_sha256"] != sha256_file(output / "model_lock.json")):
            raise ValueError("Held-out requires a passing validation of the identical locked model")
    records = read(output / config["private_files"][part])
    structure = validate_records(records)
    labels = load_judgments(labels_path, records)
    changes = scores_and_changes(records, lock["model"], labels)
    gate = allocation_gate(changes, independent=True, structure_valid=structure, loss_budget=len(records) // 50)
    report = {"partition": part, "gate": gate, "query_level_changes": changes,
              "baseline_macro_precision_at_60": sum(row["baseline_relevant"] for row in changes) / (60 * len(changes)),
              "challenger_macro_precision_at_60": sum(row["challenger_relevant"] for row in changes) / (60 * len(changes)),
              "model_lock_sha256": sha256_file(output / "model_lock.json"),
              "independent_judgments_sha256": sha256_file(labels_path),
              "official_f2": "UNMEASURED", "submission_created": False}
    write_once(result_path, report)
    print(json.dumps({"partition": part, "gate": gate}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "fit-dev", "evaluate-validation", "evaluate-held-out"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.labels is None:
        parser.error("--labels requires completed independent exact-context judgments")
    elif args.command == "fit-dev":
        fit(args.output, args.labels)
    else:
        evaluate(args.output, args.labels, "validation" if args.command == "evaluate-validation" else "held_out")
