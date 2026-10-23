"""Replay source-grounded adjacent context on completed parent-first scores, without models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.medrag.adapters.validation import canonical_payload, validate_selected
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy, select_results
from scripts.v4.audit_evidence_survival import normalize, sha


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def replay(root: Path, reports: Path, config_path: Path, output: Path) -> dict:
    root, reports, config_path, output = (
        path.resolve() for path in (root, reports, config_path, output))
    if output.exists():
        raise FileExistsError(output)
    summary = read(reports / "summary.json")
    run_lock = read(reports / "run_lock.json")
    if summary.get("status") != "PILOT_COMPLETE" or summary["query_ids"] != run_lock["query_ids"]:
        raise ValueError("Input must be a complete parent-first evidence pilot")
    config = load_config(config_path)
    policy = SelectionPolicy(**config["selection"])
    if (policy.chunk_context != "adjacent" or policy.document_selection != "independent"
            or policy.chunks_per_doc != 1):
        raise ValueError("Replay requires independent c1 adjacent-context selection")
    control_path = root / f"configs/v4_phase2_full_independent_d{policy.max_docs}_gpu3.yaml"
    control = load_config(control_path)
    for key in ("corpus_version", "registry", "index_namespace", "lexical", "dense",
                "reranker", "fusion", "retrieval", "batch"):
        if config[key] != control[key]:
            raise ValueError(f"Context replay changed the inference contract: {key}")
    pilot_control = root / "configs/v4_phase2_full_independent_d200_gpu3.yaml"
    if sha(pilot_control) != run_lock["pilot_config_sha256"]:
        raise ValueError("Pilot score contract differs from the locked d200 control")
    index = root / config["index_namespace"]
    index_manifest = read(index / "manifest.json")
    if sha(index / "manifest.json") != run_lock["full_index_manifest_sha256"]:
        raise ValueError("Context replay index differs from the pilot index")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)

    baseline_path = root / "outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip"
    if sha(baseline_path) != run_lock["baseline_submission_sha256"]:
        raise ValueError("Historical anchor submission differs from pilot lock")
    with zipfile.ZipFile(baseline_path) as bundle:
        baseline = {row["id"]: row for row in json.loads(bundle.read("predictions.json"))}
    audit_csv = root / "outputs/v4/evidence_survival_audit_20261021_v2/historical_positive_stages.csv"
    with audit_csv.open(encoding="utf-8") as stream:
        anchors = [row for row in csv.DictReader(stream)
                   if int(row["query_id"]) in summary["query_ids"]]

    selected_by_id, diagnostics, anchor_rows = {}, [], []
    for qid in summary["query_ids"]:
        checkpoint = read(reports / "queries" / f"{qid}.json")
        raw, selection_trace = select_results(
            checkpoint["diagnostics"]["ranked_candidates"], policy,
            reranker_enabled=True, limit=int(config["batch"]["result_limit"]))
        selected, context_trace = apply_chunk_context(raw, policy, sources, passages)
        official_ids = {document["official_id"] for document in selected}
        validate_selected([qid], {qid: selected}, store, policy,
                          result_limit=int(config["batch"]["result_limit"]),
                          reranker_enabled=True, official_ids=official_ids)
        selected_by_id[qid] = selected
        diagnostics.append({"id": qid, "selection": selection_trace,
                            "context": context_trace})
        raw_by_parent = {document["official_id"]: document for document in raw}
        context_by_parent = {document["official_id"]: document for document in selected}
        for row in (item for item in anchors if int(item["query_id"]) == qid):
            parent = int(row["doc_id"])
            old_text = baseline[qid]["relevant_chunks"][
                int(row["baseline_chunk_order"])]["chunk_text"]
            anchor = normalize(old_text)
            seeds = raw_by_parent.get(parent, {}).get("evidence", [])
            expanded = context_by_parent.get(parent, {}).get("evidence", [])
            anchor_rows.append({
                "query_id": qid,
                "doc_id": parent,
                "parent_selected": parent in context_by_parent,
                "seed_contains_old_anchor": any(anchor in normalize(item["text"])
                                                for item in seeds),
                "context_contains_old_anchor": any(anchor in normalize(item["text"])
                                                   for item in expanded),
            })

    predictions = validate_selected(
        summary["query_ids"], selected_by_id, store, policy,
        result_limit=int(config["batch"]["result_limit"]), reranker_enabled=True,
        official_ids={document["official_id"] for selected in selected_by_id.values()
                      for document in selected})
    payload = canonical_payload(predictions)
    output.mkdir(parents=True)
    (output / "predictions.json").write_bytes(payload + b"\n")
    (output / "selected_results.jsonl").write_text("".join(
        json.dumps({"id": qid, "selected": selected_by_id[qid]}, ensure_ascii=False,
                   separators=(",", ":"), allow_nan=False) + "\n"
        for qid in summary["query_ids"]), encoding="utf-8")
    (output / "diagnostics.jsonl").write_text("".join(
        json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        for row in diagnostics), encoding="utf-8")
    result = {
        "status": "ADJACENT_CONTEXT_REPLAY_PASS",
        "query_ids": summary["query_ids"],
        "selection": config["selection"],
        "selected_documents": sum(len(rows) for rows in selected_by_id.values()),
        "selected_chunks": sum(len(document["evidence"])
                               for rows in selected_by_id.values() for document in rows),
        "derived_chunks_validated": sum(row["context"]["derived_evidence"]
                                        for row in diagnostics),
        "expanded_chunks": sum(row["context"]["expanded_spans"] for row in diagnostics),
        "unchanged_chunks": sum(row["context"]["unchanged_spans"] for row in diagnostics),
        "historical_positive_anchors": len(anchor_rows),
        "positive_parents_selected": sum(row["parent_selected"] for row in anchor_rows),
        "seed_contains_old_anchor": sum(row["seed_contains_old_anchor"] for row in anchor_rows),
        "context_contains_old_anchor": sum(row["context_contains_old_anchor"]
                                           for row in anchor_rows),
        "newly_recovered_by_context": [row for row in anchor_rows
                                       if not row["seed_contains_old_anchor"]
                                       and row["context_contains_old_anchor"]],
        "grounding": "all derived spans rebuilt from registered source offsets",
        "model_calls": 0,
        "official_f2": "UNMEASURED",
        "config_sha256": sha(config_path),
        "pilot_run_lock_sha256": sha(reports / "run_lock.json"),
        "canonical_predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    write_json(output / "replay_report.json", result)
    write_json(output / "manifest.json", {
        **result,
        "format": "v4-adjacent-context-replay-v1",
        "implementation_sha256": {
            name: sha(root / name) for name in (
                "scripts/v4/replay_adjacent_context.py",
                "src/medrag/evidence/context.py",
                "src/medrag/evidence/selector.py",
                "src/medrag/adapters/validation.py",
                "src/medrag/adapters/vibiomir_submission.py",
            )},
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.root, args.reports, args.config, args.output),
                     ensure_ascii=False, indent=2))
