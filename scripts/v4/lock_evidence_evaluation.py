"""Re-audit F01/F03/F06 without changing frozen historical artifacts or S2."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.v4.evaluate_r5_same_input import feedback_labels, submission
from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_r8_evidence_extraction import ROOT, R6, OUTPUT as R8, S2, select_eligible
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import containment, judgment_key, normalize
from src.medrag.validation.evidence_experiment import SPLIT_VERSION, query_partition, validate_split

OUTPUT = ROOT / "outputs/v4/evidence_evaluation_lock_20261025"
TRAIN = ROOT / "outputs/v4/r3_chunk_reranker_20261024/train_triples.jsonl"


def run(output: Path) -> None:
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frozen_sha = sha256_file(S2)
    registry = {"version": SPLIT_VERSION,
                "assignments": {str(qid): query_partition(qid) for qid in range(1, 1201)},
                "scope": "Query-disjoint prospective judgments; historical labels are not a pristine holdout",
                "r3_training_sha256": sha256_file(TRAIN),
                "r3_training_path": str(TRAIN.relative_to(ROOT))}
    training_ids = {int(json.loads(line)["query_id"]) for line in TRAIN.read_text().splitlines()}
    validate_split(registry, training_ids)
    write_once(output / "query_split_registry.json", registry)

    labels, s2 = feedback_labels(), submission(S2)
    exact = Counter()
    for qid, prediction in s2.items():
        for chunk in prediction["relevant_chunks"]:
            key = judgment_key(qid, chunk["doc_id"], chunk["chunk_text"])
            if key in labels:
                exact["positive" if labels[key] else "negative"] += 1

    rows = pq.read_table(R6 / "positive_registry.parquet").to_pylist()
    _, store, _, _ = environment()
    sources = store.sources_batch({int(row["doc_id"]) for row in rows if row["source_present"]})
    corrected, counts = [], Counter()
    for row in rows:
        source = sources.get(int(row["doc_id"]))
        if source is None:
            counts["source_unavailable"] += 1
            continue
        old_start, old_end = row["source_start"], row["source_end"]
        if old_start is not None and old_end is not None:
            counts["old_offsets_present"] += 1
            counts["old_slice_mismatch"] += normalize(source.text[old_start:old_end]) != normalize(row["evidence_text"])
        matched, start, end = containment(source.text, row["evidence_text"])
        if matched and normalize(source.text[start:end]) != normalize(row["evidence_text"]):
            raise ValueError("Corrected source offset did not round trip")
        counts["corrected_contained"] += matched
        counts["corrected_roundtrip_mismatch"] += bool(matched and
            normalize(source.text[start:end]) != normalize(row["evidence_text"]))
        corrected.append({"registry_id": row["registry_id"], "query_id": row["query_id"],
                          "doc_id": row["doc_id"], "label_group": row["label_group"],
                          "source_contains_evidence": matched, "source_start": start, "source_end": end,
                          "source_sha256": source.text_sha256,
                          "metric_scope": "anchor_preservation_only"})
    with (output / "corrected_source_offsets.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(corrected[0]))
        writer.writeheader()
        writer.writerows(corrected)

    old_lock_path = R8 / "policy_lock.json"
    old_lock = json.loads(old_lock_path.read_text())
    budget = old_lock["development_loss_budget_queries"]
    summaries = {name: Counter(value) for name, value in old_lock["a1_development"].items()}
    selected = select_eligible([name for name in summaries if name != "A0"], summaries, budget)
    # Original net-loss counts are a lower bound on gross-loss query counts.
    # All policies fail even this weaker gate, so no labels need to be reopened.
    if selected is not None:
        raise ValueError("Historical gate must be recomputed from query-level identities")
    write_once(output / "r8_development_gate.json", {
        "status": "DEVELOPMENT_NO_GO", "selected_policy": None,
        "validation_labels_read": False, "source_policy_lock_sha256": sha256_file(old_lock_path),
        "basis": "All A1 policies fail even the weaker historical net-loss gate; gross-loss gate cannot admit them",
        "eligible_policy_count": 0,
    })
    report = {
        "format": "evidence-evaluation-lock-v1", "s2_zip_sha256": frozen_sha,
        "s2_zip_unchanged": sha256_file(S2) == frozen_sha,
        "f01": {"exact_identity_labels": sum(exact.values()), **dict(exact),
                "s2_unjudged_chunks": 72000 - sum(exact.values()),
                "status": "INSUFFICIENT_EXACT_IDENTITY_LABELS",
                "independent_evidence_judgments": 0},
        "f03": dict(counts),
        "f06": {"status": "DEVELOPMENT_NO_GO", "selected_policy": None,
                "validation_labels_read": False},
        "split_counts": dict(Counter(registry["assignments"].values())),
        "r3_train_queries": len(training_ids), "r3_train_evaluation_overlap": 0,
        "next_experiment_status": "AWAITING_INDEPENDENT_EVIDENCE_JUDGMENTS",
        "metric_scope": "Historical preservation, independent relevance and official F2 are separate",
        "supersedes": ["R5 inherited-parent label metrics", "R6 source offsets and derived token visibility",
                       "R8 development lock and validation-based treatment choice"],
        "token_visibility": "INVALIDATED_PENDING_RECOMPUTATION; no corrected token visibility claim",
        "implementation_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / "src/medrag/evidence/identity.py",
            ROOT / "src/medrag/validation/evidence_experiment.py",
            ROOT / "scripts/v4/evaluate_r5_same_input.py",
            ROOT / "scripts/v4/evaluate_r5_window_scoring.py",
            ROOT / "scripts/v4/run_r8_evidence_extraction.py")},
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            R6 / "positive_registry.parquet", old_lock_path, TRAIN)},
    }
    if not report["s2_zip_unchanged"] or counts["corrected_roundtrip_mismatch"]:
        raise ValueError("Evaluation lock failed")
    write_once(output / "audit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
