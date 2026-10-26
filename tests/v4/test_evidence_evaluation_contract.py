import csv
import json
from collections import Counter
from types import SimpleNamespace
import zipfile

import pytest

from scripts.v4 import evaluate_r5_same_input as r5
from scripts.v4 import run_r8_evidence_extraction as r8
from src.medrag.evidence.identity import containment, judgment_key, normalize, text_sha256
from src.medrag.validation.evidence_experiment import (
    SPLIT_VERSION, allocation_gate, paired_changes, query_partition, validate_split,
)


@pytest.mark.parametrize("source,needle", [
    ("A &amp; B\n  Điều trị VIÊM dạ dày.", "điều trị viêm"),
    ("prefix &#x1F600; tail", "😀"),
    ("A &NotEqualTilde; B", "≂̸"),
    ("prefix ﬃ suffix", "ffi"),
    ("prefix … suffix", "..."),
    ("abc Điều trị xyz", "điều trị"),
    ("before İ after", "i̇"),
    ("before 각 after", "각"),
    ("x &notit; y", "¬it;"),
    ("prefix\n \t text", "prefix text"),
])
def test_offsets_round_trip_through_original_source(source, needle):
    matched, start, end = containment(source, needle)
    assert matched
    assert 0 <= start < end <= len(source)
    assert normalize(source[start:end]) == normalize(needle)


def test_partial_compatibility_expansion_cannot_claim_an_invalid_span():
    assert containment("ﬃ", "fi") == (False, None, None)
    assert containment("ﬃ fi", "fi") == (True, 2, 4)


def test_feedback_stays_bound_to_exact_text_including_negative_contexts(tmp_path, monkeypatch):
    old = tmp_path / "old.zip"
    chunks = [{"doc_id": 10, "chunk_text": "old positive"},
              {"doc_id": 10, "chunk_text": "old negative"}]
    with zipfile.ZipFile(old, "w") as archive:
        archive.writestr("predictions.json", json.dumps([{"id": 1, "relevant_chunks": chunks}]))
    feedback = tmp_path / "feedback.csv"
    with feedback.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scoring_status", "query_id", "chunk_order",
                                                    "baseline_doc_id_int", "chunk_text_sha256", "relevant"])
        writer.writeheader()
        for order, chunk in enumerate(chunks):
            writer.writerow(dict(scoring_status="scored", query_id=1, chunk_order=order,
                                 baseline_doc_id_int=10, chunk_text_sha256=text_sha256(chunk["chunk_text"]),
                                 relevant=order == 0))
    monkeypatch.setattr(r5, "OLD_ZIP", old)
    monkeypatch.setattr(r5, "FEEDBACK", feedback)
    labels = r5.feedback_labels()
    assert labels[judgment_key(1, 10, "old positive")] is True
    assert labels[judgment_key(1, 10, "old negative")] is False
    assert judgment_key(1, 10, "new context old positive") not in labels
    assert judgment_key(1, 10, "old negative with new evidence") not in labels


def test_development_rejects_all_ineligible_policies():
    summaries = {"lossy": Counter(queries=10, query_loss=2, gained_labels=4, lost_labels=1),
                 "flat": Counter(queries=10, query_loss=0, gained_labels=0, lost_labels=0)}
    assert r8.select_eligible(list(summaries), summaries, 1) is None
    assert r8.select_eligible([], {}, 1) is None


def test_rejected_or_legacy_lock_cannot_open_validation(tmp_path, monkeypatch):
    (tmp_path / "policy_lock.json").write_text(json.dumps({"status": "DEVELOPMENT_NO_GO"}))
    monkeypatch.setattr(r8, "load_registry", lambda *_: pytest.fail("validation labels were opened"))
    with pytest.raises(ValueError, match="immutable development lock"):
        r8.evaluate(tmp_path)


def test_development_writes_null_treatment_when_every_policy_fails(tmp_path, monkeypatch):
    from src.medrag.data.phase2_batch import sha256_file

    candidates = tmp_path / "span_candidates.parquet"
    candidates.write_bytes(b"fixture")
    (tmp_path / "candidate_manifest.json").write_text(json.dumps({
        "span_candidates_sha256": sha256_file(candidates)}))
    (tmp_path / "config.json").write_text(json.dumps({"split": "historical fixture"}))
    monkeypatch.setattr(r8, "load_registry", lambda part: {1: []} if part == "development" else
                        pytest.fail("validation opened"))
    summaries = {"A0": Counter(queries=1)}
    summaries.update({f"A1/{policy['name']}": Counter(queries=1, gained_labels=0, lost_labels=1)
                      for policy in r8.A1_POLICIES})
    calls = []

    def summarize(*args):
        calls.append(args)
        return summaries, {}

    monkeypatch.setattr(r8, "summarize_policies", summarize)
    r8.select_dev(tmp_path)
    lock = json.loads((tmp_path / "policy_lock.json").read_text())
    assert lock["status"] == "DEVELOPMENT_NO_GO"
    assert lock["selected_arm"] is lock["selected_a1"] is lock["selected_a2"] is None
    assert lock["validation_labels_read"] is False
    assert len(calls) == 1  # A2 search cannot start after A1 rejection.


def test_equal_net_gain_still_counts_a_loss_query():
    changes = paired_changes({"old"}, {"new"})
    assert changes["net_change"] == 0
    assert changes["query_loss"] and changes["query_gain"]
    result = allocation_gate([changes], independent=True, structure_valid=True, loss_budget=0)
    assert result["decision"] == "NO_GO"
    assert result["gross_loss_queries"] == 1


def test_historical_proxy_or_failed_validator_cannot_pass():
    row = paired_changes(set(), {"new"})
    assert allocation_gate([row], independent=False, structure_valid=True, loss_budget=0)["decision"] == "NO_GO"
    assert allocation_gate([row], independent=True, structure_valid=False, loss_budget=0)["decision"] == "NO_GO"
    assert allocation_gate([], independent=True, structure_valid=True, loss_budget=0)["decision"] == "NO_GO"
    assert allocation_gate([row], independent=True, structure_valid=True, loss_budget=0)["decision"] == "GO"


def test_split_rejects_model_training_overlap():
    registry = {"version": SPLIT_VERSION,
                "assignments": {str(qid): query_partition(qid) for qid in range(1, 1201)}}
    assert validate_split(registry, {53, 130})
    eval_qid = next(qid for qid in range(1, 1201) if query_partition(qid) != "development")
    with pytest.raises(ValueError, match="overlap"):
        validate_split(registry, {eval_qid})


def test_runtime_validator_rejects_tampered_offsets_and_document_set():
    docs = list(range(1, 61))
    frozen = {"relevant_docs": docs,
              "relevant_chunks": [{"doc_id": doc, "chunk_text": "abc"} for doc in docs]}
    source = SimpleNamespace(text="abc", text_sha256=text_sha256("abc"))
    store = SimpleNamespace(sources_batch=lambda ids: {doc: source for doc in ids})
    rows = [dict(doc_id=doc, start=0, end=3, text="abc", source_sha256=source.text_sha256,
                 text_sha256=source.text_sha256, is_baseline_text=True) for doc in docs]
    assert r8.validate_candidate_rows(rows, frozen, store)
    rows[0]["start"] = 1
    with pytest.raises(ValueError, match="grounding"):
        r8.validate_candidate_rows(rows, frozen, store)
    with pytest.raises(ValueError, match="document set"):
        r8.validate_candidate_rows(rows[1:], frozen, store)
