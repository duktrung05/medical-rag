import csv
import json
from types import SimpleNamespace

import pytest

from scripts.v4 import run_evidence_allocation as allocation
from src.medrag.evidence.identity import text_sha256


def write_labels(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "relevant", "reviewer_id", "judgment_basis"])
        writer.writeheader()
        writer.writerows(rows)


def label(item="a", relevant="true", basis="direct_context_review", reviewer="reviewer-1"):
    return dict(item_id=item, relevant=relevant, reviewer_id=reviewer, judgment_basis=basis)


@pytest.mark.parametrize("labels", [[], [label(relevant="")], [label(reviewer="")],
                                       [label(basis="historical_parent")], [label(), label()],
                                       [label(item="foreign")]])
def test_judgments_fail_closed_on_missing_inherited_or_duplicate_labels(tmp_path, labels):
    path = tmp_path / "labels.csv"
    write_labels(path, labels)
    with pytest.raises(ValueError):
        allocation.load_judgments(path, [{"rows": [{"item_id": "a"}]}])


def test_context_pool_preserves_all_s2_chunks_and_documents():
    docs = list(range(1, 62))
    source = SimpleNamespace(text="same source context", text_sha256=text_sha256("same source context"))
    frozen = {"relevant_docs": docs,
              "relevant_chunks": [{"doc_id": doc, "chunk_text": source.text} for doc in docs[:60]]}
    rows = [{"doc_id": doc, "text": source.text, "text_sha256": source.text_sha256,
             "source_sha256": source.text_sha256, "start": 0, "end": len(source.text),
             "baseline": doc <= 60, "features": [0.0, 1.0, 1.0]} for doc in docs]
    sources = {doc: source for doc in docs}
    assert allocation.validate_frozen_query(rows, frozen, sources, list(range(1, 201)))
    assert allocation.validate_frozen_query(rows, frozen, sources, list(range(1, 177)))
    rows[-1]["end"] -= 1
    with pytest.raises(ValueError, match="source/feature"):
        allocation.validate_frozen_query(rows, frozen, sources, list(range(1, 201)))


def test_allocator_changes_only_sixty_parent_slots():
    rows = [{"doc_id": doc, "item_id": str(doc), "document_rank": doc,
             "baseline": doc <= 60, "features": [float(doc == 61), 0, 0]} for doc in range(1, 62)]
    labels = {str(doc): doc == 61 for doc in range(1, 62)}
    model = {"mean": [0, 0, 0], "scale": [1, 1, 1], "weights": [1, 0, 0]}
    changes = allocation.scores_and_changes([{"query_id": 1, "rows": rows}], model, labels)
    assert changes[0]["gross_gains"] == 1
    assert changes[0]["gross_losses"] == 0
    assert len(set(changes[0]["chosen_item_ids"])) == 60


def test_failed_development_cannot_read_validation_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(allocation, "load_experiment", lambda _: {})
    (tmp_path / "model_lock.json").write_text(json.dumps({"status": "DEVELOPMENT_NO_GO"}))
    monkeypatch.setattr(allocation, "load_judgments", lambda *_: pytest.fail("opened validation"))
    with pytest.raises(ValueError, match="eligible challenger"):
        allocation.evaluate(tmp_path, tmp_path / "missing-labels.csv", "validation")


def test_fit_and_validation_use_the_identical_frozen_model(tmp_path, monkeypatch):
    config = {"private_files": {part: f"{part}.json" for part in ("development", "validation", "held_out")},
              "allocator": {"iterations": 400, "step": 0.1, "l2": 0.01}}
    monkeypatch.setattr(allocation, "load_experiment", lambda _: config)
    monkeypatch.setattr(allocation, "validate_records", lambda _: True)
    (tmp_path / "experiment_lock.json").write_text("{}")
    paths = {}
    for part, qid in (("development", 1), ("validation", 2), ("held_out", 3)):
        rows = [{"query_id": qid, "doc_id": doc, "item_id": f"{qid}-{doc}", "document_rank": doc,
                 "baseline": doc <= 60, "features": [float(doc == 61), 0, 0]} for doc in range(1, 62)]
        (tmp_path / f"{part}.json").write_text(json.dumps([{"query_id": qid, "rows": rows}]))
        paths[part] = tmp_path / f"{part}.csv"
        write_labels(paths[part], [label(row["item_id"], "true" if row["doc_id"] == 61 else "false") for row in rows])
    allocation.fit(tmp_path, paths["development"])
    model_bytes = (tmp_path / "model_lock.json").read_bytes()
    assert json.loads(model_bytes)["status"] == "LOCKED_BEFORE_VALIDATION"
    with pytest.raises(FileNotFoundError):
        allocation.evaluate(tmp_path, paths["held_out"], "held_out")
    allocation.evaluate(tmp_path, paths["validation"], "validation")
    allocation.evaluate(tmp_path, paths["held_out"], "held_out")
    assert (tmp_path / "model_lock.json").read_bytes() == model_bytes
    assert json.loads((tmp_path / "held_out_evaluation.json").read_text())["gate"]["decision"] == "GO"
