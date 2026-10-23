"""Regression for separate document/chunk budgets through scoring, grounding and export."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.v4.replay_subset_selection import replay
from src.medrag.adapters.validation import (
    canonical_payload, package_zip, validate_payload, validate_selected, validate_zip,
)
from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.factory import build_pipeline
from src.medrag.core.pipeline import Pipeline
from src.medrag.evidence.selector import SelectionPolicy, select_results


class Store:
    def __init__(self, count=200, per_doc=2):
        self.sources, self.passages, self.candidates = {}, {}, []
        for parent in range(1, count + 1):
            parts = [f"medical treatment {parent} section {i}" for i in range(per_doc)]
            source = Source("fixture", parent, f"https://example.test/{parent}", "title",
                            "\n".join(parts), "test", "raw")
            self.sources[parent] = source
            evidence, start = [], 0
            for index, text in enumerate(parts):
                passage = Passage(f"{parent}:{index}", "fixture", parent, source.url, text,
                                  start, start + len(text), source.text_sha256, "en", "test")
                self.passages[passage.passage_id] = passage
                evidence.append({"passage_id": passage.passage_id, "text": text,
                                 "start": start, "end": start + len(text),
                                 "rerank_score": 200.0 - parent - index / 10,
                                 "score_status": "reranked"})
                start += len(text) + 1
            self.candidates.append({"official_id": parent, "evidence": evidence})

    def source(self, key):
        return self.sources[key]

    def passage(self, key):
        return self.passages[key]


@pytest.mark.parametrize("doc_cap", [120, 200])
@pytest.mark.parametrize("chunks_per_doc", [1, 2])
def test_document_budget_is_independent_and_grounded_through_zip(tmp_path, doc_cap, chunks_per_doc):
    store = Store()
    original = deepcopy(store.candidates)
    policy = SelectionPolicy(max_docs=doc_cap, max_chunks=60, chunks_per_doc=chunks_per_doc,
                             document_selection="independent")
    selected, trace = select_results(store.candidates, policy, reranker_enabled=True, limit=200)
    assert len(selected) == doc_cap
    assert sum(len(doc["evidence"]) for doc in selected) == 60
    assert trace["documents_without_exported_chunks"] == doc_cap - 60 // chunks_per_doc
    assert store.candidates == original
    assert [doc["official_id"] for doc in selected] == list(range(1, doc_cap + 1))
    predictions = validate_selected([7], {7: selected}, store, policy, result_limit=200,
                                    reranker_enabled=True, official_ids=set(store.sources))
    assert len(predictions[0]["relevant_docs"]) == doc_cap
    assert len(predictions[0]["relevant_chunks"]) == 60
    assert set(predictions[0]) == {"id", "relevant_docs", "relevant_chunks"}
    payload = canonical_payload(predictions)
    assert b"document_evidence" not in payload
    archive = tmp_path / "submission.zip"
    package_zip(archive, payload)
    assert validate_zip(archive, query_ids={7}, official_ids=set(store.sources), policy=policy,
                        result_limit=200)["status"] == "pass"
    legacy, _ = select_results(store.candidates, replace(policy, document_selection="with_chunks"),
                               reranker_enabled=True, limit=200)
    assert len(legacy) == 60 // chunks_per_doc


def test_unsupported_parents_not_promoted_and_limit_respected():
    store = Store(count=4)
    store.candidates[0]["evidence"] = []
    for item in store.candidates[1]["evidence"]:
        item.pop("rerank_score")
        item["score_status"] = "not_reranked_budget"
    policy = SelectionPolicy(max_chunks=1, document_selection="independent")
    selected, trace = select_results(store.candidates, policy, reranker_enabled=True, limit=1)
    assert [doc["official_id"] for doc in selected] == [3]
    assert trace["documents_without_eligible_evidence"] == 2
    assert trace["not_reranked_budget"] == 2
    selected, _ = select_results(store.candidates, policy, reranker_enabled=True, limit=200)
    assert [doc["official_id"] for doc in selected] == [3, 4]
    assert selected[1]["evidence"] == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0"])
def test_invalid_document_support_score_fails_even_after_chunk_cap(value):
    store = Store(count=2)
    store.candidates[1]["evidence"][0]["rerank_score"] = value
    with pytest.raises(ValueError, match="reranker score"):
        select_results(store.candidates, SelectionPolicy(max_chunks=1, document_selection="independent"),
                       reranker_enabled=True, limit=200)


def test_anchor_and_exported_chunks_each_require_grounding_and_scoring():
    store = Store(count=2)
    policy = SelectionPolicy(max_chunks=1, document_selection="independent")
    selected, _ = select_results(store.candidates, policy, reranker_enabled=True, limit=200)
    assert not selected[1]["evidence"]
    changes = [
        lambda rows: rows[1].pop("document_evidence"),
        lambda rows: rows[1]["document_evidence"].update(text="invented"),
        lambda rows: rows[1].update(document_evidence=rows[0]["document_evidence"]),
        lambda rows: rows[1]["document_evidence"].update(rerank_score=True),
        lambda rows: rows[1]["document_evidence"].update(score_status="not_reranked_budget"),
        lambda rows: rows[0]["evidence"][0].update(end=0),
    ]
    for change in changes:
        bad = deepcopy(selected)
        change(bad)
        with pytest.raises(ValueError):
            validate_selected([7], {7: bad}, store, policy, result_limit=200,
                              reranker_enabled=True, official_ids=set(store.sources))


def test_payload_doc_only_is_mode_specific_and_wrong_parents_remain_invalid():
    policy = SelectionPolicy(document_selection="independent")
    base = [{"id": 7, "relevant_docs": [1, 2],
             "relevant_chunks": [{"doc_id": 1, "chunk_text": "text"}]}]
    kwargs = {"query_ids": {7}, "official_ids": {1, 2, 3}, "result_limit": 200}
    validate_payload(canonical_payload(base), policy=policy, **kwargs)
    with pytest.raises(ValueError, match="without selected evidence"):
        validate_payload(canonical_payload(base), policy=SelectionPolicy(), **kwargs)
    for docs, chunks in [
        ([1, 2], [{"doc_id": 3, "chunk_text": "text"}]),
        ([1, 1], []), ([True], []), ([9], []),
        ([1], [{"doc_id": 1, "chunk_text": ""}]),
        ([1], [{"doc_id": 1, "chunk_text": "text"}] * 2),
    ]:
        with pytest.raises(ValueError):
            validate_payload(canonical_payload([{"id": 7, "relevant_docs": docs,
                                                  "relevant_chunks": chunks}]), policy=policy, **kwargs)


def test_parent_first_pipeline_scores_200_parents_at_same_pair_budget():
    store = Store(per_doc=3)

    class Reranker:
        def score(self, query, payload):
            assert len(payload) == 200
            return [1.0] * len(payload)

    policy = SelectionPolicy(document_selection="independent")
    pipeline = Pipeline(list(store.sources.values()), list(store.passages.values()),
                        reranker=Reranker(), rerank_allocation="parent_first", selection_policy=policy)
    selected, trace = pipeline.retrieve_with_diagnostics("medical treatment", 200,
                                                         include_candidates=True)
    assert len(selected) == 200
    assert trace["rerank_pairs"] == 200
    assert sum(len(doc["evidence"]) for doc in selected) == 60
    assert len({doc["official_id"] for doc in selected}) == 200
    validate_selected([7], {7: selected}, store, policy, result_limit=200,
                      reranker_enabled=True, official_ids=set(store.sources))


def test_factory_loads_independent_policy_without_models():
    store = Store(count=3)
    pipeline = build_pipeline(list(store.sources.values()), list(store.passages.values()), {
        "selection": {"max_docs": 3, "max_chunks": 1, "document_selection": "independent"}})
    selected = pipeline.retrieve("medical treatment", 3)
    assert len(selected) == 3
    assert len(records([7], {7: selected})[0]["relevant_chunks"]) == 1


def test_cpu_replay_can_expand_document_limit_without_inference(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"corpus_version": "fixture", "retrieval": {"rerank_allocation": "parent_first"},
              "reranker": {"enabled": True}, "selection": {"chunks_per_doc": 1}}
    trace = (json.dumps({"id": 7, "ranked_candidates": Store().candidates}) + "\n").encode()
    (source / "diagnostics.jsonl").write_bytes(trace)
    manifest = {"status": "complete", "scope": "locked_v4_subset", "config": config,
                "diagnostics_sha256": hashlib.sha256(trace).hexdigest(),
                "result_limit": 10, "query_ids": [7]}
    (source / "manifest.json").write_text(json.dumps(manifest))
    policy = {"max_docs": 200, "max_chunks": 60, "chunks_per_doc": 1,
              "document_selection": "independent"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**config, "selection": policy}))
    a = replay(source, path, tmp_path / "a", result_limit=200)
    b = replay(source, path, tmp_path / "b", result_limit=200)
    assert a["canonical_predictions_sha256"] == b["canonical_predictions_sha256"]
    assert a["selected_documents"] == 200 and a["selected_chunks"] == 60
    # A selection replay cannot invent the scores for a different allocation.
    config["retrieval"]["rerank_allocation"] = "document_order"
    path.write_text(json.dumps({**config, "selection": policy}))
    with pytest.raises(ValueError, match="inference contract: retrieval"):
        replay(source, path, tmp_path / "invalid", result_limit=200)


def test_invalid_policy_and_duplicate_candidate_ids_fail():
    with pytest.raises(ValueError, match="document_selection"):
        SelectionPolicy(document_selection="typo")
    doc = Store(count=1).candidates[0]
    with pytest.raises(ValueError, match="duplicate candidate"):
        select_results([doc, doc], SelectionPolicy(document_selection="independent"),
                       reranker_enabled=True, limit=200)


def test_new_configs_preserve_models_and_pair_budget():
    root = Path(__file__).resolve().parents[2]
    baseline = json.loads((root / "configs/v4_phase2_full_a0_gpu3.yaml").read_text())
    for cap in (120, 200):
        config = json.loads((root / f"configs/v4_phase2_full_independent_d{cap}_gpu3.yaml").read_text())
        for name in ("dense", "reranker", "batch", "corpus_version", "index_namespace"):
            assert config[name] == baseline[name]
        assert config["retrieval"] == {**baseline["retrieval"], "rerank_allocation": "parent_first"}
        assert config["selection"] == {"max_docs": cap, "max_chunks": 60, "chunks_per_doc": 1,
                                        "document_selection": "independent"}
