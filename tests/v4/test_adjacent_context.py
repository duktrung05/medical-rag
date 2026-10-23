"""Adjacent export context stays deterministic, bounded and source-grounded."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.medrag.adapters.validation import validate_selected
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.factory import build_pipeline
from src.medrag.evidence.context import apply_chunk_context, derive_adjacent_evidence
from src.medrag.evidence.selector import SelectionPolicy


class Store:
    def __init__(self, source: Source, passages: list[Passage]):
        self.sources = {source.official_id: source}
        self.passages = {row.passage_id: row for row in passages}

    def source(self, key: int) -> Source:
        return self.sources[key]

    def passage(self, key: str) -> Passage:
        return self.passages[key]

    def passages_for_parent(self, official_id: int) -> list[Passage]:
        return [row for row in self.passages.values() if row.official_id == official_id]


def fixture() -> tuple[Store, dict]:
    text = "alpha evidence\nbeta treatment\ngamma dosage"
    source = Source("release", 17, "https://example.test/17", "title", text,
                    "test", hashlib.sha256(text.encode()).hexdigest())
    parts = ["alpha evidence", "beta treatment", "gamma dosage"]
    passages = []
    for index, part in enumerate(parts):
        start = text.index(part)
        passages.append(Passage(f"17:p{index}", "release", 17, source.url, part,
                                start, start + len(part), source.text_sha256, "en", "test"))
    seed = {"passage_id": "17:p1", "text": parts[1],
            "start": passages[1].source_start, "end": passages[1].source_end,
            "rerank_score": 2.5, "score_status": "reranked"}
    return Store(source, passages), seed


def test_adjacent_context_is_exact_deterministic_source_span():
    store, seed = fixture()
    policy = SelectionPolicy(document_selection="independent", chunk_context="adjacent",
                             context_window=1, context_max_characters=4096)
    selected = [{"official_id": 17, "document_evidence": deepcopy(seed),
                 "evidence": [deepcopy(seed)]}]
    first, trace = apply_chunk_context(selected, policy, store.sources, store.passages)
    second, _ = apply_chunk_context(selected, policy, store.sources, store.passages)
    assert first == second
    assert selected[0]["evidence"][0] == seed
    evidence = first[0]["evidence"][0]
    assert evidence["text"] == store.source(17).text
    assert evidence["context_passage_ids"] == ["17:p0", "17:p1", "17:p2"]
    assert evidence["seed_evidence"] == seed
    assert evidence["score_status"] == "derived_from_reranked_seed"
    assert trace["expanded_spans"] == trace["derived_evidence"] == 1
    predictions = validate_selected([8], {8: first}, store, policy, result_limit=200,
                                    reranker_enabled=True, official_ids={17})
    assert predictions[0]["relevant_chunks"] == [
        {"doc_id": 17, "chunk_text": store.source(17).text}]


def test_context_character_cap_chooses_widest_then_earliest_span():
    store, seed = fixture()
    # Full three-passage span is too long; both two-passage choices fit at this cap.
    cap = len("alpha evidence\nbeta treatment")
    policy = SelectionPolicy(chunk_context="adjacent", context_max_characters=cap)
    evidence = derive_adjacent_evidence(
        seed, store.source(17), store.passages_for_parent(17), policy)
    assert evidence["text"] == "alpha evidence\nbeta treatment"
    assert evidence["context_passage_ids"] == ["17:p0", "17:p1"]
    with pytest.raises(ValueError, match="exceeds context_max_characters"):
        derive_adjacent_evidence(
            seed, store.source(17), store.passages_for_parent(17),
            SelectionPolicy(chunk_context="adjacent", context_max_characters=3))


@pytest.mark.parametrize("tamper", [
    lambda item: item.update(text="invented"),
    lambda item: item.update(start=item["start"] + 1),
    lambda item: item.update(passage_id="context:wrong"),
    lambda item: item.update(seed_passage_id="17:p0"),
    lambda item: item.update(context_passage_ids=["17:p1"]),
    lambda item: item["seed_evidence"].update(text="invented"),
    lambda item: item.update(score_status="reranked"),
    lambda item: item.update(rerank_score=True),
])
def test_validator_rejects_tampered_derived_contract(tamper):
    store, seed = fixture()
    policy = SelectionPolicy(document_selection="independent", chunk_context="adjacent")
    selected, _ = apply_chunk_context(
        [{"official_id": 17, "document_evidence": deepcopy(seed),
          "evidence": [deepcopy(seed)]}], policy, store.sources, store.passages)
    tamper(selected[0]["evidence"][0])
    with pytest.raises((ValueError, KeyError)):
        validate_selected([8], {8: selected}, store, policy, result_limit=200,
                          reranker_enabled=True, official_ids={17})


def test_default_passage_mode_preserves_object_and_factory_loads_context():
    store, seed = fixture()
    selected = [{"official_id": 17, "evidence": [seed]}]
    output, trace = apply_chunk_context(
        selected, SelectionPolicy(), store.sources, store.passages)
    assert output is selected and trace["derived_evidence"] == 0
    pipeline = build_pipeline(list(store.sources.values()), list(store.passages.values()), {
        "selection": {"chunk_context": "adjacent", "context_window": 0,
                      "context_max_characters": 100}})
    assert pipeline.selection_policy.chunk_context == "adjacent"
    assert pipeline.selection_policy.context_window == 0


def test_context_configs_only_extend_selection_contract():
    root = Path(__file__).resolve().parents[2]
    for cap in (120, 200):
        control = json.loads((root / f"configs/v4_phase2_full_independent_d{cap}_gpu3.yaml").read_text())
        context = json.loads((root / f"configs/v4_phase2_full_independent_d{cap}_context_gpu3.yaml").read_text())
        for key in ("corpus_version", "registry", "index_namespace", "lexical", "dense",
                    "reranker", "fusion", "retrieval", "batch"):
            assert context[key] == control[key]
        assert context["selection"] == {
            **control["selection"], "chunk_context": "adjacent", "context_window": 1,
            "context_max_characters": 4096,
        }


@pytest.mark.parametrize("kwargs", [
    {"chunk_context": "unknown"},
    {"context_window": -1},
    {"context_window": True},
    {"context_max_characters": 0},
])
def test_invalid_context_policy_fails(kwargs):
    with pytest.raises(ValueError):
        SelectionPolicy(**kwargs)
