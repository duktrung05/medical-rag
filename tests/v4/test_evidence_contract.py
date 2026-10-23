"""Regression cases for tail passage retrieval, scoring, and V4 export."""
from __future__ import annotations

import math
from hashlib import sha256

import pytest

from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.evidence.selector import SelectionPolicy, select_results


def _source(doc_id: int, parts: list[str], title: str = "generic title") -> tuple[Source, list[Passage]]:
    text = "\n".join(parts)
    url = f"https://example.org/{doc_id}"
    source = Source("fixture-v1", doc_id, url, title, text, "fixture", sha256(text.encode()).hexdigest())
    passages = []
    cursor = 0
    for index, part in enumerate(parts):
        start = text.index(part, cursor)
        end = start + len(part)
        passages.append(Passage(f"{doc_id}:{index}", source.corpus_version, doc_id, url,
                                part, start, end, source.text_sha256, "en", "fixture"))
        cursor = end
    return source, passages


class _Encoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float("sentineltail" in text), float("ordinary" in text)] for text in texts]

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _Reranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [2.0 if "sentineltail" in text else -1.0 for text in passages]


def test_tail_passage_survives_lexical_dense_rerank_and_export():
    source, passages = _source(73, [f"ordinary section {i}" for i in range(7)]
                               + ["sentineltail contains treatment evidence"])
    assert "sentineltail" not in source.title
    for encoder in (None, _Encoder()):
        pipeline = Pipeline([source], passages, encoder, _Reranker())
        result, trace = pipeline.retrieve_with_diagnostics("sentineltail", 1)
        assert result[0]["official_id"] == 73
        assert result[0]["evidence"][0]["passage_id"] == "73:7"
        assert result[0]["evidence"][0]["score_status"] == "reranked"
        assert records([5], {5: result})[0]["relevant_chunks"] == [
            {"doc_id": 73, "chunk_text": "sentineltail contains treatment evidence"}]
        assert trace["selection"]["selected_evidence"] == 1


def test_two_selected_passages_exported_without_second_slice():
    source, passages = _source(73, ["dose treatment one", "dose treatment two"])
    pipeline = Pipeline([source], passages, selection_policy=SelectionPolicy(chunks_per_doc=2))
    selected = pipeline.retrieve("dose treatment", 1)
    assert len(selected[0]["evidence"]) == 2
    exported = records([5], {5: selected})[0]
    assert [row["chunk_text"] for row in exported["relevant_chunks"]] == [
        candidate["text"] for candidate in selected[0]["evidence"]]


def test_same_text_in_distinct_parents_remains_distinct():
    a, pa = _source(1, ["same medical evidence"])
    b, pb = _source(2, ["same medical evidence"])
    selected = Pipeline([a, b], pa + pb).retrieve("medical evidence", 2)
    exported = records([11], {11: selected})[0]
    assert set(exported["relevant_docs"]) == {1, 2}
    assert len(exported["relevant_chunks"]) == 2


def test_same_text_twice_in_one_parent_is_exported_once():
    source, passages = _source(3, ["repeated medical evidence", "repeated medical evidence"])
    pipeline = Pipeline([source], passages, selection_policy=SelectionPolicy(chunks_per_doc=2))
    selected, trace = pipeline.retrieve_with_diagnostics("medical evidence", 1)
    assert len(selected[0]["evidence"]) == 1
    assert trace["selection"]["duplicate_parent_text"] == 1


def test_unscored_evidence_excluded_and_bad_score_fails():
    source, passages = _source(1, ["first treatment", "second treatment"])

    class OnePair:
        def score(self, query, pairs):
            assert len(pairs) == 1
            return [1.0]

    pipeline = Pipeline([source], passages, reranker=OnePair(), rerank_candidate_limit=1,
                        selection_policy=SelectionPolicy(chunks_per_doc=2))
    result, trace = pipeline.retrieve_with_diagnostics("treatment", 1)
    assert len(result[0]["evidence"]) == 1
    assert trace["selection"]["not_reranked_budget"] == 1

    class BadScore:
        def score(self, query, pairs):
            return [math.nan for _ in pairs]

    with pytest.raises(ValueError, match="non-finite"):
        Pipeline([source], passages, reranker=BadScore()).retrieve("treatment")


def test_selection_rejects_invalid_caps_and_never_slices_in_adapter():
    with pytest.raises(ValueError, match="chunks_per_doc"):
        SelectionPolicy(chunks_per_doc=0)
    candidate = {"official_id": 1, "evidence": [
        {"passage_id": "p0", "text": "first", "rerank_score": 1.0},
        {"passage_id": "p1", "text": "second", "rerank_score": 0.5},
    ]}
    selected, diagnostics = select_results([candidate], SelectionPolicy(chunks_per_doc=2),
                                           reranker_enabled=True, limit=1)
    assert diagnostics["selected_evidence"] == 2
    assert len(records([1], {1: selected})[0]["relevant_chunks"]) == 2


def test_parent_first_scores_more_distinct_parents_at_same_pair_budget():
    sources, passages = [], []
    for parent_id in (1, 2, 3):
        source, parts = _source(parent_id, [f"treatment {parent_id} part {i}" for i in range(3)])
        sources.append(source)
        passages.extend(parts)

    class Recorder:
        def __init__(self):
            self.pairs = []

        def score(self, query, pairs):
            self.pairs = pairs
            return [1.0] * len(pairs)

    old, diverse = Recorder(), Recorder()
    Pipeline(sources, passages, reranker=old, rerank_candidate_limit=3).retrieve("treatment", 3)
    Pipeline(sources, passages, reranker=diverse, rerank_candidate_limit=3,
             rerank_allocation="parent_first").retrieve("treatment", 3)
    assert len(old.pairs) == len(diverse.pairs) == 3
    assert len({value.split(" part ")[0] for value in old.pairs}) == 1
    assert len({value.split(" part ")[0] for value in diverse.pairs}) == 3
