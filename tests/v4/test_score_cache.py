"""Scores may be replayed only with the same exact pair and model contract."""
from __future__ import annotations

import math

import pytest

from src.medrag.ranking.score_cache import ScoreCache


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, query, payloads):
        self.calls.append((query, payloads))
        return [float(len(value)) for value in payloads]


def test_score_cache_reuses_exact_pairs_and_rejects_other_contract(tmp_path):
    contract = {"model": "test", "revision": "rev1", "max_length": 512,
                "precision": "fp16", "index_fingerprint": "snapshot1"}
    path = tmp_path / "scores.sqlite"
    cache = ScoreCache(path, contract, create=True)
    model = Reranker()
    rows = [(2, "p8", "Title\ntail evidence"), (3, "p1", "Other\nmedical text")]
    first = cache.score("query", rows, model)
    assert first == [len(row[2]) for row in rows]
    assert len(model.calls) == 1
    assert ScoreCache(path, contract).score("query", rows, model) == first
    assert len(model.calls) == 1
    changed = [(2, "p8", "Changed title\ntail evidence"), rows[1]]
    cache.score("query", changed, model)
    assert model.calls[-1][1] == [changed[0][2]]
    with pytest.raises(ValueError, match="fingerprint"):
        ScoreCache(path, {**contract, "revision": "rev2"})


def test_score_cache_does_not_commit_bad_batch(tmp_path):
    cache = ScoreCache(tmp_path / "scores.sqlite", {"model": "test"}, create=True)

    class Bad:
        def score(self, query, payloads):
            return [math.nan for _ in payloads]

    with pytest.raises(ValueError, match="non-finite"):
        cache.score("query", [(2, "p", "payload")], Bad())
    model = Reranker()
    assert cache.score("query", [(2, "p", "payload")], model) == [7.0]
    assert len(model.calls) == 1
