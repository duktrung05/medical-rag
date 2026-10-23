from __future__ import annotations

import hashlib

import pytest

from scripts.v4.evaluate_parent_first_bm25 import FORMAT, benchmark_eligibility, percentile95
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.parent_first_bm25 import ParentFirstDiskBM25


def report():
    return {"format": FORMAT, "gates": {"same_environment_algorithm_parity": "PASS",
            "same_environment_downstream_parity": "PASS",
            "historical_raw_score_reproducibility": "FAIL"}, "coverage": {"parents_checked": 144}}


def test_independent_benchmark_gate_keeps_historical_failure_visible():
    r = report()
    result = benchmark_eligibility(r)
    assert result == {"eligible": True, "historical_raw_score_reproducibility": "FAIL", "migration_authorized": False}
    assert r["gates"]["historical_raw_score_reproducibility"] == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("same_environment_algorithm_parity", "FAIL"),
    ("same_environment_downstream_parity", "FAIL"),
    ("historical_raw_score_reproducibility", None),
])
def test_algorithm_failure_or_hidden_historical_result_blocks_benchmark(field, value):
    r = report()
    r["gates"][field] = value
    with pytest.raises(ValueError):
        benchmark_eligibility(r)


def test_p95_nearest_rank_is_explicit():
    assert percentile95(list(range(1, 22))) == 20


@pytest.fixture
def many_passages(tmp_path):
    body = "common fever dose treatment điều trị 高血压 治疗 5 mg"
    sources, passages = [], []
    rows = []
    for parent in range(8):
        text = "\n".join([body] * 128)
        source = Source("fixture", parent, f"https://fixture/{parent}", f"Title {parent}", text, "fixture", "raw")
        sources.append(source)
        for i in range(128):
            start = i * (len(body) + 1)
            pid = f"{parent}:p{i:03}"
            passages.append(Passage(pid, "fixture", parent, source.url, body, start,
                                    start + len(body), source.text_sha256, "mixed", "fixture"))
            rows.append((parent, pid, body))
    old = DiskBM25(tmp_path / "many.sqlite", "passage_bm25", "fixture", create=True)
    old.append_batch(rows)
    old.complete()
    return sources, passages, old, ParentFirstDiskBM25(old.path, old.channel, old.fingerprint)


@pytest.mark.parametrize("query", [
    "common fever dose treatment 5 mg",
    "common fever dose treatment " + " ".join(f"absent{i}" for i in range(79)),
    "điều trị 高血压 治疗 common",
    "common",  # all 128 passages tie exactly
])
def test_long_multilingual_tied_128_passage_parents_and_final_hashes(many_passages, query):
    sources, passages, old, new = many_passages
    for parent in range(8):
        a, b = old.search_parent(query, parent, 129), new.search_parent(query, parent, 129)
        assert len(a) == len(b) == 128
        assert [(h.passage_id, h.score.hex(), h.rank) for h in a] == [
            (h.passage_id, h.score.hex(), h.rank) for h in b]
        assert len({h.score.hex() for h in a}) == 1

    class FallbackOnly:
        def __init__(self, backend):
            self.backend = backend

        def search(self, query, limit):
            return []

        def search_parent(self, query, parent_id, limit):
            return self.backend.search_parent(query, parent_id, limit)

    def prepare(backend):
        p = Pipeline(sources, passages, reranker=object(), rerank_allocation="parent_first")
        p.passage_bm25 = FallbackOnly(backend)
        return p.prepare_rerank(query)

    a, b = prepare(old), prepare(new)
    assert a == b and len(a[1]) == 8
    assert [(d, p, hashlib.sha256(payload.encode()).hexdigest()) for d, p, payload in a[1]] == [
        (d, p, hashlib.sha256(payload.encode()).hexdigest()) for d, p, payload in b[1]]
