"""Exact SQL-rewrite parity, including accumulation, ties and downstream pairs."""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.parent_first_bm25 import PARENT_SQL, ParentFirstDiskBM25


def exact(hits):
    return [(h.official_id, h.passage_id, h.score.hex(), h.channel, h.rank) for h in hits]


@pytest.fixture
def readers(tmp_path):
    path = tmp_path / "passages.sqlite"
    old = DiskBM25(path, "passage_bm25", "fixture", create=True)
    old.append_batch([
        (9, "z", "common fever dose 5 mg 高血压 điều trị"),
        (9, "a", "common fever dose 5 mg 高血压 điều trị"),
        (9, "b", "dose dose 10 mg treatment"),
        (9, "empty", ""),
        (3, "other", "common common treatment"),
        (6, "rare", "rare marker"),
    ])
    old.complete()
    return old, ParentFirstDiskBM25(path, "passage_bm25", "fixture")


@pytest.mark.parametrize("query", [
    "common fever dose 5 mg treatment", "dose dose 10 mg", "高血压 治疗",
    "điều trị", "điều trị", "absent", "", "!!!", "common unknown rare",
])
@pytest.mark.parametrize("parent", [9, 3, 6, 999])
@pytest.mark.parametrize("limit", [1, 2, 20])
def test_parent_hits_scores_and_ranks_are_bit_exact(readers, query, parent, limit):
    old, new = readers
    assert exact(new.search_parent(query, parent, limit)) == exact(old.search_parent(query, parent, limit))
    assert exact(new.search(query, limit, parent_id=parent)) == exact(old.search(query, limit, parent_id=parent))


def test_global_search_unchanged_and_parent_ties_use_passage_id(readers):
    old, new = readers
    assert exact(new.search("common dose", 10)) == exact(old.search("common dose", 10))
    assert [h.passage_id for h in new.search_parent("fever", 9, 20)] == ["a", "z"]


def test_parent_query_plan_uses_both_existing_indexes(readers):
    old, _ = readers
    with sqlite3.connect(old.path.as_uri() + "?mode=ro", uri=True) as db:
        plan = "\n".join(str(r[-1]) for r in db.execute("EXPLAIN QUERY PLAN " + PARENT_SQL, (9, "dose")))
    assert "docs_parent" in plan and "official_id=?" in plan
    assert "term=? AND row_id=?" in plan


def test_rewrite_does_not_write_index_and_rejects_invalid_state(readers):
    old, new = readers
    before = hashlib.sha256(old.path.read_bytes()).hexdigest()
    new.search_parent("common dose", 9)
    assert hashlib.sha256(old.path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="positive"):
        new.search_parent("dose", 9, 0)
    with pytest.raises(ValueError, match="fingerprint"):
        ParentFirstDiskBM25(old.path, "passage_bm25", "wrong")
    with sqlite3.connect(old.path) as db:
        db.execute("UPDATE meta SET value='building' WHERE key='status'")
    with pytest.raises(ValueError, match="incomplete"):
        new.search_parent("dose", 9)


def test_pair_id_payload_hash_and_full_candidates_exact(tmp_path):
    sources = [Source("fixture", doc, f"https://fixture/{doc}", f"Title {doc}",
                      "common fever treatment", "fixture", "raw") for doc in (9, 3, 6)]
    passages = [Passage(f"{s.official_id}:p", s.corpus_version, s.official_id,
                        s.url, s.text, 0, len(s.text), s.text_sha256, "en", "fixture")
                for s in sources]
    path = tmp_path / "pairs.sqlite"
    old = DiskBM25(path, "passage_bm25", "fixture", create=True)
    old.append_batch((p.official_id, p.passage_id, p.text) for p in passages)
    old.complete()

    class EmptyBranchWithFallback:
        def __init__(self, backend):
            self.backend = backend

        def search(self, query, limit):
            return []  # Every document branch hit must exercise fallback.

        def search_parent(self, query, parent_id, limit):
            return self.backend.search_parent(query, parent_id, limit)

    def prepare(backend):
        pipe = Pipeline(sources, passages, reranker=object(), rerank_allocation="parent_first")
        pipe.passage_bm25 = EmptyBranchWithFallback(backend)
        return pipe.prepare_rerank("common fever treatment")

    before = prepare(old)
    after = prepare(ParentFirstDiskBM25(path, "passage_bm25", "fixture"))
    assert after == before
    assert len(after[1]) == 3
    assert [(d, p, hashlib.sha256(text.encode()).hexdigest()) for d, p, text in after[1]] == [
        (d, p, hashlib.sha256(text.encode()).hexdigest()) for d, p, text in before[1]]


def test_adversarial_multi_term_exact_with_nonlocal_statistics(tmp_path):
    """Local IDF/avgdl or changed term accumulation must not sneak in."""
    path = tmp_path / "many.sqlite"
    old = DiskBM25(path, "passage_bm25", "fixture", create=True)
    words = [f"term{i}" for i in range(32)]
    old.append_batch((parent, f"{parent}:{i}", " ".join(
        word for k, word in enumerate(words) for _ in range(1 + (i + k) % 5)))
                     for parent in range(7) for i in range(4))
    old.complete()
    new = ParentFirstDiskBM25(path, "passage_bm25", "fixture")
    for parent in range(7):
        assert exact(new.search_parent(" ".join(words), parent, 4)) == exact(
            old.search_parent(" ".join(words), parent, 4))


def test_benchmark_fails_closed_before_creating_index_on_frozen_score_failure(tmp_path):
    import json
    import time

    from scripts.v4.benchmark_parent_first_bm25 import benchmark

    report = tmp_path / "parity.json"
    report.write_text(json.dumps({"strict_frozen_gate": "FAIL", "old_new_exact_parity": "PASS"}))
    with pytest.raises(ValueError, match="strict frozen"):
        benchmark(tmp_path, report, 100, 1, time.monotonic() + 2)
    assert not (tmp_path / "synthetic.sqlite").exists()


def test_empty_complete_corpus_returns_empty(tmp_path):
    path = tmp_path / "empty.sqlite"
    old = DiskBM25(path, "passage_bm25", "empty", create=True)
    old.complete()
    new = ParentFirstDiskBM25(path, "passage_bm25", "empty")
    assert new.search_parent("fever", 1) == old.search_parent("fever", 1) == []
