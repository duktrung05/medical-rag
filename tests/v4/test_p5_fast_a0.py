from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from scripts.v4.run_p5_2_split import ScoreCache
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.data.pair_lineage import (PairLineageResolver, process_alive,
                                          require_process_stopped, snapshot_source,
                                          write_once)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.retrieval.disk_dense import DiskDense, input_sha256
from src.medrag.retrieval.instrumented_dense import InstrumentedDiskDense
from src.medrag.retrieval.stage_timing import prepare_rerank_with_stage_timing


class Encoder:
    def encode_query(self, texts):
        return [[1.0, 0.0]]


def test_instrumented_dense_is_exact_and_reports_exclusive_stages(tmp_path):
    path = tmp_path / "dense"
    writer = DiskDense(path, Encoder(), "fixture", create=True,
                       expected_rows=3, dimension=2, block_rows=2)
    writer.append_batch([
        (3, "3:a", input_sha256("a", "")),
        (1, "1:b", input_sha256("b", "")),
        (2, "2:c", input_sha256("c", "")),
    ], np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32))
    writer.complete()
    expected = DiskDense(path, Encoder(), "fixture", block_rows=2).search("query", 3)
    timed = InstrumentedDiskDense(path, Encoder(), "fixture", block_rows=2)
    assert timed.search("query", 3) == expected
    assert set(timed.last_timing) == {
        "query_embedding_seconds", "dense_search_seconds", "dense_total_seconds"
    }
    assert timed.last_timing["dense_total_seconds"] >= sum(
        timed.last_timing[name] for name in ("query_embedding_seconds", "dense_search_seconds")
    )


def test_stage_timer_preserves_complete_pipeline_output_and_restores_backends():
    sources = [Source("v", i, f"u{i}", f"T{i}", "common fever", "x", "raw")
               for i in range(1, 4)]
    passages = [Passage(f"{s.official_id}:p", "v", s.official_id, s.url, s.text,
                        0, len(s.text), s.text_sha256, "en", "x") for s in sources]
    expected_pipeline = Pipeline(sources, passages, reranker=object(),
                                 rerank_allocation="parent_first")
    expected = expected_pipeline.prepare_rerank("common fever", 3)
    pipeline = Pipeline(sources, passages, reranker=object(), rerank_allocation="parent_first")
    original = (pipeline.doc_bm25, pipeline.passage_bm25, pipeline.dense)
    candidates, pairs, diagnostics, timing = prepare_rerank_with_stage_timing(
        pipeline, "common fever", 3
    )
    assert (candidates, pairs, diagnostics) == expected
    assert (pipeline.doc_bm25, pipeline.passage_bm25, pipeline.dense) == original
    assert timing["document_bm25_calls"] == 1
    assert timing["passage_bm25_calls"] == 1
    assert timing["fusion_calls"] == 1
    assert timing["prepare_total_seconds"] >= timing["pair_construction_seconds"] >= 0


def _contract():
    return {"format": "medrag-v4-a0-pair-score-v1", "index_manifest_sha256": "a" * 64,
            "corpus_version": "v", "model": "m", "revision": "r", "max_length": 512,
            "precision": "fp16", "pair_format": "query_title_newline_passage-v1"}


def _lock():
    return {"format": "medrag-v4-p5-2-split-lock-v1", "mode": "full",
            "config_sha256": "b" * 64, "index_manifest_sha256": "a" * 64,
            "official_queries_sha256": "c" * 64, "official_corpus_sha256": "d" * 64,
            "query_ids": list(range(1, 1201)), "score_contract": _contract(),
            "implementation_sha256": {}}


def _shard(qid, query, lock_path, cache):
    pairs, candidates = [], []
    for index in range(200):
        doc = qid * 1000 + index
        pid = f"{doc}:p"
        payload = f"Title {doc}\nPayload {index}"
        pairs.append({"official_id": doc, "passage_id": pid, "payload": payload,
                      "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "cache_key": cache._key(query, doc, pid, payload)})
        candidates.append({"official_id": doc, "score": 1.0, "ranks": {},
                           "evidence": [{"passage_id": pid, "text": f"Payload {index}",
                                         "start": 0, "end": 1,
                                         "score_status": "retrieval_only"}]})
    return {"format": "medrag-v4-p5-2-pairs-v1", "id": qid,
            "lock_sha256": sha256_file(lock_path), "query": query,
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "candidate_count": 200, "candidates": candidates,
            "diagnostics": {"rerank_pairs": 200}, "pairs": pairs}


def test_lineage_references_immutable_source_and_keeps_lock_domains_separate(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    source_lock = _lock()
    write_once(source / "lock.json", source_lock)
    source_cache = ScoreCache(source / "key_contract.sqlite", _contract(), create=True)
    write_once(source / "pairs" / "1.json", _shard(1, "q1", source / "lock.json", source_cache))
    before = (source / "pairs" / "1.json").read_bytes()
    lineage = snapshot_source(source, {1: "q1"})
    assert lineage["inherited_query_ids"] == [1]
    assert lineage["source_files_copied"] is False

    target.mkdir()
    write_once(target / "lineage.json", lineage)
    fast_lock = copy.deepcopy(source_lock)
    fast_lock.update({"format": "medrag-v4-p5-fast-a0-lock-v1",
                      "lineage_sha256": sha256_file(target / "lineage.json")})
    write_once(target / "lock.json", fast_lock)
    local_cache = ScoreCache(target / "key_contract.sqlite", _contract(), create=True)
    resolver = PairLineageResolver(target)
    assert resolver.validate(1, "q1")[1] == "inherited"
    write_once(target / "pairs" / "2.json", _shard(2, "q2", target / "lock.json", local_cache))
    assert resolver.validate(2, "q2")[1] == "local"
    manifest = resolver.build_merged_manifest({1: "q1", 2: "q2", **{
        qid: f"q{qid}" for qid in range(3, 1201)}}, require_complete=False)
    assert manifest["inherited_shards"] == manifest["local_shards"] == 1
    assert manifest["missing_query_ids"][0] == 3
    assert (source / "pairs" / "1.json").read_bytes() == before
    write_once(target / "pairs" / "1.json", _shard(1, "q1", target / "lock.json", local_cache))
    with pytest.raises(ValueError, match="conflict"):
        resolver.resolve(1)


def test_live_process_blocks_snapshot_gate():
    import os

    assert process_alive(os.getpid())
    with pytest.raises(RuntimeError, match="still active"):
        require_process_stopped(os.getpid())


def test_production_migration_requires_explicit_handoff_token(tmp_path):
    from scripts.v4.migrate_p5_fast_a0 import create_migration

    with pytest.raises(ValueError, match="GO_STOPPED_P5_CPU"):
        create_migration(tmp_path / "source", tmp_path / "target", -1,
                         enforce_production_paths=True)


def test_fast_prepare_first_publication_creates_pairs_directory_and_resumes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.v4 import run_p5_fast_a0 as runner

    target = tmp_path / 'fresh_migration'
    target.mkdir()
    write_once(target / 'lock.json', _lock())
    cache = ScoreCache(target / 'key_contract.sqlite', _contract(), create=True)
    fixture = _shard(1, 'q1', target / 'lock.json', cache)
    def validate(qid, query):
        import json
        from scripts.v4.run_p5_2_split import validate_pair_shard
        shard = json.loads((target / 'pairs' / f'{qid}.json').read_text())
        validate_pair_shard(shard, _lock(), qid, query, cache)
    resolver = SimpleNamespace(
        lock={'query_ids': [1]}, lock_path=target / 'lock.json', local_cache=cache,
        resolve=lambda qid: (target / 'pairs' / f'{qid}.json', 'local'),
        validate=validate,
        build_merged_manifest=lambda *a, **k: dict(inherited_shards=0, local_shards=1,
                                                  missing_query_ids=[], complete=True))
    monkeypatch.setattr(runner, 'verify_runtime', lambda _: ({}, [{'id': 1, 'query': 'q1'}], tmp_path, resolver))
    monkeypatch.setattr(runner, 'build_pipeline', lambda *a: object())
    calls = []
    def timed(*args):
        calls.append(1)
        return (fixture['candidates'], [(p['official_id'], p['passage_id'], p['payload'])
                                       for p in fixture['pairs']], fixture['diagnostics'],
                {'parent_fallback_calls': 0, 'prepare_total_seconds': 0.01})
    monkeypatch.setattr(runner, 'prepare_rerank_with_stage_timing', timed)
    assert not (target / 'pairs').exists()
    assert runner.prepare(target)['prepared_now'] == 1
    before = (target / 'pairs' / '1.json').read_bytes()
    assert (target / 'timings' / '1.json').is_file()
    assert runner.prepare(target)['prepared_now'] == 0
    assert (target / 'pairs' / '1.json').read_bytes() == before
    assert len(calls) == 1
