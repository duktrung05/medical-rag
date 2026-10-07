"""Behavioral checks for authentic scores, abstention, replay and calibration."""
import json
import subprocess
import sys
import types
import zipfile
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.calibrate_evidence import calibrate, evaluate, load_truth, prepare_labels
from r2ai.evidence_io import (
    fingerprint,
    read_ranking_cache,
    write_predictions,
    write_ranking_cache,
)
from r2ai.evidence_selector import (
    EvidenceCandidate,
    QueryRanking,
    SelectionConfig,
    select_evidence,
)
from r2ai.select_evidence import replay
from r2ai.validate import validate


def candidate(doc, index, score, text=None, retrieval=.99):
    return EvidenceCandidate(doc, index, text or f"evidence {doc}/{index}", score, retrieval)


def threshold(**kwargs):
    return SelectionConfig(mode="threshold", chunk_threshold=0, **kwargs)


def test_filters_negative_and_unreranked_without_backfill():
    row = QueryRanking(1, [1, 2, 3], [candidate(1, 0, 2), candidate(2, 0, -1), candidate(3, 0, None)])
    prediction, diag = select_evidence(row, threshold())
    assert prediction["relevant_docs"] == [1]
    assert [c["doc_id"] for c in prediction["relevant_chunks"]] == [1]
    assert diag["reasons"] == {"selected": 1, "below_chunk_threshold": 1, "not_reranked": 1}


@pytest.mark.parametrize("limits", [{}, {"max_docs": 0}, {"max_chunks": 0}, {"chunks_per_doc": 0}])
def test_can_abstain_with_low_scores_or_zero_limits(limits):
    row = QueryRanking(1, [1], [candidate(1, 0, -4 if not limits else 4)])
    prediction, _ = select_evidence(row, threshold(**limits))
    assert prediction == {"id": 1, "relevant_docs": [], "relevant_chunks": []}


def test_relative_gate_uses_highest_raw_score_instead_of_metadata_priority():
    row = QueryRanking(1, [2, 1], [candidate(2, 0, 1), candidate(1, 0, 5)])
    prediction, diag = select_evidence(row, threshold(chunk_delta=1))
    assert prediction["relevant_docs"] == [1]
    assert diag["chunk_relative_cutoff"] == 4


def test_caps_deduplication_and_original_unicode_text():
    text = "  Đái tháo đường\n糖尿病  "
    row = QueryRanking(1, [1, 2], [candidate(1, 0, 5, text), candidate(1, 1, 4, text),
                                 candidate(1, 2, 3), candidate(2, 0, 6)])
    prediction, diag = select_evidence(row, threshold(max_docs=1, max_chunks=3, chunks_per_doc=2))
    assert prediction["relevant_docs"] == [1]
    assert prediction["relevant_chunks"][0]["chunk_text"] == text
    assert len(prediction["relevant_chunks"]) == 2
    assert diag["reasons"]["duplicate_text"] == 1
    assert diag["reasons"]["document_cap"] == 1


def test_chunk_only_ablation_preserves_baseline_documents():
    row = QueryRanking(1, [1, 2], [candidate(1, 0, -5), candidate(2, 0, 3)])
    prediction, _ = select_evidence(row, threshold(document_mode="baseline", max_docs=2))
    assert prediction["relevant_docs"] == [1, 2]
    assert [c["doc_id"] for c in prediction["relevant_chunks"]] == [2]


def test_baseline_documents_can_include_parents_without_evidence():
    row = QueryRanking(1, [1, 2], [candidate(1, 0, 3)])
    prediction, _ = select_evidence(row, threshold(document_mode="baseline", max_docs=2))
    assert prediction["relevant_docs"] == [1, 2]
    parent_prediction, _ = select_evidence(row, threshold())
    assert parent_prediction["relevant_docs"] == [1]
    with pytest.raises(ValueError, match="contain all candidate parents"):
        QueryRanking(1, [2], [candidate(1, 0, 3)])


def test_real_baseline_preparation_checks_original_corpus_and_fingerprints(tmp_path):
    from r2ai.test_evidence_submission import prepare

    queries, corpus, baseline = (tmp_path / name for name in ("queries.parquet", "chunks.parquet", "baseline.zip"))
    pq.write_table(pa.table({"id": [1], "query": ["Question?"]}), queries)
    text = "  Nội dung gốc\n糖尿病  "
    pq.write_table(pa.table({"doc_id": [10], "chunk_index": [7], "text": [text], "title": ["Original title"]}), corpus)
    write_predictions(baseline, [{"id": 1, "relevant_docs": [10, 20],
                                  "relevant_chunks": [{"doc_id": 10, "chunk_text": text}]}])
    output = tmp_path / "run"
    output.mkdir()
    rows, metadata = prepare(baseline, corpus, queries, output)
    assert rows[0].candidates[0].key == (10, 7)
    assert rows[0].candidates[0].title == "Original title"
    assert rows[0].candidates[0].chunk_text == text
    assert rows[0].candidates[0].rerank_score is None
    assert metadata["verified_unique_chunks"] == 1
    assert prepare(baseline, corpus, queries, output)[0] == rows
    pq.write_table(pa.table({"id": [1], "query": ["Changed question"]}), queries)
    with pytest.raises(ValueError, match="fingerprints changed"):
        prepare(baseline, corpus, queries, output)


def test_document_gate_never_readmits_rejected_parents():
    row = QueryRanking(1, [1, 2, 3], [candidate(1, 0, 3), candidate(2, 0, 6), candidate(3, 0, None)])
    prediction, _ = select_evidence(row, threshold(document_mode="threshold", doc_threshold=5))
    assert prediction["relevant_docs"] == [2]
    assert [c["doc_id"] for c in prediction["relevant_chunks"]] == [2]


def test_topk_matches_existing_submission_emitter(tmp_path, monkeypatch):
    from r2ai import rank

    monkeypatch.setattr(rank, "SUB", tmp_path)
    row = QueryRanking(1, [2, 1], [candidate(2, 0, -4), candidate(2, 1, None), candidate(1, 0, 6)])
    rank.emit([1], [([2, 1], {2: [0, 1], 1: [2]})], [c.chunk_text for c in row.candidates],
              2, 2, 1, False, "baseline")
    with zipfile.ZipFile(tmp_path / "baseline.zip") as archive:
        baseline = json.loads(archive.read("predictions.json"))[0]
    selected, _ = select_evidence(row, SelectionConfig(max_docs=2, max_chunks=2))
    assert selected == baseline


@pytest.mark.parametrize("kwargs", [{"chunk_threshold": float("nan")}, {"chunk_delta": -1},
                                    {"max_docs": -1}, {"max_chunks": True}, {"doc_threshold": 1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(threshold(), **kwargs)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError):
        candidate(1, 0, score)


def test_empty_ranking_and_duplicate_id_validation():
    prediction, _ = select_evidence(QueryRanking(1, [], []), threshold())
    assert prediction["relevant_chunks"] == []
    with pytest.raises(ValueError, match="Duplicate chunk"):
        QueryRanking(1, [1], [candidate(1, 0, 3), candidate(1, 0, 2)])


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_cache_roundtrip_and_cpu_replay(tmp_path, suffix):
    rows = [QueryRanking(1, [1, 2], [candidate(1, 0, 2), candidate(2, 0, None)], "Câu hỏi?"),
            QueryRanking(2, [], [], "No evidence?")]
    cache = tmp_path / f"rankings{suffix}"
    metadata = {"reranker": {"activation": "identity"}}
    write_ranking_cache(cache, rows, metadata)
    loaded, manifest = read_ranking_cache(cache)
    assert loaded == rows and manifest == metadata
    with pytest.raises(FileExistsError):
        write_ranking_cache(cache, rows, metadata)
    output = tmp_path / "selected"
    summary = replay(loaded, manifest, threshold(), output)
    assert summary["empty_chunk_rate"] == .5
    assert summary["model_inference"] is False
    with zipfile.ZipFile(output / "submission.zip") as archive:
        records = json.loads(archive.read("predictions.json"))
    assert records[1] == {"id": 2, "relevant_docs": [], "relevant_chunks": []}
    with pytest.raises(FileExistsError):
        replay(loaded, manifest, threshold(), output)


def test_cache_detects_truncation_and_content_change(tmp_path):
    cache = tmp_path / "rankings.jsonl"
    write_ranking_cache(cache, [QueryRanking(1, [], [])], {})
    before = fingerprint(cache)
    cache.write_text(cache.read_text().splitlines()[0] + "\n")
    assert fingerprint(cache)["sha256"] != before["sha256"]
    with pytest.raises(ValueError, match="Truncated"):
        read_ranking_cache(cache)


def test_macro_metrics_include_abstaining_queries():
    rows = [QueryRanking(1, [1], [candidate(1, 0, 2)]), QueryRanking(2, [2], [candidate(2, 0, -1)])]
    truths = {1: {"docs": {1}, "chunks": {(1, 0)}, "scope": "candidate_pool"},
              2: {"docs": {2}, "chunks": {(2, 0)}, "scope": "candidate_pool"}}
    metrics, _ = evaluate(rows, truths, threshold())
    assert metrics["chunk_precision"] == .5
    assert metrics["chunk_recall"] == .5
    assert metrics["internal_macro_f2"] == .5
    assert metrics["num_queries"] == 2


def test_label_template_requires_complete_manual_judgments(tmp_path):
    rows = [QueryRanking(1, [1, 2], [candidate(1, 0, 2), candidate(2, 0, -3)], "Which evidence?")]
    path = tmp_path / "labels.jsonl"
    prepare_labels(rows, path, 100, 42)
    with pytest.raises(ValueError, match="unjudged"):
        load_truth(path, rows)
    annotation = json.loads(path.read_text())
    assert annotation["query"] == "Which evidence?"
    annotation["judgments"][0]["relevant"] = True
    annotation["judgments"][1]["relevant"] = False
    path.write_text(json.dumps(annotation) + "\n")
    truth = load_truth(path, rows)[1]
    assert truth["chunks"] == {(1, 0)}
    annotation["judgments"].pop()
    path.write_text(json.dumps(annotation) + "\n")
    with pytest.raises(ValueError, match="cover exactly"):
        load_truth(path, rows)


def test_calibration_selects_on_dev_and_does_not_retune_on_holdout(tmp_path):
    rows = [QueryRanking(1, [1, 2], [candidate(1, 0, 2), candidate(2, 0, -1)]),
            QueryRanking(2, [3], [candidate(3, 0, -2)])]
    dev = {1: {"docs": {1}, "chunks": {(1, 0)}, "scope": "candidate_pool"}}
    holdout = {2: {"docs": {3}, "chunks": {(3, 0)}, "scope": "candidate_pool"}}
    config = threshold()
    report = calibrate(rows, dev, SelectionConfig(), [config], .02, tmp_path / "calibration", {}, holdout)
    assert report["status"] == "improving_configuration_found"
    assert report["selected_config"] == json.loads((tmp_path / "calibration/best_config.json").read_text())
    assert report["selected_holdout"]["chunk_recall"] == 0
    assert report["holdout_meets_objective"] is False
    with pytest.raises(ValueError, match="disjoint"):
        calibrate(rows, dev, SelectionConfig(), [config], .02, tmp_path / "other", {}, dev)


def test_calibration_retains_baseline_when_recall_loss_is_too_large(tmp_path):
    rows = [QueryRanking(1, [1], [candidate(1, 0, -2)])]
    dev = {1: {"docs": {1}, "chunks": {(1, 0)}, "scope": "candidate_pool"}}
    baseline = SelectionConfig()
    report = calibrate(rows, dev, baseline, [threshold()], .02, tmp_path / "calibration", {})
    assert report["status"] == "no_improving_configuration"
    assert report["selected_config"] == json.loads(json.dumps(baseline.__dict__))


def install_fake_models(monkeypatch, scores):
    constructor_calls = []

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            constructor_calls.append(kwargs)
            self.model = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="fixture-revision"))

        def predict(self, pairs, **kwargs):
            assert len(pairs) == len(scores)
            return np.asarray(scores)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        float16="float16", nn=types.SimpleNamespace(Identity=lambda: "identity"),
        cuda=types.SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    return constructor_calls


def test_partial_rerank_retains_authentic_scores_and_null_tail(monkeypatch):
    from r2ai.rank import cross_encoder_rerank

    calls = install_fake_models(monkeypatch, [-2, 3])
    metadata = {}
    pools = cross_encoder_rerank([(np.array([0, 1, 2]), np.array([.9, .8, .7]))],
                                ["query"], ["a", "b", "c"], ["", "", ""], "cpu", 2, 2,
                                model_metadata=metadata)
    assert pools[0].indices.tolist() == [1, 0, 2]
    assert pools[0].rerank_scores == [3, -2, None]
    assert pools[0].retrieval_scores.tolist() == [.8, .9, .7]
    assert calls[0]["activation_fn"] == "identity"
    assert metadata["resolved_revision"] == "fixture-revision"


def test_empty_rerank_skips_model_loading(monkeypatch):
    from r2ai.rank import cross_encoder_rerank

    calls = install_fake_models(monkeypatch, [])
    result = cross_encoder_rerank([(np.array([], dtype=int), np.array([]))], ["q"], [], [], "cpu", 2)
    assert result[0].rerank_scores == []
    assert calls == []


def test_threshold_includes_boundary_and_replay_rejects_dense_only_cache(tmp_path):
    row = QueryRanking(1, [1], [candidate(1, 0, 0)])
    prediction, _ = select_evidence(row, threshold(chunk_delta=0))
    assert prediction["relevant_docs"] == [1]
    with pytest.raises(ValueError, match="reranking enabled"):
        replay([row], {"reranker": {"enabled": False}}, threshold(), tmp_path / "invalid")


def test_rank_empty_corpus_emits_every_query_without_models(tmp_path, monkeypatch):
    from r2ai import rank

    raw, out, sub = (tmp_path / name for name in ("raw", "data", "sub"))
    raw.mkdir()
    out.mkdir()
    pq.write_table(pa.table({"id": [1, 2], "query": ["first", "second"]}), raw / "query.parquet")
    pq.write_table(pa.table({"doc_id": pa.array([], type=pa.int64()),
                             "chunk_index": pa.array([], type=pa.int32()),
                             "text": pa.array([], type=pa.string()), "title": pa.array([], type=pa.string())}),
                   out / "chunks.parquet")
    for name, path in (("RAW", raw), ("OUT", out), ("SUB", sub)):
        monkeypatch.setattr(rank, name, path)
    rank.main("chunks.parquet", [10], 10, 1, 200, True, "cpu", "empty", 2,
              selection_configs=[threshold()], ranking_cache=tmp_path / "rankings.jsonl")
    with zipfile.ZipFile(sub / "empty/submission.zip") as archive:
        predictions = json.loads(archive.read("predictions.json"))
    assert [p["id"] for p in predictions] == [1, 2]
    assert all(p["relevant_docs"] == [] and p["relevant_chunks"] == [] for p in predictions)


def test_rank_integration_writes_replayable_cache_and_empty_submission(tmp_path, monkeypatch):
    from r2ai import rank

    raw, out, sub = (tmp_path / name for name in ("raw", "data", "sub"))
    raw.mkdir()
    out.mkdir()
    pq.write_table(pa.table({"id": [1, 2], "query": ["question", "empty query"]}), raw / "query.parquet")
    pq.write_table(pa.table({"doc_id": [10, 20], "chunk_index": [0, 0], "text": ["a", "b"],
                             "title": ["", ""]}), out / "chunks.parquet")
    for name, path in (("RAW", raw), ("OUT", out), ("SUB", sub)):
        monkeypatch.setattr(rank, name, path)
    monkeypatch.setattr(rank, "embed_chunks", lambda *args: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(rank, "embed_queries", lambda *args: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(rank, "dense_shortlist", lambda *args: [
        (np.array([0, 1]), np.array([.9, .8])), (np.array([], dtype=int), np.array([]))])
    install_fake_models(monkeypatch, [-2])
    cache = tmp_path / "rankings.jsonl.gz"
    rank.main("chunks.parquet", [10], 10, 1, 200, True, "cpu", "filtered", 2,
              rerank_top=1, selection_configs=[threshold()], ranking_cache=cache)
    rows, metadata = read_ranking_cache(cache)
    assert [c.rerank_score for c in rows[0].candidates] == [-2, None]
    assert metadata["inputs"]["chunks"]["sha256"] == fingerprint(out / "chunks.parquet")["sha256"]
    with zipfile.ZipFile(sub / "filtered/submission.zip") as archive:
        records = json.loads(archive.read("predictions.json"))
    assert len(records) == 2 and all(r["relevant_chunks"] == [] for r in records)
    assert rows[0].query_text == "question"
    with pytest.raises(ValueError, match="requires reranking"):
        rank.main("chunks.parquet", [10], 10, 1, 200, False, "cpu", "invalid", 2,
                  selection_configs=[threshold()])


def test_validator_checks_parent_consistency_duplicates_and_empty_queries(tmp_path):
    queries, corpus, archive = (tmp_path / name for name in ("queries.parquet", "corpus.parquet", "submission.zip"))
    pq.write_table(pa.table({"id": [1, 2]}), queries)
    pq.write_table(pa.table({"id": [10, 20]}), corpus)
    rows = [{"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "exact text"}]},
            {"id": 2, "relevant_docs": [], "relevant_chunks": []}]
    write_predictions(archive, rows)
    assert validate(archive, query_path=queries, corpus_path=corpus) == []
    rows[0]["relevant_chunks"] *= 2
    rows[0]["relevant_docs"] = [20, 20]
    write_predictions(archive, rows)
    errors = validate(archive, query_path=queries, corpus_path=corpus)
    assert any("parent document" in e for e in errors)
    assert any("duplicate evidence" in e for e in errors)
    assert any("duplicate doc" in e for e in errors)


def test_cpu_cli_replay_does_not_require_torch(tmp_path):
    cache = tmp_path / "rankings.jsonl"
    write_ranking_cache(cache, [QueryRanking(1, [1], [candidate(1, 0, 3)])], {})
    result = subprocess.run([sys.executable, "-m", "r2ai.select_evidence", "--rankings", str(cache),
                             "--selector", "threshold", "--chunk-threshold", "0", "--output", str(tmp_path / "replay")],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "replay/submission.zip").exists()
