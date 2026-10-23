"""The split boundary must preserve V4 scores and reject detached imports."""
from __future__ import annotations

import copy
import hashlib

import pytest

from scripts.v4.run_p5_2_split import validate_score_shard
from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.evidence.selector import SelectionPolicy


class Reranker:
    def score(self, query, payloads):
        return [float(len(payload)) for payload in payloads]


def test_prepare_score_finish_matches_monolithic_parent_first():
    sources = [Source("test", doc, f"https://example.test/{doc}", f"Title {doc}",
                      f"Medical text for parent {doc}", "fixture", "raw") for doc in (1, 2, 3)]
    passages = [Passage(f"{source.official_id}:p", "test", source.official_id,
                        source.url, source.text, 0, len(source.text), source.text_sha256,
                        "en", "fixture") for source in sources]
    policy = SelectionPolicy(max_docs=3, max_chunks=2, chunks_per_doc=1,
                             document_selection="independent")
    pipeline = Pipeline(sources, passages, reranker=Reranker(),
                        rerank_allocation="parent_first", selection_policy=policy)
    original, trace = pipeline.retrieve_with_diagnostics("medical text", 3,
                                                          include_candidates=True)
    candidates, pairs, diagnostics = pipeline.prepare_rerank("medical text", 3)
    scores = pipeline.reranker.score("medical text", [pair[2] for pair in pairs])
    split, split_trace = pipeline.finish_rerank(copy.deepcopy(candidates), pairs, scores, 3,
                                                diagnostics, include_candidates=True)
    assert split == original
    assert split_trace == trace


def test_score_import_requires_exact_shard_order_identity_hash_and_finite_values():
    payload = "Title\nmedical evidence"
    pair = {"cache_key": "key", "official_id": 7, "passage_id": "7:p",
            "input_sha256": hashlib.sha256(payload.encode()).hexdigest(), "payload": payload}
    pair_shard = {"id": 53, "pairs": [pair]}
    scores = {"format": "medrag-v4-p5-2-scores-v1", "id": 53,
              "pair_shard_sha256": "pair-hash", "contract_fingerprint": "contract-hash",
              "scores": [{key: pair[key] for key in
                          ("cache_key", "official_id", "passage_id", "input_sha256")}
                         | {"score": 1.25}]}
    assert validate_score_shard(scores, pair_shard, "pair-hash", "contract-hash") == [1.25]
    for altered in (
        {**scores, "pair_shard_sha256": "wrong"},
        {**scores, "contract_fingerprint": "wrong"},
        {**scores, "scores": [{**scores["scores"][0], "input_sha256": "wrong"}]},
        {**scores, "scores": [{**scores["scores"][0], "score": float("nan")}]},
        {**scores, "scores": [{**scores["scores"][0], "score": True}]},
        {**scores, "scores": []},
    ):
        with pytest.raises(ValueError):
            validate_score_shard(altered, pair_shard, "pair-hash", "contract-hash")
