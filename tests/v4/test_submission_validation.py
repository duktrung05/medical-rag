"""Submission checks must catch ungrounded output and malformed ZIPs."""
from __future__ import annotations

import json

import pytest

from src.medrag.adapters.validation import (canonical_payload, package_zip,
                                            validate_payload, validate_selected, validate_zip)
from src.medrag.core.contracts import Passage, Source
from src.medrag.evidence.selector import SelectionPolicy
from src.medrag.evidence.verifier import verify


class Store:
    def __init__(self, source, passage):
        self.source_row = source
        self.passage_row = passage

    def source(self, key):
        assert key == self.source_row.official_id
        return self.source_row

    def passage(self, key):
        assert key == self.passage_row.passage_id
        return self.passage_row


def test_selected_grounding_and_reproducible_zip(tmp_path):
    source = Source("release", 4, "https://example.test/4", "title", "head tail phrase",
                    "test", "raw")
    passage = Passage("4:p8", "release", 4, source.url, "tail phrase", 5, 16,
                      source.text_sha256, "en", "test")
    policy = SelectionPolicy(max_docs=200, max_chunks=60, chunks_per_doc=1)
    selected = {7: [{"official_id": 4, "evidence": [{
        "passage_id": passage.passage_id, "text": passage.text, "start": 5, "end": 16,
        "score_status": "reranked", "rerank_score": -1.3,
    }]}], 9: []}
    predictions = validate_selected([7, 9], selected, Store(source, passage), policy,
                                    result_limit=200, reranker_enabled=True, official_ids={4})
    payload = canonical_payload(predictions)
    assert validate_payload(payload, query_ids={7, 9}, official_ids={4},
                            policy=policy, result_limit=200) == predictions
    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    assert package_zip(first, payload) == package_zip(second, payload)
    assert validate_zip(first, query_ids={7, 9}, official_ids={4}, policy=policy,
                        result_limit=200)["status"] == "pass"
    selected[7][0]["evidence"][0]["text"] = "not source text"
    with pytest.raises(ValueError, match="Ungrounded"):
        validate_selected([7, 9], selected, Store(source, passage), policy,
                          result_limit=200, reranker_enabled=True, official_ids={4})


def test_strict_json_and_offset_negative_cases():
    policy = SelectionPolicy()
    base = {"id": 7, "relevant_docs": [4],
            "relevant_chunks": [{"doc_id": 4, "chunk_text": "tail"}]}
    kwargs = {"query_ids": {7}, "official_ids": {4}, "policy": policy, "result_limit": 200}
    for broken in [
        {**base, "id": True},
        {**base, "relevant_chunks": [{"doc_id": 5, "chunk_text": "tail"}]},
        {**base, "relevant_chunks": []},
    ]:
        with pytest.raises(ValueError):
            validate_payload(json.dumps([broken]).encode(), **kwargs)
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        validate_payload(b'[{"id":7,"id":7,"relevant_docs":[],"relevant_chunks":[]}]', **kwargs)
    with pytest.raises(ValueError, match="Non-finite JSON"):
        validate_payload(b'[{"id":7,"relevant_docs":[],"relevant_chunks":[],"x":NaN}]', **kwargs)
    source = Source("r", 4, "https://example.test/4", "", "abcd", "test", "raw")
    invalid = Passage("p", "r", 4, source.url, "abc", -4, -1, source.text_sha256, "en", "test")
    assert not verify(invalid, source)
