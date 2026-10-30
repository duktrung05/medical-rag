"""Check the full candidate's source and slot contract without a GPU."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from types import SimpleNamespace

import numpy as np
import torch

from scripts.v4.run_end_to_end_evidence_recovery import dense_batch_top800, score_query


class Tokenizer:
    def __call__(self, *texts, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [match.span() for match in re.finditer(r"\S+", " ".join(texts))]
        if return_offsets_mapping:
            return {"offset_mapping": offsets}
        return {"input_ids": list(range(len(offsets) + 2 * int(add_special_tokens)))}


class Store:
    def source(self, doc):
        text = f"key{doc} " + "medical evidence " * 12
        return SimpleNamespace(official_id=doc, title=f"Title{doc}", text=text,
                               text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def passage(self, passage_id):
        doc = int(passage_id)
        source = self.source(doc)
        return SimpleNamespace(passage_id=passage_id, official_id=doc,
                               text=source.text, source_start=0, source_end=len(source.text))

    def passages_for_parent(self, doc):
        return [self.passage(str(doc))]


class Model:
    def score(self, query, payloads):
        return [float(100 if "key62 " in text else 90 if "key61 " in text else 1)
                for text in payloads]


def test_new_parent_requires_slot_and_context_is_source_grounded():
    store = Store()
    baseline = {
        "relevant_docs": list(range(1, 62)),
        "relevant_chunks": [{"doc_id": doc, "chunk_text": store.source(doc).text}
                            for doc in range(1, 61)],
    }
    retrieval = {"parents": [{"doc_id": doc, "seed_passage_id": str(doc)}
                             for doc in range(1, 63)]}
    result = score_query(1, "medical evidence", baseline, retrieval,
                         store, Tokenizer(), Model())
    prediction = result["prediction"]
    selected = {row["doc_id"] for row in prediction["relevant_chunks"]}
    assert len(prediction["relevant_chunks"]) == 60
    assert len(selected) == 60
    assert set(baseline["relevant_docs"]).issubset(prediction["relevant_docs"])
    assert 62 in prediction["relevant_docs"] and {61, 62}.issubset(selected)
    assert len(result["swaps"]) == 2
    assert all(row["chunk_text"] in store.source(row["doc_id"]).text
               for row in prediction["relevant_chunks"])


def test_batched_dense_returns_query_specific_order_and_parent_mapping(tmp_path):
    matrix_path = tmp_path / "vectors.npy"
    np.save(matrix_path, np.asarray([[1, 0], [0, 1], [.8, .6]], dtype=np.float32))
    mapping = tmp_path / "mapping.sqlite"
    with sqlite3.connect(mapping) as db:
        db.execute("CREATE TABLE rows(row_id INTEGER, official_id INTEGER, passage_id TEXT)")
        db.executemany("INSERT INTO rows VALUES(?,?,?)",
                       [(0, 20, "p20"), (1, 10, "p10"), (2, 30, "p30")])
    class Encoder:
        def encode_query(self, texts):
            return [[1, 0] if text == "a" else [0, 1] for text in texts]
    dense = SimpleNamespace(encoder=Encoder(), matrix_path=matrix_path,
                            mapping=mapping, meta={"rows": "3", "dimension": "2"})
    result = dense_batch_top800(dense, {1: "a", 2: "b"}, torch, device="cpu")
    assert [hit.official_id for hit in result[1]] == [20, 30, 10]
    assert [hit.official_id for hit in result[2]] == [10, 30, 20]
    assert [hit.rank for hit in result[1]] == [1, 2, 3]
