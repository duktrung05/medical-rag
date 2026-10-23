"""Exact disk dense search keeps passage identities and fails on broken mappings."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.medrag.retrieval.disk_dense import DiskDense, input_sha256


class Encoder:
    def encode_query(self, texts):
        return [[1.0, 0.0]]


def test_disk_dense_search_and_resume(tmp_path):
    path = tmp_path / "dense"
    dense = DiskDense(path, Encoder(), "snapshot-model-contract", create=True,
                      expected_rows=3, dimension=2, block_rows=2)
    records = [(2, "2:first", input_sha256("first", "Title")),
               (9, "9:tail", input_sha256("tail", "Title")),
               (3, "3:middle", input_sha256("middle", ""))]
    assert dense.append_batch(records[:1], [[0.0, 1.0]]) == 1
    resumed = DiskDense(path, Encoder(), "snapshot-model-contract", block_rows=2)
    assert resumed.count() == 1
    with pytest.raises(ValueError, match="incomplete"):
        resumed.search("tail")
    resumed.append_batch(records[1:], [[1.0, 0.0], [1.0, 0.0]])
    assert resumed.complete()["rows"] == 3
    hits = DiskDense(path, Encoder(), "snapshot-model-contract", block_rows=2).search("tail", 3)
    assert [(h.official_id, h.passage_id) for h in hits] == [
        (3, "3:middle"), (9, "9:tail"), (2, "2:first")]
    with pytest.raises(ValueError, match="fingerprint"):
        DiskDense(path, Encoder(), "wrong")


def test_disk_dense_rejects_invalid_vectors(tmp_path):
    dense = DiskDense(tmp_path / "dense", Encoder(), "key", create=True,
                      expected_rows=1, dimension=2)
    row = [(1, "p", hashlib.sha256(b"payload").hexdigest())]
    with pytest.raises(ValueError, match="finite"):
        dense.append_batch(row, [[np.nan, 0.0]])
    with pytest.raises(ValueError, match="normalized"):
        dense.append_batch(row, [[2.0, 0.0]])
    assert dense.count() == 0
