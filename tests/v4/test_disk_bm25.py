"""The disk index must preserve V4 BM25 ranks and global corpus statistics."""
from __future__ import annotations

import pytest

from src.medrag.retrieval.core import BM25
from src.medrag.retrieval.disk_bm25 import DiskBM25


@pytest.mark.parametrize("channel", ["document_bm25", "passage_bm25"])
def test_disk_index_matches_in_memory_bm25_across_batches(tmp_path, channel):
    rows = [
        (8, "8:p0", "common symptom fever and dose 5 mg"),
        (8, "8:p1", "late section sentineltail treatment"),
        (3, "3:p0", "common symptom no treatment"),
        (11, "11:p0", "高血压 治疗 观察"),
    ]
    reference = BM25(rows, channel)
    disk = DiskBM25(tmp_path / "index.sqlite", channel, "source-sha", create=True)
    assert disk.append_batch(rows[:2]) == 2
    assert disk.append_batch(rows[2:]) == 2
    with pytest.raises(ValueError, match="incomplete"):
        disk.search("treatment")
    assert disk.complete()["rows"] == len(rows)
    reopened = DiskBM25(tmp_path / "index.sqlite", channel, "source-sha")
    for query in ("treatment", "common symptom", "sentineltail", "高血压 治疗"):
        expected = reference.search(query, 4)
        actual = reopened.search(query, 4)
        assert [(hit.official_id, hit.passage_id) for hit in actual] == [
            (hit.official_id, hit.passage_id) for hit in expected]
        assert [hit.score for hit in actual] == pytest.approx([hit.score for hit in expected])
    assert reopened.search_parent("treatment", 8)[0].passage_id == "8:p1"
    with pytest.raises(ValueError, match="fingerprint"):
        DiskBM25(tmp_path / "index.sqlite", channel, "different-source")


def test_disk_bm25_partition_rollback_uses_global_stats(tmp_path):
    first = [(1, None, "common first"), (2, None, "common second")]
    pending = [(3, None, "pending third")]
    index = DiskBM25(tmp_path / "index.sqlite", "document_bm25", "lock", create=True)
    index.append_batch(first)
    index.append_batch(pending)
    assert index.truncate_to_rows(2) == 2
    assert index.append_batch(pending) == 1
    index.complete()
    reference = BM25(first + pending, "document_bm25")
    assert [hit.score for hit in index.search("common pending", 3)] == pytest.approx(
        [hit.score for hit in reference.search("common pending", 3)])
