"""Sampling tests use only tiny local Parquet files."""

import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import urllib.request
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.workflows.vibiomir.prepare_vibiomir import prepare, sha256_file
from scripts.workflows.vibiomir.sample_vibiomir_urls import allocate_quotas, sample_urls, selection_hash


@pytest.fixture
def dataset(tmp_path):
    def create(counts=None, rows=None):
        counts = counts or {"a.vn": 8, "b.vn": 7, "c.vn": 4}
        if rows is None:
            rows = []
            for domain, count in counts.items():
                for number in range(count):
                    official_id = 1000 + len(rows) * 17
                    host = f"WWW.{domain.upper()}." if number == 0 else domain
                    rows.append((official_id, f"https://{host}/article/{number}"))
        corpus = tmp_path / "links_corpus.parquet"
        query = tmp_path / "query.parquet"
        inventory_dir = tmp_path / "inventory"
        pq.write_table(pa.table({"id": pa.array([10], type=pa.int64()),
                                 "query": pa.array(["test"], type=pa.string())}), query)
        pq.write_table(pa.table({"id": pa.array([id_ for id_, _ in rows], type=pa.int64()),
                                 "url": pa.array([url for _, url in rows], type=pa.string())}),
                       corpus, row_group_size=3)
        prepare(query, corpus, inventory_dir, batch_size=2, progress=lambda _: None)
        return corpus, inventory_dir / "inventory.json", tmp_path / "sample.parquet", tmp_path / "summary.json"
    return create


def run(paths, sample_size, per_domain_base=2, **kwargs):
    return sample_urls(*paths, sample_size=sample_size,
                       per_domain_base=per_domain_base, batch_size=2,
                       progress=lambda _: None, **kwargs)


def read_rows(paths):
    return pq.read_table(paths[2]).to_pylist()


def test_covers_every_domain_and_uses_base_when_available(dataset):
    paths = dataset()
    summary = run(paths, sample_size=8)
    rows = read_rows(paths)
    assert summary["sample_size_actual"] == 8
    assert summary["total_domains"] == summary["sampled_domains"] == 3
    assert Counter(row["domain"] for row in rows) == {"a.vn": 3, "b.vn": 3, "c.vn": 2}
    assert [row["sampling_reason"] for row in rows if row["domain"] == "a.vn"] == [
        "domain_base", "domain_base", "large_domain_extra"]
    assert sum(item["quota"] for item in summary["domain_allocation"]) == 8
    assert sha256_file(paths[2]) == summary["output_sha256"]


def test_small_domain_reallocates_quota_to_largest(dataset):
    paths = dataset({"a.vn": 8, "b.vn": 5, "c.vn": 1})
    summary = run(paths, sample_size=12, per_domain_base=4)
    quotas = {item["domain"]: item["quota"] for item in summary["domain_allocation"]}
    assert quotas == {"a.vn": 6, "b.vn": 5, "c.vn": 1}
    assert sum(quotas.values()) == 12
    assert [item["large_domain_extra"] for item in summary["domain_allocation"]] == [2, 1, 0]


def test_sample_smaller_than_domain_count_selects_largest_domains(dataset):
    paths = dataset({"a.vn": 8, "b.vn": 7, "c.vn": 4})
    summary = run(paths, sample_size=2, per_domain_base=5)
    assert summary["sampled_domains"] == 2
    assert {row["domain"] for row in read_rows(paths)} == {"a.vn", "b.vn"}


def test_exact_total_unique_pairs_and_source_membership(dataset):
    paths = dataset()
    before = sha256_file(paths[0])
    run(paths, sample_size=12)
    rows = read_rows(paths)
    source = pq.read_table(paths[0]).to_pylist()  # Tiny fixture only.
    source_pairs = {(row["id"], row["url"]) for row in source}
    assert len(rows) == 12
    assert len({row["id"] for row in rows}) == 12
    assert len({row["url"] for row in rows}) == 12
    assert {(row["id"], row["url"]) for row in rows} <= source_pairs
    assert before == sha256_file(paths[0])


def test_selection_is_sha256_minimum_and_order_is_deterministic(dataset):
    paths = dataset({"a.vn": 5, "b.vn": 4})
    run(paths, sample_size=4, per_domain_base=2)
    rows = read_rows(paths)
    assert rows == sorted(rows, key=lambda row: (row["domain"], row["sample_rank_in_domain"], row["id"]))
    source = pq.read_table(paths[0]).to_pylist()
    for domain in ("a.vn", "b.vn"):
        candidates = [row for row in source if domain in row["url"].lower()]
        candidates.sort(key=lambda row: (selection_hash(row["id"], row["url"]), row["id"], row["url"]))
        chosen = [row for row in rows if row["domain"] == domain]
        assert [(row["id"], row["url"]) for row in chosen] == [
            (row["id"], row["url"]) for row in candidates[:2]]
        assert [row["sample_rank_in_domain"] for row in chosen] == [0, 1]
        for row in chosen:
            assert row["selection_hash"] == hashlib.sha256(
                f"{row['id']}\0{row['url']}".encode()).hexdigest()


def test_repeat_run_is_byte_identical_for_parquet_and_summary(dataset):
    paths = dataset()
    run(paths, sample_size=8)
    original = paths[2].read_bytes(), paths[3].read_bytes()
    run(paths, sample_size=8)
    assert (paths[2].read_bytes(), paths[3].read_bytes()) == original


def test_python_hash_seed_does_not_affect_selection(dataset):
    paths = dataset()
    command = [sys.executable, "-m", "scripts.workflows.vibiomir.sample_vibiomir_urls", "--corpus-file", str(paths[0]),
               "--inventory-file", str(paths[1]), "--output-file", str(paths[2]),
               "--summary-file", str(paths[3]), "--sample-size", "8",
               "--per-domain-base", "2", "--batch-size", "2"]
    hashes = []
    for seed in ("1", "935739"):
        result = subprocess.run(command, env={**os.environ, "PYTHONHASHSEED": seed},
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        hashes.append(sha256_file(paths[2]))
    assert hashes[0] == hashes[1]


def test_sha_mismatch_stops_before_output_publication(dataset):
    paths = dataset()
    paths[2].write_bytes(b"old sample")
    paths[3].write_text("old summary")
    with paths[0].open("ab") as handle:
        handle.write(b"extra bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run(paths, sample_size=8)
    assert paths[2].read_bytes() == b"old sample"
    assert paths[3].read_text() == "old summary"
    assert not list(paths[2].parent.glob(".sample.parquet.*.tmp"))


def test_stale_domain_stats_stops_before_output_publication(dataset):
    paths = dataset()
    stats = paths[1].parent / "domain_stats.csv"
    with stats.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["domain", "url_count"])
        writer.writerows([("a.vn", 7), ("b.vn", 8), ("c.vn", 4)])
    with pytest.raises(ValueError, match="disagrees"):
        run(paths, sample_size=8)
    assert not paths[2].exists() and not paths[3].exists()


def test_corpus_is_streamed_without_full_read(dataset, monkeypatch):
    paths = dataset()
    original = pq.ParquetFile
    class StreamingCorpus:
        def __init__(self, path):
            self.file = original(path)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.file.close()
        def __getattr__(self, name):
            if name == "read":
                raise AssertionError("Full corpus read is forbidden")
            return getattr(self.file, name)
    monkeypatch.setattr(pq, "ParquetFile", lambda path: StreamingCorpus(path))
    progress = []
    sample_urls(*paths, sample_size=8, per_domain_base=2, batch_size=2, progress=progress.append)
    assert len([item for item in progress if item.startswith("Corpus batch")]) == 10


def test_sample_size_exceeding_corpus_reports_clear_error(dataset):
    paths = dataset()
    with pytest.raises(ValueError, match="exceeds available corpus URL rows"):
        run(paths, sample_size=20)
    assert not paths[2].exists()


def test_no_network_call(dataset, monkeypatch):
    paths = dataset()
    def blocked(*args, **kwargs):
        raise AssertionError("Network call attempted")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    run(paths, sample_size=8)
    assert paths[2].exists()


def test_invalid_inventory_blocks_sampling(dataset):
    paths = dataset()
    inventory = json.loads(paths[1].read_text())
    inventory["corpus_file"]["duplicate_ids"] = 1
    paths[1].write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="invalid IDs"):
        run(paths, sample_size=8)
    assert not paths[2].exists()


def test_output_paths_cannot_overwrite_input_files(dataset):
    paths = dataset()
    with pytest.raises(ValueError, match="distinct"):
        sample_urls(paths[0], paths[1], paths[0], paths[3], sample_size=8, progress=lambda _: None)


def test_quota_allocation_respects_capacity_and_ties():
    assert allocate_quotas({"z.vn": 3, "a.vn": 3, "x.vn": 1}, 6, 3) == {
        "a.vn": 3, "z.vn": 2, "x.vn": 1}
