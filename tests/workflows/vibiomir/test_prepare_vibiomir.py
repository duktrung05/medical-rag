"""Small local Parquet fixtures only; no real corpus or network access."""

import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.workflows.vibiomir.prepare_vibiomir import REPORT_FILES, main, prepare, sha256_file

INT64 = pa.int64()
STRING = pa.string()


@pytest.fixture
def dataset(tmp_path):
    def create(*, query_ids=(-3, 42), queries=("Hỏi về sức khỏe", "Ví dụ"),
               corpus_ids=(-8, 9007199254740993, 200),
               urls=("https://WWW.Example.COM./a", "http://example.com/b", "https://other.vn"),
               query_id_type=INT64, corpus_id_type=INT64,
               query_type=STRING, url_type=STRING):
        query_path, corpus_path = tmp_path / "query.parquet", tmp_path / "links_corpus.parquet"
        pq.write_table(pa.table({"id": pa.array(query_ids, type=query_id_type),
                                 "query": pa.array(queries, type=query_type)}), query_path)
        pq.write_table(pa.table({"id": pa.array(corpus_ids, type=corpus_id_type),
                                 "url": pa.array(urls, type=url_type)}), corpus_path, row_group_size=2)
        return query_path, corpus_path, tmp_path / "reports"
    return create


def run(paths, **kwargs):
    return prepare(*paths, batch_size=1, progress=lambda _: None, **kwargs)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cli_args(paths):
    return ["--query-file", str(paths[0]), "--corpus-file", str(paths[1]),
            "--output-dir", str(paths[2]), "--batch-size", "1"]


def test_valid_dataset_and_non_contiguous_official_ids(dataset):
    paths = dataset()
    report = run(paths)
    assert report["query_file"]["rows"] == 2
    assert (report["query_file"]["id_min"], report["query_file"]["id_max"]) == (-3, 42)
    corpus = report["corpus_file"]
    assert (corpus["rows"], corpus["scanned_rows"], corpus["row_groups"]) == (3, 3, 2)
    assert (corpus["id_min"], corpus["id_max"]) == (-8, 9007199254740993)
    assert corpus["schema"][0]["type"] == "int64"
    assert corpus["invalid_urls"] == corpus["duplicate_ids"] == corpus["duplicate_urls"] == 0
    assert report["validation_errors"] == []
    assert all((paths[2] / name).exists() for name in REPORT_FILES)
    assert not list(paths[2].glob(".inventory-*"))
    assert main(cli_args(paths)) == 0


@pytest.mark.parametrize("query_ids,queries,field,count", [
    ([5, 5, 5], ["a", "b", "c"], "duplicate_ids", 2),
    ([5, 8], ["a", " \t\n"], "empty_queries", 1),
    ([5, 8], ["a", None], "null_queries", 1),
    ([5, None], ["a", "b"], "null_ids", 1),
])
def test_query_errors_produce_inventory_and_nonzero_exit(dataset, query_ids, queries, field, count):
    paths = dataset(query_ids=query_ids, queries=queries)
    assert main(cli_args(paths)) == 1
    report = json.loads((paths[2] / "inventory.json").read_text())
    assert report["query_file"][field] == count
    assert f"query.{field}={count}" in report["validation_errors"]
    assert (paths[2] / "domain_stats.csv").exists()


@pytest.mark.parametrize("ids,field,count", [
    ([9, 10, 9], "duplicate_ids", 1),
    ([9, 9, 9], "duplicate_ids", 2),
    ([None, 10, None], "null_ids", 2),
])
def test_corpus_id_errors_across_batches_and_row_groups(dataset, ids, field, count):
    paths = dataset(corpus_ids=ids)
    assert main(cli_args(paths)) == 1
    report = json.loads((paths[2] / "inventory.json").read_text())
    assert report["corpus_file"][field] == count
    assert f"corpus.{field}={count}" in report["validation_errors"]


@pytest.mark.parametrize("url,reason", [
    (None, "null_url"), (" \t\n", "empty_url"),
    ("ftp://example.com/a", "unsupported_scheme"),
    ("mailto:doctor@example.com", "unsupported_scheme"),
    ("//example.com/a", "unsupported_scheme"),
    ("https:///path", "missing_hostname"),
    ("http://", "missing_hostname"),
    ("https://[bad", "malformed_url"),
])
def test_invalid_urls_report_original_values_without_failing_run(dataset, url, reason):
    paths = dataset(corpus_ids=[91], urls=[url])
    assert main(cli_args(paths)) == 0
    report = json.loads((paths[2] / "inventory.json").read_text())
    assert report["corpus_file"]["invalid_urls"] == 1
    assert report["corpus_file"]["invalid_url_reasons"] == {reason: 1}
    assert report["corpus_file"]["unique_domains"] == 0
    assert read_csv(paths[2] / "invalid_urls.csv") == [
        {"id": "91", "url": url if url is not None else "", "reason": reason}]


def test_duplicate_urls_preserve_all_official_ids_and_exact_strings(dataset):
    url = "https://example.com/a"
    paths = dataset(corpus_ids=[9007199254740993, 8, 20, 30, 99],
                    urls=[url, "https://example.com/b", url, url, " " + url])
    report = run(paths)
    assert report["corpus_file"]["duplicate_urls"] == 2
    assert report["corpus_file"]["duplicate_ids"] == 0
    assert read_csv(paths[2] / "duplicate_urls.csv") == [
        {"url": url, "id": str(id_), "occurrence_count": "3"}
        for id_ in [20, 30, 9007199254740993]]
    assert read_csv(paths[2] / "domain_stats.csv") == [{"domain": "example.com", "url_count": "5"}]


def test_duplicate_url_with_null_and_duplicate_official_ids_is_not_lost(dataset):
    paths = dataset(corpus_ids=[None, 7, 7], urls=["https://a.vn"] * 3)
    report = run(paths)
    assert report["corpus_file"]["duplicate_urls"] == 2
    rows = read_csv(paths[2] / "duplicate_urls.csv")
    assert [row["id"] for row in rows] == ["", "7", "7"]
    assert all(row["occurrence_count"] == "3" for row in rows)


def test_domain_order_normalization_and_deterministic_repeat(dataset):
    paths = dataset(corpus_ids=[10, 30, 8, 20, 42], urls=[
        "https://B.VN/a", "https://www.A.vn./x", "http://a.vn/y",
        "https://b.vn/b", "https://c.vn"])
    run(paths)
    assert read_csv(paths[2] / "domain_stats.csv") == [
        {"domain": "a.vn", "url_count": "2"},
        {"domain": "b.vn", "url_count": "2"},
        {"domain": "c.vn", "url_count": "1"}]
    original = {name: (paths[2] / name).read_bytes() for name in REPORT_FILES}
    prepare(*paths, batch_size=4, progress=lambda _: None)
    assert original == {name: (paths[2] / name).read_bytes() for name in REPORT_FILES}


def test_keep_www_option(dataset):
    paths = dataset(corpus_ids=[1, 2], urls=["https://WWW.EXAMPLE.COM./a", "https://example.com/b"])
    run(paths, strip_www=False)
    assert read_csv(paths[2] / "domain_stats.csv") == [
        {"domain": "example.com", "url_count": "1"},
        {"domain": "www.example.com", "url_count": "1"}]


def test_sha256_stable_and_bounded_reads(dataset, monkeypatch):
    paths = dataset()
    expected = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def read(self, size=-1):
            assert 0 < size <= 17
            return self.handle.read(size)
    original_open = Path.open
    def open_bounded(path, *args, **kwargs):
        return BoundedReader(original_open(path, *args, **kwargs))
    monkeypatch.setattr(Path, "open", open_bounded)
    assert sha256_file(paths[0], block_size=17) == expected
    assert sha256_file(paths[0], block_size=17) == expected


@pytest.mark.parametrize("kwargs", [
    {"query_id_type": pa.float64()}, {"query_id_type": pa.string(), "query_ids": ["1", "2"]},
    {"query_type": pa.int64(), "queries": [1, 2]},
    {"corpus_id_type": pa.int32(), "corpus_ids": [1, 2, 3]},
    {"url_type": pa.int64(), "urls": [1, 2, 3]},
])
def test_wrong_schema_fails_without_publishing(dataset, kwargs):
    paths = dataset(**kwargs)
    paths[2].mkdir()
    sentinel = paths[2] / "inventory.json"
    sentinel.write_text("old report")
    assert main(cli_args(paths)) == 2
    assert sentinel.read_text() == "old report"
    assert not list(paths[2].glob(".inventory-*"))


def test_extra_column_rejected(dataset):
    paths = dataset()
    table = pq.read_table(paths[0]).append_column("unexpected", pa.array([1, 2]))
    pq.write_table(table, paths[0])
    with pytest.raises(ValueError, match="schema columns"):
        run(paths)
    assert not (paths[2] / "inventory.json").exists()


@pytest.mark.parametrize("id_type,ids", [(pa.int32(), [-7, 45]), (pa.uint64(), [1, 2**64 - 1])])
def test_query_accepts_integer_types_without_coercing_ids(dataset, id_type, ids):
    report = run(dataset(query_ids=ids, query_id_type=id_type))
    assert report["query_file"]["id_min"] == min(ids)
    assert report["query_file"]["id_max"] == max(ids)


def test_empty_files(dataset):
    report = run(dataset(query_ids=[], queries=[], corpus_ids=[], urls=[]))
    for name in ["query_file", "corpus_file"]:
        assert report[name]["rows"] == 0
        assert report[name]["id_min"] is report[name]["id_max"] is None
        assert report[name]["duplicate_ids"] == 0
    assert report["validation_errors"] == []


def test_corpus_is_streamed_and_sources_stay_unchanged(dataset, monkeypatch):
    paths = dataset()
    before = [sha256_file(path) for path in paths[:2]]
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
    def open_parquet(path):
        return StreamingCorpus(path) if Path(path) == paths[1] else original(path)
    monkeypatch.setattr(pq, "ParquetFile", open_parquet)
    messages = []
    prepare(*paths, batch_size=1, progress=messages.append)
    assert len([message for message in messages if message.startswith("Corpus batch")]) == 3
    assert [sha256_file(path) for path in paths[:2]] == before


def test_interrupted_scan_preserves_existing_reports_and_cleans_scratch(dataset):
    paths = dataset()
    run(paths)
    before = {name: (paths[2] / name).read_bytes() for name in REPORT_FILES}
    def interrupt(message):
        if message.startswith("Corpus batch"):
            raise RuntimeError("Interrupted")
    with pytest.raises(RuntimeError, match="Interrupted"):
        prepare(*paths, batch_size=1, progress=interrupt)
    assert before == {name: (paths[2] / name).read_bytes() for name in REPORT_FILES}
    assert not list(paths[2].glob(".inventory-*"))
