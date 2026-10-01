"""Batch crawler tests use only httpx.MockTransport and tiny local Parquet files."""

import hashlib
import socket
from collections import Counter

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.crawl_vibiomir_batch import (
    crawl_batch,
    detect_file_type,
    interleave_by_domain,
)


def _sample(tmp_path, urls):
    rows = [{"id": official_id, "url": url,
             "domain": url.split("/")[2], "domain_url_count": 10}
            for official_id, url in urls]
    path = tmp_path / "sample_urls.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _run(tmp_path, sample, handler, **kwargs):
    return crawl_batch(sample, tmp_path / "crawl", limit=pq.read_table(sample).num_rows,
                       concurrency=kwargs.pop("concurrency", 2), delay_seconds=0,
                       client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
                       sleep=kwargs.pop("sleep", lambda _: None), progress=lambda _: None,
                       **kwargs)


def _results(tmp_path):
    return pq.read_table(tmp_path / "crawl/crawl_results.parquet").to_pylist()


def test_html_pdf_text_magic_and_official_ids(tmp_path):
    sample = _sample(tmp_path, [(9001, "https://a.test/a"), (7, "https://b.test/b"),
                                (830, "https://c.test/c")])
    bodies = {"a.test": (b"<html>Medical content</html>", "application/octet-stream"),
              "b.test": (b"%PDF-1.4\nPDF body", "text/plain"),
              "c.test": (b"Plain medical text", "text/plain")}
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        body, mime = bodies[request.url.host]
        return httpx.Response(200, content=body, headers={"content-type": mime})
    summary = _run(tmp_path, sample, handler)
    assert summary["success_count"] == 3
    rows = {r["id"]: r for r in _results(tmp_path)}
    assert set(rows) == {9001, 7, 830}
    assert rows[9001]["raw_file"] == "raw/html/9001.html"
    assert rows[7]["raw_file"] == "raw/pdf/7.pdf"
    assert rows[830]["raw_file"] == "raw/text/830.txt"
    for row in rows.values():
        body = (tmp_path / "crawl" / row["raw_file"]).read_bytes()
        assert row["content_sha256"] == hashlib.sha256(body).hexdigest()
        assert row["bytes_downloaded"] == len(body)


def test_redirect_robots_cache_and_resume(tmp_path):
    sample = _sample(tmp_path, [(1, "https://a.test/first"),
                                (2, "https://a.test/second")])
    calls = Counter()
    def handler(request):
        calls[str(request.url)] += 1
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/first":
            return httpx.Response(302, headers={"location": "https://b.test/final"})
        return httpx.Response(200, content=b"<html>Done</html>",
                              headers={"content-type": "text/html"})
    _run(tmp_path, sample, handler)
    assert calls["https://a.test/robots.txt"] == 1
    assert calls["https://b.test/robots.txt"] == 1
    assert {r["redirect_count"] for r in _results(tmp_path)} == {0, 1}
    before = calls.copy()
    summary = _run(tmp_path, sample, handler)
    assert summary["skipped_due_to_resume"] == 2
    assert calls == before


def test_retry_failed_and_valid_success_not_fetched(tmp_path):
    sample = _sample(tmp_path, [(1, "https://a.test/one"),
                                (2, "https://a.test/two")])
    calls = Counter()
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls[request.url.path] += 1
        if request.url.path == "/two" and calls["/two"] <= 2:
            return httpx.Response(500)
        return httpx.Response(200, content=b"<html>Recovered</html>",
                              headers={"content-type": "text/html"})
    first = _run(tmp_path, sample, handler)
    assert first["status_counts"] == {"http_error": 1, "success": 1}
    assert calls["/two"] == 2
    second = _run(tmp_path, sample, handler, retry_failed=True)
    assert second["success_count"] == 2
    assert second["skipped_due_to_resume"] == 1
    assert calls["/one"] == 1 and calls["/two"] == 3


def test_not_found_rate_limited_timeout_and_robots_denied(tmp_path):
    sample = _sample(tmp_path, [(1, "https://a.test/404"), (2, "https://b.test/429"),
                                (3, "https://c.test/timeout"), (4, "https://d.test/blocked")])
    calls = Counter()
    def handler(request):
        calls[request.url.host, request.url.path] += 1
        if request.url.path == "/robots.txt":
            if request.url.host == "d.test":
                return httpx.Response(200, text="User-agent: *\nDisallow: /blocked")
            return httpx.Response(404)
        if request.url.path == "/timeout":
            raise httpx.ReadTimeout("test timeout")
        return httpx.Response(int(request.url.path.strip("/")),
                              headers={"retry-after": "0"})
    summary = _run(tmp_path, sample, handler)
    assert summary["status_counts"] == {"not_found": 1, "rate_limited": 1,
                                        "robots_denied": 1, "timeout": 1}
    assert calls["a.test", "/404"] == 1
    assert calls["b.test", "/429"] == 2
    assert calls["c.test", "/timeout"] == 2
    assert calls["d.test", "/blocked"] == 0


def test_size_limit_unsupported_and_atomic_cleanup(tmp_path):
    sample = _sample(tmp_path, [(1, "https://a.test/large"),
                                (2, "https://b.test/image")])
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/large":
            return httpx.Response(200, content=b"x" * 100,
                                  headers={"content-type": "text/plain"})
        return httpx.Response(200, content=b"\x89PNG\r\n\x1a\nimage",
                              headers={"content-type": "image/png"})
    summary = _run(tmp_path, sample, handler, max_bytes=20)
    assert summary["status_counts"] == {"too_large": 1, "unsupported_content": 1}
    assert not list((tmp_path / "crawl/raw").rglob("*.tmp"))
    assert not list((tmp_path / "crawl/raw").rglob("*.bin"))
    assert all(r["raw_file"] is None for r in _results(tmp_path))


def test_detect_type_and_corrupt_success_reloads(tmp_path):
    assert detect_file_type(b"%PDF-1.7", "text/html") == "pdf"
    assert detect_file_type(b"<html>x", "application/pdf") == "html"
    assert detect_file_type(b"unknown", "application/octet-stream") == "unknown"
    assert detect_file_type(b"\xff\xd8\xffdata", "application/octet-stream") is None
    sample = _sample(tmp_path, [(44, "https://a.test/doc")])
    pages = []
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        pages.append(1)
        return httpx.Response(200, content=b"<html>Original</html>",
                              headers={"content-type": "text/html"})
    _run(tmp_path, sample, handler)
    raw = tmp_path / "crawl/raw/html/44.html"
    raw.write_bytes(b"corrupt")
    summary = _run(tmp_path, sample, handler)
    assert len(pages) == 2 and summary["success_count"] == 1
    assert raw.read_bytes() == b"<html>Original</html>"


def test_retry_after_and_unknown_binary(tmp_path):
    sample = _sample(tmp_path, [(4, "https://a.test/blob")])
    attempts = []
    sleeps = []
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, content=b"raw binary payload",
                              headers={"content-type": "application/octet-stream"})
    result = _run(tmp_path, sample, handler, sleep=sleeps.append)
    assert result["success_count"] == 1
    assert max(sleeps) >= 2
    assert _results(tmp_path)[0]["raw_file"] == "raw/unknown/4.bin"


def test_no_live_socket_in_mock_test(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *_: (_ for _ in ()).throw(
        AssertionError("Live network attempted")))
    sample = _sample(tmp_path, [(99, "https://a.test/doc")])
    def handler(request):
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(
            200, content=b"text", headers={"content-type": "text/plain"})
    assert _run(tmp_path, sample, handler)["success_count"] == 1


def test_interleaving_preserves_each_domain_order():
    entries = [{"id": i, "domain": domain} for i, domain in
               [(1, "a"), (2, "a"), (3, "a"), (4, "b"), (5, "b"), (6, "c")]]
    ordered = interleave_by_domain(entries)
    assert [r["id"] for r in ordered] == [1, 4, 6, 2, 5, 3]


def test_resume_cleans_orphan_raw_without_fetching(tmp_path):
    sample = _sample(tmp_path, [(44, "https://a.test/doc")])
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(
            200, content=b"<html>Article</html>", headers={"content-type": "text/html"})
    _run(tmp_path, sample, handler)
    orphan = tmp_path / "crawl/raw/pdf/44.pdf"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    before = len(calls)
    result = _run(tmp_path, sample, handler)
    assert result["skipped_due_to_resume"] == 1
    assert len(calls) == before
    assert not orphan.exists()
    assert (tmp_path / "crawl/raw/html/44.html").exists()


def test_retry_status_selects_only_robots_unavailable_and_protects_success(tmp_path):
    sample = _sample(tmp_path, [(1, "https://a.test/ok"),
                                (2, "https://b.test/retry"),
                                (3, "https://c.test/blocked"),
                                (4, "https://d.test/page403")])
    def first(request):
        if request.url.path == "/robots.txt":
            if request.url.host == "b.test":
                return httpx.Response(521)
            if request.url.host == "c.test":
                return httpx.Response(200, text="User-agent: *\nDisallow: /blocked")
            return httpx.Response(404)
        if request.url.host == "d.test":
            return httpx.Response(403)
        return httpx.Response(200, content=b"<html>Keep this raw</html>",
                              headers={"content-type": "text/html"})
    _run(tmp_path, sample, first)
    old_raw = tmp_path / "crawl/raw/html/1.html"
    old_sha = hashlib.sha256(old_raw.read_bytes()).hexdigest()
    calls = []
    def second(request):
        calls.append(str(request.url))
        if request.url.host != "b.test":
            raise AssertionError("Only robots_unavailable may be retried")
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=b"<html>New raw</html>",
                              headers={"content-type": "text/html"})
    summary = _run(tmp_path, sample, second, retry_status={"robots_unavailable"},
                   expected_retry_count=1)
    assert summary["success_count"] == 2
    assert summary["skipped_due_to_resume"] == 3
    rows = {r["id"]: r for r in _results(tmp_path)}
    assert rows[2]["previous_status"] == "robots_unavailable"
    assert rows[2]["robots_policy_status"] == "unavailable_allow"
    assert rows[3]["status"] == "robots_denied"
    assert hashlib.sha256(old_raw.read_bytes()).hexdigest() == old_sha
    assert calls == ["https://b.test/robots.txt", "https://b.test/retry"]
    assert (tmp_path / "crawl/pre_retry_success_sha256.json").exists()


def test_retry_expected_count_guard_does_not_request(tmp_path):
    sample = _sample(tmp_path, [(1, "https://a.test/doc")])
    _run(tmp_path, sample, lambda request: httpx.Response(521))
    def blocked(request):
        raise AssertionError(f"Unexpected request: {request.url}")
    import pytest
    with pytest.raises(ValueError, match="Expected 2 retry URLs"):
        _run(tmp_path, sample, blocked, retry_status={"robots_unavailable"},
             expected_retry_count=2)
